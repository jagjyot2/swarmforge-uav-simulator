"""SwarmForge FastAPI + WebSocket 3D Multi-UAV Digital Twin Server."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sim_engine.models import (  # noqa: E402
    MissionConfig,
    Obstacle3D,
    PlannerComparisonRequest,
    PlannerComparisonResult,
    PlannerMetrics,
    PlannerType,
    TelemetryFrame,
    Vec3,
    WindConfig,
)
from sim_engine.path_planners import (  # noqa: E402
    compare_planners,
    evaluate_path_metrics,
    plan_apf_3d,
    plan_astar_3d,
    plan_rrt_star_3d,
)
from sim_engine.safety import (  # noqa: E402
    HARDWARE_ACTUATION_LOCKED,
    HardwareActuationBlockedError,
    assert_hardware_actuation_locked,
)
from sim_engine.simulator import SwarmSimulator  # noqa: E402


STATIC_DIR = BACKEND_DIR / "static"
FRONTEND_DIST_DIR = BACKEND_DIR.parent / "frontend" / "dist"


class StepRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=2000)


class CommLossRequest(BaseModel):
    active: bool = True
    action: Optional[Literal["HOLD", "RTL"]] = None
    uav_ids: Optional[List[str]] = None


class EmergencyRequest(BaseModel):
    command: Literal["RTL", "LAND", "CLEAR"] = "RTL"


class PlannerSwitchRequest(BaseModel):
    algorithm: PlannerType = "astar"
    uav_id: Optional[str] = None


class SinglePlanRequest(BaseModel):
    algorithm: Literal["astar", "rrt_star", "apf"] = "astar"
    start: Vec3
    goal: Vec3
    obstacles: List[Obstacle3D] = Field(default_factory=list)
    clearance_m: float = Field(default=2.5, ge=0.5)
    seed: int = 42


class MissionIdPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mission_id: str


class HardwareActuateRequest(BaseModel):
    command: str = "ARM_MOTORS"
    target_uav_id: str = "uav-01"
    throttle_pct: float = Field(default=0.0, ge=0.0, le=100.0)


simulator = SwarmSimulator()


async def _simulation_loop(sim: SwarmSimulator) -> None:
    """Background physics loop stepping the swarm at dt_s while RUNNING."""
    while True:
        dt = sim.mission.dt_s
        if sim.status == "RUNNING":
            sim.step(1)
        await asyncio.sleep(dt)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(_simulation_loop(simulator))
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="SwarmForge Multi-UAV Digital Twin API",
    version="1.0.0",
    description=(
        "Autonomous 3D multi-UAV path planning, physics simulation, collision avoidance, "
        "geofencing, and WebSocket telemetry digital twin. HARDWARE_ACTUATION_LOCKED=True."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if (FRONTEND_DIST_DIR / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")),
        name="frontend-assets",
    )


@app.get("/", response_class=HTMLResponse)
async def serve_digital_twin_ui() -> HTMLResponse:
    """Serve the interactive 3D WebGL Digital Twin UI."""
    built_index = FRONTEND_DIST_DIR / "index.html"
    static_index = STATIC_DIR / "index.html"
    if built_index.exists():
        return HTMLResponse(content=built_index.read_text(encoding="utf-8"))
    if static_index.exists():
        return HTMLResponse(content=static_index.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>SwarmForge Digital Twin API</h1><p>UI static bundle missing.</p>",
        status_code=200,
    )


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Liveness and hardware-lock verification endpoint."""
    return {
        "status": "ok",
        "service": "swarmforge-uav-simulator",
        "hardware_actuation_locked": HARDWARE_ACTUATION_LOCKED,
        "sim_status": simulator.status,
        "mission_id": simulator.mission.mission_id,
    }


@app.get("/api/missions")
async def list_missions() -> Dict[str, Any]:
    """List sample mission presets and currently loaded mission ID."""
    return {
        "active_mission_id": simulator.mission.mission_id,
        "missions": simulator.list_sample_missions(),
    }


@app.get("/api/missions/current", response_model=MissionConfig)
async def get_current_mission() -> MissionConfig:
    """Return active MissionConfig JSON."""
    return simulator.mission


@app.get("/api/missions/export")
async def export_mission() -> Response:
    """Export active mission configuration as downloadable JSON."""
    payload = simulator.export_mission_json()
    filename = f"{simulator.mission.mission_id}.json"
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/missions/load")
async def load_mission(
    payload: Union[MissionIdPayload, MissionConfig]
) -> Dict[str, Any]:
    """Import a MissionConfig JSON or load a preset mission by mission_id."""
    if isinstance(payload, MissionIdPayload):
        try:
            loaded = simulator.load_sample_mission(payload.mission_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    else:
        loaded = simulator.load_mission(payload)
    return {
        "loaded": True,
        "mission": loaded.model_dump(mode="json"),
        "telemetry": simulator.get_telemetry_frame().model_dump(mode="json"),
    }


@app.post("/api/planners/compare", response_model=PlannerComparisonResult)
async def api_compare_planners(
    request: PlannerComparisonRequest,
    use_mission_obstacles: bool = Query(default=False),
) -> PlannerComparisonResult:
    """Benchmark 3D A*, RRT*, and APF side-by-side on the provided or active environment."""
    if use_mission_obstacles or (not request.obstacles and simulator.mission.obstacles):
        req = request.model_copy(
            update={
                "obstacles": request.obstacles or simulator.mission.obstacles,
                "geofence": request.geofence or simulator.mission.geofence,
            }
        )
    else:
        req = request
    return compare_planners(req)


@app.post("/api/planners/plan", response_model=PlannerMetrics)
async def api_plan_single(request: SinglePlanRequest) -> PlannerMetrics:
    """Compute a 3D trajectory with one specified planner (astar, rrt_star, or apf)."""
    obs = request.obstacles or simulator.mission.obstacles
    gf = simulator.mission.geofence
    s = request.start.to_np()
    g = request.goal.to_np()
    t0 = time.perf_counter()
    if request.algorithm == "astar":
        ok, pts, note = plan_astar_3d(s, g, obs, request.clearance_m, gf)
    elif request.algorithm == "rrt_star":
        ok, pts, note = plan_rrt_star_3d(s, g, obs, request.clearance_m, gf, seed=request.seed)
    else:
        ok, pts, note = plan_apf_3d(s, g, obs, request.clearance_m, gf)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return evaluate_path_metrics(request.algorithm, ok, pts, dt_ms, obs, notes=note)


@app.get("/api/simulation/state", response_model=TelemetryFrame)
async def get_simulation_state() -> TelemetryFrame:
    """Return latest swarm TelemetryFrame."""
    return simulator.get_telemetry_frame()


@app.post("/api/simulation/start", response_model=TelemetryFrame)
async def start_simulation() -> TelemetryFrame:
    """Start or resume real-time swarm simulation."""
    return simulator.start()


@app.post("/api/simulation/pause", response_model=TelemetryFrame)
async def pause_simulation() -> TelemetryFrame:
    """Pause swarm simulation."""
    return simulator.pause()


@app.post("/api/simulation/reset", response_model=TelemetryFrame)
async def reset_simulation() -> TelemetryFrame:
    """Reset swarm to mission start conditions."""
    return simulator.reset()


@app.post("/api/simulation/step", response_model=TelemetryFrame)
async def step_simulation(req: StepRequest) -> TelemetryFrame:
    """Advance simulation deterministically by N physics ticks."""
    return simulator.step(req.steps)


@app.post("/api/simulation/wind", response_model=TelemetryFrame)
async def update_simulation_wind(wind: WindConfig) -> TelemetryFrame:
    """Update steady wind vector and Dryden turbulence intensity live."""
    return simulator.update_wind(wind)


@app.post("/api/simulation/comm-loss", response_model=TelemetryFrame)
async def simulate_comm_loss(req: CommLossRequest) -> TelemetryFrame:
    """Trigger or restore communication link drop (autonomous Hold/RTL failsafe)."""
    return simulator.set_comm_loss(
        active=req.active, action=req.action, uav_ids=req.uav_ids
    )


@app.post("/api/simulation/emergency", response_model=TelemetryFrame)
async def simulate_emergency(req: EmergencyRequest) -> TelemetryFrame:
    """Trigger swarm-wide emergency RTL / Landing or clear override."""
    return simulator.trigger_emergency(req.command)


@app.post("/api/simulation/planner", response_model=TelemetryFrame)
async def switch_swarm_planner(req: PlannerSwitchRequest) -> TelemetryFrame:
    """Switch path planner algorithm live for a UAV or the entire swarm."""
    return simulator.set_planner(req.algorithm, uav_id=req.uav_id)


@app.post("/api/hardware/actuate")
@app.post("/api/hardware/arm")
async def hardware_actuation_guard(req: HardwareActuateRequest) -> JSONResponse:
    """Permanent safety lock guard blocking any physical hardware actuation attempts."""
    try:
        assert_hardware_actuation_locked(f"{req.command}:{req.target_uav_id}")
    except HardwareActuationBlockedError as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "HARDWARE_ACTUATION_LOCKED",
                "hardware_actuation_locked": HARDWARE_ACTUATION_LOCKED,
                "message": str(exc),
            },
        ) from exc
    return JSONResponse(status_code=403, content={"hardware_actuation_locked": True})


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket) -> None:
    """Stream live TelemetryFrame updates at ~20 Hz and accept interactive control commands."""
    await websocket.accept()
    last_tick = -1
    last_status = ""
    idle_cycles = 0

    try:
        # Send immediate snapshot on connect
        initial = simulator.get_telemetry_frame()
        last_tick = initial.tick
        last_status = initial.status
        await websocket.send_json(initial.model_dump(mode="json"))

        while True:
            try:
                raw_msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                msg = json.loads(raw_msg)
                cmd = msg.get("command")
                if cmd == "start":
                    simulator.start()
                elif cmd == "pause":
                    simulator.pause()
                elif cmd == "reset":
                    simulator.reset()
                elif cmd == "step":
                    simulator.step(int(msg.get("steps", 1)))
                elif cmd == "wind" and isinstance(msg.get("wind"), dict):
                    simulator.update_wind(WindConfig.model_validate(msg["wind"]))
                elif cmd == "comm_loss":
                    simulator.set_comm_loss(
                        active=bool(msg.get("active", True)),
                        action=msg.get("action"),
                        uav_ids=msg.get("uav_ids"),
                    )
                elif cmd == "emergency":
                    simulator.trigger_emergency(msg.get("action", "RTL"))
                elif cmd == "planner":
                    simulator.set_planner(
                        algorithm=msg.get("algorithm", "astar"),
                        uav_id=msg.get("uav_id"),
                    )

                frame = simulator.get_telemetry_frame()
                last_tick = frame.tick
                last_status = frame.status
                idle_cycles = 0
                await websocket.send_json(frame.model_dump(mode="json"))
            except asyncio.TimeoutError:
                frame = simulator.get_telemetry_frame()
                idle_cycles += 1
                if (
                    frame.status == "RUNNING"
                    or frame.tick != last_tick
                    or frame.status != last_status
                    or idle_cycles >= 5
                ):
                    last_tick = frame.tick
                    last_status = frame.status
                    idle_cycles = 0
                    await websocket.send_json(frame.model_dump(mode="json"))
    except WebSocketDisconnect:
        return
