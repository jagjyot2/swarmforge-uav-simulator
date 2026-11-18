"""Stateful SwarmSimulator orchestrating mission lifecycle, planning, safety, and telemetry."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np

from .models import (
    MissionConfig,
    PlannerType,
    SimStatus,
    TelemetryFrame,
    Vec3,
    WindConfig,
)
from .path_planners import plan_waypoint_chain
from .safety import (
    HARDWARE_ACTUATION_LOCKED,
    resolve_comm_loss_mode,
    resolve_emergency_mode,
)
from .uav_dynamics import DrydenWindModel, UAVState, step_swarm_physics


DEFAULT_MISSIONS_DIR = Path(__file__).resolve().parent.parent / "sample_missions"


class SwarmSimulator:
    """Thread-safe multi-UAV Digital Twin simulation coordinator."""

    def __init__(self, missions_dir: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        self.missions_dir = missions_dir or DEFAULT_MISSIONS_DIR
        self.mission: MissionConfig = self._load_default_mission()
        self.wind_model = DrydenWindModel(self.mission.wind)
        self.last_wind_vec = np.array(
            [
                self.mission.wind.steady_vx_mps,
                self.mission.wind.steady_vy_mps,
                self.mission.wind.steady_vz_mps,
            ],
            dtype=np.float64,
        )
        self.states: List[UAVState] = []
        self.sim_time_s: float = 0.0
        self.tick: int = 0
        self.status: SimStatus = "STOPPED"
        self.comm_loss_active: bool = False
        self.emergency_active: bool = False
        self.events: List[str] = []
        self._build_swarm_states()

    def _load_default_mission(self) -> MissionConfig:
        urban_file = self.missions_dir / "urban_corridor.json"
        if urban_file.exists():
            return MissionConfig.model_validate_json(urban_file.read_text(encoding="utf-8"))
        return MissionConfig()

    def list_sample_missions(self) -> List[Dict[str, object]]:
        """List available preset sample mission templates from disk."""
        templates: List[Dict[str, object]] = []
        if not self.missions_dir.exists():
            return templates
        for path in sorted(self.missions_dir.glob("*.json")):
            try:
                cfg = MissionConfig.model_validate_json(path.read_text(encoding="utf-8"))
                templates.append(
                    {
                        "mission_id": cfg.mission_id,
                        "name": cfg.name,
                        "description": cfg.description,
                        "uav_count": len(cfg.uavs),
                        "obstacle_count": len(cfg.obstacles),
                        "filename": path.name,
                    }
                )
            except Exception:
                continue
        return templates

    def load_sample_mission(self, mission_id: str) -> MissionConfig:
        """Load a preset sample mission by mission_id (or filename stem)."""
        if self.missions_dir.exists():
            for path in sorted(self.missions_dir.glob("*.json")):
                cfg = MissionConfig.model_validate_json(path.read_text(encoding="utf-8"))
                if cfg.mission_id == mission_id or path.stem == mission_id:
                    return self.load_mission(cfg)
        raise FileNotFoundError(f"Sample mission '{mission_id}' not found.")

    def load_mission(self, config: MissionConfig) -> MissionConfig:
        """Load a validated MissionConfig, plan all UAV routes, and reset simulation state."""
        with self._lock:
            self.mission = config.model_copy(deep=True)
            self.wind_model = DrydenWindModel(self.mission.wind)
            self.last_wind_vec = np.array(
                [
                    self.mission.wind.steady_vx_mps,
                    self.mission.wind.steady_vy_mps,
                    self.mission.wind.steady_vz_mps,
                ],
                dtype=np.float64,
            )
            self.sim_time_s = 0.0
            self.tick = 0
            self.status = "STOPPED"
            self.comm_loss_active = False
            self.emergency_active = False
            self.events = [f"Loaded mission '{self.mission.name}' ({len(self.mission.uavs)} UAVs)."]
            self._build_swarm_states()
            return self.mission.model_copy(deep=True)

    def _build_swarm_states(self) -> None:
        self.states = []
        for idx, uav_cfg in enumerate(self.mission.uavs):
            raw_wps = [w.to_np() for w in uav_cfg.waypoints]
            planned = plan_waypoint_chain(
                start=uav_cfg.start_position.to_np(),
                waypoints=raw_wps,
                obstacles=self.mission.obstacles,
                algorithm=uav_cfg.planner_type,
                clearance_m=self.mission.obstacle_clearance_m,
                geofence=self.mission.geofence,
                seed=42 + idx * 17,
            )
            self.states.append(UAVState.from_config(uav_cfg, planned_path=planned))

    def start(self) -> TelemetryFrame:
        """Start or resume real-time simulation."""
        with self._lock:
            if self.status in ("STOPPED", "PAUSED"):
                self.status = "RUNNING"
                self._append_event("Simulation started.")
            return self.get_telemetry_frame()

    def pause(self) -> TelemetryFrame:
        """Pause real-time simulation."""
        with self._lock:
            if self.status == "RUNNING":
                self.status = "PAUSED"
                self._append_event("Simulation paused.")
            return self.get_telemetry_frame()

    def reset(self) -> TelemetryFrame:
        """Reset simulation time, wind turbulence, and UAV states to initial mission conditions."""
        with self._lock:
            self.sim_time_s = 0.0
            self.tick = 0
            self.status = "STOPPED"
            self.comm_loss_active = False
            self.emergency_active = False
            self.wind_model.reset()
            self.last_wind_vec = np.array(
                [
                    self.mission.wind.steady_vx_mps,
                    self.mission.wind.steady_vy_mps,
                    self.mission.wind.steady_vz_mps,
                ],
                dtype=np.float64,
            )
            self.events = [f"Reset mission '{self.mission.name}'."]
            self._build_swarm_states()
            return self.get_telemetry_frame()

    def step(self, steps: int = 1) -> TelemetryFrame:
        """Advance simulation deterministically by `steps` physics ticks."""
        with self._lock:
            n_steps = max(1, min(int(steps), 2000))
            dt = self.mission.dt_s
            for _ in range(n_steps):
                self.last_wind_vec = self.wind_model.step(dt)
                tick_events = step_swarm_physics(
                    states=self.states,
                    dt_s=dt,
                    wind_vec=self.last_wind_vec,
                    obstacles=self.mission.obstacles,
                    geofence=self.mission.geofence,
                    min_separation_m=self.mission.min_separation_m,
                    obstacle_clearance_m=self.mission.obstacle_clearance_m,
                )
                for ev in tick_events:
                    self._append_event(ev)
                self.tick += 1
                self.sim_time_s += dt
            return self.get_telemetry_frame()

    def update_wind(self, wind: WindConfig) -> TelemetryFrame:
        """Update wind velocity vector and Dryden turbulence intensity live."""
        with self._lock:
            self.mission.wind = wind
            self.wind_model.update_config(wind)
            self.last_wind_vec = np.array(
                [wind.steady_vx_mps, wind.steady_vy_mps, wind.steady_vz_mps],
                dtype=np.float64,
            )
            self._append_event(
                f"Wind updated: ({wind.steady_vx_mps:.1f}, {wind.steady_vy_mps:.1f}, "
                f"{wind.steady_vz_mps:.1f}) m/s, turb={wind.turbulence_intensity:.2f}"
            )
            return self.get_telemetry_frame()

    def set_comm_loss(
        self,
        active: bool,
        action: Optional[Literal["HOLD", "RTL"]] = None,
        uav_ids: Optional[List[str]] = None,
    ) -> TelemetryFrame:
        """Simulate GCS communication link loss (autonomous Hold/RTL) or link restoration."""
        with self._lock:
            eff_action = action or self.mission.comm_loss_action
            self.mission.comm_loss_action = eff_action
            self.comm_loss_active = active
            targets = set(uav_ids) if uav_ids else None

            for st in self.states:
                if targets is not None and st.config.uav_id not in targets:
                    continue
                if active:
                    st.comm_link_active = False
                    st.mode = resolve_comm_loss_mode(st.mode, eff_action)
                    if st.mode == "COMM_LOSS_HOLD":
                        st.hold_anchor = st.position.copy()
                else:
                    st.comm_link_active = True
                    if st.mode in ("COMM_LOSS_HOLD", "COMM_LOSS_RTL"):
                        st.mode = (
                            "MISSION"
                            if st.active_waypoint_idx < len(st.planned_path)
                            else "HOLD"
                        )
            state_lbl = f"ACTIVE ({eff_action})" if active else "RESTORED"
            self._append_event(f"Communication link {state_lbl}.")
            return self.get_telemetry_frame()

    def trigger_emergency(
        self, command: Literal["RTL", "LAND", "CLEAR"]
    ) -> TelemetryFrame:
        """Trigger swarm-wide emergency RTL / Landing or clear emergency override."""
        with self._lock:
            if command == "CLEAR":
                self.emergency_active = False
                for st in self.states:
                    if st.mode in ("RTL", "EMERGENCY_LAND") and st.battery_soc_pct > st.config.rtl_soc_threshold_pct:
                        st.mode = (
                            "MISSION"
                            if st.active_waypoint_idx < len(st.planned_path)
                            else "HOLD"
                        )
                self._append_event("Emergency override cleared.")
                return self.get_telemetry_frame()

            self.emergency_active = True
            for st in self.states:
                st.mode = resolve_emergency_mode(st.mode, command)
            self._append_event(f"EMERGENCY COMMAND ISSUED: {command}.")
            return self.get_telemetry_frame()

    def set_planner(
        self, algorithm: PlannerType, uav_id: Optional[str] = None
    ) -> TelemetryFrame:
        """Switch path planner algorithm live for one UAV or the whole swarm and replan."""
        with self._lock:
            for idx, st in enumerate(self.states):
                if uav_id is not None and st.config.uav_id != uav_id:
                    continue
                st.config.planner_type = algorithm
                if idx < len(self.mission.uavs):
                    self.mission.uavs[idx].planner_type = algorithm
                raw_wps = [w.to_np() for w in st.config.waypoints]
                st.planned_path = plan_waypoint_chain(
                    start=st.position.copy(),
                    waypoints=raw_wps,
                    obstacles=self.mission.obstacles,
                    algorithm=algorithm,
                    clearance_m=self.mission.obstacle_clearance_m,
                    geofence=self.mission.geofence,
                    seed=42 + idx * 17,
                )
                st.active_waypoint_idx = 0
                if st.mode == "HOLD" and st.planned_path:
                    st.mode = "MISSION"
            scope = uav_id or "ALL UAVs"
            self._append_event(f"Replanned {scope} with {algorithm.upper()}.")
            return self.get_telemetry_frame()

    def export_mission_json(self) -> str:
        """Export active MissionConfig as formatted JSON."""
        with self._lock:
            return json.dumps(self.mission.model_dump(mode="json"), indent=2)

    def get_telemetry_frame(self) -> TelemetryFrame:
        """Return synchronized TelemetryFrame snapshot."""
        with self._lock:
            return TelemetryFrame(
                sim_time_s=round(self.sim_time_s, 3),
                tick=self.tick,
                status=self.status,
                mission_id=self.mission.mission_id,
                mission_name=self.mission.name,
                hardware_actuation_locked=HARDWARE_ACTUATION_LOCKED,
                safety_banner="SIMULATION ONLY — HARDWARE ACTUATION LOCKED",
                wind_vector_mps=Vec3.from_np(self.last_wind_vec),
                comm_loss_active=self.comm_loss_active,
                emergency_active=self.emergency_active,
                geofence=self.mission.geofence,
                obstacles=self.mission.obstacles,
                uavs=[st.to_telemetry(self.last_wind_vec) for st in self.states],
                events=list(self.events[-12:]),
            )

    def _append_event(self, text: str) -> None:
        stamp = f"[{self.sim_time_s:06.2f}s] {text}"
        self.events.append(stamp)
        if len(self.events) > 40:
            self.events = self.events[-40:]
