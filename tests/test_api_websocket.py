"""API and WebSocket telemetry integration tests via FastAPI TestClient."""

from fastapi.testclient import TestClient

from main import app


def test_root_ui_and_health_hardware_lock():
    with TestClient(app) as client:
        r_ui = client.get("/")
        assert r_ui.status_code == 200
        assert "SWARMFORGE" in r_ui.text

        r_health = client.get("/health")
        assert r_health.status_code == 200
        data = r_health.json()
        assert data["status"] == "ok"
        assert data["hardware_actuation_locked"] is True


def test_hardware_actuation_endpoint_blocked_with_403():
    with TestClient(app) as client:
        r = client.post(
            "/api/hardware/actuate",
            json={"command": "ARM_MOTORS", "target_uav_id": "uav-01", "throttle_pct": 50.0},
        )
        assert r.status_code == 403
        detail = r.json()["detail"]
        assert detail["error"] == "HARDWARE_ACTUATION_LOCKED"
        assert detail["hardware_actuation_locked"] is True


def test_missions_list_load_export_and_compare_planners():
    with TestClient(app) as client:
        r_list = client.get("/api/missions")
        assert r_list.status_code == 200
        missions = r_list.json()["missions"]
        assert len(missions) >= 3

        r_load = client.post("/api/missions/load", json={"mission_id": "head_on_crossing"})
        assert r_load.status_code == 200
        loaded = r_load.json()
        assert loaded["mission"]["mission_id"] == "head_on_crossing"
        assert len(loaded["telemetry"]["uavs"]) == 4

        r_export = client.get("/api/missions/export")
        assert r_export.status_code == 200
        exported = r_export.json()
        assert exported["mission_id"] == "head_on_crossing"

        # Re-import exported mission JSON cleanly
        r_reimport = client.post("/api/missions/load", json=exported)
        assert r_reimport.status_code == 200
        assert r_reimport.json()["mission"]["mission_id"] == "head_on_crossing"

        # Planner comparison endpoint using active mission obstacles
        r_cmp = client.post(
            "/api/planners/compare?use_mission_obstacles=true",
            json={
                "start": {"x": -48.0, "y": 14.0, "z": 18.0},
                "goal": {"x": 48.0, "y": -14.0, "z": 22.0},
                "clearance_m": 2.5,
            },
        )
        assert r_cmp.status_code == 200
        cmp_data = r_cmp.json()
        assert len(cmp_data["results"]) == 3
        assert {m["algorithm"] for m in cmp_data["results"]} == {"astar", "rrt_star", "apf"}
        assert all(m["success"] for m in cmp_data["results"])


def test_simulation_controls_and_websocket_telemetry_stream():
    with TestClient(app) as client:
        client.post("/api/missions/load", json={"mission_id": "urban_corridor"})
        client.post("/api/simulation/reset")

        # REST step + wind + comm-loss + emergency
        r_wind = client.post(
            "/api/simulation/wind",
            json={"steady_vx_mps": 3.5, "steady_vy_mps": -1.5, "turbulence_intensity": 0.9},
        )
        assert r_wind.status_code == 200
        assert r_wind.json()["wind_vector_mps"]["x"] != 0.0

        r_step = client.post("/api/simulation/step", json={"steps": 15})
        assert r_step.status_code == 200
        assert r_step.json()["tick"] == 15

        # WebSocket /ws/telemetry live stream + interactive command over WS
        with client.websocket_connect("/ws/telemetry") as ws:
            initial_frame = ws.receive_json()
            assert initial_frame["hardware_actuation_locked"] is True
            assert initial_frame["mission_id"] == "urban_corridor"
            assert initial_frame["tick"] == 15
            assert len(initial_frame["uavs"]) == 3

            # Trigger step command via WebSocket and verify updated telemetry frame
            ws.send_json({"command": "step", "steps": 10})
            stepped_frame = ws.receive_json()
            assert stepped_frame["tick"] == 25
            assert stepped_frame["sim_time_s"] > initial_frame["sim_time_s"]

            # Trigger comm-loss over WebSocket
            ws.send_json({"command": "comm_loss", "active": True, "action": "RTL"})
            comm_frame = ws.receive_json()
            assert comm_frame["comm_loss_active"] is True
            assert all(u["mode"] == "COMM_LOSS_RTL" for u in comm_frame["uavs"])

            # Live start over WebSocket streams advancing frames automatically
            ws.send_json({"command": "comm_loss", "active": False})
            _ = ws.receive_json()
            ws.send_json({"command": "start"})
            running_frame = ws.receive_json()
            assert running_frame["status"] == "RUNNING"

        # Switch planner live and test emergency lifecycle via REST
        r_plan = client.post(
            "/api/simulation/planner",
            json={"algorithm": "rrt_star", "uav_id": "uav-01"},
        )
        assert r_plan.status_code == 200
        u1 = next(u for u in r_plan.json()["uavs"] if u["uav_id"] == "uav-01")
        assert u1["planner_type"] == "rrt_star"

        r_em = client.post("/api/simulation/emergency", json={"command": "LAND"})
        assert r_em.status_code == 200
        assert r_em.json()["emergency_active"] is True
        assert all(u["mode"] in ("EMERGENCY_LAND", "LANDED") for u in r_em.json()["uavs"])

        r_pause = client.post("/api/simulation/pause")
        assert r_pause.status_code == 200
        assert r_pause.json()["status"] == "PAUSED"

