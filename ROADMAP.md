# SwarmForge Engineering Roadmap

## v1.0 — Core Digital Twin & Benchmark Suite (Current Release)
- [x] Pydantic v2 domain schemas (`UAVConfig`, `Obstacle3D`, `Geofence`, `WindConfig`, `MissionConfig`, `TelemetryFrame`, `PlannerComparisonResult`)
- [x] 3D A* voxel grid planner with vectorized occupancy + Line-of-Sight shortcut smoothing
- [x] 3D RRT* sampling planner with rewiring & continuous segment obstacle checking
- [x] 3D Artificial Potential Field (APF) planner with tangential curl local-minimum escape
- [x] Quantitative `compare_planners()` engine (length, compute time, smoothness, clearance, energy)
- [x] Multi-UAV physics engine (cascaded PID, attitude estimation, Dryden wind/turbulence, non-linear LiPo battery discharge + voltage sag + Auto-RTL)
- [x] Pairwise CPA collision avoidance, 3D cylindrical/polygonal geofencing, comm-loss Hold/RTL, and permanent `HARDWARE_ACTUATION_LOCKED = True` interlock
- [x] FastAPI + 20 Hz `/ws/telemetry` WebSocket server + interactive Three.js WebGL Digital Twin UI + React/TypeScript source in `frontend/`
- [x] Full pytest verification suite covering planners, physics, safety, REST, and WebSockets

## v1.1 — SITL Bridge & Flight Log Replay
- [ ] Optional read-only MAVLink telemetry playback ingest (maintaining strict `HARDWARE_ACTUATION_LOCKED = True` outbound lock)
- [ ] Mission telemetry export to CSV / PX4 `.ulg`-compatible tabular telemetry logs
- [ ] Dynamic moving no-fly zones & time-parameterized trajectory scheduling (4D space-time A*)

## v1.2 — Terrain Elevation & Swarm Task Allocation
- [ ] Digital Elevation Model (DEM) terrain heightmap meshing in Three.js & voxel grid planner
- [ ] Hungarian / auction-based dynamic multi-UAV task reallocation when a unit triggers Low-Battery Auto-RTL
- [ ] Energy-optimal Dubins / minimum-snap spline trajectory generation post-planner smoothing

## v2.0 — Distributed Simulation Scaling
- [ ] Optional WebWorker / NumPy-vectorized physics solver supporting 50+ simultaneous UAVs at 60 Hz
- [ ] Deterministic Monte Carlo wind & failure stress-test runner via CLI batch report generator
