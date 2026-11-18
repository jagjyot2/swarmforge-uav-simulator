"""Pydantic v2 domain models for SwarmForge 3D Multi-UAV simulation."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field, model_validator


PlannerType = Literal["astar", "rrt_star", "apf", "direct"]
UAVMode = Literal[
    "IDLE",
    "ARMED",
    "MISSION",
    "HOLD",
    "RTL",
    "EMERGENCY_LAND",
    "LANDED",
    "COMM_LOSS_HOLD",
    "COMM_LOSS_RTL",
]
SimStatus = Literal["STOPPED", "RUNNING", "PAUSED", "COMPLETED"]


class Vec3(BaseModel):
    """3D ENU vector (x=East, y=North, z=Altitude AGL in meters)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_np(self) -> np.ndarray:
        """Convert to a 3-element float64 NumPy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_np(cls, arr: np.ndarray) -> "Vec3":
        """Create a Vec3 from a 3-element array-like."""
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))


class Waypoint(BaseModel):
    """3D mission waypoint with optional hold dwell time and segment target speed."""

    x: float
    y: float
    z: float = Field(default=15.0, ge=0.0)
    hold_time_s: float = Field(default=0.0, ge=0.0)
    speed_mps: Optional[float] = Field(default=None, gt=0.0)
    label: Optional[str] = None

    def to_np(self) -> np.ndarray:
        """Convert waypoint position to NumPy vector."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)


class Obstacle3D(BaseModel):
    """3D structural obstacle (axis-aligned box building or vertical cylinder)."""

    id: str
    name: str = "Obstacle"
    kind: Literal["box", "cylinder"] = "box"
    x: float
    y: float
    base_z: float = Field(default=0.0, ge=0.0)
    width_x: float = Field(default=12.0, gt=0.0)
    depth_y: float = Field(default=12.0, gt=0.0)
    height_z: float = Field(default=30.0, gt=0.0)
    radius: float = Field(default=8.0, gt=0.0)
    color: str = "#475569"

    def min_corner(self) -> np.ndarray:
        """Minimum AABB corner [x_min, y_min, z_min]."""
        if self.kind == "cylinder":
            return np.array(
                [self.x - self.radius, self.y - self.radius, self.base_z],
                dtype=np.float64,
            )
        return np.array(
            [self.x - 0.5 * self.width_x, self.y - 0.5 * self.depth_y, self.base_z],
            dtype=np.float64,
        )

    def max_corner(self) -> np.ndarray:
        """Maximum AABB corner [x_max, y_max, z_max]."""
        if self.kind == "cylinder":
            return np.array(
                [self.x + self.radius, self.y + self.radius, self.base_z + self.height_z],
                dtype=np.float64,
            )
        return np.array(
            [
                self.x + 0.5 * self.width_x,
                self.y + 0.5 * self.depth_y,
                self.base_z + self.height_z,
            ],
            dtype=np.float64,
        )


class Geofence(BaseModel):
    """3D cylindrical + optional polygonal XY geofence with altitude limits."""

    enabled: bool = True
    center_x: float = 0.0
    center_y: float = 0.0
    radius_m: float = Field(default=140.0, gt=5.0)
    min_altitude_m: float = Field(default=1.0, ge=0.0)
    max_altitude_m: float = Field(default=85.0, gt=2.0)
    soft_margin_m: float = Field(default=8.0, ge=0.5)
    polygon_xy: Optional[List[Tuple[float, float]]] = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "Geofence":
        if self.max_altitude_m <= self.min_altitude_m:
            raise ValueError("max_altitude_m must be strictly greater than min_altitude_m")
        if self.polygon_xy is not None and len(self.polygon_xy) < 3:
            raise ValueError("polygon_xy must have at least 3 vertices when provided")
        return self


class WindConfig(BaseModel):
    """Steady wind vector + Dryden-like stochastic turbulence configuration."""

    steady_vx_mps: float = 0.0
    steady_vy_mps: float = 0.0
    steady_vz_mps: float = 0.0
    turbulence_intensity: float = Field(default=0.75, ge=0.0, le=12.0)
    gust_frequency_hz: float = Field(default=0.35, gt=0.01, le=5.0)
    seed: Optional[int] = 42


class UAVConfig(BaseModel):
    """Configuration for a single simulated multi-rotor UAV."""

    uav_id: str
    callsign: str = "SWARM-01"
    mass_kg: float = Field(default=1.85, gt=0.2)
    max_speed_mps: float = Field(default=14.0, gt=1.0)
    max_accel_mps2: float = Field(default=6.5, gt=0.5)
    cruise_speed_mps: float = Field(default=8.5, gt=0.5)
    drag_coeff: float = Field(default=0.045, ge=0.0)
    hover_power_w: float = Field(default=210.0, gt=10.0)
    accel_power_coeff: float = Field(default=1.8, ge=0.0)
    battery_capacity_wh: float = Field(default=99.9, gt=5.0)
    nominal_voltage_v: float = Field(default=22.2, gt=3.0)
    cells: int = Field(default=6, ge=1, le=16)
    initial_soc_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    rtl_soc_threshold_pct: float = Field(default=22.0, ge=5.0, le=80.0)
    start_position: Vec3 = Field(default_factory=lambda: Vec3(x=-45.0, y=-45.0, z=0.5))
    waypoints: List[Waypoint] = Field(default_factory=list)
    planner_type: PlannerType = "astar"
    color: str = "#38bdf8"


class MissionConfig(BaseModel):
    """Full mission configuration including swarm fleet, environment, wind, and geofence."""

    mission_id: str = "mission-urban-corridor"
    name: str = "Urban Corridor Inspection Swarm"
    description: str = (
        "Multi-UAV 3D urban obstacle navigation with geofencing and dynamic wind."
    )
    dt_s: float = Field(default=0.05, ge=0.01, le=0.2)
    min_separation_m: float = Field(default=4.5, ge=1.0)
    obstacle_clearance_m: float = Field(default=2.5, ge=0.5)
    comm_loss_action: Literal["HOLD", "RTL"] = "RTL"
    geofence: Geofence = Field(default_factory=Geofence)
    wind: WindConfig = Field(default_factory=WindConfig)
    obstacles: List[Obstacle3D] = Field(default_factory=list)
    uavs: List[UAVConfig] = Field(default_factory=list)


class UAVTelemetry(BaseModel):
    """Real-time telemetry snapshot for one simulated UAV."""

    uav_id: str
    callsign: str
    color: str
    planner_type: PlannerType
    position: Vec3
    velocity: Vec3
    speed_mps: float
    airspeed_mps: float
    attitude_deg: Vec3
    rotor_rpm: float
    battery_soc_pct: float
    voltage_v: float
    power_w: float
    energy_consumed_wh: float
    mode: UAVMode
    comm_link_active: bool
    active_waypoint_idx: int
    total_waypoints: int
    mission_waypoints: List[Vec3]
    planned_path: List[Vec3]
    geofence_warning: bool
    geofence_breached: bool
    collision_avoidance_active: bool
    nearest_neighbor_m: Optional[float] = None
    nearest_obstacle_m: Optional[float] = None
    alerts: List[str] = Field(default_factory=list)


class TelemetryFrame(BaseModel):
    """Synchronized swarm telemetry frame emitted by simulation engine and WebSocket."""

    sim_time_s: float
    tick: int
    status: SimStatus
    mission_id: str
    mission_name: str
    hardware_actuation_locked: bool = True
    safety_banner: str = "SIMULATION ONLY — HARDWARE ACTUATION LOCKED"
    wind_vector_mps: Vec3
    comm_loss_active: bool
    emergency_active: bool
    geofence: Geofence
    obstacles: List[Obstacle3D]
    uavs: List[UAVTelemetry]
    events: List[str] = Field(default_factory=list)


class PlannerMetrics(BaseModel):
    """Quantitative metrics evaluating a 3D path planner output."""

    algorithm: Literal["astar", "rrt_star", "apf"]
    success: bool
    path_length_m: float
    compute_time_ms: float
    smoothness_rad: float
    min_obstacle_clearance_m: float
    estimated_energy_wh: float
    num_waypoints: int
    waypoints: List[Vec3]
    notes: Optional[str] = None


class PlannerComparisonRequest(BaseModel):
    """Request payload for benchmarking A*, RRT*, and APF in 3D obstacle space."""

    start: Vec3 = Field(default_factory=lambda: Vec3(x=-55.0, y=-45.0, z=18.0))
    goal: Vec3 = Field(default_factory=lambda: Vec3(x=55.0, y=45.0, z=24.0))
    obstacles: List[Obstacle3D] = Field(default_factory=list)
    geofence: Optional[Geofence] = None
    clearance_m: float = Field(default=2.5, ge=0.5, le=15.0)
    cruise_speed_mps: float = Field(default=8.5, gt=0.5)
    hover_power_w: float = Field(default=210.0, gt=10.0)
    drag_coeff: float = Field(default=0.045, ge=0.0)
    mass_kg: float = Field(default=1.85, gt=0.2)
    seed: int = 42


class PlannerComparisonResult(BaseModel):
    """Side-by-side benchmark results across A*, RRT*, and APF."""

    start: Vec3
    goal: Vec3
    recommended_algorithm: str
    results: List[PlannerMetrics]
