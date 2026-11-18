/** Strict TypeScript domain contracts matching SwarmForge FastAPI Pydantic schemas. */

export type PlannerType = "astar" | "rrt_star" | "apf" | "direct";

export type UAVMode =
  | "IDLE"
  | "ARMED"
  | "MISSION"
  | "HOLD"
  | "RTL"
  | "EMERGENCY_LAND"
  | "LANDED"
  | "COMM_LOSS_HOLD"
  | "COMM_LOSS_RTL";

export type SimStatus = "STOPPED" | "RUNNING" | "PAUSED" | "COMPLETED";

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Obstacle3D {
  id: string;
  name: string;
  kind: "box" | "cylinder";
  x: number;
  y: number;
  base_z: number;
  width_x: number;
  depth_y: number;
  height_z: number;
  radius: number;
  color: string;
}

export interface Geofence {
  enabled: boolean;
  center_x: number;
  center_y: number;
  radius_m: number;
  min_altitude_m: number;
  max_altitude_m: number;
  soft_margin_m: number;
  polygon_xy?: [number, number][] | null;
}

export interface WindConfig {
  steady_vx_mps: number;
  steady_vy_mps: number;
  steady_vz_mps: number;
  turbulence_intensity: number;
  gust_frequency_hz: number;
  seed?: number | null;
}

export interface UAVTelemetry {
  uav_id: string;
  callsign: string;
  color: string;
  planner_type: PlannerType;
  position: Vec3;
  velocity: Vec3;
  speed_mps: number;
  airspeed_mps: number;
  attitude_deg: Vec3;
  rotor_rpm: number;
  battery_soc_pct: number;
  voltage_v: number;
  power_w: number;
  energy_consumed_wh: number;
  mode: UAVMode;
  comm_link_active: boolean;
  active_waypoint_idx: number;
  total_waypoints: number;
  mission_waypoints: Vec3[];
  planned_path: Vec3[];
  geofence_warning: boolean;
  geofence_breached: boolean;
  collision_avoidance_active: boolean;
  nearest_neighbor_m?: number | null;
  nearest_obstacle_m?: number | null;
  alerts: string[];
}

export interface TelemetryFrame {
  sim_time_s: number;
  tick: number;
  status: SimStatus;
  mission_id: string;
  mission_name: string;
  hardware_actuation_locked: boolean;
  safety_banner: string;
  wind_vector_mps: Vec3;
  comm_loss_active: boolean;
  emergency_active: boolean;
  geofence: Geofence;
  obstacles: Obstacle3D[];
  uavs: UAVTelemetry[];
  events: string[];
}

export interface MissionSummary {
  mission_id: string;
  name: string;
  description: string;
  uav_count: number;
  obstacle_count: number;
  filename: string;
}

export interface PlannerMetrics {
  algorithm: "astar" | "rrt_star" | "apf";
  success: boolean;
  path_length_m: number;
  compute_time_ms: number;
  smoothness_rad: number;
  min_obstacle_clearance_m: number;
  estimated_energy_wh: number;
  num_waypoints: number;
  waypoints: Vec3[];
  notes?: string | null;
}

export interface PlannerComparisonResult {
  start: Vec3;
  goal: Vec3;
  recommended_algorithm: string;
  results: PlannerMetrics[];
}
