"""Multi-UAV 3D dynamics, cascaded PID controller, Dryden wind/turbulence, and LiPo battery physics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .models import (
    Geofence,
    Obstacle3D,
    UAVConfig,
    UAVMode,
    UAVTelemetry,
    Vec3,
    WindConfig,
)
from .safety import (
    clamp_outside_obstacles,
    compute_obstacle_repulsion,
    compute_pairwise_collision_avoidance,
    enforce_geofence,
)

GRAVITY_MPS2 = 9.81


class DrydenWindModel:
    """Steady wind vector combined with temporally correlated Dryden-like 3D turbulence."""

    def __init__(self, config: WindConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.turbulence = np.zeros(3, dtype=np.float64)

    def update_config(self, config: WindConfig) -> None:
        """Update steady wind and turbulence parameters while preserving RNG continuity if seed unchanged."""
        seed_changed = config.seed != self.config.seed
        self.config = config
        if seed_changed:
            self.rng = np.random.default_rng(config.seed)
        if config.turbulence_intensity <= 0.0:
            self.turbulence.fill(0.0)

    def reset(self) -> None:
        """Reset turbulence state and RNG."""
        self.rng = np.random.default_rng(self.config.seed)
        self.turbulence.fill(0.0)

    def step(self, dt_s: float) -> np.ndarray:
        """Advance Dryden Gauss-Markov turbulence filter by dt_s and return total wind vector [m/s]."""
        steady = np.array(
            [
                self.config.steady_vx_mps,
                self.config.steady_vy_mps,
                self.config.steady_vz_mps,
            ],
            dtype=np.float64,
        )
        sigma = float(self.config.turbulence_intensity)
        if sigma <= 0.0:
            self.turbulence.fill(0.0)
            return steady

        tau = 1.0 / (2.0 * np.pi * max(self.config.gust_frequency_hz, 0.05))
        alpha = float(np.exp(-dt_s / tau))
        scale = float(np.sqrt(max(0.0, 1.0 - alpha * alpha)))
        noise = self.rng.normal(0.0, 1.0, size=3)
        sigma_vec = np.array([sigma, sigma, 0.65 * sigma], dtype=np.float64)
        self.turbulence = alpha * self.turbulence + scale * sigma_vec * noise
        return steady + self.turbulence


def compute_instantaneous_power_w(
    config: UAVConfig,
    airspeed_mps: float,
    vertical_vel_mps: float,
    accel_mag_mps2: float,
) -> float:
    """Non-linear multirotor electrical power draw: hover + drag*v^3 + accel_penalty + climb."""
    hover = config.hover_power_w
    aero_drag_power = config.drag_coeff * (max(0.0, airspeed_mps) ** 3)
    accel_penalty = config.accel_power_coeff * config.mass_kg * (accel_mag_mps2 ** 2)
    # Climb adds potential energy work; descent slightly unloads rotors (bounded >= -0.22 * hover)
    if vertical_vel_mps >= 0.0:
        climb_power = (config.mass_kg * GRAVITY_MPS2 * vertical_vel_mps) / 0.72
    else:
        climb_power = max(
            -0.22 * hover,
            0.35 * (config.mass_kg * GRAVITY_MPS2 * vertical_vel_mps),
        )
    return float(max(15.0, hover + aero_drag_power + accel_penalty + climb_power))


def compute_lipo_voltage_v(soc_pct: float, power_w: float, cells: int = 6) -> float:
    """Non-linear LiPo OCV vs SoC curve with internal-resistance load voltage sag."""
    s = float(np.clip(soc_pct / 100.0, 0.0, 1.0))
    # Smooth realistic per-cell OCV: ~4.20V at 100%, ~3.82V at 50%, ~3.68V at 20%, ~3.22V at 0%
    ocv_cell = (
        3.22
        + 0.45 * (1.0 - np.exp(-9.0 * s))
        + 0.31 * s
        + 0.22 * (s ** 3)
    )
    ocv_pack = float(cells * ocv_cell)
    current_a = power_w / max(ocv_pack, 3.0)
    r_internal_ohm = 0.018 * cells
    v_terminal = ocv_pack - current_a * r_internal_ohm
    return float(max(cells * 3.0, v_terminal))


def estimate_attitude_deg(
    accel_cmd: np.ndarray,
    velocity: np.ndarray,
    prev_yaw_rad: float,
    target_dir_xy: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """Estimate roll, pitch, yaw (degrees) from thrust/acceleration vector and velocity."""
    # Thrust vector in ENU required for commanded acceleration + counteracting gravity
    thrust = np.array([accel_cmd[0], accel_cmd[1], accel_cmd[2] + GRAVITY_MPS2], dtype=np.float64)
    horiz_speed = float(np.hypot(velocity[0], velocity[1]))
    if horiz_speed > 0.65:
        yaw = float(np.arctan2(velocity[1], velocity[0]))
    elif target_dir_xy is not None and float(np.hypot(target_dir_xy[0], target_dir_xy[1])) > 0.25:
        yaw = float(np.arctan2(target_dir_xy[1], target_dir_xy[0]))
    else:
        yaw = prev_yaw_rad

    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
    # Project thrust into yaw-aligned body-level frame (forward, left, up)
    t_fwd = cy * thrust[0] + sy * thrust[1]
    t_left = -sy * thrust[0] + cy * thrust[1]
    t_up = max(0.5, float(thrust[2]))

    max_tilt = np.deg2rad(38.0)
    pitch = float(np.clip(np.arctan2(t_fwd, t_up), -max_tilt, max_tilt))
    roll = float(np.clip(np.arctan2(-t_left, np.hypot(t_fwd, t_up)), -max_tilt, max_tilt))
    return np.array([np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)], dtype=np.float64), yaw


@dataclass
class UAVState:
    """Mutable runtime state for one simulated multi-rotor UAV."""

    config: UAVConfig
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    attitude_deg: np.ndarray
    rotor_rpm: float
    battery_soc_pct: float
    voltage_v: float
    power_w: float
    energy_consumed_wh: float
    mode: UAVMode
    comm_link_active: bool
    home_position: np.ndarray
    planned_path: List[np.ndarray]
    active_waypoint_idx: int = 0
    hold_anchor: Optional[np.ndarray] = None
    vel_integral: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    prev_vel_err: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_yaw_rad: float = 0.0
    geofence_warning: bool = False
    geofence_breached: bool = False
    collision_avoidance_active: bool = False
    nearest_neighbor_m: Optional[float] = None
    nearest_obstacle_m: Optional[float] = None
    alerts: List[str] = field(default_factory=list)
    rtl_climbing: bool = False

    @classmethod
    def from_config(
        cls, cfg: UAVConfig, planned_path: Optional[List[np.ndarray]] = None
    ) -> "UAVState":
        """Initialize UAVState from UAVConfig and optional planned 3D trajectory."""
        start = cfg.start_position.to_np()
        path = (
            [np.asarray(p, dtype=np.float64) for p in planned_path]
            if planned_path is not None
            else [w.to_np() for w in cfg.waypoints]
        )
        init_mode: UAVMode = "MISSION" if path else "HOLD"
        init_v = compute_lipo_voltage_v(cfg.initial_soc_pct, cfg.hover_power_w, cells=cfg.cells)
        return cls(
            config=cfg,
            position=start.copy(),
            velocity=np.zeros(3, dtype=np.float64),
            acceleration=np.zeros(3, dtype=np.float64),
            attitude_deg=np.zeros(3, dtype=np.float64),
            rotor_rpm=0.0,
            battery_soc_pct=cfg.initial_soc_pct,
            voltage_v=init_v,
            power_w=0.0,
            energy_consumed_wh=0.0,
            mode=init_mode,
            comm_link_active=True,
            home_position=start.copy(),
            planned_path=path,
            active_waypoint_idx=0,
            hold_anchor=start.copy() if init_mode == "HOLD" else None,
        )

    def to_telemetry(self, wind_vec: np.ndarray) -> UAVTelemetry:
        """Serialize current UAV state into a TelemetryFrame UAVTelemetry payload."""
        speed = float(np.linalg.norm(self.velocity))
        airspeed = float(np.linalg.norm(self.velocity - wind_vec))
        return UAVTelemetry(
            uav_id=self.config.uav_id,
            callsign=self.config.callsign,
            color=self.config.color,
            planner_type=self.config.planner_type,
            position=Vec3.from_np(self.position),
            velocity=Vec3.from_np(self.velocity),
            speed_mps=round(speed, 3),
            airspeed_mps=round(airspeed, 3),
            attitude_deg=Vec3(
                x=round(float(self.attitude_deg[0]), 2),
                y=round(float(self.attitude_deg[1]), 2),
                z=round(float(self.attitude_deg[2]), 2),
            ),
            rotor_rpm=round(self.rotor_rpm, 1),
            battery_soc_pct=round(self.battery_soc_pct, 3),
            voltage_v=round(self.voltage_v, 3),
            power_w=round(self.power_w, 2),
            energy_consumed_wh=round(self.energy_consumed_wh, 4),
            mode=self.mode,
            comm_link_active=self.comm_link_active,
            active_waypoint_idx=min(self.active_waypoint_idx, len(self.planned_path)),
            total_waypoints=len(self.planned_path),
            mission_waypoints=[Vec3(x=w.x, y=w.y, z=w.z) for w in self.config.waypoints],
            planned_path=[Vec3.from_np(p) for p in self.planned_path],
            geofence_warning=self.geofence_warning,
            geofence_breached=self.geofence_breached,
            collision_avoidance_active=self.collision_avoidance_active,
            nearest_neighbor_m=(
                round(self.nearest_neighbor_m, 2)
                if self.nearest_neighbor_m is not None
                else None
            ),
            nearest_obstacle_m=(
                round(self.nearest_obstacle_m, 2)
                if self.nearest_obstacle_m is not None
                else None
            ),
            alerts=list(self.alerts),
        )


def _compute_guidance_velocity(state: UAVState, geofence: Geofence) -> np.ndarray:
    """Compute desired 3D velocity vector from outer-loop waypoint/RTL/hold guidance."""
    cfg = state.config
    if state.mode in ("IDLE", "LANDED"):
        return np.zeros(3, dtype=np.float64)

    if state.mode == "EMERGENCY_LAND":
        return np.array([0.0, 0.0, -2.5], dtype=np.float64)

    if state.mode in ("HOLD", "COMM_LOSS_HOLD"):
        if state.hold_anchor is None:
            state.hold_anchor = state.position.copy()
        err = state.hold_anchor - state.position
        dist = float(np.linalg.norm(err))
        if dist < 0.05:
            return np.zeros(3, dtype=np.float64)
        spd = min(cfg.cruise_speed_mps * 0.6, 1.4 * dist)
        return (err / dist) * spd

    if state.mode in ("RTL", "COMM_LOSS_RTL"):
        safe_alt = max(
            float(state.home_position[2]) + 16.0,
            float(state.position[2]),
        )
        if geofence.enabled:
            safe_alt = min(safe_alt, geofence.max_altitude_m - 3.0)
        horiz_diff = state.home_position[:2] - state.position[:2]
        horiz_dist = float(np.hypot(horiz_diff[0], horiz_diff[1]))
        if horiz_dist > 2.2:
            target = np.array(
                [state.home_position[0], state.home_position[1], safe_alt],
                dtype=np.float64,
            )
            diff = target - state.position
            dist = float(np.linalg.norm(diff))
            spd = min(cfg.cruise_speed_mps, max(1.5, 1.1 * dist))
            return (diff / max(dist, 1e-6)) * spd
        # Descend smoothly at home
        target = np.array(
            [state.home_position[0], state.home_position[1], max(0.25, state.home_position[2])],
            dtype=np.float64,
        )
        diff = target - state.position
        dist = float(np.linalg.norm(diff))
        if dist < 0.8 and state.position[2] <= max(0.8, state.home_position[2] + 0.5):
            state.mode = "LANDED"
            state.velocity.fill(0.0)
            return np.zeros(3, dtype=np.float64)
        spd = min(3.2, max(0.8, 1.2 * dist))
        return (diff / max(dist, 1e-6)) * spd

    # MISSION / ARMED waypoint tracking
    if not state.planned_path or state.active_waypoint_idx >= len(state.planned_path):
        state.mode = "HOLD"
        state.hold_anchor = state.position.copy()
        return np.zeros(3, dtype=np.float64)

    target = state.planned_path[state.active_waypoint_idx]
    diff = target - state.position
    dist = float(np.linalg.norm(diff))
    is_final = state.active_waypoint_idx == len(state.planned_path) - 1
    arrival_r = 1.35 if is_final else 1.5

    while dist <= arrival_r:
        state.active_waypoint_idx += 1
        if state.active_waypoint_idx >= len(state.planned_path):
            state.mode = "HOLD"
            state.hold_anchor = target.copy()
            return np.zeros(3, dtype=np.float64)
        target = state.planned_path[state.active_waypoint_idx]
        diff = target - state.position
        dist = float(np.linalg.norm(diff))
        is_final = state.active_waypoint_idx == len(state.planned_path) - 1
        arrival_r = 1.35 if is_final else 1.5

    cruise = min(cfg.cruise_speed_mps, cfg.max_speed_mps)
    # Slow gently into final waypoint, cruise smoothly through intermediate nodes
    desired_spd = min(cruise, max(1.4, 1.35 * dist)) if is_final else cruise
    return (diff / max(dist, 1e-6)) * desired_spd


def step_swarm_physics(
    states: Sequence[UAVState],
    dt_s: float,
    wind_vec: np.ndarray,
    obstacles: Sequence[Obstacle3D],
    geofence: Geofence,
    min_separation_m: float,
    obstacle_clearance_m: float,
) -> List[str]:
    """Advance all UAVs by dt_s using cascaded PID, collision avoidance, geofencing, and battery model."""
    events: List[str] = []
    if not states:
        return events

    desired_vels: List[np.ndarray] = []
    active_mask: List[bool] = []
    for st in states:
        st.alerts.clear()
        v_des = _compute_guidance_velocity(st, geofence)
        desired_vels.append(v_des)
        active_mask.append(st.mode not in ("IDLE", "LANDED"))

    positions = [st.position for st in states]
    velocities = [st.velocity for st in states]
    max_speeds = [st.config.max_speed_mps for st in states]

    ca_vels, ca_flags, nearest_neighbors = compute_pairwise_collision_avoidance(
        positions=positions,
        velocities=velocities,
        desired_velocities=desired_vels,
        min_separation_m=min_separation_m,
        max_speeds=max_speeds,
        active_mask=active_mask,
    )

    kp_v, ki_v, kd_v = 2.8, 0.75, 0.22

    for idx, st in enumerate(states):
        st.collision_avoidance_active = ca_flags[idx]
        st.nearest_neighbor_m = nearest_neighbors[idx]
        if ca_flags[idx]:
            st.alerts.append("COLLISION_AVOIDANCE_ACTIVE")

        if st.mode in ("IDLE", "LANDED"):
            st.velocity.fill(0.0)
            st.acceleration.fill(0.0)
            st.rotor_rpm = 0.0
            st.power_w = 4.0
            st.geofence_warning = False
            st.geofence_breached = False
            continue

        cfg = st.config
        allow_landing = st.mode in ("RTL", "COMM_LOSS_RTL", "EMERGENCY_LAND")

        # Reactive obstacle shield + geofence velocity shaping
        v_obs_safe, nearest_obs, obs_deflected = compute_obstacle_repulsion(
            st.position, ca_vels[idx], obstacles, min_clearance_m=obstacle_clearance_m
        )
        st.nearest_obstacle_m = nearest_obs
        if obs_deflected:
            st.alerts.append("OBSTACLE_PROXIMITY_SHIELD")

        gf_pre = enforce_geofence(
            st.position, v_obs_safe, geofence, allow_ground_landing=allow_landing
        )
        v_target = gf_pre.velocity

        # Inner-loop velocity PID with anti-windup
        vel_err = v_target - st.velocity
        st.vel_integral = np.clip(st.vel_integral + vel_err * dt_s, -4.5, 4.5)
        d_err = (vel_err - st.prev_vel_err) / max(dt_s, 1e-4)
        st.prev_vel_err = vel_err.copy()

        a_cmd = kp_v * vel_err + ki_v * st.vel_integral + kd_v * d_err
        a_cmd_mag = float(np.linalg.norm(a_cmd))
        if a_cmd_mag > cfg.max_accel_mps2:
            a_cmd = a_cmd * (cfg.max_accel_mps2 / a_cmd_mag)
            a_cmd_mag = cfg.max_accel_mps2

        # Wind relative airspeed & quadratic aerodynamic drag
        v_air = st.velocity - wind_vec
        v_air_mag = float(np.linalg.norm(v_air))
        a_drag = -(0.028 / cfg.mass_kg) * v_air_mag * v_air
        a_total = a_cmd + a_drag

        v_next = st.velocity + a_total * dt_s
        v_next_mag = float(np.linalg.norm(v_next))
        max_physical = cfg.max_speed_mps + 4.0
        if v_next_mag > max_physical:
            v_next = v_next * (max_physical / v_next_mag)
        p_next = st.position + v_next * dt_s

        # Post-integration hard obstacle + geofence safety clamp
        p_obs, v_obs, obs_clamped = clamp_outside_obstacles(
            p_next, v_next, obstacles, min_surface_m=0.8
        )
        if obs_clamped and "OBSTACLE_PROXIMITY_SHIELD" not in st.alerts:
            st.alerts.append("OBSTACLE_PROXIMITY_SHIELD")

        gf_post = enforce_geofence(
            p_obs, v_obs, geofence, allow_ground_landing=allow_landing
        )
        st.position = gf_post.position
        st.velocity = gf_post.velocity
        st.acceleration = a_total
        st.geofence_warning = gf_pre.warning or gf_post.warning
        st.geofence_breached = gf_pre.breached or gf_post.breached
        for al in gf_post.alerts:
            if al not in st.alerts:
                st.alerts.append(al)

        # Ground touchdown check for emergency land
        if st.mode == "EMERGENCY_LAND" and st.position[2] <= 0.35:
            st.position[2] = 0.0
            st.velocity.fill(0.0)
            st.mode = "LANDED"
            st.rotor_rpm = 0.0
            events.append(f"{cfg.callsign} completed emergency landing.")

        # Attitude & rotor RPM estimation
        target_dir_xy = v_target[:2] if float(np.hypot(v_target[0], v_target[1])) > 0.2 else None
        st.attitude_deg, st.last_yaw_rad = estimate_attitude_deg(
            a_cmd, st.velocity, st.last_yaw_rad, target_dir_xy=target_dir_xy
        )
        thrust_spec = float(np.linalg.norm(a_cmd + np.array([0.0, 0.0, GRAVITY_MPS2])))
        st.rotor_rpm = 5200.0 * float(np.sqrt(max(0.15, thrust_spec / GRAVITY_MPS2)))

        # Non-linear battery discharge & voltage sag
        v_air_after = float(np.linalg.norm(st.velocity - wind_vec))
        st.power_w = compute_instantaneous_power_w(
            cfg, v_air_after, float(st.velocity[2]), a_cmd_mag
        )
        d_wh = (st.power_w * dt_s) / 3600.0
        st.energy_consumed_wh += d_wh
        st.battery_soc_pct = max(
            0.0, st.battery_soc_pct - 100.0 * (d_wh / cfg.battery_capacity_wh)
        )
        st.voltage_v = compute_lipo_voltage_v(st.battery_soc_pct, st.power_w, cells=cfg.cells)

        if st.battery_soc_pct <= 4.0 and st.mode not in ("EMERGENCY_LAND", "LANDED"):
            st.mode = "EMERGENCY_LAND"
            st.alerts.append("CRITICAL_BATTERY_LAND")
            events.append(f"{cfg.callsign} critical battery ({st.battery_soc_pct:.1f}%) -> EMERGENCY_LAND")
        elif st.battery_soc_pct <= cfg.rtl_soc_threshold_pct and st.mode in (
            "MISSION",
            "HOLD",
            "ARMED",
        ):
            st.mode = "RTL"
            st.alerts.append("LOW_BATTERY_AUTO_RTL")
            events.append(f"{cfg.callsign} reached {st.battery_soc_pct:.1f}% SoC -> AUTO RTL")

    return events
