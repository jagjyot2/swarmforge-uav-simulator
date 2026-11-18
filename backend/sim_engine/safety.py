"""Safety systems: Hardware lock, pairwise collision avoidance, 3D geofencing, and comm-loss failsafes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np

from .models import Geofence, Obstacle3D, UAVMode
from .path_planners import (
    distance_to_polygon_edge_2d,
    point_in_polygon_2d,
    point_obstacle_distance,
    point_obstacle_normal,
)

# Permanent safety interlock: SwarmForge is a pure Digital Twin simulator.
HARDWARE_ACTUATION_LOCKED: bool = True


class HardwareActuationBlockedError(PermissionError):
    """Raised whenever any call attempts to dispatch physical UAV hardware actuation."""


def assert_hardware_actuation_locked(action: str = "physical_actuation") -> None:
    """Enforce the permanent simulation-only hardware interlock."""
    if HARDWARE_ACTUATION_LOCKED:
        raise HardwareActuationBlockedError(
            f"Blocked '{action}': HARDWARE_ACTUATION_LOCKED=True. "
            "SwarmForge forbids sending physical actuator/MAVLink commands."
        )


@dataclass
class GeofenceResult:
    """Outcome of 3D geofence clamping and soft-braking check."""

    position: np.ndarray
    velocity: np.ndarray
    warning: bool
    breached: bool
    alerts: List[str]


def _closest_point_on_polygon(
    x: float, y: float, polygon: Sequence[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (closest_point_2d, inward_unit_normal_2d) for a closed 2D polygon."""
    p = np.array([x, y], dtype=np.float64)
    best_pt = p.copy()
    best_dist = float("inf")
    best_edge = np.array([1.0, 0.0], dtype=np.float64)
    n = len(polygon)
    # Compute centroid to orient inward normal reliably
    centroid = np.mean(np.asarray(polygon, dtype=np.float64), axis=0)
    for i in range(n):
        a = np.array(polygon[i], dtype=np.float64)
        b = np.array(polygon[(i + 1) % n], dtype=np.float64)
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0
        cand = a + t * ab
        d = float(np.linalg.norm(p - cand))
        if d < best_dist:
            best_dist = d
            best_pt = cand
            best_edge = ab

    # Two candidate normals perpendicular to edge; pick one pointing toward centroid
    en = float(np.linalg.norm(best_edge))
    if en > 1e-8:
        n1 = np.array([-best_edge[1], best_edge[0]], dtype=np.float64) / en
        if np.dot(centroid - best_pt, n1) < 0.0:
            n1 = -n1
    else:
        n1 = centroid - p
        ln = float(np.linalg.norm(n1))
        n1 = n1 / ln if ln > 1e-8 else np.array([0.0, 0.0])
    return best_pt, n1


def enforce_geofence(
    position: np.ndarray,
    velocity: np.ndarray,
    geofence: Geofence,
    allow_ground_landing: bool = False,
) -> GeofenceResult:
    """Enforce 3D cylindrical/polygonal and altitude geofence with soft braking and hard clamp."""
    pos = np.asarray(position, dtype=np.float64).copy()
    vel = np.asarray(velocity, dtype=np.float64).copy()
    alerts: List[str] = []
    warning = False
    breached = False

    if not geofence.enabled:
        return GeofenceResult(pos, vel, False, False, alerts)

    # 1. Horizontal cylindrical geofence
    rx = pos[0] - geofence.center_x
    ry = pos[1] - geofence.center_y
    r = float(np.hypot(rx, ry))
    soft_r = max(1.0, geofence.radius_m - geofence.soft_margin_m)
    if r > 1e-6:
        r_hat = np.array([rx / r, ry / r], dtype=np.float64)
        v_xy = vel[:2]
        v_rad = float(np.dot(v_xy, r_hat))

        if r >= geofence.radius_m:
            breached = True
            warning = True
            scale = (geofence.radius_m - 0.02) / r
            pos[0] = geofence.center_x + rx * scale
            pos[1] = geofence.center_y + ry * scale
            if v_rad > 0.0:
                vel[:2] = v_xy - v_rad * r_hat
            vel[:2] -= 1.2 * r_hat
            alerts.append("GEOFENCE_RADIAL_CLAMP")
        elif r >= soft_r:
            warning = True
            frac = float(np.clip((geofence.radius_m - r) / max(geofence.soft_margin_m, 0.5), 0.0, 1.0))
            if v_rad > 0.0:
                vel[:2] = v_xy - (1.0 - frac) * v_rad * r_hat
            vel[:2] -= (1.0 - frac) * 1.5 * r_hat
            alerts.append("GEOFENCE_RADIAL_WARNING")

    # 2. Optional polygonal XY geofence
    if geofence.polygon_xy is not None:
        inside = point_in_polygon_2d(float(pos[0]), float(pos[1]), geofence.polygon_xy)
        edge_d = distance_to_polygon_edge_2d(float(pos[0]), float(pos[1]), geofence.polygon_xy)
        if not inside:
            breached = True
            warning = True
            closest_pt, inward_n = _closest_point_on_polygon(
                float(pos[0]), float(pos[1]), geofence.polygon_xy
            )
            pos[0] = float(closest_pt[0] + 0.05 * inward_n[0])
            pos[1] = float(closest_pt[1] + 0.05 * inward_n[1])
            v_in = float(np.dot(vel[:2], inward_n))
            if v_in < 0.0:
                vel[:2] -= v_in * inward_n
            vel[:2] += 1.0 * inward_n
            alerts.append("GEOFENCE_POLYGON_CLAMP")
        elif edge_d < geofence.soft_margin_m:
            warning = True
            _, inward_n = _closest_point_on_polygon(
                float(pos[0]), float(pos[1]), geofence.polygon_xy
            )
            frac = float(np.clip(edge_d / max(geofence.soft_margin_m, 0.5), 0.0, 1.0))
            v_in = float(np.dot(vel[:2], inward_n))
            if v_in < 0.0:
                vel[:2] -= (1.0 - frac) * v_in * inward_n
            alerts.append("GEOFENCE_POLYGON_WARNING")

    # 3. Vertical ceiling & floor geofence
    ceil_soft = geofence.max_altitude_m - min(geofence.soft_margin_m, 6.0)
    if pos[2] >= geofence.max_altitude_m:
        breached = True
        warning = True
        pos[2] = geofence.max_altitude_m - 0.02
        if vel[2] > 0.0:
            vel[2] = 0.0
        vel[2] -= 0.8
        alerts.append("GEOFENCE_CEILING_CLAMP")
    elif pos[2] >= ceil_soft:
        warning = True
        frac = float(
            np.clip(
                (geofence.max_altitude_m - pos[2]) / max(geofence.max_altitude_m - ceil_soft, 0.5),
                0.0,
                1.0,
            )
        )
        if vel[2] > 0.0:
            vel[2] *= frac
        alerts.append("GEOFENCE_CEILING_WARNING")

    if not allow_ground_landing:
        floor_soft = geofence.min_altitude_m + min(geofence.soft_margin_m * 0.5, 3.0)
        if pos[2] <= geofence.min_altitude_m:
            breached = True
            warning = True
            pos[2] = geofence.min_altitude_m + 0.02
            if vel[2] < 0.0:
                vel[2] = 0.0
            vel[2] += 0.8
            alerts.append("GEOFENCE_FLOOR_CLAMP")
        elif pos[2] <= floor_soft:
            warning = True
            frac = float(
                np.clip(
                    (pos[2] - geofence.min_altitude_m)
                    / max(floor_soft - geofence.min_altitude_m, 0.5),
                    0.0,
                    1.0,
                )
            )
            if vel[2] < 0.0:
                vel[2] *= frac
            alerts.append("GEOFENCE_FLOOR_WARNING")
    else:
        if pos[2] < 0.0:
            pos[2] = 0.0
            if vel[2] < 0.0:
                vel[2] = 0.0

    return GeofenceResult(pos, vel, warning, breached, alerts)


def compute_pairwise_collision_avoidance(
    positions: Sequence[np.ndarray],
    velocities: Sequence[np.ndarray],
    desired_velocities: Sequence[np.ndarray],
    min_separation_m: float,
    max_speeds: Sequence[float],
    active_mask: Optional[Sequence[bool]] = None,
) -> Tuple[List[np.ndarray], List[bool], List[Optional[float]]]:
    """Reciprocal multi-agent collision avoidance using CPA prediction and repulsive velocity blending."""
    n = len(positions)
    adjusted = [np.asarray(v, dtype=np.float64).copy() for v in desired_velocities]
    ca_active = [False] * n
    nearest_dists: List[Optional[float]] = [None] * n

    if n < 2:
        return adjusted, ca_active, nearest_dists

    influence_dist = max(min_separation_m * 2.4, min_separation_m + 4.0)
    cpa_horizon_s = 3.0

    for i in range(n):
        for j in range(i + 1, n):
            if active_mask is not None and (not active_mask[i] or not active_mask[j]):
                continue
            diff = positions[i] - positions[j]
            dist = float(np.linalg.norm(diff))

            if nearest_dists[i] is None or dist < nearest_dists[i]:
                nearest_dists[i] = dist
            if nearest_dists[j] is None or dist < nearest_dists[j]:
                nearest_dists[j] = dist

            v_rel = desired_velocities[i] - desired_velocities[j]
            v_rel2 = float(np.dot(v_rel, v_rel))
            t_cpa = float(np.clip(-np.dot(diff, v_rel) / v_rel2, 0.0, cpa_horizon_s)) if v_rel2 > 1e-4 else 0.0
            cpa_vec = diff + v_rel * t_cpa
            d_cpa = float(np.linalg.norm(cpa_vec))

            if dist >= influence_dist and d_cpa >= min_separation_m * 1.35:
                continue

            ca_active[i] = True
            ca_active[j] = True

            n_ij = diff / dist if dist > 1e-4 else np.array([1.0, 0.0, 0.0], dtype=np.float64)

            # Right-hand starboard tangential swirl + slight vertical stagger prevents head-on deadlock
            horiz_n = np.array([n_ij[0], n_ij[1], 0.0], dtype=np.float64)
            hn = float(np.linalg.norm(horiz_n))
            if hn > 1e-4:
                horiz_unit = horiz_n / hn
                starboard = np.array([-horiz_unit[1], horiz_unit[0], 0.0], dtype=np.float64)
            else:
                starboard = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            z_stagger = np.array([0.0, 0.0, 0.45], dtype=np.float64)

            # Repulsion strength from both current distance and predicted CPA miss distance
            w_dist = float(
                np.clip(
                    (influence_dist - dist) / max(influence_dist - min_separation_m * 0.75, 0.5),
                    0.0,
                    2.5,
                )
            )
            w_cpa = float(
                np.clip(
                    (min_separation_m * 1.4 - d_cpa) / max(min_separation_m, 0.5),
                    0.0,
                    2.0,
                )
            )

            # Cancel closing relative velocity if closer than safety envelope
            closing_speed = float(np.dot(adjusted[i] - adjusted[j], -n_ij))
            if dist < min_separation_m * 1.3 and closing_speed > 0.0:
                cancel = 0.65 * closing_speed * n_ij
                adjusted[i] += cancel
                adjusted[j] -= cancel

            push_mag = 4.2 * w_dist + 3.2 * w_cpa
            swirl_mag = 3.6 * max(w_dist, w_cpa)

            delta_i = push_mag * n_ij + swirl_mag * starboard + 1.4 * max(w_dist, w_cpa) * z_stagger
            delta_j = -push_mag * n_ij - swirl_mag * starboard - 1.4 * max(w_dist, w_cpa) * z_stagger

            adjusted[i] += delta_i
            adjusted[j] += delta_j

    for i in range(n):
        spd = float(np.linalg.norm(adjusted[i]))
        if spd > max_speeds[i] and spd > 1e-6:
            adjusted[i] = adjusted[i] * (max_speeds[i] / spd)

    return adjusted, ca_active, nearest_dists


def compute_obstacle_repulsion(
    position: np.ndarray,
    desired_velocity: np.ndarray,
    obstacles: Sequence[Obstacle3D],
    min_clearance_m: float,
) -> Tuple[np.ndarray, Optional[float], bool]:
    """Reactive obstacle safety layer deflecting UAV velocity away from nearby 3D structures."""
    if not obstacles:
        return desired_velocity.copy(), None, False

    vel = desired_velocity.copy()
    nearest = float("inf")
    reactive_zone = min_clearance_m + 2.8
    deflected = False

    for obs in obstacles:
        d = point_obstacle_distance(position, obs)
        if d < nearest:
            nearest = d
        if d < reactive_zone:
            deflected = True
            normal = point_obstacle_normal(position, obs)
            v_into = float(np.dot(vel, -normal))
            if v_into > 0.0:
                vel += 1.15 * v_into * normal
            penetration = max(0.0, reactive_zone - d)
            vel += (6.5 * penetration / max(reactive_zone, 0.5)) * normal

    return vel, nearest, deflected


def clamp_outside_obstacles(
    position: np.ndarray,
    velocity: np.ndarray,
    obstacles: Sequence[Obstacle3D],
    min_surface_m: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Hard post-integration safety clamp keeping UAV outside physical obstacle volumes."""
    pos = position.copy()
    vel = velocity.copy()
    clamped = False
    for obs in obstacles:
        d = point_obstacle_distance(pos, obs)
        if d < min_surface_m:
            clamped = True
            normal = point_obstacle_normal(pos, obs)
            pos = pos + (min_surface_m - d) * normal
            v_into = float(np.dot(vel, -normal))
            if v_into > 0.0:
                vel = vel + v_into * normal
    return pos, vel, clamped


def resolve_comm_loss_mode(
    current_mode: UAVMode, action: Literal["HOLD", "RTL"]
) -> UAVMode:
    """Determine onboard autonomous mode when GCS communication link drops."""
    if current_mode in ("IDLE", "LANDED", "EMERGENCY_LAND"):
        return current_mode
    return "COMM_LOSS_HOLD" if action == "HOLD" else "COMM_LOSS_RTL"


def resolve_emergency_mode(
    current_mode: UAVMode, emergency: Literal["RTL", "LAND"]
) -> UAVMode:
    """Determine UAV mode when global swarm emergency command is triggered."""
    if current_mode == "LANDED":
        return "LANDED"
    return "EMERGENCY_LAND" if emergency == "LAND" else "RTL"
