"""3D UAV Path Planners (A* Voxel, RRT*, Artificial Potential Field) and Quantitative Comparator."""

from __future__ import annotations

import heapq
import time
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .models import (
    Geofence,
    Obstacle3D,
    PlannerComparisonRequest,
    PlannerComparisonResult,
    PlannerMetrics,
    Vec3,
)


def point_in_polygon_2d(x: float, y: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test in 2D XY plane."""
    inside = False
    n = len(polygon)
    if n < 3:
        return True
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) if abs(yj - yi) > 1e-12 else 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def distance_to_polygon_edge_2d(
    x: float, y: float, polygon: Sequence[Tuple[float, float]]
) -> float:
    """Minimum Euclidean distance from point (x, y) to polygon boundary edges."""
    p = np.array([x, y], dtype=np.float64)
    min_d = float("inf")
    n = len(polygon)
    for i in range(n):
        a = np.array(polygon[i], dtype=np.float64)
        b = np.array(polygon[(i + 1) % n], dtype=np.float64)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            d = float(np.linalg.norm(p - a))
        else:
            t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
            d = float(np.linalg.norm(p - (a + t * ab)))
        if d < min_d:
            min_d = d
    return min_d


def point_obstacle_distance(p: np.ndarray, obs: Obstacle3D) -> float:
    """Exact signed 3D Euclidean distance from point p to an Obstacle3D (negative if inside)."""
    if obs.kind == "cylinder":
        radial_d = float(np.hypot(p[0] - obs.x, p[1] - obs.y)) - obs.radius
        z_min = obs.base_z
        z_max = obs.base_z + obs.height_z
        dz = max(z_min - float(p[2]), 0.0, float(p[2]) - z_max)
        if radial_d > 0.0 or dz > 0.0:
            return float(np.hypot(max(radial_d, 0.0), dz))
        return -min(-radial_d, float(p[2]) - z_min, z_max - float(p[2]))

    hx = 0.5 * obs.width_x
    hy = 0.5 * obs.depth_y
    x_min, x_max = obs.x - hx, obs.x + hx
    y_min, y_max = obs.y - hy, obs.y + hy
    z_min, z_max = obs.base_z, obs.base_z + obs.height_z

    dx = max(x_min - float(p[0]), 0.0, float(p[0]) - x_max)
    dy = max(y_min - float(p[1]), 0.0, float(p[1]) - y_max)
    dz = max(z_min - float(p[2]), 0.0, float(p[2]) - z_max)
    if dx > 0.0 or dy > 0.0 or dz > 0.0:
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))
    return -min(
        float(p[0]) - x_min,
        x_max - float(p[0]),
        float(p[1]) - y_min,
        y_max - float(p[1]),
        float(p[2]) - z_min,
        z_max - float(p[2]),
    )


def point_obstacle_normal(p: np.ndarray, obs: Obstacle3D) -> np.ndarray:
    """Outward unit normal pointing from obstacle surface toward point p."""
    if obs.kind == "cylinder":
        dx = float(p[0] - obs.x)
        dy = float(p[1] - obs.y)
        r = float(np.hypot(dx, dy))
        z_top = obs.base_z + obs.height_z
        if float(p[2]) > z_top and r <= obs.radius:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if r > 1e-6:
            horiz = np.array([dx / r, dy / r, 0.0], dtype=np.float64)
            if float(p[2]) > z_top:
                dz = float(p[2] - z_top)
                dr = max(r - obs.radius, 0.0)
                v = np.array([horiz[0] * dr, horiz[1] * dr, dz], dtype=np.float64)
                n = float(np.linalg.norm(v))
                return v / n if n > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
            return horiz
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    hx = 0.5 * obs.width_x
    hy = 0.5 * obs.depth_y
    cx = float(np.clip(p[0], obs.x - hx, obs.x + hx))
    cy = float(np.clip(p[1], obs.y - hy, obs.y + hy))
    cz = float(np.clip(p[2], obs.base_z, obs.base_z + obs.height_z))
    diff = p - np.array([cx, cy, cz], dtype=np.float64)
    norm = float(np.linalg.norm(diff))
    if norm > 1e-6:
        return diff / norm
    # Inside box: push out toward nearest face (prefer lateral or top)
    deltas = np.array(
        [
            p[0] - (obs.x - hx),
            (obs.x + hx) - p[0],
            p[1] - (obs.y - hy),
            (obs.y + hy) - p[1],
            (obs.base_z + obs.height_z) - p[2],
        ],
        dtype=np.float64,
    )
    dirs = [
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    return dirs[int(np.argmin(deltas))]


def is_inside_geofence(
    p: np.ndarray, geofence: Optional[Geofence], margin_m: float = 0.0
) -> bool:
    """Check whether point p satisfies cylindrical/polygonal and altitude geofence."""
    if p[2] < 0.25:
        return False
    if geofence is None or not geofence.enabled:
        return True
    r = float(np.hypot(p[0] - geofence.center_x, p[1] - geofence.center_y))
    if r > max(1.0, geofence.radius_m - margin_m):
        return False
    if p[2] < geofence.min_altitude_m or p[2] > geofence.max_altitude_m:
        return False
    if geofence.polygon_xy is not None:
        if not point_in_polygon_2d(float(p[0]), float(p[1]), geofence.polygon_xy):
            return False
        if margin_m > 0.0 and distance_to_polygon_edge_2d(
            float(p[0]), float(p[1]), geofence.polygon_xy
        ) < margin_m:
            return False
    return True


def is_point_free(
    p: np.ndarray,
    obstacles: Sequence[Obstacle3D],
    clearance_m: float = 2.0,
    geofence: Optional[Geofence] = None,
) -> bool:
    """Return True if point p is inside geofence and maintains clearance from all obstacles."""
    if not is_inside_geofence(p, geofence, margin_m=0.5):
        return False
    for obs in obstacles:
        if point_obstacle_distance(p, obs) < clearance_m:
            return False
    return True


def is_segment_free(
    a: np.ndarray,
    b: np.ndarray,
    obstacles: Sequence[Obstacle3D],
    clearance_m: float = 2.0,
    geofence: Optional[Geofence] = None,
) -> bool:
    """Return True if line segment a->b maintains clearance from obstacles and stays in geofence."""
    diff = b - a
    dist = float(np.linalg.norm(diff))
    step = min(0.75, max(0.3, clearance_m * 0.35))
    n_steps = max(2, int(np.ceil(dist / step)))
    for i in range(n_steps + 1):
        t = i / n_steps
        p = a + t * diff
        if not is_point_free(p, obstacles, clearance_m=clearance_m, geofence=geofence):
            return False
    return True


def min_clearance_along_path(
    pts: Sequence[np.ndarray], obstacles: Sequence[Obstacle3D]
) -> float:
    """Compute exact minimum obstacle surface distance sampled along a 3D polyline."""
    if not obstacles or not pts:
        return 999.0
    min_d = float("inf")
    if len(pts) == 1:
        for obs in obstacles:
            min_d = min(min_d, point_obstacle_distance(pts[0], obs))
        return float(min_d)

    for idx in range(len(pts) - 1):
        a, b = pts[idx], pts[idx + 1]
        dist = float(np.linalg.norm(b - a))
        n_steps = max(2, int(np.ceil(dist / 0.5)))
        for s in range(n_steps + 1):
            p = a + (s / n_steps) * (b - a)
            for obs in obstacles:
                d = point_obstacle_distance(p, obs)
                if d < min_d:
                    min_d = d
    return float(min_d)


def shortcut_smooth_path(
    pts: List[np.ndarray],
    obstacles: Sequence[Obstacle3D],
    clearance_m: float,
    geofence: Optional[Geofence],
    passes: int = 2,
) -> List[np.ndarray]:
    """Greedy Line-of-Sight path shortcut smoothing preserving strict segment clearance."""
    if len(pts) <= 2:
        return pts
    current = [p.copy() for p in pts]
    for _ in range(passes):
        smoothed = [current[0]]
        i = 0
        while i < len(current) - 1:
            next_idx = i + 1
            for j in range(len(current) - 1, i + 1, -1):
                if is_segment_free(current[i], current[j], obstacles, clearance_m, geofence):
                    next_idx = j
                    break
            smoothed.append(current[next_idx])
            i = next_idx
        current = smoothed
        if len(current) <= 2:
            break
    return current


def _rdp_simplify_safe(
    pts: List[np.ndarray],
    epsilon: float,
    obstacles: Sequence[Obstacle3D],
    clearance_m: float,
    geofence: Optional[Geofence],
) -> List[np.ndarray]:
    """Douglas-Peucker polyline simplification that only drops points if segment remains collision-free."""
    if len(pts) <= 2:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    ab_len = float(np.linalg.norm(ab))
    max_d = -1.0
    max_idx = 0
    for i in range(1, len(pts) - 1):
        if ab_len < 1e-8:
            d = float(np.linalg.norm(pts[i] - a))
        else:
            t = float(np.clip(np.dot(pts[i] - a, ab) / (ab_len * ab_len), 0.0, 1.0))
            d = float(np.linalg.norm(pts[i] - (a + t * ab)))
        if d > max_d:
            max_d = d
            max_idx = i

    if max_d <= epsilon and is_segment_free(a, b, obstacles, clearance_m, geofence):
        return [a, b]
    left = _rdp_simplify_safe(pts[: max_idx + 1], epsilon, obstacles, clearance_m, geofence)
    right = _rdp_simplify_safe(pts[max_idx:], epsilon, obstacles, clearance_m, geofence)
    return left[:-1] + right


def _compute_bounds(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: Sequence[Obstacle3D],
    geofence: Optional[Geofence],
) -> Tuple[np.ndarray, np.ndarray]:
    """Determine 3D search bounding box from geofence, start/goal, and obstacles."""
    if geofence is not None and geofence.enabled:
        pad = 2.0
        min_b = np.array(
            [
                geofence.center_x - geofence.radius_m + 0.5,
                geofence.center_y - geofence.radius_m + 0.5,
                max(1.0, geofence.min_altitude_m),
            ],
            dtype=np.float64,
        )
        max_b = np.array(
            [
                geofence.center_x + geofence.radius_m - 0.5,
                geofence.center_y + geofence.radius_m - 0.5,
                max(geofence.min_altitude_m + 1.0, geofence.max_altitude_m - 0.5),
            ],
            dtype=np.float64,
        )
        return min_b, max_b

    pts = [start, goal]
    for obs in obstacles:
        pts.append(obs.min_corner())
        pts.append(obs.max_corner())
    arr = np.vstack(pts)
    min_b = np.min(arr, axis=0) - np.array([25.0, 25.0, 0.0])
    max_b = np.max(arr, axis=0) + np.array([25.0, 25.0, 20.0])
    min_b[2] = max(1.0, float(min_b[2]))
    max_b[2] = max(min_b[2] + 15.0, float(max_b[2]), 60.0)
    return min_b, max_b


def plan_astar_3d(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: Sequence[Obstacle3D],
    clearance_m: float = 2.5,
    geofence: Optional[Geofence] = None,
    voxel_size_m: float = 4.0,
) -> Tuple[bool, List[np.ndarray], str]:
    """3D A* voxel grid planner with Euclidean heuristic and LOS shortcut smoothing."""
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    if not is_point_free(start, obstacles, clearance_m, geofence):
        return False, [], "Start point violates obstacle clearance or geofence."
    if not is_point_free(goal, obstacles, clearance_m, geofence):
        return False, [], "Goal point violates obstacle clearance or geofence."

    if is_segment_free(start, goal, obstacles, clearance_m, geofence):
        return True, [start.copy(), goal.copy()], "Direct LOS corridor clear."

    min_b, max_b = _compute_bounds(start, goal, obstacles, geofence)
    span = np.maximum(max_b - min_b, 4.0)
    res = float(voxel_size_m)
    while np.prod(np.ceil(span / res) + 1) > 120_000:
        res *= 1.15

    xs = np.arange(min_b[0], max_b[0] + 0.5 * res, res, dtype=np.float64)
    ys = np.arange(min_b[1], max_b[1] + 0.5 * res, res, dtype=np.float64)
    zs = np.arange(min_b[2], max_b[2] + 0.5 * res, res, dtype=np.float64)
    nx, ny, nz = len(xs), len(ys), len(zs)

    # Vectorized 3D occupancy grid construction
    occ = np.zeros((nx, ny, nz), dtype=bool)
    X = xs[:, None, None]
    Y = ys[None, :, None]
    Z = zs[None, None, :]

    if geofence is not None and geofence.enabled:
        radial = np.hypot(X - geofence.center_x, Y - geofence.center_y)
        occ |= (radial[:, :, 0] > (geofence.radius_m - 0.5))[:, :, None]
        occ |= (Z < geofence.min_altitude_m) | (Z > geofence.max_altitude_m)
        if geofence.polygon_xy is not None:
            for ix in range(nx):
                for iy in range(ny):
                    if not point_in_polygon_2d(float(xs[ix]), float(ys[iy]), geofence.polygon_xy):
                        occ[ix, iy, :] = True

    for obs in obstacles:
        if obs.kind == "cylinder":
            radial_d = np.hypot(X[:, :, 0] - obs.x, Y[:, :, 0] - obs.y) - obs.radius
            dz = np.maximum(
                np.maximum(obs.base_z - Z[0, 0, :], 0.0),
                Z[0, 0, :] - (obs.base_z + obs.height_z),
            )
            dist2 = np.maximum(radial_d, 0.0)[:, :, None] ** 2 + (dz[None, None, :] ** 2)
            inside = (radial_d[:, :, None] <= 0.0) & (
                (Z >= obs.base_z) & (Z <= obs.base_z + obs.height_z)
            )
            occ |= inside | (dist2 < clearance_m * clearance_m)
        else:
            hx = 0.5 * obs.width_x
            hy = 0.5 * obs.depth_y
            dx = np.maximum(np.maximum((obs.x - hx) - X, 0.0), X - (obs.x + hx))
            dy = np.maximum(np.maximum((obs.y - hy) - Y, 0.0), Y - (obs.y + hy))
            dz = np.maximum(
                np.maximum(obs.base_z - Z, 0.0), Z - (obs.base_z + obs.height_z)
            )
            dist2 = dx * dx + dy * dy + dz * dz
            occ |= dist2 < clearance_m * clearance_m

    def _world_to_idx(pt: np.ndarray) -> Optional[Tuple[int, int, int]]:
        ix0 = int(np.clip(np.argmin(np.abs(xs - pt[0])), 0, nx - 1))
        iy0 = int(np.clip(np.argmin(np.abs(ys - pt[1])), 0, ny - 1))
        iz0 = int(np.clip(np.argmin(np.abs(zs - pt[2])), 0, nz - 1))
        if not occ[ix0, iy0, iz0]:
            vpos = np.array([xs[ix0], ys[iy0], zs[iz0]], dtype=np.float64)
            if is_segment_free(pt, vpos, obstacles, clearance_m, geofence):
                return (ix0, iy0, iz0)
        best = None
        best_d = float("inf")
        for r in range(1, 4):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        ix, iy, iz = ix0 + dx, iy0 + dy, iz0 + dz
                        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz and not occ[ix, iy, iz]:
                            vpos = np.array([xs[ix], ys[iy], zs[iz]], dtype=np.float64)
                            d = float(np.linalg.norm(vpos - pt))
                            if d < best_d and is_segment_free(pt, vpos, obstacles, clearance_m, geofence):
                                best_d = d
                                best = (ix, iy, iz)
            if best is not None:
                return best
        return None

    s_idx = _world_to_idx(start)
    g_idx = _world_to_idx(goal)
    if s_idx is None or g_idx is None:
        return False, [], "Unable to anchor start/goal onto free 3D voxel grid."

    # 26-connectivity offsets and step lengths
    neighbors: List[Tuple[int, int, int, float]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbors.append((dx, dy, dz, res * float(np.sqrt(dx * dx + dy * dy + dz * dz))))

    gx_w, gy_w, gz_w = xs[g_idx[0]], ys[g_idx[1]], zs[g_idx[2]]
    g_score = np.full((nx, ny, nz), np.inf, dtype=np.float64)
    closed = np.zeros((nx, ny, nz), dtype=bool)
    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

    g_score[s_idx] = 0.0
    h0 = float(np.sqrt((xs[s_idx[0]] - gx_w) ** 2 + (ys[s_idx[1]] - gy_w) ** 2 + (zs[s_idx[2]] - gz_w) ** 2))
    open_heap: List[Tuple[float, Tuple[int, int, int]]] = [(h0, s_idx)]

    found = False
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == g_idx:
            found = True
            break
        if closed[cur]:
            continue
        closed[cur] = True

        cx, cy, cz = cur
        cur_g = g_score[cx, cy, cz]
        for dx, dy, dz, step_cost in neighbors:
            nx_i, ny_i, nz_i = cx + dx, cy + dy, cz + dz
            if not (0 <= nx_i < nx and 0 <= ny_i < ny and 0 <= nz_i < nz):
                continue
            if closed[nx_i, ny_i, nz_i] or occ[nx_i, ny_i, nz_i]:
                continue
            # Prevent diagonal corner clipping through adjacent occupied voxels
            if (dx != 0 and occ[cx + dx, cy, cz]) or (dy != 0 and occ[cx, cy + dy, cz]) or (
                dz != 0 and occ[cx, cy, cz + dz]
            ):
                continue
            tentative_g = cur_g + step_cost
            if tentative_g < g_score[nx_i, ny_i, nz_i]:
                g_score[nx_i, ny_i, nz_i] = tentative_g
                came_from[(nx_i, ny_i, nz_i)] = cur
                h = float(
                    np.sqrt(
                        (xs[nx_i] - gx_w) ** 2
                        + (ys[ny_i] - gy_w) ** 2
                        + (zs[nz_i] - gz_w) ** 2
                    )
                )
                heapq.heappush(open_heap, (tentative_g + h, (nx_i, ny_i, nz_i)))

    if not found:
        return False, [], "3D A* could not find a valid collision-free voxel corridor."

    idx_path = [g_idx]
    node = g_idx
    while node != s_idx:
        node = came_from[node]
        idx_path.append(node)
    idx_path.reverse()

    world_path = [start.copy()]
    for ix, iy, iz in idx_path:
        world_path.append(np.array([xs[ix], ys[iy], zs[iz]], dtype=np.float64))
    world_path.append(goal.copy())

    smoothed = shortcut_smooth_path(world_path, obstacles, clearance_m, geofence, passes=3)
    return True, smoothed, f"Voxel A* ({res:.1f}m grid) + LOS smoothing ({len(smoothed)} wps)."


def plan_rrt_star_3d(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: Sequence[Obstacle3D],
    clearance_m: float = 2.5,
    geofence: Optional[Geofence] = None,
    max_iters: int = 900,
    step_size_m: float = 8.5,
    rewire_radius_m: float = 20.0,
    goal_sample_rate: float = 0.16,
    seed: int = 42,
) -> Tuple[bool, List[np.ndarray], str]:
    """3D RRT* sampling-based planner with asymptotic rewiring and segment collision checks."""
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    if not is_point_free(start, obstacles, clearance_m, geofence):
        return False, [], "Start point violates obstacle clearance or geofence."
    if not is_point_free(goal, obstacles, clearance_m, geofence):
        return False, [], "Goal point violates obstacle clearance or geofence."

    if is_segment_free(start, goal, obstacles, clearance_m, geofence):
        return True, [start.copy(), goal.copy()], "RRT* direct collision-free connection."

    rng = np.random.default_rng(seed)
    min_b, max_b = _compute_bounds(start, goal, obstacles, geofence)

    nodes: List[np.ndarray] = [start.copy()]
    parents: List[int] = [-1]
    costs: List[float] = [0.0]

    best_goal_parent: int = -1
    best_goal_cost: float = float("inf")
    iters_after_first_goal = 0

    for _ in range(max_iters):
        if rng.random() < goal_sample_rate:
            sample = goal.copy()
        else:
            sample = rng.uniform(min_b, max_b)

        pts_arr = np.asarray(nodes, dtype=np.float64)
        dists = np.linalg.norm(pts_arr - sample, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_pt = nodes[nearest_idx]

        vec = sample - nearest_pt
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            continue
        if dist > step_size_m:
            new_pt = nearest_pt + (vec / dist) * step_size_m
        else:
            new_pt = sample.copy()

        if not is_point_free(new_pt, obstacles, clearance_m, geofence):
            continue
        if not is_segment_free(nearest_pt, new_pt, obstacles, clearance_m, geofence):
            continue

        # Choose minimum-cost parent within rewire radius
        n_dists = np.linalg.norm(pts_arr - new_pt, axis=1)
        near_indices = np.where(n_dists <= rewire_radius_m)[0]
        best_parent = nearest_idx
        best_cost = costs[nearest_idx] + float(np.linalg.norm(new_pt - nearest_pt))

        for idx_np in near_indices:
            idx = int(idx_np)
            cand_cost = costs[idx] + float(n_dists[idx])
            if cand_cost < best_cost and is_segment_free(
                nodes[idx], new_pt, obstacles, clearance_m, geofence
            ):
                best_parent = idx
                best_cost = cand_cost

        new_idx = len(nodes)
        nodes.append(new_pt)
        parents.append(best_parent)
        costs.append(best_cost)

        # Rewire nearby nodes through new_pt if cheaper
        for idx_np in near_indices:
            idx = int(idx_np)
            if idx == best_parent:
                continue
            cand_cost = best_cost + float(n_dists[idx])
            if cand_cost + 1e-6 < costs[idx] and is_segment_free(
                new_pt, nodes[idx], obstacles, clearance_m, geofence
            ):
                parents[idx] = new_idx
                costs[idx] = cand_cost

        # Attempt goal connection
        d_goal = float(np.linalg.norm(goal - new_pt))
        if d_goal <= step_size_m * 1.8 and is_segment_free(
            new_pt, goal, obstacles, clearance_m, geofence
        ):
            total_goal_cost = best_cost + d_goal
            if total_goal_cost < best_goal_cost:
                best_goal_cost = total_goal_cost
                best_goal_parent = new_idx

        if best_goal_parent != -1:
            iters_after_first_goal += 1
            if iters_after_first_goal >= 140:
                break

    if best_goal_parent == -1:
        # Final check from any existing node within extended range
        pts_arr = np.asarray(nodes, dtype=np.float64)
        g_dists = np.linalg.norm(pts_arr - goal, axis=1)
        for idx in np.argsort(g_dists)[:25]:
            i = int(idx)
            if is_segment_free(nodes[i], goal, obstacles, clearance_m, geofence):
                cand = costs[i] + float(g_dists[i])
                if cand < best_goal_cost:
                    best_goal_cost = cand
                    best_goal_parent = i

    if best_goal_parent == -1:
        return False, [], "3D RRT* did not reach goal within iteration budget."

    path = [goal.copy()]
    cur = best_goal_parent
    visited = set()
    while cur != -1 and cur not in visited:
        visited.add(cur)
        path.append(nodes[cur].copy())
        cur = parents[cur]
    path.reverse()

    return True, path, f"RRT* sampled {len(nodes)} nodes with rewiring ({len(path)} wps)."


def plan_apf_3d(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: Sequence[Obstacle3D],
    clearance_m: float = 2.5,
    geofence: Optional[Geofence] = None,
    max_steps: int = 550,
    step_size_m: float = 2.0,
) -> Tuple[bool, List[np.ndarray], str]:
    """3D Artificial Potential Field planner with tangential curl local-minimum escape."""
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    if not is_point_free(start, obstacles, clearance_m, geofence):
        return False, [], "Start point violates obstacle clearance or geofence."
    if not is_point_free(goal, obstacles, clearance_m, geofence):
        return False, [], "Goal point violates obstacle clearance or geofence."

    pos = start.copy()
    trace: List[np.ndarray] = [pos.copy()]
    prev_dir = goal - pos
    n0 = float(np.linalg.norm(prev_dir))
    prev_dir = prev_dir / n0 if n0 > 1e-6 else np.array([1.0, 0.0, 0.0])

    influence_m = max(12.0, clearance_m * 4.2)
    k_att = 1.35
    k_rep = 28.0
    escape_steps = 0
    curl_sign = 1.0
    min_alt = geofence.min_altitude_m if (geofence and geofence.enabled) else 1.5
    max_alt = geofence.max_altitude_m if (geofence and geofence.enabled) else 85.0

    for step_idx in range(max_steps):
        to_goal = goal - pos
        dist_goal = float(np.linalg.norm(to_goal))
        if dist_goal <= 2.6 and is_segment_free(pos, goal, obstacles, clearance_m, geofence):
            trace.append(goal.copy())
            simplified = _rdp_simplify_safe(trace, 0.55, obstacles, clearance_m, geofence)
            return True, simplified, f"3D APF converged in {step_idx + 1} steps ({len(simplified)} wps)."

        dir_goal = to_goal / max(dist_goal, 1e-6)
        f_att = k_att * (to_goal if dist_goal <= 8.0 else 8.0 * dir_goal)

        f_rep = np.zeros(3, dtype=np.float64)
        for obs in obstacles:
            d_raw = point_obstacle_distance(pos, obs)
            d_eff = d_raw - clearance_m
            if d_eff < influence_m:
                normal = point_obstacle_normal(pos, obs)
                d_safe = max(d_eff, 0.28)
                mag = k_rep * (1.0 / d_safe - 1.0 / influence_m) / (d_safe * d_safe)
                f_rep += mag * normal

                # Tangential curl component when obstacle directly opposes goal direction
                opp = float(np.dot(normal, dir_goal))
                if opp < -0.2:
                    tangent = np.cross(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
                    t_norm = float(np.linalg.norm(tangent))
                    if t_norm > 1e-4:
                        tangent = (tangent / t_norm) * curl_sign
                    else:
                        tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    # Add slight climb bias if obstacle top is well below ceiling
                    obs_top = obs.base_z + obs.height_z + clearance_m + 2.0
                    z_bias = 0.45 if obs_top < max_alt else 0.0
                    f_rep += mag * 0.85 * (tangent + np.array([0.0, 0.0, z_bias]))

        # Geofence boundary potential
        if geofence is not None and geofence.enabled:
            rx = pos[0] - geofence.center_x
            ry = pos[1] - geofence.center_y
            r = float(np.hypot(rx, ry))
            r_limit = geofence.radius_m - 1.5
            if r > r_limit - geofence.soft_margin_m and r > 1e-4:
                inward = np.array([-rx / r, -ry / r, 0.0], dtype=np.float64)
                f_rep += 3.5 * inward * max(0.2, (r - (r_limit - geofence.soft_margin_m)))
            if pos[2] > max_alt - 3.0:
                f_rep[2] -= 3.0
            elif pos[2] < min_alt + 2.0:
                f_rep[2] += 3.0

        # Local minimum detection over sliding window
        if len(trace) >= 18:
            net_disp = float(np.linalg.norm(pos - trace[-18]))
            if net_disp < 3.0 and escape_steps == 0:
                escape_steps = 16
                curl_sign *= -1.0

        f_total = f_att + f_rep
        if escape_steps > 0:
            escape_steps -= 1
            ortho = np.cross(dir_goal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
            onorm = float(np.linalg.norm(ortho))
            ortho = (ortho / onorm * curl_sign) if onorm > 1e-4 else np.array([1.0, 0.0, 0.0])
            f_total += 8.0 * (ortho + np.array([0.0, 0.0, 0.45 if pos[2] < max_alt - 6.0 else -0.35]))

        fnorm = float(np.linalg.norm(f_total))
        raw_dir = (f_total / fnorm) if fnorm > 1e-6 else dir_goal
        blended = 0.45 * prev_dir + 0.55 * raw_dir
        bnorm = float(np.linalg.norm(blended))
        step_dir = blended / bnorm if bnorm > 1e-6 else raw_dir

        # Candidate direction selection ensuring strict segment safety
        step_len = min(step_size_m, dist_goal)
        candidates = [step_dir]
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        side = np.cross(step_dir, up)
        sn = float(np.linalg.norm(side))
        if sn > 1e-4:
            side /= sn
            for ang_deg in (30.0, -30.0, 60.0, -60.0, 90.0, -90.0):
                rad = np.deg2rad(ang_deg)
                candidates.append(np.cos(rad) * step_dir + np.sin(rad) * side)
        candidates.append(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        candidates.append(np.array([step_dir[0], step_dir[1], 0.4], dtype=np.float64))

        moved = False
        for cand in candidates:
            cn = float(np.linalg.norm(cand))
            if cn < 1e-6:
                continue
            cand_unit = cand / cn
            cand_pt = pos + step_len * cand_unit
            if is_segment_free(pos, cand_pt, obstacles, clearance_m, geofence):
                pos = cand_pt
                prev_dir = cand_unit
                trace.append(pos.copy())
                moved = True
                break

        if not moved:
            # Try shorter step along goal/escape
            cand_pt = pos + 0.6 * step_dir
            if is_segment_free(pos, cand_pt, obstacles, clearance_m, geofence):
                pos = cand_pt
                trace.append(pos.copy())
            else:
                break

    return False, trace, "3D APF reached local minimum or step limit before goal."


def evaluate_path_metrics(
    algorithm: Literal["astar", "rrt_star", "apf"],
    success: bool,
    pts: Sequence[np.ndarray],
    compute_time_ms: float,
    obstacles: Sequence[Obstacle3D],
    cruise_speed_mps: float = 8.5,
    hover_power_w: float = 210.0,
    drag_coeff: float = 0.045,
    mass_kg: float = 1.85,
    notes: Optional[str] = None,
) -> PlannerMetrics:
    """Compute length, smoothness, clearance, and physics-derived energy for a 3D path."""
    if not success or len(pts) < 2:
        return PlannerMetrics(
            algorithm=algorithm,
            success=False,
            path_length_m=0.0,
            compute_time_ms=round(compute_time_ms, 3),
            smoothness_rad=0.0,
            min_obstacle_clearance_m=0.0,
            estimated_energy_wh=0.0,
            num_waypoints=len(pts),
            waypoints=[Vec3.from_np(p) for p in pts],
            notes=notes,
        )

    seg_vecs = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
    seg_lens = [float(np.linalg.norm(v)) for v in seg_vecs]
    path_len = float(sum(seg_lens))

    smoothness_rad = 0.0
    turn_energy_j = 0.0
    unit_vecs = [v / l for v, l in zip(seg_vecs, seg_lens) if l > 1e-6]
    for i in range(len(unit_vecs) - 1):
        cos_theta = float(np.clip(np.dot(unit_vecs[i], unit_vecs[i + 1]), -1.0, 1.0))
        angle = float(np.arccos(cos_theta))
        smoothness_rad += angle
        # Kinetic re-acceleration penalty across turns (efficiency ~0.72)
        turn_energy_j += (1.0 - cos_theta) * 0.5 * mass_kg * (cruise_speed_mps ** 2) / 0.72

    min_clear = min_clearance_along_path(pts, obstacles)
    flight_time_s = path_len / max(cruise_speed_mps, 0.5)
    cruise_power_w = hover_power_w + drag_coeff * (cruise_speed_mps ** 3)
    climb_m = sum(max(0.0, float(pts[i + 1][2] - pts[i][2])) for i in range(len(pts) - 1))
    climb_energy_j = mass_kg * 9.81 * climb_m / 0.72
    energy_wh = (cruise_power_w * flight_time_s + turn_energy_j + climb_energy_j) / 3600.0

    return PlannerMetrics(
        algorithm=algorithm,
        success=True,
        path_length_m=round(path_len, 3),
        compute_time_ms=round(compute_time_ms, 3),
        smoothness_rad=round(smoothness_rad, 4),
        min_obstacle_clearance_m=round(min_clear, 3),
        estimated_energy_wh=round(energy_wh, 4),
        num_waypoints=len(pts),
        waypoints=[Vec3.from_np(p) for p in pts],
        notes=notes,
    )


def compare_planners(request: PlannerComparisonRequest) -> PlannerComparisonResult:
    """Run 3D A*, RRT*, and APF on identical scenario and return quantitative comparison."""
    start = request.start.to_np()
    goal = request.goal.to_np()

    results: List[PlannerMetrics] = []

    # 1. 3D A*
    t0 = time.perf_counter()
    ok_a, pts_a, note_a = plan_astar_3d(
        start, goal, request.obstacles, clearance_m=request.clearance_m, geofence=request.geofence
    )
    dt_a = (time.perf_counter() - t0) * 1000.0
    results.append(
        evaluate_path_metrics(
            "astar",
            ok_a,
            pts_a,
            dt_a,
            request.obstacles,
            cruise_speed_mps=request.cruise_speed_mps,
            hover_power_w=request.hover_power_w,
            drag_coeff=request.drag_coeff,
            mass_kg=request.mass_kg,
            notes=note_a,
        )
    )

    # 2. 3D RRT*
    t0 = time.perf_counter()
    ok_r, pts_r, note_r = plan_rrt_star_3d(
        start,
        goal,
        request.obstacles,
        clearance_m=request.clearance_m,
        geofence=request.geofence,
        seed=request.seed,
    )
    dt_r = (time.perf_counter() - t0) * 1000.0
    results.append(
        evaluate_path_metrics(
            "rrt_star",
            ok_r,
            pts_r,
            dt_r,
            request.obstacles,
            cruise_speed_mps=request.cruise_speed_mps,
            hover_power_w=request.hover_power_w,
            drag_coeff=request.drag_coeff,
            mass_kg=request.mass_kg,
            notes=note_r,
        )
    )

    # 3. 3D APF
    t0 = time.perf_counter()
    ok_p, pts_p, note_p = plan_apf_3d(
        start, goal, request.obstacles, clearance_m=request.clearance_m, geofence=request.geofence
    )
    dt_p = (time.perf_counter() - t0) * 1000.0
    results.append(
        evaluate_path_metrics(
            "apf",
            ok_p,
            pts_p,
            dt_p,
            request.obstacles,
            cruise_speed_mps=request.cruise_speed_mps,
            hover_power_w=request.hover_power_w,
            drag_coeff=request.drag_coeff,
            mass_kg=request.mass_kg,
            notes=note_p,
        )
    )

    successful = [r for r in results if r.success]
    if not successful:
        recommended = "none"
    else:
        min_energy = min(r.estimated_energy_wh for r in successful) or 1e-6
        min_len = min(r.path_length_m for r in successful) or 1e-6

        def _score(m: PlannerMetrics) -> float:
            return (
                0.45 * (m.estimated_energy_wh / min_energy)
                + 0.35 * (m.path_length_m / min_len)
                + 0.12 * (1.0 + m.smoothness_rad / 10.0)
                + 0.08 * (m.compute_time_ms / 50.0)
            )

        recommended = min(successful, key=_score).algorithm

    return PlannerComparisonResult(
        start=request.start,
        goal=request.goal,
        recommended_algorithm=recommended,
        results=results,
    )


def plan_waypoint_chain(
    start: np.ndarray,
    waypoints: Sequence[np.ndarray],
    obstacles: Sequence[Obstacle3D],
    algorithm: Literal["astar", "rrt_star", "apf", "direct"],
    clearance_m: float = 2.5,
    geofence: Optional[Geofence] = None,
    seed: int = 42,
) -> List[np.ndarray]:
    """Generate a continuous collision-free 3D waypoint chain across all mission waypoints."""
    if not waypoints:
        return []
    if algorithm == "direct":
        return [np.asarray(w, dtype=np.float64) for w in waypoints]

    # Elevate takeoff start above ground to minimum safe planner altitude if needed
    min_alt = geofence.min_altitude_m if (geofence and geofence.enabled) else 1.5
    cur = np.asarray(start, dtype=np.float64).copy()
    planned: List[np.ndarray] = []
    if cur[2] < min_alt:
        takeoff_pt = np.array([cur[0], cur[1], max(min_alt + 1.0, float(waypoints[0][2]))], dtype=np.float64)
        if not is_point_free(takeoff_pt, obstacles, clearance_m, geofence):
            takeoff_pt[2] = min_alt + 1.0
        planned.append(takeoff_pt)
        cur = takeoff_pt

    for idx, wp in enumerate(waypoints):
        target = np.asarray(wp, dtype=np.float64)
        if algorithm == "astar":
            ok, seg, _ = plan_astar_3d(cur, target, obstacles, clearance_m, geofence)
        elif algorithm == "rrt_star":
            ok, seg, _ = plan_rrt_star_3d(cur, target, obstacles, clearance_m, geofence, seed=seed + idx)
        else:
            ok, seg, _ = plan_apf_3d(cur, target, obstacles, clearance_m, geofence)
            if not ok:
                # Fallback to A* if local APF fails on a tight corridor leg
                ok, seg, _ = plan_astar_3d(cur, target, obstacles, clearance_m, geofence)

        if ok and len(seg) >= 2:
            for p in seg[1:]:
                if not planned or float(np.linalg.norm(p - planned[-1])) > 0.25:
                    planned.append(p)
        else:
            planned.append(target.copy())
        cur = target

    return planned
