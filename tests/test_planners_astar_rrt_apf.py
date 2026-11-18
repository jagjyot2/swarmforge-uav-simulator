"""Unit and integration tests for 3D A*, RRT*, APF planners and quantitative comparator."""

import numpy as np
import pytest

from sim_engine.models import Geofence, Obstacle3D, PlannerComparisonRequest, Vec3
from sim_engine.path_planners import (
    compare_planners,
    min_clearance_along_path,
    plan_apf_3d,
    plan_astar_3d,
    plan_rrt_star_3d,
    point_obstacle_distance,
)


@pytest.fixture
def urban_obstacles():
    return [
        Obstacle3D(
            id="wall-center",
            name="Central Block",
            kind="box",
            x=0.0,
            y=0.0,
            base_z=0.0,
            width_x=18.0,
            depth_y=30.0,
            height_z=42.0,
        ),
        Obstacle3D(
            id="tower-side",
            name="Comms Mast",
            kind="cylinder",
            x=20.0,
            y=22.0,
            base_z=0.0,
            height_z=50.0,
            radius=7.5,
        ),
    ]


@pytest.fixture
def arena_geofence():
    return Geofence(
        enabled=True,
        center_x=0.0,
        center_y=0.0,
        radius_m=110.0,
        min_altitude_m=2.0,
        max_altitude_m=65.0,
        soft_margin_m=8.0,
    )


def test_astar_avoids_box_and_cylinder_with_clearance(urban_obstacles, arena_geofence):
    start = np.array([-40.0, -5.0, 18.0])
    goal = np.array([40.0, 15.0, 20.0])
    clearance = 2.5

    ok, pts, note = plan_astar_3d(
        start, goal, urban_obstacles, clearance_m=clearance, geofence=arena_geofence
    )
    assert ok is True, note
    assert len(pts) >= 2
    assert np.allclose(pts[0], start)
    assert np.allclose(pts[-1], goal)
    assert min_clearance_along_path(pts, urban_obstacles) >= clearance - 1e-6


def test_rrt_star_finds_valid_rewired_path(urban_obstacles, arena_geofence):
    start = np.array([-42.0, 0.0, 16.0])
    goal = np.array([42.0, 5.0, 22.0])
    clearance = 2.5

    ok, pts, note = plan_rrt_star_3d(
        start, goal, urban_obstacles, clearance_m=clearance, geofence=arena_geofence, seed=42
    )
    assert ok is True, note
    assert len(pts) >= 2
    assert np.allclose(pts[0], start)
    assert np.allclose(pts[-1], goal)
    assert min_clearance_along_path(pts, urban_obstacles) >= clearance - 1e-6


def test_apf_escapes_symmetric_head_on_obstacle(urban_obstacles, arena_geofence):
    # Start and goal lie straight through center of wall-center box at x=0,y=0
    start = np.array([-38.0, 0.0, 18.0])
    goal = np.array([38.0, 0.0, 18.0])
    clearance = 2.5

    ok, pts, note = plan_apf_3d(
        start, goal, urban_obstacles, clearance_m=clearance, geofence=arena_geofence
    )
    assert ok is True, note
    assert len(pts) >= 3
    assert np.allclose(pts[0], start)
    assert np.allclose(pts[-1], goal)
    assert min_clearance_along_path(pts, urban_obstacles) >= clearance - 1e-6


def test_compare_planners_metrics_and_recommendation(urban_obstacles, arena_geofence):
    req = PlannerComparisonRequest(
        start=Vec3(x=-42.0, y=-6.0, z=18.0),
        goal=Vec3(x=42.0, y=10.0, z=22.0),
        obstacles=urban_obstacles,
        geofence=arena_geofence,
        clearance_m=2.5,
        cruise_speed_mps=8.5,
        seed=42,
    )
    res = compare_planners(req)
    algos = {m.algorithm: m for m in res.results}
    assert set(algos.keys()) == {"astar", "rrt_star", "apf"}
    assert res.recommended_algorithm in ("astar", "rrt_star", "apf")

    straight_dist = float(np.linalg.norm(req.goal.to_np() - req.start.to_np()))
    for algo, metrics in algos.items():
        assert metrics.success is True, f"{algo} failed: {metrics.notes}"
        assert metrics.path_length_m >= straight_dist - 1e-3
        assert metrics.compute_time_ms >= 0.0
        assert metrics.smoothness_rad >= 0.0
        assert metrics.min_obstacle_clearance_m >= 2.5 - 1e-3
        assert metrics.estimated_energy_wh > 0.0
        assert len(metrics.waypoints) >= 2


def test_planners_reject_goal_inside_obstacle(urban_obstacles, arena_geofence):
    start = np.array([-40.0, 0.0, 18.0])
    blocked_goal = np.array([0.0, 0.0, 18.0])  # Inside wall-center box
    assert point_obstacle_distance(blocked_goal, urban_obstacles[0]) < 0.0

    for fn in (plan_astar_3d, plan_rrt_star_3d, plan_apf_3d):
        ok, _, _ = fn(start, blocked_goal, urban_obstacles, 2.5, arena_geofence)
        assert ok is False
