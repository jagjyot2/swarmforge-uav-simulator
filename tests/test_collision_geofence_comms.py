"""Tests for pairwise collision avoidance, 3D geofencing, comm-loss, and hardware safety lock."""

import numpy as np
import pytest

from sim_engine.models import Geofence, MissionConfig, UAVConfig, Vec3, Waypoint
from sim_engine.safety import (
    HARDWARE_ACTUATION_LOCKED,
    HardwareActuationBlockedError,
    assert_hardware_actuation_locked,
    enforce_geofence,
)
from sim_engine.simulator import SwarmSimulator
from sim_engine.uav_dynamics import UAVState, step_swarm_physics


def test_hardware_actuation_lock_enforced():
    assert HARDWARE_ACTUATION_LOCKED is True
    with pytest.raises(HardwareActuationBlockedError):
        assert_hardware_actuation_locked("ARM_MOTORS")


def test_pairwise_collision_avoidance_maintains_separation():
    uav_a = UAVConfig(
        uav_id="a",
        callsign="A",
        cruise_speed_mps=9.0,
        start_position=Vec3(x=-35.0, y=0.0, z=20.0),
        waypoints=[Waypoint(x=35.0, y=0.0, z=20.0)],
    )
    uav_b = UAVConfig(
        uav_id="b",
        callsign="B",
        cruise_speed_mps=9.0,
        start_position=Vec3(x=35.0, y=0.0, z=20.0),
        waypoints=[Waypoint(x=-35.0, y=0.0, z=20.0)],
    )
    st_a = UAVState.from_config(uav_a)
    st_b = UAVState.from_config(uav_b)
    gf = Geofence(enabled=True, radius_m=120.0, min_altitude_m=2.0, max_altitude_m=70.0)

    min_sep_req = 5.0
    observed_min_dist = float("inf")
    ca_triggered = False

    for _ in range(220):
        step_swarm_physics([st_a, st_b], 0.05, np.zeros(3), [], gf, min_sep_req, 2.5)
        d = float(np.linalg.norm(st_a.position - st_b.position))
        observed_min_dist = min(observed_min_dist, d)
        if st_a.collision_avoidance_active or st_b.collision_avoidance_active:
            ca_triggered = True

    assert ca_triggered is True
    assert observed_min_dist >= min_sep_req - 0.5, f"Minimum separation violated: {observed_min_dist:.2f}m"


def test_3d_cylindrical_polygonal_and_altitude_geofence():
    gf = Geofence(
        enabled=True,
        center_x=0.0,
        center_y=0.0,
        radius_m=50.0,
        min_altitude_m=3.0,
        max_altitude_m=40.0,
        soft_margin_m=6.0,
        polygon_xy=[(-45.0, -45.0), (45.0, -45.0), (45.0, 45.0), (-45.0, 45.0)],
    )

    # Soft warning inside margin (r=46m > 44m, inside square [-45,45])
    res_soft = enforce_geofence(
        np.array([44.0, 0.0, 20.0]), np.array([8.0, 0.0, 0.0]), gf
    )
    assert res_soft.warning is True
    assert res_soft.breached is False
    assert res_soft.velocity[0] < 8.0

    # Hard radial clamp beyond radius_m and ceiling clamp
    res_hard = enforce_geofence(
        np.array([62.0, 0.0, 52.0]), np.array([10.0, 0.0, 5.0]), gf
    )
    assert res_hard.breached is True
    assert np.hypot(res_hard.position[0], res_hard.position[1]) <= 50.0
    assert res_hard.position[2] <= 40.0
    assert res_hard.velocity[2] <= 0.0

    # Floor clamp when airborne landing disabled
    res_floor = enforce_geofence(
        np.array([0.0, 0.0, 1.5]), np.array([0.0, 0.0, -3.0]), gf, allow_ground_landing=False
    )
    assert res_floor.breached is True
    assert res_floor.position[2] >= 3.0


def test_comm_loss_hold_and_rtl_failsafes():
    mission = MissionConfig(
        mission_id="comm-test",
        name="Comm Loss Failsafe Test",
        uavs=[
            UAVConfig(
                uav_id="u1",
                callsign="U1",
                start_position=Vec3(x=0.0, y=0.0, z=15.0),
                waypoints=[Waypoint(x=70.0, y=0.0, z=18.0)],
                planner_type="direct",
            )
        ],
    )
    sim = SwarmSimulator()
    sim.load_mission(mission)
    sim.step(60)  # Fly outward ~25m

    pos_before_hold = sim.states[0].position.copy()
    sim.set_comm_loss(active=True, action="HOLD")
    assert sim.states[0].mode == "COMM_LOSS_HOLD"
    assert sim.states[0].comm_link_active is False
    sim.step(60)
    # UAV should hold near the link-drop coordinate rather than flying to x=70
    assert np.linalg.norm(sim.states[0].position - pos_before_hold) < 3.0

    # Switch to RTL on comm loss -> navigates back toward home (0,0)
    sim.set_comm_loss(active=True, action="RTL")
    assert sim.states[0].mode == "COMM_LOSS_RTL"
    dist_start_rtl = float(np.hypot(sim.states[0].position[0], sim.states[0].position[1]))
    sim.step(60)
    dist_after_rtl = float(np.hypot(sim.states[0].position[0], sim.states[0].position[1]))
    assert dist_after_rtl < dist_start_rtl

    # Restore link -> resumes MISSION
    sim.set_comm_loss(active=False)
    assert sim.states[0].comm_link_active is True
    assert sim.states[0].mode == "MISSION"


def test_head_on_crossing_sample_mission_deconflicts_4_uavs():
    sim = SwarmSimulator()
    sim.load_sample_mission("head_on_crossing")
    min_sep = sim.mission.min_separation_m
    observed_min_neighbor = float("inf")
    observed_min_obs = float("inf")

    for _ in range(400):
        frame = sim.step(1)
        for u in frame.uavs:
            assert u.geofence_breached is False
            if u.nearest_neighbor_m is not None:
                observed_min_neighbor = min(observed_min_neighbor, u.nearest_neighbor_m)
            if u.nearest_obstacle_m is not None:
                observed_min_obs = min(observed_min_obs, u.nearest_obstacle_m)

    assert observed_min_neighbor >= min_sep
    assert observed_min_obs >= 0.8

