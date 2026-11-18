"""Tests for multi-UAV 3D physics, PID tracking, attitude estimation, wind/turbulence, and LiPo battery."""

import numpy as np

from sim_engine.models import Geofence, UAVConfig, Vec3, Waypoint, WindConfig
from sim_engine.uav_dynamics import (
    DrydenWindModel,
    UAVState,
    compute_instantaneous_power_w,
    compute_lipo_voltage_v,
    estimate_attitude_deg,
    step_swarm_physics,
)


def test_pid_waypoint_tracking_and_hover_stability():
    cfg = UAVConfig(
        uav_id="uav-pid",
        callsign="PID-TEST",
        cruise_speed_mps=9.0,
        start_position=Vec3(x=0.0, y=0.0, z=15.0),
        waypoints=[Waypoint(x=45.0, y=30.0, z=25.0)],
    )
    st = UAVState.from_config(cfg)
    gf = Geofence(enabled=True, radius_m=150.0, min_altitude_m=1.0, max_altitude_m=80.0)
    wind = np.zeros(3)

    # Simulate 14 seconds at 20 Hz
    for _ in range(280):
        step_swarm_physics([st], 0.05, wind, [], gf, 4.5, 2.5)

    target = np.array([45.0, 30.0, 25.0])
    dist = float(np.linalg.norm(st.position - target))
    assert dist < 1.6
    assert st.mode == "HOLD"


def test_attitude_estimation_tilts_along_acceleration():
    # Accelerating East (+X) while flying East -> positive pitch (nose down forward)
    att, yaw = estimate_attitude_deg(
        accel_cmd=np.array([4.0, 0.0, 0.0]),
        velocity=np.array([6.0, 0.0, 0.0]),
        prev_yaw_rad=0.0,
    )
    assert abs(att[2]) < 1.0  # Yaw ~ 0 deg (East)
    assert att[1] > 15.0  # Positive forward pitch
    assert abs(att[0]) < 1.0

    # Lateral Left acceleration (+Y) while facing East -> negative roll (bank left)
    att_lat, _ = estimate_attitude_deg(
        accel_cmd=np.array([0.0, 4.0, 0.0]),
        velocity=np.array([6.0, 0.0, 0.0]),
        prev_yaw_rad=yaw,
    )
    assert att_lat[0] < -15.0


def test_dryden_wind_and_turbulence_affect_airspeed():
    steady_cfg = WindConfig(steady_vx_mps=5.0, steady_vy_mps=-2.0, turbulence_intensity=0.0)
    calm_model = DrydenWindModel(steady_cfg)
    w_calm = calm_model.step(0.05)
    assert np.allclose(w_calm, np.array([5.0, -2.0, 0.0]))

    turb_cfg = WindConfig(
        steady_vx_mps=5.0, steady_vy_mps=-2.0, turbulence_intensity=1.8, gust_frequency_hz=0.4, seed=99
    )
    gust_model = DrydenWindModel(turb_cfg)
    samples = np.array([gust_model.step(0.05) for _ in range(100)])
    # Correlated fluctuations should produce non-zero variance around steady wind
    assert np.std(samples[:, 0]) > 0.2
    assert np.std(samples[:, 1]) > 0.2


def test_nonlinear_battery_power_voltage_sag_and_auto_rtl():
    cfg = UAVConfig(
        uav_id="uav-bat",
        callsign="BAT-01",
        hover_power_w=200.0,
        drag_coeff=0.05,
        accel_power_coeff=1.8,
        battery_capacity_wh=50.0,
        initial_soc_pct=23.0,
        rtl_soc_threshold_pct=22.0,
        start_position=Vec3(x=60.0, y=0.0, z=20.0),
        waypoints=[Waypoint(x=95.0, y=0.0, z=20.0)],
    )

    p_hover = compute_instantaneous_power_w(cfg, airspeed_mps=0.0, vertical_vel_mps=0.0, accel_mag_mps2=0.0)
    p_cruise = compute_instantaneous_power_w(cfg, airspeed_mps=8.0, vertical_vel_mps=0.0, accel_mag_mps2=0.0)
    p_sprint = compute_instantaneous_power_w(cfg, airspeed_mps=14.0, vertical_vel_mps=0.0, accel_mag_mps2=3.0)
    assert p_sprint > p_cruise > p_hover
    # Verify cubic aerodynamic term drag*v^3
    assert np.isclose(p_cruise - p_hover, cfg.drag_coeff * (8.0 ** 3))

    # Voltage sags under heavier electrical load at same SoC
    v_light = compute_lipo_voltage_v(soc_pct=50.0, power_w=180.0, cells=6)
    v_heavy = compute_lipo_voltage_v(soc_pct=50.0, power_w=650.0, cells=6)
    assert v_light > v_heavy

    # Crossing rtl_soc_threshold_pct automatically triggers RTL
    st = UAVState.from_config(cfg)
    gf = Geofence(enabled=True, radius_m=150.0, min_altitude_m=1.0, max_altitude_m=80.0)
    triggered_rtl = False
    for _ in range(400):
        events = step_swarm_physics([st], 0.1, np.array([-4.0, 0.0, 0.0]), [], gf, 4.5, 2.5)
        if st.mode == "RTL" or any("AUTO RTL" in ev for ev in events):
            triggered_rtl = True
            break
    assert triggered_rtl is True
    assert st.battery_soc_pct <= cfg.rtl_soc_threshold_pct
    assert st.energy_consumed_wh > 0.0
