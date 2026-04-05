"""
integrator.py — RK4 integrator for the full 6DOF state.

State vector integrated at each step:
    rx, ry, rz           ECI position [m]
    vx_eci, vy_eci, vz_eci  ECI velocity [m/s]
    qw, qx, qy, qz       body-to-ECI quaternion
    omega_x, omega_y, omega_z  body angular rates [rad/s]

The arc-length accumulator (state.x) is updated separately using the
tangential speed at the end of each step.
"""

import math

from src.atmosphere.us_standard_1976 import dynamic_pressure, speed_of_sound
from src.attitude.quaternion import quat_normalize
from src.events.event import Event
from src.guidance.base import GuidanceBase
from src.propagator.equations_of_motion import compute_control_commands, compute_derivatives
from src.propagator.state import SimState
from src.vehicle.vehicle import Vehicle


_OMEGA_EARTH = 7.2921150e-5  # Earth rotation rate [rad/s]


def _atm_relative_velocity(state: SimState) -> tuple[float, float, float]:
    """Return velocity relative to the rotating atmosphere [m/s]."""
    vx_atm = -_OMEGA_EARTH * state.ry
    vy_atm =  _OMEGA_EARTH * state.rx
    return (
        state.vx_eci - vx_atm,
        state.vy_eci - vy_atm,
        state.vz_eci,
    )


def _update_derived(state: SimState) -> None:
    """Refresh cached scalar quantities from the primary ECI state."""
    alt = state.altitude

    # Mach, dynamic pressure, and FPA all use the atmosphere-relative velocity.
    # (At launch the vehicle is stationary w.r.t. Earth → v_rel ≈ 0, Mach ≈ 0.)
    vrx, vry, vrz = _atm_relative_velocity(state)
    v_rel = math.sqrt(vrx * vrx + vry * vry + vrz * vrz)

    state.dynamic_pressure_Pa = dynamic_pressure(alt, v_rel)
    a_s = speed_of_sound(alt)
    state.mach = v_rel / a_s if a_s > 0.0 else 0.0

    if v_rel > 1.0:
        r = state.r_mag
        if r > 1.0:
            rhat_x = state.rx / r
            rhat_y = state.ry / r
            rhat_z = state.rz / r
            vr = vrx * rhat_x + vry * rhat_y + vrz * rhat_z
            vt = math.sqrt(max(0.0, v_rel * v_rel - vr * vr))
            state.flight_path_angle_deg = math.degrees(math.atan2(vr, vt))
    else:
        state.flight_path_angle_deg = 90.0   # vertical at liftoff


def _rk4_step(
    state: SimState,
    vehicle: Vehicle,
    guidance: GuidanceBase,
    dt: float,
) -> None:
    """Advance the full 6DOF state by one RK4 step of size dt."""
    # Snapshot the current primary state
    rx0, ry0, rz0 = state.rx, state.ry, state.rz
    vx0, vy0, vz0 = state.vx_eci, state.vy_eci, state.vz_eci
    qw0, qx0, qy0, qz0 = state.qw, state.qx, state.qy, state.qz
    wx0, wy0, wz0 = state.omega_x, state.omega_y, state.omega_z

    # Evaluate guidance + attitude controller ONCE (zero-order hold over the step).
    # Re-evaluating at intermediate RK4 stages with modified state would couple
    # the feedback loop into the integrator and cause numerical instability.
    delta_pitch, delta_yaw = compute_control_commands(state, vehicle, guidance)

    def _unpack(d):
        return (d[0],d[1],d[2],  d[3],d[4],d[5],
                d[6],d[7],d[8],d[9],  d[10],d[11],d[12])

    # k1 — derivative at current state
    k1 = _unpack(compute_derivatives(state, vehicle, delta_pitch, delta_yaw))

    # k2 — derivative at midpoint using k1
    state.rx,  state.ry,  state.rz  = rx0+0.5*dt*k1[0],  ry0+0.5*dt*k1[1],  rz0+0.5*dt*k1[2]
    state.vx_eci, state.vy_eci, state.vz_eci = vx0+0.5*dt*k1[3], vy0+0.5*dt*k1[4], vz0+0.5*dt*k1[5]
    qm = quat_normalize((qw0+0.5*dt*k1[6], qx0+0.5*dt*k1[7], qy0+0.5*dt*k1[8], qz0+0.5*dt*k1[9]))
    state.qw, state.qx, state.qy, state.qz = qm
    state.omega_x, state.omega_y, state.omega_z = wx0+0.5*dt*k1[10], wy0+0.5*dt*k1[11], wz0+0.5*dt*k1[12]
    k2 = _unpack(compute_derivatives(state, vehicle, delta_pitch, delta_yaw))

    # k3 — derivative at midpoint using k2
    state.rx,  state.ry,  state.rz  = rx0+0.5*dt*k2[0],  ry0+0.5*dt*k2[1],  rz0+0.5*dt*k2[2]
    state.vx_eci, state.vy_eci, state.vz_eci = vx0+0.5*dt*k2[3], vy0+0.5*dt*k2[4], vz0+0.5*dt*k2[5]
    qm = quat_normalize((qw0+0.5*dt*k2[6], qx0+0.5*dt*k2[7], qy0+0.5*dt*k2[8], qz0+0.5*dt*k2[9]))
    state.qw, state.qx, state.qy, state.qz = qm
    state.omega_x, state.omega_y, state.omega_z = wx0+0.5*dt*k2[10], wy0+0.5*dt*k2[11], wz0+0.5*dt*k2[12]
    k3 = _unpack(compute_derivatives(state, vehicle, delta_pitch, delta_yaw))

    # k4 — derivative at full step using k3
    state.rx,  state.ry,  state.rz  = rx0+dt*k3[0],  ry0+dt*k3[1],  rz0+dt*k3[2]
    state.vx_eci, state.vy_eci, state.vz_eci = vx0+dt*k3[3], vy0+dt*k3[4], vz0+dt*k3[5]
    qm = quat_normalize((qw0+dt*k3[6], qx0+dt*k3[7], qy0+dt*k3[8], qz0+dt*k3[9]))
    state.qw, state.qx, state.qy, state.qz = qm
    state.omega_x, state.omega_y, state.omega_z = wx0+dt*k3[10], wy0+dt*k3[11], wz0+dt*k3[12]
    k4 = _unpack(compute_derivatives(state, vehicle, delta_pitch, delta_yaw))

    # Final RK4 combination
    def rk4(s0, k1v, k2v, k3v, k4v):
        return s0 + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)

    state.rx     = rk4(rx0, k1[0],  k2[0],  k3[0],  k4[0])
    state.ry     = rk4(ry0, k1[1],  k2[1],  k3[1],  k4[1])
    state.rz     = rk4(rz0, k1[2],  k2[2],  k3[2],  k4[2])
    state.vx_eci = rk4(vx0, k1[3],  k2[3],  k3[3],  k4[3])
    state.vy_eci = rk4(vy0, k1[4],  k2[4],  k3[4],  k4[4])
    state.vz_eci = rk4(vz0, k1[5],  k2[5],  k3[5],  k4[5])

    qw_new = rk4(qw0, k1[6],  k2[6],  k3[6],  k4[6])
    qx_new = rk4(qx0, k1[7],  k2[7],  k3[7],  k4[7])
    qy_new = rk4(qy0, k1[8],  k2[8],  k3[8],  k4[8])
    qz_new = rk4(qz0, k1[9],  k2[9],  k3[9],  k4[9])
    q_final = quat_normalize((qw_new, qx_new, qy_new, qz_new))
    state.qw, state.qx, state.qy, state.qz = q_final

    state.omega_x = rk4(wx0, k1[10], k2[10], k3[10], k4[10])
    state.omega_y = rk4(wy0, k1[11], k2[11], k3[11], k4[11])
    state.omega_z = rk4(wz0, k1[12], k2[12], k3[12], k4[12])

    # Update arc-length accumulator (tangential speed × dt)
    state.x += state.vx * dt


def _fire_ready_events(state: SimState, events: list[Event]) -> bool:
    for event in events:
        if event.check(state):
            event.trigger(state)
            if event.terminal:
                state.record()
                return True
    return False


def run(
    state: SimState,
    vehicle: Vehicle,
    guidance: GuidanceBase,
    events: list[Event],
    t_end_s: float,
    dt: float,
) -> SimState:
    from src.atmosphere.us_standard_1976 import pressure

    while state.t < t_end_s:
        _update_derived(state)
        # Update guidance internal state (PEG checks cutoff, etc.) via a pitch query
        _ = compute_control_commands(state, vehicle, guidance)
        if _fire_ready_events(state, events):
            return state

        ambient_pressure = pressure(state.altitude)
        vehicle.mass_model.burn(dt, ambient_pressure, engine_on=state.engine_on)
        # Sync bookkeeping fields on state
        state.mass_kg = vehicle.mass
        state.propellant_remaining_kg = vehicle.mass_model.propellant_remaining_kg

        _rk4_step(state, vehicle, guidance, dt)
        state.t += dt

        _update_derived(state)
        if _fire_ready_events(state, events):
            return state

        # Terminal condition: vehicle hit the ground
        if state.altitude < 0.0:
            # Clamp to surface (nudge r_mag to R_EARTH)
            r = state.r_mag
            if r > 0:
                scale = (r + abs(state.altitude)) / r
                state.rx *= scale
                state.ry *= scale
                state.rz *= scale
            return state

        state.record()

    return state
