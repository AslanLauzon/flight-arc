"""
equations_of_motion.py — Full 6DOF equations of motion (ECI frame).

State vector:
    Translational:  rx, ry, rz [m]          — ECI position
                    vx, vy, vz [m/s]         — ECI velocity
    Rotational:     qw, qx, qy, qz           — body-to-ECI quaternion
                    ωx, ωy, ωz [rad/s]       — body angular rates

Translational dynamics
----------------------
    dr/dt = v_eci
    dv/dt = a_gravity + a_thrust + a_drag

Rotational kinematics
---------------------
    dq/dt = 0.5 * q ⊗ [0, ωx, ωy, ωz]

Rotational dynamics (Euler's equations in body frame)
------------------------------------------------------
    I · dω/dt = τ_total − ω × (I · ω)

    τ_total = τ_gimbal (thrust vector control) + τ_aero (aerodynamic moments)

Guidance coupling
-----------------
The guidance still outputs a pitch angle [deg].  This is converted to a
commanded ECI thrust direction in the orbital plane (spanned by r_hat and
the orbital-plane prograde direction).  The PD attitude controller turns
that into gimbal deflections, which produce both attitude torques AND a
small transverse force component through the TVC nozzle.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.attitude.controller import PdAttitudeController
from src.attitude.quaternion import (
    Vec3,
    quat_kinematics,
    quat_rotate,
    vec3_cross,
    vec3_normalize,
    vec3_sub,
)
from src.gravity.gravity import gravity_vector_eci
from src.guidance.base import GuidanceBase
from src.propagator.state import SimState
from src.vehicle.vehicle import Vehicle

if TYPE_CHECKING:
    pass

R_EARTH = 6_371_000.0


def _commanded_thrust_dir(state: SimState, pitch_deg: float) -> Vec3:
    """
    Convert a guidance pitch angle [deg] to a unit ECI thrust direction.

    Pitch convention:  90° = radial-up (vertical),  0° = prograde (horizontal).

    We work in the local orbital plane:
        radial  = r_hat (outward from Earth centre)
        tangent = unit vector in the orbital plane, perpendicular to r_hat,
                  pointing in the direction of increasing angular momentum
                  (approximately prograde).
    """
    pitch_rad = math.radians(pitch_deg)

    r_mag = math.sqrt(state.rx ** 2 + state.ry ** 2 + state.rz ** 2)
    if r_mag < 1.0:
        return (1.0, 0.0, 0.0)

    # Radial unit vector
    r_hat: Vec3 = (state.rx / r_mag, state.ry / r_mag, state.rz / r_mag)

    # Prograde: v_eci minus its radial component, then normalised.
    # At liftoff (v_eci ≈ 0) fall back to a eastward reference.
    v = (state.vx_eci, state.vy_eci, state.vz_eci)
    v_sq = v[0] ** 2 + v[1] ** 2 + v[2] ** 2

    if v_sq < 1.0:
        # No velocity yet — use East as prograde fallback
        lat = math.asin(max(-1.0, min(1.0, state.rz / r_mag)))
        lon = math.atan2(state.ry, state.rx)
        east: Vec3 = (-math.sin(lon), math.cos(lon), 0.0)
        tangent = vec3_normalize(
            vec3_sub(east, _proj(east, r_hat))
        )
    else:
        vr = v[0] * r_hat[0] + v[1] * r_hat[1] + v[2] * r_hat[2]
        t = (v[0] - vr * r_hat[0], v[1] - vr * r_hat[1], v[2] - vr * r_hat[2])
        t_mag = math.sqrt(t[0] ** 2 + t[1] ** 2 + t[2] ** 2)
        if t_mag < 1e-6:
            lat = math.asin(max(-1.0, min(1.0, state.rz / r_mag)))
            lon = math.atan2(state.ry, state.rx)
            east = (-math.sin(lon), math.cos(lon), 0.0)
            tangent = vec3_normalize(vec3_sub(east, _proj(east, r_hat)))
        else:
            tangent = (t[0] / t_mag, t[1] / t_mag, t[2] / t_mag)

    # Thrust direction: sin(pitch)*r_hat + cos(pitch)*tangent
    sp = math.sin(pitch_rad)
    cp = math.cos(pitch_rad)
    tx = sp * r_hat[0] + cp * tangent[0]
    ty = sp * r_hat[1] + cp * tangent[1]
    tz = sp * r_hat[2] + cp * tangent[2]
    n  = math.sqrt(tx * tx + ty * ty + tz * tz)
    if n < 1e-9:
        return r_hat
    return (tx / n, ty / n, tz / n)


def _proj(v: Vec3, onto: Vec3) -> Vec3:
    """Project v onto unit vector 'onto'."""
    d = v[0] * onto[0] + v[1] * onto[1] + v[2] * onto[2]
    return (d * onto[0], d * onto[1], d * onto[2])


def compute_control_commands(
    state: SimState,
    vehicle: Vehicle,
    guidance: GuidanceBase,
) -> tuple[float, float]:
    """
    Evaluate guidance and attitude controller at the current state.

    Returns (delta_pitch, delta_yaw) gimbal deflections [rad].
    Called ONCE per integration step; the result is held constant through
    all four RK4 sub-evaluations (zero-order hold approximation).
    """
    pitch_deg = guidance.pitch_angle_deg(state)
    thrust_dir_eci = _commanded_thrust_dir(state, pitch_deg)

    stage = vehicle.mass_model.current_stage
    ctrl = PdAttitudeController(
        kp=stage.attitude_kp,
        kd=stage.attitude_kd,
        max_deflection_deg=math.degrees(stage.gimbal_max_deflection_rad),
    )
    q: tuple[float, float, float, float] = (state.qw, state.qx, state.qy, state.qz)
    omega: Vec3 = (state.omega_x, state.omega_y, state.omega_z)
    return ctrl.gimbal_deflection(q, omega, thrust_dir_eci)


def compute_derivatives(
    state: SimState,
    vehicle: Vehicle,
    delta_pitch: float,
    delta_yaw: float,
) -> tuple[
    float, float, float,          # drx, dry, drz
    float, float, float,          # dvx, dvy, dvz
    float, float, float, float,   # dqw, dqx, dqy, dqz
    float, float, float,          # dωx, dωy, dωz
]:
    """
    Return all 13 time derivatives of the 6DOF state given fixed gimbal deflections.

    Called four times per RK4 step with the same (delta_pitch, delta_yaw).
    """

    q: Vec3 = (state.qw, state.qx, state.qy, state.qz)  # type: ignore[assignment]
    omega: Vec3 = (state.omega_x, state.omega_y, state.omega_z)
    stage = vehicle.mass_model.current_stage

    # ── Translational: position derivatives ─────────────────────────────
    drx, dry, drz = state.vx_eci, state.vy_eci, state.vz_eci

    # Gravity (ECI vector)
    gx, gy, gz = gravity_vector_eci(state.rx, state.ry, state.rz)

    # Thrust force (ECI vector, including TVC side force)
    alt = state.altitude
    Tx, Ty, Tz = vehicle.thrust_vector_eci(
        alt, state.engine_on,
        state.qw, state.qx, state.qy, state.qz,
        delta_pitch, delta_yaw,
    )

    # Drag force (ECI vector)
    Dx, Dy, Dz = vehicle.drag_vector_eci(
        state.rx, state.ry, state.rz,
        state.vx_eci, state.vy_eci, state.vz_eci,
        alt,
    )

    mass = vehicle.mass
    dvx = gx + (Tx + Dx) / mass
    dvy = gy + (Ty + Dy) / mass
    dvz = gz + (Tz + Dz) / mass

    # ── Rotational: attitude kinematics ─────────────────────────────────
    dq = quat_kinematics(q, omega)
    dqw, dqx, dqy, dqz = dq

    # ── Rotational: Euler's equations in body frame ──────────────────────
    Ixx, Iyy, Izz = vehicle.inertia_tensor(state.propellant_remaining_kg)

    # Gimbal torque: τ = T * sin(δ) * L_arm  (small angle ≈ T * δ * L_arm)
    T_mag = vehicle.thrust(alt) if state.engine_on else 0.0
    L = stage.gimbal_moment_arm_m
    tau_pitch = T_mag * math.sin(delta_pitch) * L
    tau_yaw   = T_mag * math.sin(delta_yaw)   * L
    tau_x_body, tau_y_body, tau_z_body = 0.0, tau_pitch, tau_yaw

    # Aerodynamic moment (body frame)
    mx, my, mz = vehicle.aero_moment_body(
        state.qw, state.qx, state.qy, state.qz,
        state.rx, state.ry,
        state.vx_eci, state.vy_eci, state.vz_eci,
        alt,
        state.dynamic_pressure_Pa,
    )
    tau_x_body += mx
    tau_y_body += my
    tau_z_body += mz

    # Euler: I dω/dt = τ − ω × (I ω)
    Iω_x = Ixx * omega[0]
    Iω_y = Iyy * omega[1]
    Iω_z = Izz * omega[2]

    # ω × (I ω)
    gyro_x = omega[1] * Iω_z - omega[2] * Iω_y
    gyro_y = omega[2] * Iω_x - omega[0] * Iω_z
    gyro_z = omega[0] * Iω_y - omega[1] * Iω_x

    dωx = (tau_x_body - gyro_x) / max(Ixx, 1e-3)
    dωy = (tau_y_body - gyro_y) / max(Iyy, 1e-3)
    dωz = (tau_z_body - gyro_z) / max(Izz, 1e-3)

    return (
        drx, dry, drz,
        dvx, dvy, dvz,
        dqw, dqx, dqy, dqz,
        dωx, dωy, dωz,
    )
