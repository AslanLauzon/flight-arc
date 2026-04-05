"""
vehicle.py — Vehicle model for 6DOF simulation.

Exposes:
  thrust(altitude_m)               — scalar thrust magnitude [N]
  drag(alt, vx, vy)                — backward-compat 2D drag tuple (unused in 6DOF EOM)
  drag_vector_eci(state)           — 3D ECI drag force vector [N]
  thrust_vector_eci(state, δp, δy) — 3D ECI thrust force vector [N]
  aero_moment_body(state)          — 3D body-frame aero torque [N·m]
  inertia_tensor(state)            — (Ixx, Iyy, Izz) [kg·m²]
"""
from __future__ import annotations

import math

from src.atmosphere.us_standard_1976 import density, pressure, speed_of_sound
from src.attitude.quaternion import quat_rotate, quat_conj
from src.config import VehicleConfig
from src.vehicle.mass_model import MassModel
from src.vehicle.stage import Stage

OMEGA_EARTH = 7.2921150e-5  # Earth rotation rate [rad/s]


class Vehicle:
    def __init__(self, cfg: VehicleConfig) -> None:
        self.name = cfg.name
        self.reference_area_m2 = cfg.reference_area_m2
        self.stages = [Stage(stage_cfg) for stage_cfg in cfg.stages]
        self.mass_model = MassModel(
            self.stages,
            cfg.payload.mass_kg,
            cfg.payload.fairing_mass_kg,
            fairing_jettisoned=cfg.payload.fairing_jettisoned,
        )

    @property
    def mass(self) -> float:
        return self.mass_model.total_mass()

    # ------------------------------------------------------------------
    # Scalar thrust (kept for guidance / event code compatibility)
    # ------------------------------------------------------------------
    def thrust(self, altitude_m: float) -> float:
        stage = self.mass_model.current_stage
        ambient_pressure_pa = pressure(altitude_m)
        return stage.thrust_vac_N * (stage.effective_isp(ambient_pressure_pa) / stage.isp_vac_s)

    # ------------------------------------------------------------------
    # 3D drag force in ECI frame
    # ------------------------------------------------------------------
    def drag_vector_eci(
        self,
        rx: float, ry: float, rz: float,
        vx_eci: float, vy_eci: float, vz_eci: float,
        altitude_m: float,
    ) -> tuple[float, float, float]:
        """
        Return drag force [N] in ECI frame.

        Uses velocity relative to the rotating atmosphere:
            v_rel = v_eci − ω_E × r_eci
        """
        # Atmospheric rotation (ω × r), ω = [0, 0, ω_E]
        v_atm_x = -OMEGA_EARTH * ry
        v_atm_y =  OMEGA_EARTH * rx
        v_atm_z = 0.0

        vrel_x = vx_eci - v_atm_x
        vrel_y = vy_eci - v_atm_y
        vrel_z = vz_eci - v_atm_z

        v_rel = math.sqrt(vrel_x ** 2 + vrel_y ** 2 + vrel_z ** 2)
        if v_rel < 1e-6:
            return (0.0, 0.0, 0.0)

        rho  = density(altitude_m)
        a_s  = speed_of_sound(altitude_m)
        mach = v_rel / a_s if a_s > 0.0 else 0.0
        cd   = self.mass_model.current_stage.drag_coefficient(mach)
        d    = 0.5 * rho * v_rel ** 2 * cd * self.reference_area_m2

        # Drag opposes relative velocity
        scale = -d / v_rel
        return (scale * vrel_x, scale * vrel_y, scale * vrel_z)

    # ------------------------------------------------------------------
    # 3D thrust force in ECI frame (with TVC gimbal deflection)
    # ------------------------------------------------------------------
    def thrust_vector_eci(
        self,
        altitude_m: float,
        engine_on: bool,
        qw: float, qx: float, qy: float, qz: float,
        delta_pitch: float,  # gimbal deflection in body pitch plane [rad]
        delta_yaw: float,    # gimbal deflection in body yaw plane [rad]
    ) -> tuple[float, float, float]:
        """
        Return thrust force [N] in ECI frame.

        The nominal thrust axis is body-x (+1, 0, 0).  TVC deflects the
        nozzle by (delta_pitch, delta_yaw) to produce a net side force
        for attitude control.

        Thrust direction in body frame (small-angle TVC):
            t_body ≈ (cos δ_p cos δ_y,  sin δ_p,  sin δ_y)
        normalised to unit length.
        """
        if not engine_on:
            return (0.0, 0.0, 0.0)

        T = self.thrust(altitude_m)

        # Thrust direction in body frame (TVC nozzle deflection)
        tx = math.cos(delta_pitch) * math.cos(delta_yaw)
        ty = math.sin(delta_pitch)
        tz = math.sin(delta_yaw)
        n  = math.sqrt(tx * tx + ty * ty + tz * tz)
        t_body = (tx / n, ty / n, tz / n)

        # Rotate body→ECI
        q = (qw, qx, qy, qz)
        t_eci = quat_rotate(q, t_body)
        return (T * t_eci[0], T * t_eci[1], T * t_eci[2])

    # ------------------------------------------------------------------
    # Aerodynamic pitch moment in body frame
    # ------------------------------------------------------------------
    def aero_moment_body(
        self,
        qw: float, qx: float, qy: float, qz: float,
        rx: float, ry: float,
        vx_eci: float, vy_eci: float, vz_eci: float,
        altitude_m: float,
        dynamic_pressure_Pa: float,
    ) -> tuple[float, float, float]:
        """
        Return aerodynamic moment [N·m] in body frame.

        Only pitch and yaw moments are modelled (via Cm·α); roll moment is zero.

        Angle-of-attack is the angle between the body-x axis and the airspeed
        vector (relative to the rotating atmosphere) in each body-frame plane.
        """
        if dynamic_pressure_Pa < 1.0:
            return (0.0, 0.0, 0.0)

        q = (qw, qx, qy, qz)
        # Velocity relative to rotating atmosphere in ECI frame
        v_rel_eci = (
            vx_eci - (-OMEGA_EARTH * ry),
            vy_eci - ( OMEGA_EARTH * rx),
            vz_eci,
        )
        # Airspeed in body frame
        v_body = quat_rotate(quat_conj(q), v_rel_eci)

        vb_x, vb_y, vb_z = v_body
        v_mag = math.sqrt(vb_x ** 2 + vb_y ** 2 + vb_z ** 2)
        if v_mag < 1.0:
            return (0.0, 0.0, 0.0)

        stage = self.mass_model.current_stage
        q_ref = dynamic_pressure_Pa * self.reference_area_m2 * stage.aero_ref_length_m

        # AoA in pitch plane: atan2(vb_y, vb_x)
        alpha_pitch = math.atan2(vb_y, vb_x)
        # AoA in yaw plane: atan2(vb_z, vb_x)
        alpha_yaw = math.atan2(vb_z, vb_x)

        cm = stage.cm_alpha_per_rad
        m_pitch = cm * alpha_pitch * q_ref   # pitch moment (body-y axis)
        m_yaw   = cm * alpha_yaw   * q_ref   # yaw moment (body-z axis)

        return (0.0, m_pitch, m_yaw)

    # ------------------------------------------------------------------
    # Inertia tensor (diagonal)
    # ------------------------------------------------------------------
    def inertia_tensor(self, propellant_remaining_kg: float) -> tuple[float, float, float]:
        """Return (Ixx, Iyy, Izz) [kg·m²] for the current stage and propellant load."""
        return self.mass_model.current_stage.inertia_tensor(propellant_remaining_kg)

    # ------------------------------------------------------------------
    # Backward-compat 2D drag (used only by legacy code paths)
    # ------------------------------------------------------------------
    def drag(self, altitude_m: float, vx: float, vy: float) -> tuple[float, float]:
        speed = (vx ** 2 + vy ** 2) ** 0.5
        if speed < 1e-6:
            return 0.0, 0.0
        rho  = density(altitude_m)
        mach = speed / speed_of_sound(altitude_m)
        cd   = self.mass_model.current_stage.drag_coefficient(mach)
        drag_total = 0.5 * rho * speed ** 2 * cd * self.reference_area_m2
        return -drag_total * (vx / speed), -drag_total * (vy / speed)
