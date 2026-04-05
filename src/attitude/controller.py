"""
controller.py — Proportional-Derivative attitude controller for 6DOF.

Converts a commanded thrust direction (ECI unit vector) into pitch and yaw
gimbal deflection angles.  The actual body torque from those deflections is
computed in equations_of_motion.py.

Control law (small-angle PD):
    δ_pitch = kp * θ_err_pitch − kd * q_rate
    δ_yaw   = kp * θ_err_yaw   − kd * r_rate

where θ_err_{pitch,yaw} are the pitch/yaw components of the error angle
between the current body-x axis and the commanded thrust direction.
"""
from __future__ import annotations

import math

from src.attitude.quaternion import (
    Quat,
    Vec3,
    quat_conj,
    quat_rotate,
)


class PdAttitudeController:
    def __init__(
        self,
        kp: float = 2.0,
        kd: float = 0.8,
        max_deflection_deg: float = 5.0,
    ) -> None:
        self.kp = kp
        self.kd = kd
        self.max_deflection_rad = math.radians(max_deflection_deg)

    def gimbal_deflection(
        self,
        q: Quat,
        omega: Vec3,
        thrust_dir_eci: Vec3,
    ) -> tuple[float, float]:
        """
        Return (delta_pitch, delta_yaw) gimbal deflections in radians.

        Parameters
        ----------
        q              : current body-to-ECI unit quaternion
        omega          : (p, q_rate, r) body angular rates [rad/s]
        thrust_dir_eci : desired thrust direction as ECI unit vector
        """
        # Desired thrust direction in body frame
        # q rotates body→ECI, so q* rotates ECI→body
        desired_body = quat_rotate(quat_conj(q), thrust_dir_eci)

        # Error angles: desired_body should be (1, 0, 0) when aligned.
        # dy = pitch error (body-y component), dz = yaw error (body-z component).
        # Small-angle: atan2(err, x_component) for robustness at large errors.
        bx = desired_body[0]
        by = desired_body[1]
        bz = desired_body[2]

        err_pitch = math.atan2(by, bx) if abs(bx) > 1e-6 else (math.pi / 2 if by > 0 else -math.pi / 2)
        err_yaw   = math.atan2(bz, bx) if abs(bx) > 1e-6 else (math.pi / 2 if bz > 0 else -math.pi / 2)

        _, q_rate, r_rate = omega  # p=roll, q=pitch, r=yaw rates

        delta_pitch = self.kp * err_pitch - self.kd * q_rate
        delta_yaw   = self.kp * err_yaw   - self.kd * r_rate

        # Saturate at structural limit
        delta_pitch = max(-self.max_deflection_rad, min(self.max_deflection_rad, delta_pitch))
        delta_yaw   = max(-self.max_deflection_rad, min(self.max_deflection_rad, delta_yaw))

        return delta_pitch, delta_yaw
