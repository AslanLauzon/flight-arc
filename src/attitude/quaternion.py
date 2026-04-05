"""
quaternion.py — Unit quaternion math for 6DOF attitude representation.

Convention: scalar-first  q = (qw, qx, qy, qz)
  where q represents the rotation from body frame to ECI frame.

To rotate a vector v from body frame to ECI frame:
    v_eci = quat_rotate(q, v_body)

To rotate a vector from ECI to body frame:
    v_body = quat_rotate(quat_conj(q), v_eci)
"""
from __future__ import annotations

import math

Quat = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]


def quat_mult(q1: Quat, q2: Quat) -> Quat:
    """Hamilton product q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_conj(q: Quat) -> Quat:
    """Conjugate (= inverse for unit quaternion)."""
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_normalize(q: Quat) -> Quat:
    """Return unit quaternion; returns identity if near-zero magnitude."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    """
    Rotate vector v by quaternion q.

    If q is the body-to-ECI quaternion, passing a body-frame vector
    returns the ECI-frame vector.
    """
    # v' = q ⊗ [0, v] ⊗ q*
    qv: Quat = (0.0, v[0], v[1], v[2])
    result = quat_mult(quat_mult(q, qv), quat_conj(q))
    return (result[1], result[2], result[3])


def quat_kinematics(q: Quat, omega: Vec3) -> Quat:
    """
    Compute dq/dt from current quaternion and body angular rates [rad/s].

    Implements: dq/dt = 0.5 * q ⊗ [0, ωx, ωy, ωz]
    """
    omega_q: Quat = (0.0, omega[0], omega[1], omega[2])
    dq = quat_mult(q, omega_q)
    return (0.5 * dq[0], 0.5 * dq[1], 0.5 * dq[2], 0.5 * dq[3])


def quat_from_two_vectors(v_from: Vec3, v_to: Vec3) -> Quat:
    """
    Shortest-arc quaternion rotating unit vector v_from onto unit vector v_to.
    Both inputs must be unit vectors.
    """
    dot = v_from[0] * v_to[0] + v_from[1] * v_to[1] + v_from[2] * v_to[2]
    dot = max(-1.0, min(1.0, dot))

    # Anti-parallel: 180-degree rotation about any perpendicular axis
    if dot < -1.0 + 1e-9:
        perp: Vec3 = (0.0, 1.0, 0.0) if abs(v_from[0]) < 0.9 else (0.0, 0.0, 1.0)
        ax = v_from[1] * perp[2] - v_from[2] * perp[1]
        ay = v_from[2] * perp[0] - v_from[0] * perp[2]
        az = v_from[0] * perp[1] - v_from[1] * perp[0]
        n = math.sqrt(ax * ax + ay * ay + az * az)
        return (0.0, ax / n, ay / n, az / n)

    # General case: w = 1 + cos θ, axis = sin θ * n_hat
    w = math.sqrt((1.0 + dot) / 2.0)
    if w < 1e-12:
        return (0.0, 1.0, 0.0, 0.0)
    s = 1.0 / (2.0 * w)
    ax = (v_from[1] * v_to[2] - v_from[2] * v_to[1]) * s
    ay = (v_from[2] * v_to[0] - v_from[0] * v_to[2]) * s
    az = (v_from[0] * v_to[1] - v_from[1] * v_to[0]) * s
    return quat_normalize((w, ax, ay, az))


def vec3_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec3_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_norm(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def vec3_normalize(a: Vec3) -> Vec3:
    n = vec3_norm(a)
    if n < 1e-12:
        return (1.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def vec3_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def vec3_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
