import math

G0 = 9.80665        # standard gravity at sea level [m/s²]
R_EARTH = 6_371_000.0  # mean Earth radius [m]
MU = 3.986004418e14    # Earth gravitational parameter [m³/s²]


def gravity(altitude_m: float) -> float:
    """
    Return gravitational acceleration magnitude [m/s²] at altitude.
    Uses inverse-square law: g(h) = G0 * (R / (R + h))²
    Always positive — direction is handled by the caller.
    """
    return G0 * (R_EARTH / (R_EARTH + altitude_m)) ** 2


def gravity_vector_eci(rx: float, ry: float, rz: float) -> tuple[float, float, float]:
    """
    Return gravitational acceleration vector [m/s²] in ECI frame.

    a_grav = -μ / |r|³ * r_vec   (points toward Earth center)
    """
    r_sq = rx * rx + ry * ry + rz * rz
    r = math.sqrt(r_sq)
    if r < 1.0:
        return (0.0, 0.0, 0.0)
    coeff = -MU / (r_sq * r)
    return (coeff * rx, coeff * ry, coeff * rz)
