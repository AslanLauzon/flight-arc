G0 = 9.80665        # standard gravity at sea level [m/s²]
R_EARTH = 6_371_000.0  # mean Earth radius [m]


def gravity(altitude_m: float) -> float:
    """
    Return gravitational acceleration [m/s²] at altitude.
    Uses inverse-square law: g(h) = G0 * (R / (R + h))²
    Always positive — direction is handled in the EOM.
    """
    return G0 * (R_EARTH / (R_EARTH + altitude_m)) ** 2
