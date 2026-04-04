"""
elements.py — Compute classical orbital elements from a 3DOF SECO state.

Coordinate convention (polar-coordinate 2D):
    x  — arc-length downrange [m]  (θ = x / R_Earth)
    y  — altitude above surface [m] (r = R_Earth + y)
    vx — tangential velocity [m/s]  (prograde)
    vy — radial velocity [m/s]      (positive outbound)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.propagator.state import SimState

MU = 3.986004418e14   # Earth gravitational parameter [m³/s²]
R_EARTH = 6_371_000.0  # Earth mean radius [m]


@dataclass
class OrbitalElements:
    semi_major_axis_m: float        # a [m]
    eccentricity: float             # e [-]
    perigee_alt_km: float           # h_p [km]
    apogee_alt_km: float            # h_a [km]
    specific_energy_J_kg: float     # ε [J/kg]
    specific_angular_momentum: float  # h [m²/s]
    orbital_period_s: float         # T [s]
    circular_velocity_m_s: float    # v_c at insertion altitude [m/s]

    def __str__(self) -> str:
        return (
            f"Perigee:  {self.perigee_alt_km:.1f} km\n"
            f"Apogee:   {self.apogee_alt_km:.1f} km\n"
            f"SMA:      {self.semi_major_axis_m/1e3:.1f} km\n"
            f"Ecc:      {self.eccentricity:.6f}\n"
            f"Period:   {self.orbital_period_s/60:.1f} min\n"
            f"v_circ:   {self.circular_velocity_m_s:.1f} m/s"
        )


def elements_from_state(state: SimState) -> OrbitalElements:
    """
    Derive orbital elements from the 2D flat-Earth state at SECO.

    r — radial distance from Earth centre (R_earth + altitude)
    v_t — tangential velocity (vx, downrange)
    v_r — radial velocity (vy, vertical)
    """
    r = R_EARTH + state.y
    v_t = state.vx   # tangential
    v_r = state.vy   # radial

    v_sq = v_t**2 + v_r**2

    # specific orbital energy: ε = v²/2 - μ/r
    epsilon = v_sq / 2.0 - MU / r

    # semi-major axis: a = -μ / (2ε)
    # negative epsilon = bound orbit
    if epsilon >= 0:
        raise ValueError(f"Vehicle is not on a bound orbit. ε = {epsilon:.2f} J/kg")
    a = -MU / (2.0 * epsilon)

    # specific angular momentum: h = r × v_t (scalar in 2D)
    h = r * v_t

    # eccentricity: e = sqrt(1 + 2ε h² / μ²)
    e = math.sqrt(max(0.0, 1.0 + (2.0 * epsilon * h**2) / MU**2))

    # periapsis and apoapsis radii
    r_p = a * (1.0 - e)
    r_a = a * (1.0 + e)

    # orbital period: T = 2π sqrt(a³/μ)
    period = 2.0 * math.pi * math.sqrt(a**3 / MU)

    # circular velocity at insertion altitude
    v_circ = math.sqrt(MU / r)

    return OrbitalElements(
        semi_major_axis_m=a,
        eccentricity=e,
        perigee_alt_km=(r_p - R_EARTH) / 1e3,
        apogee_alt_km=(r_a - R_EARTH) / 1e3,
        specific_energy_J_kg=epsilon,
        specific_angular_momentum=h,
        orbital_period_s=period,
        circular_velocity_m_s=v_circ,
    )
