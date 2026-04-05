"""
elements.py — Compute classical orbital elements from the 6DOF ECI state.

Uses the full 3D position (rx, ry, rz) and velocity (vx_eci, vy_eci, vz_eci)
vectors in the Earth-Centered Inertial frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.propagator.state import SimState

MU = 3.986004418e14    # Earth gravitational parameter [m³/s²]
R_EARTH = 6_371_000.0  # Earth mean radius [m]


@dataclass
class OrbitalElements:
    semi_major_axis_m: float
    eccentricity: float
    perigee_alt_km: float
    apogee_alt_km: float
    inclination_deg: float
    specific_energy_J_kg: float
    specific_angular_momentum: float   # magnitude of h = r × v
    orbital_period_s: float
    circular_velocity_m_s: float

    def __str__(self) -> str:
        return (
            f"Perigee:      {self.perigee_alt_km:.1f} km\n"
            f"Apogee:       {self.apogee_alt_km:.1f} km\n"
            f"SMA:          {self.semi_major_axis_m/1e3:.1f} km\n"
            f"Eccentricity: {self.eccentricity:.6f}\n"
            f"Inclination:  {self.inclination_deg:.2f} deg\n"
            f"Period:       {self.orbital_period_s/60:.1f} min\n"
            f"v_circ:       {self.circular_velocity_m_s:.1f} m/s"
        )


def elements_from_state(state: SimState) -> OrbitalElements:
    """
    Derive classical orbital elements from the 6DOF ECI state at SECO.

    Works with full 3D r and v vectors.
    """
    rx, ry, rz       = state.rx, state.ry, state.rz
    vx, vy, vz       = state.vx_eci, state.vy_eci, state.vz_eci

    r_mag = math.sqrt(rx*rx + ry*ry + rz*rz)
    v_sq  = vx*vx + vy*vy + vz*vz

    # Specific orbital energy
    epsilon = v_sq / 2.0 - MU / r_mag
    if epsilon >= 0.0:
        raise ValueError(f"Vehicle is not on a bound orbit.  ε = {epsilon:.2f} J/kg")

    # Semi-major axis
    a = -MU / (2.0 * epsilon)

    # Specific angular momentum vector  h = r × v
    hx = ry*vz - rz*vy
    hy = rz*vx - rx*vz
    hz = rx*vy - ry*vx
    h_mag = math.sqrt(hx*hx + hy*hy + hz*hz)

    # Eccentricity
    e = math.sqrt(max(0.0, 1.0 + (2.0 * epsilon * h_mag**2) / MU**2))

    # Periapsis and apoapsis radii
    r_p = a * (1.0 - e)
    r_a = a * (1.0 + e)

    # Inclination: angle between h and z-axis
    inclination_deg = math.degrees(math.acos(max(-1.0, min(1.0, hz / h_mag)))) if h_mag > 0 else 0.0

    # Orbital period
    period = 2.0 * math.pi * math.sqrt(a**3 / MU)

    # Circular velocity at current altitude
    v_circ = math.sqrt(MU / r_mag)

    return OrbitalElements(
        semi_major_axis_m=a,
        eccentricity=e,
        perigee_alt_km=(r_p - R_EARTH) / 1e3,
        apogee_alt_km=(r_a - R_EARTH) / 1e3,
        inclination_deg=inclination_deg,
        specific_energy_J_kg=epsilon,
        specific_angular_momentum=h_mag,
        orbital_period_s=period,
        circular_velocity_m_s=v_circ,
    )
