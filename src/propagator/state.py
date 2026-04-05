"""
state.py — SimState dataclass for the full 6DOF simulation.

Primary state vector (ECI frame):
    rx, ry, rz          [m]     — position in Earth-Centered Inertial frame
    vx_eci, vy_eci, vz_eci [m/s] — velocity in ECI frame
    qw, qx, qy, qz      [-]     — unit quaternion, body → ECI rotation
    omega_x, omega_y, omega_z [rad/s] — body-frame angular rates (p, q, r)

Backward-compatible properties (derived from ECI state):
    altitude / y  — radial altitude above Earth surface [m]
    vx            — tangential (prograde) speed [m/s]  (always ≥ 0)
    vy            — radial speed [m/s]  (positive outward)
    x             — accumulated arc-length downrange [m]  (integrated)

These allow all existing guidance, event, and report code to work
unchanged against the new 6DOF state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

R_EARTH = 6_371_000.0  # [m]


@dataclass
class SimState:
    # ── Simulation time ──────────────────────────────────────────────────────
    t: float = 0.0

    # ── 3D ECI position [m] ──────────────────────────────────────────────────
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    # ── 3D ECI velocity [m/s] ────────────────────────────────────────────────
    vx_eci: float = 0.0
    vy_eci: float = 0.0
    vz_eci: float = 0.0

    # ── Attitude quaternion (body → ECI), scalar-first ───────────────────────
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0

    # ── Body-frame angular rates [rad/s]: (p=roll, q=pitch, r=yaw) ──────────
    omega_x: float = 0.0  # p — roll  rate
    omega_y: float = 0.0  # q — pitch rate
    omega_z: float = 0.0  # r — yaw   rate

    # ── Backward-compatible arc-length accumulator ───────────────────────────
    x: float = 0.0   # downrange arc-length [m]  (updated by integrator)

    # ── Engine control ────────────────────────────────────────────────────────
    engine_on: bool = True

    # ── Vehicle bookkeeping ──────────────────────────────────────────────────
    mass_kg: float = 0.0
    stage_index: int = 0
    propellant_remaining_kg: float = 0.0

    # ── Derived / cached (updated each RK4 step) ─────────────────────────────
    dynamic_pressure_Pa: float = 0.0
    mach: float = 0.0
    flight_path_angle_deg: float = 90.0

    # ── Event log ─────────────────────────────────────────────────────────────
    events_triggered: list[str] = field(default_factory=list)

    # ── Trajectory history ────────────────────────────────────────────────────
    history: list[dict[str, Any]] = field(default_factory=list)

    # ── Derived properties (backward compatibility) ───────────────────────────

    @property
    def r_mag(self) -> float:
        """ECI radial distance from Earth center [m]."""
        return math.sqrt(self.rx * self.rx + self.ry * self.ry + self.rz * self.rz)

    @property
    def altitude(self) -> float:
        """Altitude above Earth's surface [m]."""
        return self.r_mag - R_EARTH

    @property
    def y(self) -> float:
        """Alias for altitude — backward compatibility."""
        return self.altitude

    @property
    def speed(self) -> float:
        """ECI speed [m/s]."""
        return math.sqrt(self.vx_eci ** 2 + self.vy_eci ** 2 + self.vz_eci ** 2)

    @property
    def vy(self) -> float:
        """
        Radial velocity component [m/s] (positive = moving away from Earth).
        dot(v_eci, r_hat)
        """
        r = self.r_mag
        if r < 1.0:
            return 0.0
        return (self.vx_eci * self.rx + self.vy_eci * self.ry + self.vz_eci * self.rz) / r

    @property
    def vx(self) -> float:
        """
        Tangential (prograde) speed [m/s] — component of velocity perpendicular
        to the radial direction.  Always non-negative.
        """
        v_sq = self.vx_eci ** 2 + self.vy_eci ** 2 + self.vz_eci ** 2
        vr = self.vy
        return math.sqrt(max(0.0, v_sq - vr * vr))

    def snapshot(self) -> dict[str, Any]:
        """Return a lightweight dict of the current state (no history copy)."""
        return {
            "t":    self.t,
            # ECI primary state
            "rx":   self.rx,
            "ry":   self.ry,
            "rz":   self.rz,
            "vx_eci": self.vx_eci,
            "vy_eci": self.vy_eci,
            "vz_eci": self.vz_eci,
            # Attitude
            "qw": self.qw,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            # Angular rates
            "omega_x": self.omega_x,
            "omega_y": self.omega_y,
            "omega_z": self.omega_z,
            # Backward-compat derived
            "x":    self.x,
            "y":    self.y,
            "vx":   self.vx,
            "vy":   self.vy,
            # Bookkeeping
            "mass_kg":                   self.mass_kg,
            "stage_index":               self.stage_index,
            "propellant_remaining_kg":   self.propellant_remaining_kg,
            "dynamic_pressure_Pa":       self.dynamic_pressure_Pa,
            "mach":                      self.mach,
            "flight_path_angle_deg":     self.flight_path_angle_deg,
        }

    def record(self) -> None:
        """Append current snapshot to history."""
        self.history.append(self.snapshot())
