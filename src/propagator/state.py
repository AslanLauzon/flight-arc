"""
state.py — SimState dataclass: the single object that flows through every module.

State vector — polar-coordinate 2D (3DOF):
    x  [m]   — arc-length downrange (θ = x / R_Earth radians)
    y  [m]   — altitude above surface (r = R_Earth + y)
    vx [m/s] — tangential velocity   (positive prograde)
    vy [m/s] — radial velocity        (positive outward / ascending)

Additional bookkeeping fields track propellant, current stage, and events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimState:
    # ── Primary state vector ─────────────────────────────────────────────────
    t:    float = 0.0      # simulation time [s]
    x:    float = 0.0      # downrange position [m]
    y:    float = 0.0      # altitude [m]
    vx:   float = 0.0      # downrange velocity [m/s]
    vy:   float = 0.0      # vertical velocity [m/s]

    # ── Engine control ──────────────────────────────────────────────────────
    engine_on: bool = True        # False during coast between burns

    # ── Derived / cached quantities (updated each step) ──────────────────────
    dynamic_pressure_Pa: float = 0.0
    mach:                float = 0.0
    flight_path_angle_deg: float = 90.0   # 90° = vertical

    # ── Event log ────────────────────────────────────────────────────────────
    events_triggered: list[str] = field(default_factory=list)

    # ── Trajectory history (accumulated by integrator) ────────────────────────
    history: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        """Return a lightweight dict of the current state (no history copy)."""
        return {
            "t":    self.t,
            "x":    self.x,
            "y":    self.y,
            "vx":   self.vx,
            "vy":   self.vy,
            "dynamic_pressure_Pa": self.dynamic_pressure_Pa,
            "mach":                self.mach,
            "flight_path_angle_deg": self.flight_path_angle_deg,
        }

    def record(self) -> None:
        """Append current snapshot to history."""
        self.history.append(self.snapshot())

    @property
    def speed(self) -> float:
        return (self.vx**2 + self.vy**2) ** 0.5

    @property
    def altitude(self) -> float:
        return self.y
