"""
insertion.py — Compare achieved orbital elements against the target orbit.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import TargetOrbitConfig
from src.orbital.elements import OrbitalElements


@dataclass
class InsertionResult:
    achieved: OrbitalElements
    target_perigee_km: float
    target_apogee_km: float
    delta_perigee_km: float    # achieved - target
    delta_apogee_km: float
    success: bool              # True if both within tolerance

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "MISS"
        return (
            f"Insertion [{status}]\n"
            f"  Perigee: {self.achieved.perigee_alt_km:.1f} km "
            f"(target {self.target_perigee_km:.1f}, d{self.delta_perigee_km:+.1f} km)\n"
            f"  Apogee:  {self.achieved.apogee_alt_km:.1f} km "
            f"(target {self.target_apogee_km:.1f}, d{self.delta_apogee_km:+.1f} km)"
        )


def evaluate_insertion(
    achieved: OrbitalElements,
    target: TargetOrbitConfig,
    tolerance_km: float = 5.0,
) -> InsertionResult:
    """
    Compare achieved orbit to target. Success if both perigee and apogee
    are within tolerance_km of their targets.
    """
    d_perigee = achieved.perigee_alt_km - target.perigee_km
    d_apogee = achieved.apogee_alt_km - target.apogee_km

    success = abs(d_perigee) <= tolerance_km and abs(d_apogee) <= tolerance_km

    return InsertionResult(
        achieved=achieved,
        target_perigee_km=target.perigee_km,
        target_apogee_km=target.apogee_km,
        delta_perigee_km=d_perigee,
        delta_apogee_km=d_apogee,
        success=success,
    )
