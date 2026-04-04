"""
equations_of_motion.py — Polar-coordinate 2D equations of motion.

Coordinate system
-----------------
The simulation uses a 2D polar frame centred on Earth:

    y  [m]   — altitude above surface  (r = R_Earth + y, positive outward)
    x  [m]   — arc-length downrange    (θ = x / R_Earth, positive east)
    vx [m/s] — tangential velocity     (positive prograde)
    vy [m/s] — radial velocity          (positive outward / ascending)

Equations of motion
-------------------
For a point mass in a central-force field with thrust and drag:

    ay = (Ty + Dy) / m  -  μ/r²  +  vx²/r      (radial)
    ax = (Tx + Dx) / m  -  vy·vx / r            (tangential)

where
    μ/r²     — gravitational acceleration (inward)
    vx²/r    — centrifugal acceleration   (outward)
    vy·vx/r  — Coriolis term              (tangential, opposes prograde when climbing)

Physical consistency: at circular orbital velocity (vx = √(μ/r)), centrifugal
exactly cancels gravity, so ay = 0 and the vehicle maintains altitude —
consistent with Keplerian orbit mechanics.  Without the centrifugal term the
propagator is inconsistent with the orbital-element math used for insertion
assessment, and closed-loop guidance (PEG) cannot converge.
"""

import math

from src.gravity.gravity import gravity
from src.guidance.base import GuidanceBase
from src.propagator.state import SimState
from src.vehicle.vehicle import Vehicle

R_EARTH = 6_371_000.0   # [m]


def compute_accelerations(
    state: SimState,
    vehicle: Vehicle,
    guidance: GuidanceBase,
) -> tuple[float, float]:
    """
    Return (ax, ay) [m/s²] — tangential and radial accelerations.

    Pitch convention: 0° = tangential (prograde), 90° = radial (straight up).
    """
    pitch_rad = math.radians(guidance.pitch_angle_deg(state))
    T  = vehicle.thrust(state.y) if state.engine_on else 0.0
    Tx = T * math.cos(pitch_rad)   # tangential component
    Ty = T * math.sin(pitch_rad)   # radial component

    Dx, Dy = vehicle.drag(state.y, state.vx, state.vy)

    mass = vehicle.mass
    r    = R_EARTH + state.y

    centrifugal = state.vx ** 2 / r          # outward; cancels gravity at orbital speed
    coriolis    = state.vy * state.vx / r    # tangential; decelerates prograde when climbing

    ax = (Tx + Dx) / mass - coriolis
    ay = (Ty + Dy) / mass - gravity(state.y) + centrifugal

    return ax, ay
