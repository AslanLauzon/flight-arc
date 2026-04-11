import math

from src.atmosphere.us_standard_1976 import density_and_mach
from src.events.event import Event
from src.guidance.base import GuidanceBase
from src.propagator.equations_of_motion import compute_accelerations
from src.propagator.state import SimState
from src.vehicle.vehicle import Vehicle


def _update_derived(state: SimState) -> None:
    # Called twice per loop iteration (before and after RK4) so events
    # and guidance always see fresh q/mach values.
    speed = state.speed
    rho, mach = density_and_mach(state.y, speed)
    state.dynamic_pressure_Pa = 0.5 * rho * speed * speed
    state.mach = mach
    if speed > 1e-6:
        state.flight_path_angle_deg = math.degrees(math.atan2(state.vy, state.vx))


def _rk4_step(
    state: SimState,
    vehicle: Vehicle,
    guidance: GuidanceBase,
    dt: float,
) -> None:
    x0, y0, vx0, vy0 = state.x, state.y, state.vx, state.vy

    # k1 — derivatives at current state
    ax1, ay1 = compute_accelerations(state, vehicle, guidance)

    # k2 — derivatives at midpoint using k1 derivatives
    state.x, state.y = x0 + 0.5 * dt * vx0, y0 + 0.5 * dt * vy0
    state.vx, state.vy = vx0 + 0.5 * dt * ax1, vy0 + 0.5 * dt * ay1
    ax2, ay2 = compute_accelerations(state, vehicle, guidance)

    # k3 — derivatives at midpoint using k2 derivatives.
    # Position uses the k1-corrected midpoint velocity (not just vx0) to get
    # a more accurate midpoint position estimate than k2 used.
    state.x, state.y = x0 + 0.5 * dt * (vx0 + 0.5 * dt * ax1), y0 + 0.5 * dt * (vy0 + 0.5 * dt * ay1)
    state.vx, state.vy = vx0 + 0.5 * dt * ax2, vy0 + 0.5 * dt * ay2
    ax3, ay3 = compute_accelerations(state, vehicle, guidance)

    # k4 — derivatives at end of step using k3 derivatives
    state.x, state.y = x0 + dt * (vx0 + 0.5 * dt * ax2), y0 + dt * (vy0 + 0.5 * dt * ay2)
    state.vx, state.vy = vx0 + dt * ax3, vy0 + dt * ay3
    ax4, ay4 = compute_accelerations(state, vehicle, guidance)

    # Weighted combination: position uses k1-k3 velocities, velocity uses all four slopes
    state.x = x0 + dt * vx0 + (dt**2 / 6.0) * (ax1 + ax2 + ax3)
    state.y = y0 + dt * vy0 + (dt**2 / 6.0) * (ay1 + ay2 + ay3)
    state.vx = vx0 + (dt / 6.0) * (ax1 + 2 * ax2 + 2 * ax3 + ax4)
    state.vy = vy0 + (dt / 6.0) * (ay1 + 2 * ay2 + 2 * ay3 + ay4)


def _fire_ready_events(state: SimState, events: list[Event]) -> bool:
    for event in events:
        if event.check(state):
            event.trigger(state)
            if event.terminal:
                state.record()  # always capture the terminal state, even if record_history=False
                return True
    return False


def run(
    state: SimState,
    vehicle: Vehicle,
    guidance: GuidanceBase,
    events: list[Event],
    t_end_s: float,
    dt: float,
    record_history: bool = True,
) -> SimState:
    from src.atmosphere.us_standard_1976 import pressure

    while state.t < t_end_s:
        _update_derived(state)
        guidance.pitch_angle_deg(state)
        if _fire_ready_events(state, events):
            return state

        ambient_pressure = pressure(state.y)
        # Mass is burned once per step (zero-order hold) — not inside each RK4
        # substep. Thrust is evaluated at the sub-step altitude during RK4, but
        # the mass used in all four substep force calculations is the pre-burn mass.
        vehicle.mass_model.burn(dt, ambient_pressure, engine_on=state.engine_on)

        _rk4_step(state, vehicle, guidance, dt)
        state.t += dt

        _update_derived(state)
        if _fire_ready_events(state, events):
            return state

        if state.y < 0.0:
            state.y = 0.0
            return state

        if record_history:
            state.record()

    return state
