import math

from src.atmosphere.us_standard_1976 import dynamic_pressure, speed_of_sound
from src.events.event import Event
from src.guidance.base import GuidanceBase
from src.propagator.equations_of_motion import compute_accelerations
from src.propagator.state import SimState
from src.vehicle.vehicle import Vehicle


def _update_derived(state: SimState) -> None:
    speed = state.speed
    state.dynamic_pressure_Pa = dynamic_pressure(state.y, speed)
    state.mach = speed / speed_of_sound(state.y) if speed_of_sound(state.y) > 0 else 0.0
    if speed > 1e-6:
        state.flight_path_angle_deg = math.degrees(math.atan2(state.vy, state.vx))


def _rk4_step(
    state: SimState,
    vehicle: Vehicle,
    guidance: GuidanceBase,
    dt: float,
) -> None:
    x0, y0, vx0, vy0 = state.x, state.y, state.vx, state.vy

    ax1, ay1 = compute_accelerations(state, vehicle, guidance)

    state.x, state.y = x0 + 0.5 * dt * vx0, y0 + 0.5 * dt * vy0
    state.vx, state.vy = vx0 + 0.5 * dt * ax1, vy0 + 0.5 * dt * ay1
    ax2, ay2 = compute_accelerations(state, vehicle, guidance)

    state.x, state.y = x0 + 0.5 * dt * (vx0 + 0.5 * dt * ax1), y0 + 0.5 * dt * (vy0 + 0.5 * dt * ay1)
    state.vx, state.vy = vx0 + 0.5 * dt * ax2, vy0 + 0.5 * dt * ay2
    ax3, ay3 = compute_accelerations(state, vehicle, guidance)

    state.x, state.y = x0 + dt * (vx0 + 0.5 * dt * ax2), y0 + dt * (vy0 + 0.5 * dt * ay2)
    state.vx, state.vy = vx0 + dt * ax3, vy0 + dt * ay3
    ax4, ay4 = compute_accelerations(state, vehicle, guidance)

    state.x = x0 + dt * vx0 + (dt**2 / 6.0) * (ax1 + ax2 + ax3)
    state.y = y0 + dt * vy0 + (dt**2 / 6.0) * (ay1 + ay2 + ay3)
    state.vx = vx0 + (dt / 6.0) * (ax1 + 2 * ax2 + 2 * ax3 + ax4)
    state.vy = vy0 + (dt / 6.0) * (ay1 + 2 * ay2 + 2 * ay3 + ay4)


def _fire_ready_events(state: SimState, events: list[Event]) -> bool:
    for event in events:
        if event.check(state):
            event.trigger(state)
            if event.terminal:
                state.record()
                return True
    return False


def run(
    state: SimState,
    vehicle: Vehicle,
    guidance: GuidanceBase,
    events: list[Event],
    t_end_s: float,
    dt: float,
) -> SimState:
    from src.atmosphere.us_standard_1976 import pressure

    while state.t < t_end_s:
        _update_derived(state)
        guidance.pitch_angle_deg(state)
        if _fire_ready_events(state, events):
            return state

        ambient_pressure = pressure(state.y)
        vehicle.mass_model.burn(dt, ambient_pressure, engine_on=state.engine_on)

        _rk4_step(state, vehicle, guidance, dt)
        state.t += dt

        _update_derived(state)
        if _fire_ready_events(state, events):
            return state

        if state.y < 0.0:
            state.y = 0.0
            return state

        state.record()

    return state
