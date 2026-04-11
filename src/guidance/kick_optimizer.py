from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize

from src.config import MissionToolkitConfig
from src.events.autosequence import build_autosequence
from src.orbital.elements import elements_from_state
from src.propagator.integrator import run
from src.propagator.state import SimState
from src.vehicle.vehicle import Vehicle


@dataclass(frozen=True)
class KickSolution:
    kick_time_s: float
    kick_angle_deg: float
    objective: float


def optimize_kick(
    cfg: MissionToolkitConfig,
    initial_kick_time_s: float,
    initial_kick_angle_deg: float,
    update_interval_s: float = 2.0,
    time_bounds_s: tuple[float, float] = (8.0, 35.0),
    angle_bounds_deg: tuple[float, float] = (1.0, 45.0),
    coarse_time_step_s: float = 4.0,
    coarse_angle_step_deg: float = 3.0,
) -> KickSolution:
    target = cfg.mission.target_orbit

    def orbit_error(params: tuple[float, float] | np.ndarray) -> float:
        from src.guidance.peg import PEG

        kick_time_s = float(params[0])
        kick_angle_deg = float(params[1])

        vehicle = Vehicle(cfg.vehicle)
        guidance = PEG(
            vehicle=vehicle,
            target_orbit=target,
            kick_time_s=kick_time_s,
            kick_angle_deg=kick_angle_deg,
            update_interval_s=update_interval_s,
            allow_two_burn=False,
        )
        state = SimState(t=cfg.simulation.t_start_s)
        events = build_autosequence(cfg, vehicle, guidance=guidance)
        final = run(
            state=state,
            vehicle=vehicle,
            guidance=guidance,
            events=events,
            t_end_s=cfg.simulation.t_end_s,
            dt=cfg.simulation.max_step_s,
        )

        staging_event = next((event for event in events if event.name == "staging" and event.result), None)
        if staging_event is None:
            return 1e6

        staging = staging_event.result.state_snapshot
        stage_vx = float(staging["vx"])
        stage_vy = float(staging["vy"])
        flight_path_deg = abs(math.degrees(math.atan2(stage_vy, max(stage_vx, 1.0e-6))))
        # Penalise staging conditions that make S2 work harder:
        # - flight path angle > 50 deg means staging too steeply (gravity losses)
        # - tangential velocity < 900 m/s means too little downrange speed at sep
        staging_penalty = (
            0.75 * max(0.0, flight_path_deg - 50.0) ** 2
            + 0.0005 * max(0.0, 900.0 - stage_vx) ** 2
        )

        try:
            elements = elements_from_state(final)
        except ValueError:
            return 1e6

        orbit_penalty = float(
            abs(elements.perigee_alt_km - target.perigee_km)
            + abs(elements.apogee_alt_km - target.apogee_km)
        )
        return orbit_penalty + staging_penalty

    best_time = float(initial_kick_time_s)
    best_angle = float(initial_kick_angle_deg)
    best_error = orbit_error((best_time, best_angle))

    time_grid = np.arange(time_bounds_s[0], time_bounds_s[1] + coarse_time_step_s * 0.5, coarse_time_step_s)
    angle_grid = np.arange(angle_bounds_deg[0], angle_bounds_deg[1] + coarse_angle_step_deg * 0.5, coarse_angle_step_deg)

    for kick_time_s in time_grid:
        for kick_angle_deg in angle_grid:
            err = orbit_error((kick_time_s, kick_angle_deg))
            if err < best_error:
                best_time = float(kick_time_s)
                best_angle = float(kick_angle_deg)
                best_error = float(err)

    result = minimize(
        orbit_error,
        x0=np.array([best_time, best_angle], dtype=float),
        method="L-BFGS-B",
        bounds=[time_bounds_s, angle_bounds_deg],
        options={"maxiter": 30},
    )

    if result.success and float(result.fun) < best_error:
        best_time = float(result.x[0])
        best_angle = float(result.x[1])
        best_error = float(result.fun)

    return KickSolution(
        kick_time_s=best_time,
        kick_angle_deg=best_angle,
        objective=best_error,
    )
