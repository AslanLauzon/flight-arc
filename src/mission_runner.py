from __future__ import annotations

from dataclasses import dataclass
import logging

from src.config import MissionToolkitConfig, load_config
from src.events.autosequence import build_autosequence
from src.events.event import Event
from src.guidance.base import GuidanceBase
from src.guidance.gravity_turn import GravityTurn
from src.guidance.peg import PEG
from src.guidance.pitch_program import PitchProgram
from src.orbital.elements import OrbitalElements, elements_from_state
from src.orbital.insertion import InsertionResult, evaluate_insertion
from src.propagator.integrator import run
from src.propagator.state import SimState
from src.vehicle.vehicle import Vehicle

logger = logging.getLogger("flight_arc.mission")


@dataclass
class NominalRunResult:
    cfg: MissionToolkitConfig
    vehicle: Vehicle
    guidance: GuidanceBase
    events: list[Event]
    final_state: SimState
    elements: OrbitalElements | None
    insertion: InsertionResult | None


def build_guidance(cfg: MissionToolkitConfig, vehicle: Vehicle) -> GuidanceBase:
    mode = cfg.mission.guidance.mode

    if mode == "pitch_program":
        return PitchProgram(cfg.mission.guidance.pitch_program["points"])
    if mode == "gravity_turn":
        gt = cfg.mission.guidance.gravity_turn
        return GravityTurn(
            kick_time_s=gt["kick_time_s"],
            kick_angle_deg=gt["kick_angle_deg"],
        )
    if mode == "peg":
        from src.guidance.kick_optimizer import optimize_kick

        gt = cfg.mission.guidance.gravity_turn
        solution = optimize_kick(
            cfg=cfg,
            initial_kick_time_s=gt["kick_time_s"],
            initial_kick_angle_deg=gt["kick_angle_deg"],
            update_interval_s=cfg.mission.guidance.peg.update_interval_s,
        )
        logger.info(
            "peg_kick_optimiser kick_time_s=%.2f kick_angle_deg=%.2f objective=%.2f",
            solution.kick_time_s,
            solution.kick_angle_deg,
            solution.objective,
        )
        return PEG(
            vehicle=vehicle,
            target_orbit=cfg.mission.target_orbit,
            kick_time_s=solution.kick_time_s,
            kick_angle_deg=solution.kick_angle_deg,
            update_interval_s=cfg.mission.guidance.peg.update_interval_s,
            allow_two_burn=False,
        )

    raise ValueError(f"Unknown guidance mode: {mode}")


def run_nominal_mission(cfg: MissionToolkitConfig | None = None) -> NominalRunResult:
    if cfg is None:
        cfg = load_config()

    vehicle = Vehicle(cfg.vehicle)
    guidance = build_guidance(cfg, vehicle)
    state = SimState(t=cfg.simulation.t_start_s, mass_kg=vehicle.mass)
    events = build_autosequence(cfg, vehicle, guidance=guidance)

    final_state = run(
        state=state,
        vehicle=vehicle,
        guidance=guidance,
        events=events,
        t_end_s=cfg.simulation.t_end_s,
        dt=cfg.simulation.max_step_s,
    )

    elements: OrbitalElements | None = None
    insertion: InsertionResult | None = None
    try:
        elements = elements_from_state(final_state)
        insertion = evaluate_insertion(elements, cfg.mission.target_orbit)
    except ValueError:
        pass

    return NominalRunResult(
        cfg=cfg,
        vehicle=vehicle,
        guidance=guidance,
        events=events,
        final_state=final_state,
        elements=elements,
        insertion=insertion,
    )
