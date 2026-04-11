"""
runner.py — Execute N dispersed trajectory simulations.

Each run draws a sample from the uncertainty distributions, builds a dispersed
copy of the vehicle + guidance, runs the propagator, and returns a result dict.
Runs are parallelised with joblib (n_jobs=-1 → all CPU cores).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from src.config import MissionToolkitConfig
from src.events.autosequence import build_autosequence
from src.guidance.kick_optimizer import KickSolution
from src.mission_runner import build_guidance
from src.montecarlo.dispersions import draw_dispersions
from src.orbital.elements import elements_from_state
from src.orbital.insertion import evaluate_insertion
from src.propagator.integrator import run
from src.propagator.state import SimState
from src.vehicle.vehicle import Vehicle


def _apply_dispersions(cfg: MissionToolkitConfig, d: dict[str, float]) -> MissionToolkitConfig:
    """
    Return a deep-copied config with dispersions applied.
    Multipliers (e.g. thrust_fraction=1.02) scale the nominal value.
    Offsets (e.g. guidance_timing_offset_s) are added.
    """
    # Deep copy so that multipliers applied below don't mutate the shared
    # nominal config — each MC run must start from the original values.
    cfg = copy.deepcopy(cfg)

    tf  = d.get("thrust_fraction", 1.0)
    isf = d.get("isp_fraction", 1.0)
    pmf = d.get("propellant_mass_fraction", 1.0)
    cdf = d.get("drag_coefficient_fraction", 1.0)
    gto = d.get("guidance_timing_offset_s", 0.0)

    for stage in cfg.vehicle.stages:
        stage.thrust_vac_N       *= tf
        stage.isp_vac_s          *= isf
        stage.isp_sl_s           *= isf
        stage.propellant_mass_kg *= pmf
        stage.cd_table = [[m, cd * cdf] for m, cd in stage.cd_table]

    guidance = cfg.mission.guidance
    if guidance.pitch_program and "points" in guidance.pitch_program:
        guidance.pitch_program["points"] = [
            [t + gto, angle] for t, angle in guidance.pitch_program["points"]
        ]

    return cfg


def _run_one(
    cfg: MissionToolkitConfig,
    dispersions: dict[str, float],
    nominal_kick: KickSolution | None = None,
) -> dict[str, Any]:
    """Run a single dispersed trajectory. Returns a metrics dict."""
    dcfg    = _apply_dispersions(cfg, dispersions)
    vehicle = Vehicle(dcfg.vehicle)

    if nominal_kick is not None and dcfg.mission.guidance.mode == "peg":
        from src.guidance.peg import PEG
        guidance = PEG(
            vehicle=vehicle,
            target_orbit=dcfg.mission.target_orbit,
            kick_time_s=nominal_kick.kick_time_s,
            kick_angle_deg=nominal_kick.kick_angle_deg,
            update_interval_s=dcfg.mission.guidance.peg.update_interval_s,
            allow_two_burn=False,
        )
    else:
        guidance = build_guidance(dcfg, vehicle)

    state  = SimState(t=dcfg.simulation.t_start_s)
    events = build_autosequence(dcfg, vehicle, guidance=guidance)

    final = run(
        state=state,
        vehicle=vehicle,
        guidance=guidance,
        events=events,
        t_end_s=dcfg.simulation.t_end_s,
        dt=dcfg.simulation.max_step_s,
        record_history=False,  # MC runs only need the terminal state, not the full trajectory
    )

    # Collect event results
    fired = {ev.name: ev.result for ev in events if ev.result}
    seco  = fired.get("seco")
    max_q = fired.get("max_q")

    # Orbital elements (may fail if suborbital)
    perigee_km = apogee_km = None
    delta_perigee_km = delta_apogee_km = None
    insertion_success = False
    try:
        elems = elements_from_state(final)
        perigee_km = elems.perigee_alt_km
        apogee_km  = elems.apogee_alt_km
        res = evaluate_insertion(elems, dcfg.mission.target_orbit)
        delta_perigee_km = res.delta_perigee_km
        delta_apogee_km = res.delta_apogee_km
        insertion_success = res.success
    except ValueError:
        pass

    return {
        "t_seco_s":          seco.t_trigger if seco else None,
        "alt_seco_km":       final.y / 1e3,
        "vx_seco_m_s":       final.vx,
        "vy_seco_m_s":       final.vy,
        "speed_seco_m_s":    final.speed,
        "mass_seco_kg":      vehicle.mass,
        "max_q_Pa":          max_q.state_snapshot["dynamic_pressure_Pa"] if max_q else None,
        "perigee_km":        perigee_km,
        "apogee_km":         apogee_km,
        "delta_perigee_km":  delta_perigee_km,
        "delta_apogee_km":   delta_apogee_km,
        "insertion_success": insertion_success,
        "events_fired":      list(fired.keys()),
        "dispersions":       dispersions,
    }


def run_montecarlo(cfg: MissionToolkitConfig) -> list[dict[str, Any]]:
    """
    Run N Monte Carlo trajectories and return all result dicts.

    Uses the seed and n_jobs from cfg.uncertainties.montecarlo.
    For PEG guidance, the kick solution is optimized once on the nominal config
    and reused across all runs — PEG corrects in-flight so the kick only needs
    to be "good enough" to start the gravity turn.
    """
    mc  = cfg.uncertainties.montecarlo
    rng = np.random.default_rng(mc.seed)
    all_dispersions = [
        draw_dispersions(cfg.uncertainties, rng) for _ in range(mc.n_runs)
    ]

    # Pre-compute kick solution once from nominal config (avoids re-running the
    # optimizer ~125 times — one per MC run — which dominates wall-clock time).
    nominal_kick: KickSolution | None = None
    if cfg.mission.guidance.mode == "peg":
        from src.guidance.kick_optimizer import optimize_kick
        gt = cfg.mission.guidance.gravity_turn
        nominal_kick = optimize_kick(
            cfg=cfg,
            initial_kick_time_s=gt["kick_time_s"],
            initial_kick_angle_deg=gt["kick_angle_deg"],
            update_interval_s=cfg.mission.guidance.peg.update_interval_s,
        )

    results: list[dict[str, Any]] = Parallel(n_jobs=mc.n_jobs, backend="loky")(
        delayed(_run_one)(cfg, d, nominal_kick) for d in all_dispersions
    )
    return results
