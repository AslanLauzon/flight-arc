"""
Integration test: run a full nominal trajectory and verify key constraints.
Keeps n_runs=1, no Monte Carlo overhead.
"""

import pytest
from src.config import load_config
from src.events.autosequence import build_autosequence
from src.guidance.peg import PEG
from src.orbital.elements import elements_from_state
from src.propagator.integrator import run
from src.propagator.state import SimState
from src.vehicle.vehicle import Vehicle


@pytest.fixture(scope="module")
def sim_result():
    cfg     = load_config()
    vehicle = Vehicle(cfg.vehicle)
    guidance = PEG(
        vehicle=vehicle,
        target_orbit=cfg.mission.target_orbit,
        stage1_pitch_points=cfg.mission.guidance.pitch_program["points"],
        update_interval_s=cfg.mission.guidance.peg.update_interval_s,
    )
    state  = SimState(t=cfg.simulation.t_start_s, mass_kg=vehicle.mass)
    events = build_autosequence(cfg, vehicle, guidance=guidance)
    final  = run(
        state=state,
        vehicle=vehicle,
        guidance=guidance,
        events=events,
        t_end_s=cfg.simulation.t_end_s,
        dt=cfg.simulation.max_step_s,
    )
    fired = {ev.name: ev.result for ev in events if ev.result}
    return final, fired, vehicle


def test_simulation_runs_to_seco(sim_result):
    _, fired, _ = sim_result
    assert "seco" in fired, "SECO event never fired"


def test_liftoff_occurs(sim_result):
    _, fired, _ = sim_result
    assert "liftoff" in fired


def test_staging_occurs(sim_result):
    _, fired, _ = sim_result
    assert "staging" in fired


def test_fairing_deploys(sim_result):
    _, fired, _ = sim_result
    assert "fairing_deploy" in fired


def test_staging_before_seco(sim_result):
    _, fired, _ = sim_result
    assert fired["staging"].t_trigger < fired["seco"].t_trigger


def test_altitude_positive_at_seco(sim_result):
    final, _, _ = sim_result
    assert final.y > 0, f"Vehicle underground at SECO: {final.y:.0f} m"


def test_positive_horizontal_velocity_at_seco(sim_result):
    final, _, _ = sim_result
    assert final.vx > 1000, f"Insufficient vx at SECO: {final.vx:.0f} m/s"


def test_nearly_horizontal_at_seco(sim_result):
    """vy should be small relative to vx — guidance is doing its job."""
    final, _, _ = sim_result
    assert abs(final.vy) < 0.1 * abs(final.vx), (
        f"Large vy at SECO: vy={final.vy:.0f}, vx={final.vx:.0f}"
    )


def test_orbit_is_bound(sim_result):
    """Achieved orbit should be a closed (bound) orbit."""
    final, _, _ = sim_result
    elems = elements_from_state(final)
    assert elems.specific_energy_J_kg < 0


def test_perigee_positive(sim_result):
    """Perigee should be above Earth's surface."""
    final, _, _ = sim_result
    elems = elements_from_state(final)
    assert elems.perigee_alt_km > 0, f"Suborbital perigee: {elems.perigee_alt_km:.1f} km"


def test_trajectory_history_populated(sim_result):
    final, _, _ = sim_result
    assert len(final.history) > 100, "Very few history points recorded"
