"""
Integration test: run a full nominal trajectory and verify key constraints.
Keeps n_runs=1, no Monte Carlo overhead.
"""

import pytest
from src.mission_runner import run_nominal_mission
from src.orbital.elements import elements_from_state


@pytest.fixture(scope="module")
def sim_result():
    return run_nominal_mission()


def test_simulation_runs_to_seco(sim_result):
    fired = {ev.name for ev in sim_result.events if ev.result}
    assert "seco" in fired, "SECO event never fired"


def test_liftoff_occurs(sim_result):
    fired = {ev.name for ev in sim_result.events if ev.result}
    assert "liftoff" in fired


def test_staging_occurs(sim_result):
    fired = {ev.name for ev in sim_result.events if ev.result}
    assert "staging" in fired


def test_fairing_deploys(sim_result):
    fired = {ev.name for ev in sim_result.events if ev.result}
    assert "fairing_deploy" in fired


def test_staging_before_seco(sim_result):
    by_name = {ev.name: ev.result for ev in sim_result.events if ev.result}
    assert by_name["staging"].t_trigger < by_name["seco"].t_trigger


def test_altitude_positive_at_seco(sim_result):
    assert sim_result.final_state.y > 0, f"Vehicle underground at SECO: {sim_result.final_state.y:.0f} m"


def test_positive_horizontal_velocity_at_seco(sim_result):
    assert sim_result.final_state.vx > 1000, f"Insufficient vx at SECO: {sim_result.final_state.vx:.0f} m/s"


def test_nearly_horizontal_at_seco(sim_result):
    """vy should be small relative to vx — guidance is doing its job."""
    vx = sim_result.final_state.vx
    vy = sim_result.final_state.vy
    assert abs(vy) < 0.1 * abs(vx), f"Large vy at SECO: vy={vy:.0f}, vx={vx:.0f}"


def test_orbit_is_bound(sim_result):
    """Achieved orbit should be a closed (bound) orbit."""
    elems = elements_from_state(sim_result.final_state)
    assert elems.specific_energy_J_kg < 0


def test_perigee_positive(sim_result):
    """Perigee should be above Earth's surface."""
    elems = elements_from_state(sim_result.final_state)
    assert elems.perigee_alt_km > 0, f"Suborbital perigee: {elems.perigee_alt_km:.1f} km"


def test_trajectory_history_populated(sim_result):
    assert len(sim_result.final_state.history) > 100, "Very few history points recorded"
