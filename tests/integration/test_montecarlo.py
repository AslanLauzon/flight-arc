"""
Integration test: run a small Monte Carlo batch and verify basic properties.
Uses a tiny n_runs override to keep the test fast.
"""

import copy
import pytest
from src.config import load_config
from src.montecarlo.analysis import compute_statistics
from src.montecarlo.dispersions import draw_dispersions
from src.montecarlo.runner import run_montecarlo, _run_one
import numpy as np


@pytest.fixture(scope="module")
def small_mc_results():
    cfg = load_config()
    # override to 5 runs for speed
    cfg = copy.deepcopy(cfg)
    cfg.uncertainties.montecarlo.n_runs = 5
    cfg.uncertainties.montecarlo.n_jobs = 1   # sequential for test determinism
    return run_montecarlo(cfg)


def test_all_runs_complete(small_mc_results):
    assert len(small_mc_results) == 5


def test_all_runs_have_seco(small_mc_results):
    for r in small_mc_results:
        assert r["t_seco_s"] is not None, "A run did not reach SECO"


def test_results_have_expected_keys(small_mc_results):
    required = {"t_seco_s", "alt_seco_km", "vx_seco_m_s", "perigee_km", "apogee_km"}
    for r in small_mc_results:
        assert required.issubset(r.keys())


def test_dispersions_vary():
    """Different seeds should produce different dispersions."""
    cfg = load_config()
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(1)
    d1 = draw_dispersions(cfg.uncertainties, rng1)
    d2 = draw_dispersions(cfg.uncertainties, rng2)
    # at least one parameter should differ
    assert any(d1[k] != d2[k] for k in d1)


def test_statistics_computed(small_mc_results):
    stats = compute_statistics(small_mc_results, percentile_vals=[50])
    assert stats.n_runs == 5
    assert "t_seco_s" in stats.mean
    assert len(stats.percentile_rows) == 1


def test_seco_times_physically_reasonable(small_mc_results):
    for r in small_mc_results:
        assert 50 < r["t_seco_s"] < 700, f"Unreasonable SECO time: {r['t_seco_s']}"


def test_altitudes_positive(small_mc_results):
    for r in small_mc_results:
        assert r["alt_seco_km"] > 0, f"Vehicle underground at SECO: {r['alt_seco_km']} km"
