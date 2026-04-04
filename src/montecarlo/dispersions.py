"""
dispersions.py — Sample one set of per-run uncertain parameters.

Each call to draw_dispersions() returns a flat dict of
  param_name → sampled value
that is applied to the nominal config before a Monte Carlo run.
"""

from __future__ import annotations

import numpy as np

from src.config import UncertaintiesConfig, UncertaintyParam


def _sample(param: UncertaintyParam, rng: np.random.Generator) -> float:
    if param.dist == "normal":
        assert param.mu is not None and param.sigma is not None
        return float(rng.normal(param.mu, param.sigma))
    if param.dist == "uniform":
        assert param.low is not None and param.high is not None
        return float(rng.uniform(param.low, param.high))
    raise ValueError(f"Unknown distribution: {param.dist!r}")


def draw_dispersions(
    cfg: UncertaintiesConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Return one sampled dispersion dict for all uncertain parameters."""
    return {name: _sample(param, rng) for name, param in cfg.uncertainties.items()}
