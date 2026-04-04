"""
analysis.py — Compute statistics over a Monte Carlo result set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Metrics extracted from each run result for statistical analysis
_SCALAR_METRICS = [
    "t_seco_s",
    "alt_seco_km",
    "vx_seco_m_s",
    "vy_seco_m_s",
    "speed_seco_m_s",
    "mass_seco_kg",
    "max_q_Pa",
    "perigee_km",
    "apogee_km",
]


@dataclass
class PercentileRow:
    pct: int
    values: dict[str, float]   # metric_name → value at this percentile


@dataclass
class MonteCarloStats:
    n_runs: int
    n_success: int
    success_rate_pct: float
    mean: dict[str, float]
    std:  dict[str, float]
    percentile_rows: list[PercentileRow] = field(default_factory=list)


def compute_statistics(
    results: list[dict[str, Any]],
    percentile_vals: list[int] | None = None,
) -> MonteCarloStats:
    """Aggregate scalar metrics across all MC runs."""
    if percentile_vals is None:
        percentile_vals = [5, 50, 95]

    n_runs    = len(results)
    n_success = sum(1 for r in results if r.get("insertion_success", False))

    # Build arrays, filtering None values
    arrays: dict[str, np.ndarray] = {}
    for key in _SCALAR_METRICS:
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            arrays[key] = np.array(vals, dtype=float)

    mean = {k: float(np.mean(v)) for k, v in arrays.items()}
    std  = {k: float(np.std(v))  for k, v in arrays.items()}

    pct_rows = []
    for p in percentile_vals:
        row_vals = {k: float(np.percentile(v, p)) for k, v in arrays.items()}
        pct_rows.append(PercentileRow(pct=p, values=row_vals))

    return MonteCarloStats(
        n_runs=n_runs,
        n_success=n_success,
        success_rate_pct=100.0 * n_success / n_runs if n_runs else 0.0,
        mean=mean,
        std=std,
        percentile_rows=pct_rows,
    )
