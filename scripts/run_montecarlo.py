"""
run_montecarlo.py - Monte Carlo trajectory dispersions analysis.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from src.api.serializers import OUTPUT_DIR, build_montecarlo_payload, write_payload_json
from src.config import load_config
from src.montecarlo.analysis import compute_statistics
from src.montecarlo.runner import run_montecarlo
from src.report.tables import montecarlo_stats_table

_console = Console()


def main() -> None:
    cfg = load_config()
    mc = cfg.uncertainties.montecarlo

    _console.print(
        f"\n[bold]Monte Carlo: {cfg.mission.name}[/bold]  "
        f"n={mc.n_runs}  seed={mc.seed}  jobs={mc.n_jobs}\n"
    )

    results = run_montecarlo(cfg)

    stats = compute_statistics(results, mc.output_percentiles)
    _console.print(montecarlo_stats_table(stats))
    _console.print()

    payload = build_montecarlo_payload(cfg, results, request_id="cli")
    json_path = write_payload_json(
        payload.model_dump(mode="json"),
        Path(OUTPUT_DIR) / "montecarlo_latest.json",
    )
    _console.print(f"JSON payload written to {json_path}\n")


if __name__ == "__main__":
    main()
