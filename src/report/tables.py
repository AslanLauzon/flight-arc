"""
tables.py — Rich table builders for mission data.

Each function returns a rich.table.Table ready to be printed via a Console.
"""

from __future__ import annotations

from rich.table import Table
from rich import box

from src.events.event import Event
from src.montecarlo.analysis import MonteCarloStats
from src.orbital.elements import OrbitalElements
from src.orbital.insertion import InsertionResult


def event_timeline_table(events: list[Event]) -> Table:
    t = Table(title="Event Timeline", box=box.SIMPLE_HEAVY, show_header=True)
    t.add_column("Time (s)",  justify="right", style="cyan",  width=10)
    t.add_column("Event",     style="bold white", width=20)
    t.add_column("Message",   style="dim")

    for ev in events:
        if ev.result:
            r = ev.result
            t.add_row(f"{r.t_trigger:7.1f}", r.name, r.message)

    return t


def orbital_elements_table(elements: OrbitalElements) -> Table:
    t = Table(title="Achieved Orbit", box=box.SIMPLE_HEAVY)
    t.add_column("Parameter", style="bold")
    t.add_column("Value",     justify="right", style="green")

    rows = [
        ("Perigee",         f"{elements.perigee_alt_km:.1f} km"),
        ("Apogee",          f"{elements.apogee_alt_km:.1f} km"),
        ("Semi-major axis", f"{elements.semi_major_axis_m/1e3:.1f} km"),
        ("Eccentricity",    f"{elements.eccentricity:.6f}"),
        ("Period",          f"{elements.orbital_period_s/60:.1f} min"),
        ("v_circ",          f"{elements.circular_velocity_m_s:.1f} m/s"),
    ]
    for name, val in rows:
        t.add_row(name, val)

    return t


def insertion_table(result: InsertionResult) -> Table:
    status_color = "green" if result.success else "red"
    status_text  = "[green]SUCCESS[/green]" if result.success else "[red]MISS[/red]"
    t = Table(title=f"Insertion Assessment  {status_text}", box=box.SIMPLE_HEAVY)
    t.add_column("Parameter", style="bold")
    t.add_column("Achieved",  justify="right")
    t.add_column("Target",    justify="right")
    t.add_column("Delta",     justify="right")

    def _delta_style(d: float) -> str:
        return "green" if abs(d) <= 5.0 else "red"

    t.add_row(
        "Perigee",
        f"{result.achieved.perigee_alt_km:.1f} km",
        f"{result.target_perigee_km:.1f} km",
        f"[{_delta_style(result.delta_perigee_km)}]{result.delta_perigee_km:+.1f} km[/]",
    )
    t.add_row(
        "Apogee",
        f"{result.achieved.apogee_alt_km:.1f} km",
        f"{result.target_apogee_km:.1f} km",
        f"[{_delta_style(result.delta_apogee_km)}]{result.delta_apogee_km:+.1f} km[/]",
    )
    return t


def montecarlo_stats_table(stats: MonteCarloStats) -> Table:
    t = Table(title=f"Monte Carlo  ({stats.n_runs} runs)", box=box.SIMPLE_HEAVY)
    t.add_column("Metric",    style="bold", width=22)
    t.add_column("Mean",      justify="right", width=12)
    t.add_column("Std",       justify="right", width=10)

    for prow in stats.percentile_rows:
        t.add_column(f"P{prow.pct}", justify="right", width=12)

    _LABELS = {
        "t_seco_s":       "SECO time (s)",
        "alt_seco_km":    "Alt at SECO (km)",
        "vx_seco_m_s":    "vx at SECO (m/s)",
        "vy_seco_m_s":    "vy at SECO (m/s)",
        "speed_seco_m_s": "Speed at SECO (m/s)",
        "mass_seco_kg":   "Mass at SECO (kg)",
        "max_q_Pa":       "Max-Q (Pa)",
        "perigee_km":     "Perigee (km)",
        "apogee_km":      "Apogee (km)",
    }

    for key, label in _LABELS.items():
        if key not in stats.mean:
            continue
        row = [
            label,
            f"{stats.mean[key]:.1f}",
            f"{stats.std[key]:.1f}",
        ]
        for prow in stats.percentile_rows:
            row.append(f"{prow.values.get(key, float('nan')):.1f}")
        t.add_row(*row)

    # Success rate as footer
    sr_color = "green" if stats.success_rate_pct >= 99 else "yellow" if stats.success_rate_pct >= 90 else "red"
    t.add_section()
    t.add_row(
        "Insertion success",
        f"[{sr_color}]{stats.success_rate_pct:.1f} %[/]",
        f"{stats.n_success}/{stats.n_runs}",
        *[""] * len(stats.percentile_rows),
    )

    return t
