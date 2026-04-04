"""
mission_report.py — Rich-formatted mission report printed to the terminal.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.table import Table

from src.config import MissionToolkitConfig
from src.events.event import Event
from src.orbital.elements import OrbitalElements
from src.orbital.insertion import InsertionResult
from src.propagator.state import SimState
from src.report.tables import (
    event_timeline_table,
    insertion_table,
    orbital_elements_table,
)


_console = Console()


def _header(cfg: MissionToolkitConfig) -> Panel:
    text = Text()
    text.append(f"  Mission : ", style="dim")
    text.append(cfg.mission.name, style="bold cyan")
    text.append(f"\n  Vehicle : ", style="dim")
    text.append(cfg.vehicle.name, style="bold")
    text.append(f"\n  Guidance: ", style="dim")
    text.append(cfg.mission.guidance.mode, style="bold")
    text.append(f"\n  Target  : ", style="dim")
    to = cfg.mission.target_orbit
    text.append(f"{to.perigee_km:.0f} x {to.apogee_km:.0f} km @ {to.inclination_deg:.1f}°")
    return Panel(text, title="[bold]Mission Summary[/bold]", box=box.DOUBLE_EDGE)


def _final_state_table(state: SimState) -> Table:
    t = Table(title="Final State", box=box.SIMPLE_HEAVY)
    t.add_column("Parameter", style="bold")
    t.add_column("Value", justify="right", style="cyan")

    rows = [
        ("Time",      f"{state.t:.1f} s"),
        ("Altitude",  f"{state.y/1e3:.1f} km"),
        ("Downrange", f"{state.x/1e3:.1f} km"),
        ("Speed",     f"{state.speed:.1f} m/s"),
        ("vx",        f"{state.vx:.1f} m/s"),
        ("vy",        f"{state.vy:.1f} m/s"),
    ]
    for name, val in rows:
        t.add_row(name, val)

    return t


def generate_report(
    cfg: MissionToolkitConfig,
    final_state: SimState,
    events: list[Event],
    elements: OrbitalElements | None = None,
    insertion: InsertionResult | None = None,
) -> None:
    """Print a full rich-formatted mission report to stdout."""
    _console.print()
    _console.print(_header(cfg))
    _console.print()
    _console.print(event_timeline_table(events))
    _console.print()
    _console.print(_final_state_table(final_state))

    if elements is not None:
        _console.print()
        _console.print(orbital_elements_table(elements))

    if insertion is not None:
        _console.print()
        _console.print(insertion_table(insertion))

    _console.print()
