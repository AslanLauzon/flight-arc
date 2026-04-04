"""
validate_config.py — Load and validate all YAML config files, print a summary.

Usage:
    python -m scripts.validate_config
"""

from rich.console import Console
from rich.table import Table
from rich import box

from src.config import load_config

_console = Console()


def main() -> None:
    try:
        cfg = load_config()
    except Exception as e:
        _console.print(f"[red]Config validation FAILED:[/red] {e}")
        raise SystemExit(1)

    _console.print("\n[green bold]Config OK[/green bold]\n")

    # Vehicle summary
    t = Table(title="Vehicle Stages", box=box.SIMPLE_HEAVY)
    t.add_column("Stage",           style="bold")
    t.add_column("Prop (kg)",       justify="right")
    t.add_column("Dry (kg)",        justify="right")
    t.add_column("Thrust vac (kN)", justify="right")
    t.add_column("Isp vac (s)",     justify="right")

    for s in cfg.vehicle.stages:
        t.add_row(
            s.name,
            f"{s.propellant_mass_kg:,.0f}",
            f"{s.dry_mass_kg:,.0f}",
            f"{s.thrust_vac_N/1e3:.1f}",
            f"{s.isp_vac_s:.0f}",
        )

    t.add_section()
    total_prop = sum(s.propellant_mass_kg for s in cfg.vehicle.stages)
    total_dry  = sum(s.dry_mass_kg        for s in cfg.vehicle.stages)
    total_mass = total_prop + total_dry + cfg.vehicle.payload.mass_kg + cfg.vehicle.payload.fairing_mass_kg
    t.add_row("TOTAL", f"{total_prop:,.0f}", f"{total_dry:,.0f}", "", "")
    t.add_row(f"Liftoff mass", f"{total_mass:,.0f} kg", "", "", "")
    _console.print(t)

    # Mission summary
    to = cfg.mission.target_orbit
    _console.print(
        f"Mission : [bold]{cfg.mission.name}[/bold]  |  "
        f"Guidance: [bold]{cfg.mission.guidance.mode}[/bold]  |  "
        f"Target: {to.perigee_km:.0f} x {to.apogee_km:.0f} km @ {to.inclination_deg:.1f}°\n"
    )


if __name__ == "__main__":
    main()
