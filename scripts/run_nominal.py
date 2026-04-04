from __future__ import annotations

from pathlib import Path

from src.api.serializers import OUTPUT_DIR, build_nominal_payload, write_payload_json
from src.mission_runner import NominalRunResult, run_nominal_mission
from src.report.plots import plot_trajectory


def _print_event_timeline(result: NominalRunResult) -> None:
    print("Event Timeline")
    for event in result.events:
        if event.result is None:
            continue
        record = event.result
        print(f"  t={record.t_trigger:7.1f}s  {record.name:<20}  {record.message}")


def _print_event_snapshots(result: NominalRunResult) -> None:
    print("\nEvent State Snapshots")
    for event in result.events:
        if event.result is None:
            continue
        record = event.result
        snap = record.state_snapshot
        print(
            f"  {record.name:<20} "
            f"alt={snap['y']/1e3:7.1f} km  "
            f"vx={snap['vx']:8.1f} m/s  "
            f"vy={snap['vy']:8.1f} m/s  "
            f"mach={snap['mach']:6.2f}  "
            f"q={snap['dynamic_pressure_Pa']/1e3:7.1f} kPa"
        )


def _print_orbit_summary(result: NominalRunResult) -> None:
    print("\nAchieved Orbit")
    if result.elements is None:
        print("Orbit computation failed: run did not reach a bound orbit")
        return

    print(result.elements)
    if result.insertion is not None:
        print("\nInsertion Assessment")
        print(result.insertion)


def _print_final_state(result: NominalRunResult) -> None:
    final_state = result.final_state
    print("\nFinal State")
    print(f"  t        = {final_state.t:.1f} s")
    print(f"  altitude = {final_state.y / 1e3:.1f} km")
    print(f"  downrange= {final_state.x / 1e3:.1f} km")
    print(f"  speed    = {final_state.speed:.1f} m/s")
    print(f"  vx       = {final_state.vx:.1f} m/s")
    print(f"  vy       = {final_state.vy:.1f} m/s")
    print(f"  mass     = {result.vehicle.mass:.1f} kg")
    print(f"  events   = {final_state.events_triggered}")


def main() -> None:
    result = run_nominal_mission()
    cfg = result.cfg

    print(f"Running nominal trajectory: {cfg.mission.name}")
    print(f"Vehicle: {cfg.vehicle.name}")
    print(f"Guidance: {cfg.mission.guidance.mode}\n")

    _print_event_timeline(result)
    _print_event_snapshots(result)
    _print_orbit_summary(result)
    _print_final_state(result)

    payload = build_nominal_payload(result)
    json_path = write_payload_json(
        payload.model_dump(mode="json"),
        Path(OUTPUT_DIR) / "nominal_latest.json",
    )
    print(f"\nJSON payload written to {json_path}")

    plot_trajectory(result.final_state, result.events, elements=result.elements)


if __name__ == "__main__":
    main()
