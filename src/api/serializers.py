from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.api.models import (
    AxisModel,
    EventModel,
    MissionMetadataModel,
    MissionTargetModel,
    MonteCarloMetricModel,
    MonteCarloPayload,
    MonteCarloRunResultModel,
    NominalRunPayload,
    PlotAnnotationModel,
    PlotModel,
    PlotSeriesModel,
    StatusModel,
    VehicleMetadataModel,
    VehicleStageModel,
)
from src.config import MissionToolkitConfig
from src.events.event import Event
from src.mission_runner import NominalRunResult
from src.montecarlo.analysis import MonteCarloStats, compute_statistics
from src.orbital.elements import OrbitalElements
from src.orbital.insertion import InsertionResult
from src.propagator.state import SimState

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


def create_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def write_payload_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def serialize_config_target(cfg: MissionToolkitConfig) -> MissionTargetModel:
    return MissionTargetModel(
        type="orbit",
        parameters=cfg.mission.target_orbit.model_dump(),
    )


def serialize_vehicle(cfg: MissionToolkitConfig) -> VehicleMetadataModel:
    return VehicleMetadataModel(
        vehicle_type="launch_vehicle",
        reference_area_m2=cfg.vehicle.reference_area_m2,
        stages=[
            VehicleStageModel(
                name=stage.name,
                dry_mass_kg=stage.dry_mass_kg,
                propellant_mass_kg=stage.propellant_mass_kg,
                thrust_vac_n=stage.thrust_vac_N,
                isp_vac_s=stage.isp_vac_s,
                burn_time_s=stage.burn_time_s,
            )
            for stage in cfg.vehicle.stages
        ],
        payload=cfg.vehicle.payload.model_dump(),
    )


def serialize_mission(cfg: MissionToolkitConfig) -> MissionMetadataModel:
    return MissionMetadataModel(
        name=cfg.mission.name,
        vehicle_name=cfg.vehicle.name,
        vehicle_type="launch_vehicle",
        guidance_mode=cfg.mission.guidance.mode,
        target=serialize_config_target(cfg),
    )


def _event_annotations(events: list[Event]) -> list[PlotAnnotationModel]:
    return [
        PlotAnnotationModel(
            kind="vertical_line",
            value=event.result.t_trigger,
            label=event.name,
        )
        for event in events
        if event.result is not None
    ]


def _eci_track(state: SimState) -> tuple[list[float], list[float]]:
    earth_radius_km = 6371.0
    eci_x: list[float] = []
    eci_y: list[float] = []
    for row in state.history:
        radius_km = earth_radius_km + row["y"] / 1e3
        theta = row["x"] / (earth_radius_km * 1e3)
        eci_x.append(float(radius_km * np.cos(theta)))
        eci_y.append(float(radius_km * np.sin(theta)))
    return eci_x, eci_y


def _trajectory_plots(
    state: SimState,
    events: list[Event],
    elements: OrbitalElements | None,
) -> list[PlotModel]:
    history = state.history
    if not history:
        return []

    t = [float(row["t"]) for row in history]
    x_km = [float(row["x"] / 1e3) for row in history]
    y_km = [float(row["y"] / 1e3) for row in history]
    vx = [float(row["vx"]) for row in history]
    vy = [float(row["vy"]) for row in history]
    speed_km_s = [float(np.hypot(row["vx"], row["vy"]) / 1e3) for row in history]
    q_kpa = [float(row["dynamic_pressure_Pa"] / 1e3) for row in history]
    mach = [float(row["mach"]) for row in history]
    annotations = _event_annotations(events)

    plots = [
        PlotModel(
            id="altitude_vs_time",
            title="Altitude vs Time",
            kind="line",
            x_axis=AxisModel(key="time_s", label="Time", unit="s"),
            y_axis=AxisModel(key="altitude_km", label="Altitude", unit="km"),
            series=[PlotSeriesModel(key="altitude_km", label="Altitude", x_data=t, y_data=y_km, unit="km")],
            annotations=annotations,
        ),
        PlotModel(
            id="speed_vs_time",
            title="Speed vs Time",
            kind="line",
            x_axis=AxisModel(key="time_s", label="Time", unit="s"),
            y_axis=AxisModel(key="speed_km_s", label="Speed", unit="km/s"),
            series=[PlotSeriesModel(key="speed_km_s", label="Speed", x_data=t, y_data=speed_km_s, unit="km/s")],
            annotations=annotations,
        ),
        PlotModel(
            id="dynamic_pressure_vs_time",
            title="Dynamic Pressure vs Time",
            kind="line",
            x_axis=AxisModel(key="time_s", label="Time", unit="s"),
            y_axis=AxisModel(key="dynamic_pressure_kpa", label="Dynamic Pressure", unit="kPa"),
            series=[PlotSeriesModel(key="dynamic_pressure_kpa", label="Dynamic Pressure", x_data=t, y_data=q_kpa, unit="kPa")],
            annotations=annotations,
        ),
        PlotModel(
            id="mach_vs_time",
            title="Mach vs Time",
            kind="line",
            x_axis=AxisModel(key="time_s", label="Time", unit="s"),
            y_axis=AxisModel(key="mach", label="Mach"),
            series=[PlotSeriesModel(key="mach", label="Mach", x_data=t, y_data=mach)],
            annotations=annotations,
        ),
        PlotModel(
            id="velocity_components_vs_time",
            title="Velocity Components vs Time",
            kind="line",
            x_axis=AxisModel(key="time_s", label="Time", unit="s"),
            y_axis=AxisModel(key="velocity_m_s", label="Velocity", unit="m/s"),
            series=[
                PlotSeriesModel(key="vx_m_s", label="vx", x_data=t, y_data=vx, unit="m/s"),
                PlotSeriesModel(key="vy_m_s", label="vy", x_data=t, y_data=vy, unit="m/s"),
            ],
            annotations=annotations,
        ),
        PlotModel(
            id="trajectory_shape",
            title="Trajectory Shape",
            kind="line",
            x_axis=AxisModel(key="downrange_km", label="Downrange", unit="km"),
            y_axis=AxisModel(key="altitude_km", label="Altitude", unit="km"),
            series=[PlotSeriesModel(key="trajectory", label="Trajectory", x_data=x_km, y_data=y_km)],
        ),
    ]

    eci_x, eci_y = _eci_track(state)
    plots.append(
        PlotModel(
            id="orbit_view",
            title="Orbit Around Earth",
            kind="xy",
            x_axis=AxisModel(key="eci_x_km", label="ECI x", unit="km"),
            y_axis=AxisModel(key="eci_y_km", label="ECI y", unit="km"),
            series=[
                PlotSeriesModel(
                    key="ascent_eci",
                    label="Ascent trajectory",
                    x_data=eci_x,
                    y_data=eci_y,
                    unit="km",
                )
            ],
            metadata={
                "earth_radius_km": 6371.0,
                "achieved_orbit": (
                    {
                        "perigee_km": elements.perigee_alt_km,
                        "apogee_km": elements.apogee_alt_km,
                        "eccentricity": elements.eccentricity,
                    }
                    if elements is not None
                    else None
                ),
            },
        )
    )

    return plots


def _serialize_events(events: list[Event]) -> list[EventModel]:
    return [
        EventModel(
            name=result.name,
            time_s=result.t_trigger,
            message=result.message,
            snapshot=result.state_snapshot,
        )
        for event in events
        if (result := event.result) is not None
    ]


def _nominal_status(
    elements: OrbitalElements | None,
    insertion: InsertionResult | None,
) -> StatusModel:
    if insertion is not None and insertion.success:
        return StatusModel(state="success", message="Target achieved")
    if elements is not None:
        return StatusModel(state="completed", message="Run completed with a bound orbit")
    return StatusModel(state="suborbital", message="Run completed without a bound orbit")


def build_nominal_payload(
    result: NominalRunResult,
    request_id: str,
    run_id: str | None = None,
) -> NominalRunPayload:
    if run_id is None:
        run_id = create_run_id("nominal")

    max_q_event = next((event for event in result.events if event.name == "max_q" and event.result), None)

    outcomes: dict[str, Any] = {}
    if result.elements is not None:
        outcomes["orbit"] = {
            "achieved": {
                "perigee_km": result.elements.perigee_alt_km,
                "apogee_km": result.elements.apogee_alt_km,
                "semi_major_axis_km": result.elements.semi_major_axis_m / 1e3,
                "eccentricity": result.elements.eccentricity,
                "period_min": result.elements.orbital_period_s / 60.0,
                "circular_velocity_m_s": result.elements.circular_velocity_m_s,
            }
        }
        if result.insertion is not None:
            outcomes["orbit"]["assessment"] = {
                "success": bool(result.insertion.success),
                "delta_perigee_km": float(result.insertion.delta_perigee_km),
                "delta_apogee_km": float(result.insertion.delta_apogee_km),
                "target_perigee_km": float(result.insertion.target_perigee_km),
                "target_apogee_km": float(result.insertion.target_apogee_km),
            }

    if max_q_event is not None and max_q_event.result is not None:
        observed_q = float(max_q_event.result.state_snapshot["dynamic_pressure_Pa"])
        outcomes["constraints"] = {
            "max_q_pa": {
                "observed": observed_q,
                "limit": float(result.cfg.simulation.constraints.max_q_Pa),
                "violated": bool(observed_q > result.cfg.simulation.constraints.max_q_Pa),
            }
        }

    return NominalRunPayload(
        request_id=request_id,
        run_id=run_id,
        run_type="nominal",
        mission=serialize_mission(result.cfg),
        vehicle=serialize_vehicle(result.cfg),
        status=_nominal_status(result.elements, result.insertion),
        summary={
            "final_time_s": result.final_state.t,
            "altitude_km": result.final_state.y / 1e3,
            "downrange_km": result.final_state.x / 1e3,
            "speed_m_s": result.final_state.speed,
            "vx_m_s": result.final_state.vx,
            "vy_m_s": result.final_state.vy,
            "mass_kg": result.vehicle.mass,
            "event_count": len(result.final_state.events_triggered),
        },
        events=_serialize_events(result.events),
        outcomes=outcomes,
        plots=_trajectory_plots(result.final_state, result.events, result.elements),
    )


def _serialize_mc_metrics(stats: MonteCarloStats) -> dict[str, MonteCarloMetricModel]:
    metric_names = set(stats.mean) | set(stats.std)
    metric_models: dict[str, MonteCarloMetricModel] = {}
    for name in metric_names:
        percentiles = {
            f"p{row.pct}": row.values[name]
            for row in stats.percentile_rows
            if name in row.values
        }
        metric_models[name] = MonteCarloMetricModel(
            mean=stats.mean[name],
            std=stats.std[name],
            percentiles=percentiles,
        )
    return metric_models


def build_montecarlo_payload(
    cfg: MissionToolkitConfig,
    results: list[dict[str, Any]],
    request_id: str,
    run_id: str | None = None,
) -> MonteCarloPayload:
    if run_id is None:
        run_id = create_run_id("montecarlo")

    stats = compute_statistics(results, cfg.uncertainties.montecarlo.output_percentiles)
    metrics = _serialize_mc_metrics(stats)
    run_results = [MonteCarloRunResultModel(**result) for result in results]

    success_rate_plot = PlotModel(
        id="insertion_success",
        title="Insertion Success",
        kind="bar",
        x_axis=AxisModel(key="category", label="Category"),
        y_axis=AxisModel(key="count", label="Runs"),
        series=[
            PlotSeriesModel(
                key="success_counts",
                label="Runs",
                x_data=[0.0, 1.0],
                y_data=[float(stats.n_success), float(stats.n_runs - stats.n_success)],
                metadata={"categories": ["success", "miss"]},
            )
        ],
    )

    return MonteCarloPayload(
        request_id=request_id,
        run_id=run_id,
        run_type="montecarlo",
        mission=serialize_mission(cfg),
        vehicle=serialize_vehicle(cfg),
        status=StatusModel(state="completed", message=f"Completed {stats.n_runs} Monte Carlo runs"),
        summary={
            "n_runs": stats.n_runs,
            "n_success": stats.n_success,
            "success_rate_pct": stats.success_rate_pct,
            "insertion_tolerance_km": 5.0,
        },
        metrics=metrics,
        runs=run_results,
        plots=[success_rate_plot],
    )
