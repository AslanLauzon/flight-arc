from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AxisModel(BaseModel):
    key: str
    label: str
    unit: str | None = None


class PlotAnnotationModel(BaseModel):
    kind: str
    value: float
    label: str | None = None


class PlotSeriesModel(BaseModel):
    key: str
    label: str
    x_data: list[float] | None = None
    y_data: list[float]
    unit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlotModel(BaseModel):
    id: str
    title: str
    kind: str
    x_axis: AxisModel | None = None
    y_axis: AxisModel | None = None
    series: list[PlotSeriesModel]
    annotations: list[PlotAnnotationModel] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactImageModel(BaseModel):
    id: str
    mime_type: str
    encoding: str
    data: str


class EventModel(BaseModel):
    name: str
    time_s: float
    message: str
    snapshot: dict[str, Any]


class MissionTargetModel(BaseModel):
    type: str
    parameters: dict[str, Any]


class MissionMetadataModel(BaseModel):
    name: str
    vehicle_name: str
    vehicle_type: str
    guidance_mode: str
    target: MissionTargetModel


class VehicleStageModel(BaseModel):
    name: str
    dry_mass_kg: float
    propellant_mass_kg: float
    thrust_vac_n: float
    isp_vac_s: float
    burn_time_s: float


class VehicleMetadataModel(BaseModel):
    vehicle_type: str
    reference_area_m2: float
    stages: list[VehicleStageModel]
    payload: dict[str, Any]


class StatusModel(BaseModel):
    state: str
    message: str


class NominalRunPayload(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    run_type: str
    mission: MissionMetadataModel
    vehicle: VehicleMetadataModel
    status: StatusModel
    summary: dict[str, Any]
    events: list[EventModel]
    outcomes: dict[str, Any]
    plots: list[PlotModel]
    artifacts: dict[str, list[ArtifactImageModel]]


class MonteCarloMetricModel(BaseModel):
    mean: float
    std: float
    percentiles: dict[str, float]


class MonteCarloPayload(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    run_type: str
    mission: MissionMetadataModel
    vehicle: VehicleMetadataModel
    status: StatusModel
    summary: dict[str, Any]
    metrics: dict[str, MonteCarloMetricModel]
    plots: list[PlotModel]
    artifacts: dict[str, list[ArtifactImageModel]]
