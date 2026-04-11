"""
config.py — load, validate, and expose all YAML config as typed Pydantic models.

Usage:
    from src.config import load_config
    cfg = load_config()          # reads all four YAML files
    cfg.vehicle.stages[0].thrust_vac_N
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).parent.parent / "config"


# ---------------------------------------------------------------------------
# Vehicle models
# ---------------------------------------------------------------------------
class StageConfig(BaseModel):
    id: int
    name: str
    propellant_mass_kg: float
    dry_mass_kg: float
    thrust_vac_N: float
    isp_vac_s: float
    isp_sl_s: float
    burn_time_s: float           # informational only — not enforced by the integrator
    cd_table: list[list[float]]  # raw [[mach, cd], ...] from YAML


class PayloadConfig(BaseModel):
    mass_kg: float
    fairing_mass_kg: float
    fairing_jettisoned: bool = True


class VehicleConfig(BaseModel):
    name: str
    reference_area_m2: float
    stages: list[StageConfig]
    payload: PayloadConfig


# ---------------------------------------------------------------------------
# Mission models
# ---------------------------------------------------------------------------
class LaunchSiteConfig(BaseModel):
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    azimuth_deg: float


class TargetOrbitConfig(BaseModel):
    apogee_km: float
    perigee_km: float
    inclination_deg: float


class PEGConfig(BaseModel):
    update_interval_s: float = 2.0


class GuidanceConfig(BaseModel):
    mode: Literal["pitch_program", "gravity_turn", "peg"]
    pitch_program: dict | None = None
    gravity_turn: dict | None = None
    peg: PEGConfig | None = None


class MissionConfig(BaseModel):
    name: str
    launch_site: LaunchSiteConfig
    target_orbit: TargetOrbitConfig
    guidance: GuidanceConfig


# ---------------------------------------------------------------------------
# Simulation models
# ---------------------------------------------------------------------------
class ConstraintsConfig(BaseModel):
    max_q_Pa: float
    fairing_deploy_altitude_m: float
    min_staging_altitude_m: float


class SimulationConfig(BaseModel):
    t_start_s: float
    t_end_s: float
    max_step_s: float
    constraints: ConstraintsConfig


# ---------------------------------------------------------------------------
# Monte Carlo models
# ---------------------------------------------------------------------------
class UncertaintyParam(BaseModel):
    dist: Literal["normal", "uniform"]
    mu: float | None = None
    sigma: float | None = None
    low: float | None = None
    high: float | None = None


class MonteCarloRunConfig(BaseModel):
    n_runs: int
    seed: int
    n_jobs: int
    output_percentiles: list[int]


class UncertaintiesConfig(BaseModel):
    uncertainties: dict[str, UncertaintyParam]
    montecarlo: MonteCarloRunConfig


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------
class MissionToolkitConfig(BaseModel):
    vehicle: VehicleConfig
    mission: MissionConfig
    simulation: SimulationConfig
    uncertainties: UncertaintiesConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(config_dir: Path = CONFIG_DIR) -> MissionToolkitConfig:
    """Load and validate all YAML config files."""
    vehicle_raw = _load_yaml(config_dir / "vehicle.yaml")
    mission_raw = _load_yaml(config_dir / "mission.yaml")
    sim_raw     = _load_yaml(config_dir / "simulation.yaml")
    unc_raw     = _load_yaml(config_dir / "uncertainties.yaml")

    # Remap pitch_program list into the expected dict key
    guidance = mission_raw["mission"]["guidance"]
    if "pitch_program" in guidance and isinstance(guidance["pitch_program"], list):
        guidance["pitch_program"] = {"points": guidance["pitch_program"]}

    return MissionToolkitConfig(
        vehicle=VehicleConfig(**vehicle_raw["vehicle"]),
        mission=MissionConfig(**mission_raw["mission"]),
        simulation=SimulationConfig(**sim_raw["simulation"]),
        uncertainties=UncertaintiesConfig(
            uncertainties={
                k: UncertaintyParam(**v)
                for k, v in unc_raw["uncertainties"].items()
            },
            montecarlo=MonteCarloRunConfig(**unc_raw["montecarlo"]),
        ),
    )
