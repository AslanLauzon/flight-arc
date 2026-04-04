"""Unit tests for Vehicle, MassModel, and Stage."""

import pytest
from src.config import load_config
from src.vehicle.vehicle import Vehicle

G0 = 9.80665


@pytest.fixture
def vehicle():
    cfg = load_config()
    return Vehicle(cfg.vehicle)


@pytest.fixture
def cfg():
    return load_config()


def test_initial_mass(vehicle, cfg):
    expected = (
        sum(s.propellant_mass_kg + s.dry_mass_kg for s in cfg.vehicle.stages)
        + cfg.vehicle.payload.mass_kg
        + cfg.vehicle.payload.fairing_mass_kg
    )
    assert vehicle.mass == pytest.approx(expected, rel=1e-9)


def test_initial_stage_is_zero(vehicle):
    assert vehicle.mass_model.current_stage_index == 0


def test_burn_reduces_propellant(vehicle, cfg):
    from src.atmosphere.us_standard_1976 import pressure
    initial_prop = vehicle.mass_model.propellant_remaining_kg
    vehicle.mass_model.burn(dt=1.0, ambient_pressure_Pa=pressure(0))
    assert vehicle.mass_model.propellant_remaining_kg < initial_prop


def test_burn_never_negative(vehicle, cfg):
    from src.atmosphere.us_standard_1976 import pressure
    # burn for a very long time
    for _ in range(10_000):
        vehicle.mass_model.burn(dt=1.0, ambient_pressure_Pa=0.0)
    assert vehicle.mass_model.propellant_remaining_kg == 0.0


def test_staging_advances_stage(vehicle):
    vehicle.mass_model.propellant_remaining_kg = 0.0
    assert vehicle.mass_model.propellant_exhausted
    vehicle.mass_model.jettison("stage")
    assert vehicle.mass_model.current_stage_index == 1


def test_staging_resets_propellant(vehicle, cfg):
    vehicle.mass_model.jettison("stage")
    expected = cfg.vehicle.stages[1].propellant_mass_kg
    assert vehicle.mass_model.propellant_remaining_kg == pytest.approx(expected)


def test_fairing_jettison_reduces_mass(vehicle, cfg):
    mass_before = vehicle.mass
    vehicle.mass_model.jettison("fairing")
    assert vehicle.mass == pytest.approx(mass_before - cfg.vehicle.payload.fairing_mass_kg)


def test_payload_jettison_reduces_mass(vehicle, cfg):
    mass_before = vehicle.mass
    vehicle.mass_model.jettison("payload")
    assert vehicle.mass == pytest.approx(mass_before - cfg.vehicle.payload.mass_kg)


def test_thrust_positive_at_sea_level(vehicle):
    assert vehicle.thrust(0.0) > 0


def test_thrust_higher_in_vacuum(vehicle):
    assert vehicle.thrust(200_000.0) > vehicle.thrust(0.0)


def test_stage_effective_isp_sea_level(cfg):
    from src.vehicle.stage import Stage
    s = Stage(cfg.vehicle.stages[0])
    assert s.effective_isp(101325.0) == pytest.approx(cfg.vehicle.stages[0].isp_sl_s, rel=1e-9)


def test_stage_effective_isp_vacuum(cfg):
    from src.vehicle.stage import Stage
    s = Stage(cfg.vehicle.stages[0])
    assert s.effective_isp(0.0) == pytest.approx(cfg.vehicle.stages[0].isp_vac_s, rel=1e-9)
