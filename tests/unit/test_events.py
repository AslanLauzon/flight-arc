"""Unit tests for the event system."""

import pytest
from src.config import load_config
from src.events.standard_events import (
    FairingDeployEvent,
    SECOEvent,
    StagingEvent,
)
from src.propagator.state import SimState
from src.vehicle.vehicle import Vehicle


@pytest.fixture
def vehicle():
    return Vehicle(load_config().vehicle)


def _state_at_alt(alt_m: float) -> SimState:
    s = SimState()
    s.y = alt_m
    return s


# ── Staging ──────────────────────────────────────────────────────────────────

def test_staging_does_not_fire_with_propellant(vehicle):
    ev = StagingEvent(vehicle)
    assert not ev.check(_state_at_alt(0))


def test_staging_fires_when_exhausted(vehicle):
    vehicle.mass_model.propellant_remaining_kg = 0.0
    ev = StagingEvent(vehicle)
    assert ev.check(_state_at_alt(70_000))


def test_staging_does_not_fire_on_stage_2(vehicle):
    vehicle.mass_model.jettison("stage")   # advance to stage 2
    vehicle.mass_model.propellant_remaining_kg = 0.0
    ev = StagingEvent(vehicle)
    assert not ev.check(_state_at_alt(0))


# ── Fairing deploy ────────────────────────────────────────────────────────────

def test_fairing_does_not_fire_below_altitude(vehicle):
    ev = FairingDeployEvent(vehicle, deploy_altitude_m=120_000)
    assert not ev.check(_state_at_alt(100_000))


def test_fairing_fires_at_altitude(vehicle):
    ev = FairingDeployEvent(vehicle, deploy_altitude_m=120_000)
    assert ev.check(_state_at_alt(120_000))


def test_fairing_fires_above_altitude(vehicle):
    ev = FairingDeployEvent(vehicle, deploy_altitude_m=120_000)
    assert ev.check(_state_at_alt(130_000))


def test_fairing_fires_only_once(vehicle):
    ev = FairingDeployEvent(vehicle, deploy_altitude_m=120_000)
    s  = _state_at_alt(130_000)
    ev.trigger(s)
    assert not ev.check(s)   # one_shot — should not fire again


# ── SECO ──────────────────────────────────────────────────────────────────────

def test_seco_does_not_fire_on_stage_1(vehicle):
    vehicle.mass_model.propellant_remaining_kg = 0.0
    ev = SECOEvent(vehicle)
    # still on stage 1 → should NOT fire
    assert not ev.check(_state_at_alt(0))


def test_seco_fires_on_stage_2_exhaustion(vehicle):
    vehicle.mass_model.jettison("stage")
    vehicle.mass_model.propellant_remaining_kg = 0.0
    ev = SECOEvent(vehicle)
    assert ev.check(_state_at_alt(250_000))


def test_guidance_commanded_seco(vehicle):
    from src.guidance.base import GuidanceBase
    from src.propagator.state import SimState as SS

    class _FakeGuidance(GuidanceBase):
        cutoff_commanded = True
        def pitch_angle_deg(self, state: SS) -> float:
            return 0.0

    vehicle.mass_model.jettison("stage")   # on stage 2 with propellant remaining
    ev = SECOEvent(vehicle, guidance=_FakeGuidance())
    assert ev.check(_state_at_alt(270_000))


def test_propellant_exhausted_seco_no_guidance(vehicle):
    vehicle.mass_model.jettison("stage")
    vehicle.mass_model.propellant_remaining_kg = 0.0
    # no guidance passed — should still fire on propellant exhaustion
    ev = SECOEvent(vehicle, guidance=None)
    assert ev.check(_state_at_alt(250_000))
