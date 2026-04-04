"""Unit tests for orbital element computation."""

import math
import pytest
from src.orbital.elements import elements_from_state
from src.propagator.state import SimState

MU      = 3.986004418e14
R_EARTH = 6_371_000.0


def _circular_state(alt_m: float) -> SimState:
    """Return a state that sits on a circular orbit at alt_m."""
    r = R_EARTH + alt_m
    v = math.sqrt(MU / r)
    s = SimState()
    s.y  = alt_m
    s.vx = v
    s.vy = 0.0
    return s


def test_circular_orbit_eccentricity():
    s = _circular_state(400_000)
    el = elements_from_state(s)
    assert el.eccentricity == pytest.approx(0.0, abs=1e-6)


def test_circular_orbit_perigee_apogee_equal():
    s = _circular_state(300_000)
    el = elements_from_state(s)
    assert el.perigee_alt_km == pytest.approx(el.apogee_alt_km, abs=0.1)


def test_circular_orbit_altitude():
    alt_km = 250.0
    s = _circular_state(alt_km * 1e3)
    el = elements_from_state(s)
    assert el.perigee_alt_km == pytest.approx(alt_km, abs=0.5)


def test_elliptical_orbit_perigee_lt_apogee():
    """Elliptical orbit: perigee < apogee."""
    s = SimState()
    r_p = R_EARTH + 200_000   # 200 km perigee
    r_a = R_EARTH + 500_000   # 500 km apogee
    a   = (r_p + r_a) / 2
    # velocity at perigee
    s.y  = 200_000
    s.vx = math.sqrt(MU * (2 / r_p - 1 / a))
    s.vy = 0.0
    el = elements_from_state(s)
    assert el.perigee_alt_km < el.apogee_alt_km
    assert el.perigee_alt_km == pytest.approx(200.0, abs=1.0)
    assert el.apogee_alt_km  == pytest.approx(500.0, abs=1.0)


def test_hyperbolic_raises():
    s = SimState()
    s.y  = 200_000
    r    = R_EARTH + 200_000
    s.vx = math.sqrt(2 * MU / r) * 1.1   # above escape velocity
    s.vy = 0.0
    with pytest.raises(ValueError):
        elements_from_state(s)


def test_period_positive():
    s = _circular_state(400_000)
    el = elements_from_state(s)
    assert el.orbital_period_s > 0


def test_iss_like_orbit():
    """~400 km circular orbit should have ~92 min period."""
    s = _circular_state(400_000)
    el = elements_from_state(s)
    assert el.orbital_period_s / 60 == pytest.approx(92.0, abs=2.0)
