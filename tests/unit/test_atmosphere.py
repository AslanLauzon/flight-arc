"""Unit tests for the US Standard Atmosphere 1976 model."""

import pytest
from src.atmosphere.us_standard_1976 import (
    temperature, pressure, density, speed_of_sound, dynamic_pressure
)

# Published reference values (NOAA, 1976)
@pytest.mark.parametrize("alt_m, T_ref, P_ref, rho_ref", [
    (      0,  288.15, 101_325.0, 1.2250),
    ( 11_000,  216.65,  22_632.1, 0.3639),
    ( 20_000,  216.65,   5_474.9, 0.0880),
    ( 32_000,  228.65,     868.0, 0.0132),
])
def test_standard_values(alt_m, T_ref, P_ref, rho_ref):
    assert temperature(alt_m) == pytest.approx(T_ref, rel=1e-3)
    assert pressure(alt_m)    == pytest.approx(P_ref, rel=1e-3)
    assert density(alt_m)     == pytest.approx(rho_ref, rel=1e-2)


def test_sea_level_speed_of_sound():
    # ~340.3 m/s at SL
    assert speed_of_sound(0) == pytest.approx(340.3, rel=1e-3)


def test_vacuum_above_86km():
    assert pressure(90_000) == 0.0
    assert density(90_000) == 0.0


def test_dynamic_pressure_zero_speed():
    assert dynamic_pressure(0, 0) == 0.0


def test_dynamic_pressure_positive():
    q = dynamic_pressure(0, 100.0)   # ~0.5 * 1.225 * 10000
    assert q == pytest.approx(6125.0, rel=1e-2)
