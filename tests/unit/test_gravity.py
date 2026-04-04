"""Unit tests for the gravity model."""

import pytest
from src.gravity.gravity import gravity

G0      = 9.80665
R_EARTH = 6_371_000.0


def test_sea_level_value():
    assert gravity(0) == pytest.approx(G0, rel=1e-6)


def test_decreases_with_altitude():
    assert gravity(100_000) < gravity(0)
    assert gravity(400_000) < gravity(100_000)


def test_inverse_square_law():
    h = 200_000.0
    expected = G0 * (R_EARTH / (R_EARTH + h)) ** 2
    assert gravity(h) == pytest.approx(expected, rel=1e-9)


def test_positive_everywhere():
    for h in [0, 1000, 10_000, 100_000, 400_000, 800_000]:
        assert gravity(h) > 0
