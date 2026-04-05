"""
us_standard_1976.py — US Standard Atmosphere 1976, 0–86 km.

Reference: NOAA/NASA/USAF, "U.S. Standard Atmosphere, 1976."

Functions are stateless and operate on scalar altitudes.

Performance note
----------------
Layer lookups use pure-Python bisect on a list of 8 breakpoints.
This avoids numpy overhead (~1 μs/call) for what would otherwise be
a very cheap operation — the numpy searchsorted call on a tiny array
turns out to be slower than a Python for-loop at this scale.
"""

from __future__ import annotations

import math
from bisect import bisect_right

# ── Constants ────────────────────────────────────────────────────────────────
R_AIR  = 287.058    # specific gas constant [J/(kg·K)]
GAMMA  = 1.4        # ratio of specific heats
G0     = 9.80665    # standard gravity [m/s²]
R_STAR = 8.31432    # universal gas constant [J/(mol·K)]
M0     = 0.0289644  # molar mass of air [kg/mol]
_GM_R  = G0 * M0 / R_STAR   # = 0.034163  (precomputed constant)

# ── Layer definitions ────────────────────────────────────────────────────────
# Tuples of (h_base [m], T_base [K], lapse_rate [K/m])
_LAYER_H   = [      0.0,  11_000.0,  20_000.0,  32_000.0,  47_000.0,  51_000.0,  71_000.0,  86_000.0]
_LAYER_T   = [  288.150,    216.650,   216.650,   228.650,   270.650,   270.650,   214.650,   186.870]
_LAYER_L   = [ -0.00650,    0.00000,   0.00100,   0.00280,   0.00000,  -0.00280,  -0.00200,   0.00000]

# ── Pre-computed base pressures ──────────────────────────────────────────────
_LAYER_P = [101_325.0]
for _i in range(1, 8):
    _h0, _T0, _L = _LAYER_H[_i - 1], _LAYER_T[_i - 1], _LAYER_L[_i - 1]
    _h1, _T1     = _LAYER_H[_i],      _LAYER_T[_i]
    _dh = _h1 - _h0
    if abs(_L) < 1e-12:
        _LAYER_P.append(_LAYER_P[-1] * math.exp(-_GM_R * _dh / _T0))
    else:
        _LAYER_P.append(_LAYER_P[-1] * (_T1 / _T0) ** (-_GM_R / _L))

_MAX_ALT = 86_000.0


def _layer(alt: float) -> int:
    """Return index of the atmospheric layer containing *alt* [m]."""
    return bisect_right(_LAYER_H, alt) - 1


def temperature(altitude_m: float) -> float:
    """Return ambient temperature [K] at altitude [m]. Clamps above 86 km."""
    alt = min(altitude_m, _MAX_ALT)
    if alt < 0.0:
        alt = 0.0
    i = _layer(alt)
    return _LAYER_T[i] + _LAYER_L[i] * (alt - _LAYER_H[i])


def _pressure_at(alt: float, i: int, T: float) -> float:
    """Return pressure [Pa] given layer index and temperature (avoids re-lookup)."""
    L = _LAYER_L[i]
    P0, T0, h0 = _LAYER_P[i], _LAYER_T[i], _LAYER_H[i]
    if abs(L) < 1e-12:
        return P0 * math.exp(-_GM_R * (alt - h0) / T0)
    return P0 * (T / T0) ** (-_GM_R / L)


def pressure(altitude_m: float) -> float:
    """Return static pressure [Pa] at altitude [m]. Returns 0 above 86 km."""
    if altitude_m > _MAX_ALT:
        return 0.0
    alt = max(altitude_m, 0.0)
    i = _layer(alt)
    T = _LAYER_T[i] + _LAYER_L[i] * (alt - _LAYER_H[i])
    return _pressure_at(alt, i, T)


def density(altitude_m: float) -> float:
    """Return air density [kg/m³] at altitude [m]."""
    if altitude_m > _MAX_ALT:
        return 0.0
    alt = max(altitude_m, 0.0)
    i = _layer(alt)
    T = _LAYER_T[i] + _LAYER_L[i] * (alt - _LAYER_H[i])
    P = _pressure_at(alt, i, T)
    return P / (R_AIR * T)


def speed_of_sound(altitude_m: float) -> float:
    """Return speed of sound [m/s] at altitude [m]."""
    T = temperature(altitude_m)
    return (GAMMA * R_AIR * T) ** 0.5


def dynamic_pressure(altitude_m: float, speed_m_s: float) -> float:
    """Return dynamic pressure q = 0.5 * rho * v² [Pa]."""
    return 0.5 * density(altitude_m) * speed_m_s * speed_m_s


def density_and_mach(altitude_m: float, speed_m_s: float) -> tuple[float, float]:
    """
    Return (rho [kg/m³], mach [-]) in one layer lookup.
    Use this instead of calling density() + speed_of_sound() separately.
    """
    if altitude_m > _MAX_ALT or speed_m_s < 1e-6:
        return 0.0, 0.0
    alt = max(altitude_m, 0.0)
    i = _layer(alt)
    T = _LAYER_T[i] + _LAYER_L[i] * (alt - _LAYER_H[i])
    P = _pressure_at(alt, i, T)
    rho = P / (R_AIR * T)
    a   = (GAMMA * R_AIR * T) ** 0.5
    return rho, speed_m_s / a
