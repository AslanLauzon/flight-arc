"""
us_standard_1976.py — US Standard Atmosphere 1976, 0–86 km.

Reference: NOAA/NASA/USAF, "U.S. Standard Atmosphere, 1976."

Functions are stateless and operate on scalar or numpy array altitudes.
"""

from __future__ import annotations

import numpy as np

# Constants
R_AIR  = 287.058   # specific gas constant [J/(kg·K)]
GAMMA  = 1.4       # ratio of specific heats
G0     = 9.80665   # standard gravity [m/s²]
R_STAR = 8.31432   # universal gas constant [J/(mol·K)]
M0     = 0.0289644 # molar mass of air [kg/mol]

# Layer base altitudes [m], temperatures [K], lapse rates [K/m]
_LAYERS = np.array([
    # h_base,   T_base,   L (K/m)
    [     0.0,  288.150,  -0.0065],
    [ 11_000.0, 216.650,   0.0000],
    [ 20_000.0, 216.650,   0.0010],
    [ 32_000.0, 228.650,   0.0028],
    [ 47_000.0, 270.650,   0.0000],
    [ 51_000.0, 270.650,  -0.0028],
    [ 71_000.0, 214.650,  -0.0020],
    [ 86_000.0, 186.870,   0.0000],  # sentinel
])

# Pressure at each layer base [Pa] — computed once at import
_P_BASE = np.zeros(len(_LAYERS))
_P_BASE[0] = 101_325.0
for _i in range(1, len(_LAYERS)):
    _h0, _T0, _L = _LAYERS[_i - 1]
    _h1, _T1, _   = _LAYERS[_i]
    _dh = _h1 - _h0
    if abs(_L) < 1e-12:  # isothermal
        _P_BASE[_i] = _P_BASE[_i - 1] * np.exp(-G0 * M0 * _dh / (R_STAR * _T0))
    else:
        _P_BASE[_i] = _P_BASE[_i - 1] * (_T1 / _T0) ** (-G0 * M0 / (R_STAR * _L))


def temperature(altitude_m: float) -> float:
    """Return ambient temperature [K] at altitude [m]. Clamps above 86 km."""
    alt = min(float(altitude_m), 86_000.0)
    alt = max(alt, 0.0)
    idx = int(np.searchsorted(_LAYERS[:, 0], alt, side="right")) - 1
    h0, T0, L = _LAYERS[idx]
    return T0 + L * (alt - h0)


def pressure(altitude_m: float) -> float:
    """Return static pressure [Pa] at altitude [m]. Returns 0 above 86 km."""
    if altitude_m > 86_000.0:
        return 0.0
    alt = max(float(altitude_m), 0.0)
    idx = int(np.searchsorted(_LAYERS[:, 0], alt, side="right")) - 1
    h0, T0, L = _LAYERS[idx]
    T = T0 + L * (alt - h0)
    P0 = _P_BASE[idx]
    if abs(L) < 1e-12:
        return P0 * np.exp(-G0 * M0 * (alt - h0) / (R_STAR * T0))
    return P0 * (T / T0) ** (-G0 * M0 / (R_STAR * L))


def density(altitude_m: float) -> float:
    """Return air density [kg/m³] at altitude [m]."""
    T = temperature(altitude_m)
    P = pressure(altitude_m)
    if T < 1.0:
        return 0.0
    return P / (R_AIR * T)


def speed_of_sound(altitude_m: float) -> float:
    """Return speed of sound [m/s] at altitude [m]."""
    T = temperature(altitude_m)
    return (GAMMA * R_AIR * T) ** 0.5


def dynamic_pressure(altitude_m: float, speed_m_s: float) -> float:
    """Return dynamic pressure q = 0.5 * rho * v² [Pa]."""
    rho = density(altitude_m)
    return 0.5 * rho * speed_m_s ** 2
