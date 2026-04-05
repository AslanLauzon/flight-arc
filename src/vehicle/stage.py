from __future__ import annotations

from bisect import bisect_right

from src.config import StageConfig


class Stage:
    """
    Immutable data and per-stage calculations.
    Holds fixed properties from config — never mutates after init.
    """

    def __init__(self, cfg: StageConfig) -> None:
        self.id = cfg.id
        self.name = cfg.name
        self.propellant_mass_kg = cfg.propellant_mass_kg
        self.dry_mass_kg = cfg.dry_mass_kg
        self.thrust_vac_N = cfg.thrust_vac_N
        self.isp_vac_s = cfg.isp_vac_s
        self.isp_sl_s = cfg.isp_sl_s
        self.burn_time_s = cfg.burn_time_s
        self.cd_table = cfg.cd_table

        # Pre-split cd_table for fast lookup (avoids rebuilding on every call)
        self._cd_machs: list[float] = [row[0] for row in cfg.cd_table]
        self._cd_vals:  list[float] = [row[1] for row in cfg.cd_table]

    def effective_isp(self, ambient_pressure_Pa: float) -> float:
        """
        Linearly interpolate between sea level and vacuum Isp based on ambient pressure.
        At P_amb = P_sl returns isp_sl, at P_amb = 0 returns isp_vac.
        """
        P_sl = 101325.0
        return self.isp_sl_s + (self.isp_vac_s - self.isp_sl_s) * (1.0 - ambient_pressure_Pa / P_sl)

    def mass_flow(self, ambient_pressure_Pa: float) -> float:
        """Mass flow rate [kg/s] at current ambient pressure."""
        G0 = 9.80665
        return self.thrust_vac_N / (self.effective_isp(ambient_pressure_Pa) * G0)

    def drag_coefficient(self, mach: float) -> float:
        """
        Linearly interpolate Cd from cd_table at given Mach number.
        Uses pure-Python bisect — faster than np.interp for small tables.
        Clamps flat beyond table bounds.
        """
        machs = self._cd_machs
        cds   = self._cd_vals

        if mach <= machs[0]:
            return cds[0]
        if mach >= machs[-1]:
            return cds[-1]

        i = bisect_right(machs, mach) - 1
        t = (mach - machs[i]) / (machs[i + 1] - machs[i])
        return cds[i] + t * (cds[i + 1] - cds[i])
