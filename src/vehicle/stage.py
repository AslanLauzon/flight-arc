from src.config import StageConfig
import numpy as np


class Stage:
    """
    Immutable data and per-stage calculations.
    Holds fixed properties from config — never mutates after init.
    """

    def __init__(self, cfg: StageConfig) -> None:
        """Stores all stage properties from config."""
        self.id = cfg.id
        self.name = cfg.name
        self.propellant_mass_kg = cfg.propellant_mass_kg
        self.dry_mass_kg = cfg.dry_mass_kg
        self.thrust_vac_N = cfg.thrust_vac_N
        self.isp_vac_s = cfg.isp_vac_s
        self.isp_sl_s = cfg.isp_sl_s
        self.burn_time_s = cfg.burn_time_s
        self.cd_table = cfg.cd_table

    def effective_isp(self, ambient_pressure_Pa: float) -> float:
        """
        Linearly interpolate between sea level and vacuum Isp based on ambient pressure.
        At P_amb = P_sl returns isp_sl, at P_amb = 0 returns isp_vac.
        """
        P_sl = 101325.0
        return self.isp_sl_s + (self.isp_vac_s - self.isp_sl_s) * (1.0 - ambient_pressure_Pa / P_sl)

    def mass_flow(self, ambient_pressure_Pa: float) -> float:
        """
        Mass flow rate at current ambient pressure.
        mdot = thrust_vac_N / (effective_isp * G0)
        """
        G0 = 9.80665
        return self.thrust_vac_N / (self.effective_isp(ambient_pressure_Pa) * G0)

    def drag_coefficient(self, mach: float) -> float:
        """
        Linearly interpolate Cd from cd_table at given Mach number.
        Clamps flat beyond table bounds (no extrapolation).
        """
        machs = [point[0] for point in self.cd_table]
        cds = [point[1] for point in self.cd_table]
        return np.interp(mach, machs, cds, left=cds[0], right=cds[-1])
