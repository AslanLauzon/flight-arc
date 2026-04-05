from src.config import StageConfig
import numpy as np


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

        # 6DOF — rotational properties
        self._inertia_cfg = cfg.inertia
        self.gimbal_max_deflection_rad = np.radians(cfg.gimbal.max_deflection_deg)
        self.gimbal_moment_arm_m = cfg.gimbal.moment_arm_m
        self.aero_ref_length_m = cfg.aero_moments.reference_length_m
        self.cm_alpha_per_rad = cfg.aero_moments.cm_alpha_per_rad
        self.attitude_kp = cfg.attitude_controller.kp
        self.attitude_kd = cfg.attitude_controller.kd

    def effective_isp(self, ambient_pressure_Pa: float) -> float:
        P_sl = 101325.0
        return self.isp_sl_s + (self.isp_vac_s - self.isp_sl_s) * (1.0 - ambient_pressure_Pa / P_sl)

    def mass_flow(self, ambient_pressure_Pa: float) -> float:
        G0 = 9.80665
        return self.thrust_vac_N / (self.effective_isp(ambient_pressure_Pa) * G0)

    def drag_coefficient(self, mach: float) -> float:
        machs = [point[0] for point in self.cd_table]
        cds = [point[1] for point in self.cd_table]
        return np.interp(mach, machs, cds, left=cds[0], right=cds[-1])

    def inertia_tensor(self, propellant_remaining_kg: float) -> tuple[float, float, float]:
        """
        Return (Ixx, Iyy, Izz) [kg·m²] for the current propellant load.
        Linearly interpolates between dry and wet inertia values.
        """
        if self._inertia_cfg is None:
            # Rough estimate if not configured: point-mass approximation
            total_mass = self.dry_mass_kg + propellant_remaining_kg
            i_trans = total_mass * (self.aero_ref_length_m ** 2) / 12.0
            return (total_mass * 0.1, i_trans, i_trans)

        frac = propellant_remaining_kg / max(self.propellant_mass_kg, 1.0)
        ixx = self._inertia_cfg.ixx_dry_kg_m2 + frac * (self._inertia_cfg.ixx_wet_kg_m2 - self._inertia_cfg.ixx_dry_kg_m2)
        iyy = self._inertia_cfg.iyy_dry_kg_m2 + frac * (self._inertia_cfg.iyy_wet_kg_m2 - self._inertia_cfg.iyy_dry_kg_m2)
        izz = self._inertia_cfg.izz_dry_kg_m2 + frac * (self._inertia_cfg.izz_wet_kg_m2 - self._inertia_cfg.izz_dry_kg_m2)
        return (ixx, iyy, izz)
