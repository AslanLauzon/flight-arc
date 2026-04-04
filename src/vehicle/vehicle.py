from src.atmosphere.us_standard_1976 import density, pressure, speed_of_sound
from src.config import VehicleConfig
from src.vehicle.mass_model import MassModel
from src.vehicle.stage import Stage


class Vehicle:
    def __init__(self, cfg: VehicleConfig) -> None:
        self.name = cfg.name
        self.reference_area_m2 = cfg.reference_area_m2
        self.stages = [Stage(stage_cfg) for stage_cfg in cfg.stages]
        self.mass_model = MassModel(
            self.stages,
            cfg.payload.mass_kg,
            cfg.payload.fairing_mass_kg,
            fairing_jettisoned=cfg.payload.fairing_jettisoned,
        )

    @property
    def mass(self) -> float:
        return self.mass_model.total_mass()

    def thrust(self, altitude_m: float) -> float:
        stage = self.mass_model.current_stage
        ambient_pressure_pa = pressure(altitude_m)
        return stage.thrust_vac_N * (stage.effective_isp(ambient_pressure_pa) / stage.isp_vac_s)

    def drag(self, altitude_m: float, vx: float, vy: float) -> tuple[float, float]:
        speed = (vx**2 + vy**2) ** 0.5
        if speed < 1e-6:
            return 0.0, 0.0

        rho = density(altitude_m)
        mach = speed / speed_of_sound(altitude_m)
        cd = self.mass_model.current_stage.drag_coefficient(mach)
        drag_total = 0.5 * rho * speed**2 * cd * self.reference_area_m2
        return -drag_total * (vx / speed), -drag_total * (vy / speed)
