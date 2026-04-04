from src.config import VehicleConfig
from src.vehicle.stage import Stage
from src.vehicle.mass_model import MassModel
from src.atmosphere.us_standard_1976 import pressure, density, speed_of_sound

class Vehicle:
    """
    Top-level vehicle object. Answers physic questions for the EOM.
    Owns a MassModel does not store mass state itself.
    """

    def __init__(self, cfg: VehicleConfig) -> None:
        """Initialises stages from config, then mass model."""
        self.name = cfg.name
        self.reference_area_m2 = cfg.reference_area_m2
        self.stages = [Stage(stage_cfg) for stage_cfg in cfg.stages]
        self.mass_model = MassModel(self.stages, cfg.payload.mass_kg, cfg.payload.fairing_mass_kg)

    @property
    def mass(self) -> float:
        """Current total mass from mass model."""
        return self.mass_model.total_mass()
    
    def thrust(self, altitude_m: float) -> float:
        """Current thrust [N] adjusted for ambient pressure."""
        stage = self.mass_model.current_stage
        ambient_pressure_Pa = pressure(altitude_m)
        return stage.thrust_vac_N * (stage.effective_isp(ambient_pressure_Pa) / stage.isp_vac_s)

    def drag(self, altitude_m: float, vx: float, vy: float) -> tuple[float, float]:
        """
        Aerodynamic drag force components [N].
        D = 0.5 * rho * v^2 * Cd * A_ref, decomposed along velocity vector.
        Returns (drag_x_N, drag_y_N) — both negative (opposes motion).
        """
        v = (vx**2 + vy**2) ** 0.5
        if v < 1e-6:
            return 0.0, 0.0
        rho = density(altitude_m)
        mach = v / speed_of_sound(altitude_m)
        cd = self.mass_model.current_stage.drag_coefficient(mach)
        d_total = 0.5 * rho * v**2 * cd * self.reference_area_m2
        return -d_total * (vx / v), -d_total * (vy / v)

    