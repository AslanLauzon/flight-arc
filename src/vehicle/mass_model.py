from src.vehicle.stage import Stage


class MassModel:
    def __init__(
        self,
        stages: list[Stage],
        payload_mass: float,
        fairing_mass: float,
        fairing_jettisoned: bool = True,
    ) -> None:
        self.stages = stages
        self.payload_mass = payload_mass
        self.fairing_mass = fairing_mass
        self.fairing_jettisoned = fairing_jettisoned
        self.current_stage_index = 0
        self.propellant_remaining_kg = stages[0].propellant_mass_kg
        self.fairing_attached = True
        self.payload_attached = True

    def total_mass(self) -> float:
        mass = self.propellant_remaining_kg
        mass += self.current_stage.dry_mass_kg
        for stage in self.stages[self.current_stage_index + 1:]:
            mass += stage.dry_mass_kg + stage.propellant_mass_kg
        if self.fairing_attached:
            mass += self.fairing_mass
        if self.payload_attached:
            mass += self.payload_mass
        return mass

    def burn(self, dt: float, ambient_pressure_pa: float, engine_on: bool = True) -> None:
        if not engine_on:
            return

        mass_flow_kg = self.current_stage.mass_flow(ambient_pressure_pa)
        self.propellant_remaining_kg -= mass_flow_kg * dt
        self.propellant_remaining_kg = max(0.0, self.propellant_remaining_kg)

    def jettison(self, item: str) -> None:
        if item == "stage":
            self.current_stage_index += 1
            self.propellant_remaining_kg = self.current_stage.propellant_mass_kg
            return

        if item == "fairing":
            if self.fairing_jettisoned:
                self.fairing_attached = False
            return

        if item == "payload":
            self.payload_attached = False

    @property
    def current_stage(self) -> Stage:
        return self.stages[self.current_stage_index]

    @property
    def propellant_exhausted(self) -> bool:
        return self.propellant_remaining_kg <= 0.0
