from src.vehicle.stage import Stage


class MassModel:
    """
    Tracks all mass accounting for the vehicle throughout flight.
    Mutated by events (jettison) and the integrator (burn).
    """

    def __init__(self, stages: list[Stage], payload_mass: float, fairing_mass: float) -> None:
        """
        Stores references to all stages, sets initial propellant to stage 0's full load.
        Initialises fairing_attached and payload_attached to True.
        """
        self.stages = stages
        self.payload_mass = payload_mass
        self.fairing_mass = fairing_mass
        self.current_stage_index = 0
        self.propellant_remaining_kg = stages[0].propellant_mass_kg
        self.fairing_attached = True
        self.payload_attached = True

    def total_mass(self) -> float:
        """
        Sum: current stage dry + propellant remaining
             + all upper stages (dry + propellant, still stacked)
             + fairing if attached
             + payload if attached
        """
        mass = self.propellant_remaining_kg
        mass += self.current_stage.dry_mass_kg
        for stage in self.stages[self.current_stage_index + 1:]:
            mass += stage.dry_mass_kg + stage.propellant_mass_kg
        if self.fairing_attached:
            mass += self.fairing_mass
        if self.payload_attached:
            mass += self.payload_mass
        return mass

    def burn(self, dt: float, ambient_pressure_Pa: float, engine_on: bool = True) -> None:
        """
        Deducts propellant consumed this timestep.
        No-op when engine_on is False (coast phase between burns).
        """
        if not engine_on:
            return
        mass_flow_kg = self.current_stage.mass_flow(ambient_pressure_Pa)
        self.propellant_remaining_kg -= mass_flow_kg * dt
        self.propellant_remaining_kg = max(0.0, self.propellant_remaining_kg)

    def jettison(self, item: str) -> None:
        """
        Drop a mass item. item is one of: "stage", "fairing", "payload".
        For "stage": increment current_stage_index,
                     reset propellant_remaining to new stage's full load.
        For "fairing"/"payload": flip the corresponding bool to False.
        """
        if item == "stage":
            self.current_stage_index += 1
            self.propellant_remaining_kg = self.current_stage.propellant_mass_kg
        elif item == "fairing":
            self.fairing_attached = False
        elif item == "payload":
            self.payload_attached = False

    @property
    def current_stage(self) -> Stage:
        """Return the Stage object at current_stage_index."""
        return self.stages[self.current_stage_index]

    @property
    def propellant_exhausted(self) -> bool:
        """Return True if propellant_remaining_kg is at or below zero."""
        return self.propellant_remaining_kg <= 0.0
