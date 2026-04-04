from __future__ import annotations

from src.events.event import Event, EventResult
from src.guidance.base import GuidanceBase
from src.propagator.state import SimState
from src.vehicle.vehicle import Vehicle


class IgnitionEvent(Event):
    name = "ignition"

    def _condition(self, state: SimState) -> bool:
        return state.t >= 0.0

    def _action(self, state: SimState) -> EventResult:
        return EventResult(
            name=self.name,
            t_trigger=state.t,
            state_snapshot=state.snapshot(),
            message="Main engine ignition command",
        )


class LiftoffEvent(Event):
    name = "liftoff"

    def _condition(self, state: SimState) -> bool:
        return state.y > 0.1

    def _action(self, state: SimState) -> EventResult:
        return EventResult(
            name=self.name,
            t_trigger=state.t,
            state_snapshot=state.snapshot(),
            message=f"Liftoff! Alt={state.y:.1f} m",
        )


class MaxQEvent(Event):
    name = "max_q"

    def __init__(self) -> None:
        super().__init__()
        self._prev_q = 0.0
        self._peak_q = 0.0

    def _condition(self, state: SimState) -> bool:
        q = state.dynamic_pressure_Pa
        if q > self._prev_q:
            self._prev_q = q
            self._peak_q = q
            return False
        if self._prev_q > 1_000.0 and q < self._prev_q:
            self._prev_q = q
            return True
        self._prev_q = q
        return False

    def _action(self, state: SimState) -> EventResult:
        return EventResult(
            name=self.name,
            t_trigger=state.t,
            state_snapshot=state.snapshot(),
            message=f"Max-Q = {self._peak_q:.0f} Pa",
        )


class StagingEvent(Event):
    name = "staging"

    def __init__(self, vehicle: Vehicle, guidance: GuidanceBase | None = None) -> None:
        super().__init__()
        self.vehicle = vehicle
        self.guidance = guidance

    def _condition(self, state: SimState) -> bool:
        if self.vehicle.mass_model.current_stage_index != 0:
            return False
        if self.vehicle.mass_model.propellant_exhausted:
            return True
        return (
            self.guidance is not None
            and getattr(self.guidance, "stage1_meco_commanded", False)
        )

    def _action(self, state: SimState) -> EventResult:
        self.vehicle.mass_model.propellant_remaining_kg = 0.0
        self.vehicle.mass_model.jettison("stage")
        state.stage_index = self.vehicle.mass_model.current_stage_index

        early_meco = (
            self.guidance is not None
            and getattr(self.guidance, "stage1_meco_commanded", False)
        )
        state.engine_on = not early_meco

        return EventResult(
            name=self.name,
            t_trigger=state.t,
            state_snapshot=state.snapshot(),
            message=(
                f"Stage 1 sep at t={state.t:.1f}s, Alt={state.y/1e3:.1f} km "
                f"vx={state.vx:.1f} m/s vy={state.vy:.1f} m/s "
                f"({'coast' if early_meco else 'S2 ignition'})"
            ),
        )


class Burn1MECOEvent(Event):
    name = "burn1_meco"
    terminal = False

    def __init__(self, guidance: GuidanceBase | None = None) -> None:
        super().__init__()
        self.guidance = guidance

    def _condition(self, state: SimState) -> bool:
        return (
            self.guidance is not None
            and getattr(self.guidance, "burn1_meco_commanded", False)
        )

    def _action(self, state: SimState) -> EventResult:
        state.engine_on = False
        return EventResult(
            name=self.name,
            t_trigger=state.t,
            state_snapshot=state.snapshot(),
            message=(
                f"Burn-1 MECO at t={state.t:.1f}s "
                f"alt={state.y/1e3:.1f}km  vx={state.vx:.0f} vy={state.vy:.0f} m/s"
            ),
        )


class RelightEvent(Event):
    name = "relight"
    terminal = False

    def __init__(self, guidance: GuidanceBase | None = None) -> None:
        super().__init__()
        self.guidance = guidance

    def _condition(self, state: SimState) -> bool:
        return (
            self.guidance is not None
            and getattr(self.guidance, "relight_commanded", False)
        )

    def _action(self, state: SimState) -> EventResult:
        state.engine_on = True
        return EventResult(
            name=self.name,
            t_trigger=state.t,
            state_snapshot=state.snapshot(),
            message=(
                f"Stage-2 relight at t={state.t:.1f}s "
                f"alt={state.y/1e3:.1f}km  vx={state.vx:.0f} vy={state.vy:.0f} m/s"
            ),
        )


class FairingDeployEvent(Event):
    name = "fairing_deploy"

    def __init__(self, vehicle: Vehicle, deploy_altitude_m: float) -> None:
        super().__init__()
        self.vehicle = vehicle
        self.deploy_altitude_m = deploy_altitude_m

    def _condition(self, state: SimState) -> bool:
        return state.y >= self.deploy_altitude_m

    def _action(self, state: SimState) -> EventResult:
        self.vehicle.mass_model.jettison("fairing")
        return EventResult(
            name=self.name,
            t_trigger=state.t,
            state_snapshot=state.snapshot(),
            message=f"Fairing jettisoned at {state.y/1e3:.1f} km",
        )


class SECOEvent(Event):
    name = "seco"
    terminal = True

    def __init__(self, vehicle: Vehicle, guidance: GuidanceBase | None = None) -> None:
        super().__init__()
        self.vehicle = vehicle
        self.guidance = guidance

    def _condition(self, state: SimState) -> bool:
        if self.vehicle.mass_model.current_stage_index != 1:
            return False
        propellant_exhausted = self.vehicle.mass_model.propellant_exhausted
        guidance_cutoff = self.guidance is not None and self.guidance.cutoff_commanded
        return propellant_exhausted or guidance_cutoff

    def _action(self, state: SimState) -> EventResult:
        cause = "guidance-commanded" if self.guidance and self.guidance.cutoff_commanded else "propellant exhausted"
        return EventResult(
            name=self.name,
            t_trigger=state.t,
            state_snapshot=state.snapshot(),
            message=f"SECO at t={state.t:.1f}s [{cause}] | vx={state.vx:.1f} m/s vy={state.vy:.1f} m/s",
        )


class PayloadSeparationEvent(Event):
    name = "payload_sep"

    def __init__(self, vehicle: Vehicle, sep_delay_s: float = 10.0) -> None:
        super().__init__()
        self.vehicle = vehicle
        self.sep_delay_s = sep_delay_s
        self.seco_time_s: float | None = None

    def notify_seco(self, t_seco: float) -> None:
        self.seco_time_s = t_seco

    def _condition(self, state: SimState) -> bool:
        if self.seco_time_s is None:
            return False
        return state.t >= self.seco_time_s + self.sep_delay_s

    def _action(self, state: SimState) -> EventResult:
        self.vehicle.mass_model.jettison("payload")
        return EventResult(
            name=self.name,
            t_trigger=state.t,
            state_snapshot=state.snapshot(),
            message=f"Payload separation at t={state.t:.1f}s, Alt={state.y/1e3:.1f} km",
        )
