"""
autosequence.py — Builds the mission event list from config.

The autosequence is the single place that knows which events exist,
in what order they fire, and how they're wired together.
The integrator just iterates the list — it knows nothing about mission specifics.
"""

from src.config import MissionToolkitConfig
from src.events.event import Event
from src.events.standard_events import (
    Burn1MECOEvent,
    FairingDeployEvent,
    IgnitionEvent,
    LiftoffEvent,
    MaxQEvent,
    PayloadSeparationEvent,
    RelightEvent,
    SECOEvent,
    StagingEvent,
)
from src.guidance.base import GuidanceBase
from src.vehicle.vehicle import Vehicle


def build_autosequence(
    cfg: MissionToolkitConfig,
    vehicle: Vehicle,
    guidance: GuidanceBase | None = None,
) -> list[Event]:
    """
    Construct and return the ordered event list for a nominal ascent.

    Events are checked every integrator step in the order returned.
    Inter-event wiring (e.g. SECO notifying PayloadSep) is done here.

    Pass guidance to enable guidance-commanded engine cutoff (PEG SECO).
    """
    seco = SECOEvent(vehicle, guidance=guidance)
    payload_sep = PayloadSeparationEvent(vehicle, sep_delay_s=10.0)

    # wire SECO → PayloadSep so separation knows when to fire
    def _on_seco_trigger(original_action):
        def wrapped(state):
            result = original_action(state)
            payload_sep.notify_seco(state.t)
            return result
        return wrapped

    seco._action = _on_seco_trigger(seco._action)

    return [
        IgnitionEvent(),
        LiftoffEvent(),
        MaxQEvent(),
        StagingEvent(vehicle, guidance=guidance),
        FairingDeployEvent(vehicle, cfg.simulation.constraints.fairing_deploy_altitude_m),
        Burn1MECOEvent(guidance=guidance),
        RelightEvent(guidance=guidance),
        seco,
        payload_sep,
    ]
