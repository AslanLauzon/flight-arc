"""
event.py — Event protocol and base class.

Every event in the auto-sequence implements this interface.
The propagator calls check() each step; when it returns True, trigger() fires.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.propagator.state import SimState


@dataclass
class EventResult:
    name: str
    t_trigger: float
    state_snapshot: dict
    message: str = ""
    constraint_violations: list[str] = field(default_factory=list)


class Event(ABC):
    """Abstract base for all mission events."""

    name: str = "base_event"
    terminal: bool = False        # if True, propagation halts after trigger
    one_shot: bool = True         # if True, fires at most once

    def __init__(self) -> None:
        self.triggered: bool = False
        self.result: EventResult | None = None

    def check(self, state: SimState) -> bool:
        """Return True when the event condition is met."""
        if self.one_shot and self.triggered:
            return False
        return self._condition(state)

    def trigger(self, state: SimState) -> EventResult:
        """Execute event side-effects and return a result record."""
        self.triggered = True
        result = self._action(state)
        self.result = result
        state.events_triggered.append(self.name)
        return result

    @abstractmethod
    def _condition(self, state: SimState) -> bool: ...

    @abstractmethod
    def _action(self, state: SimState) -> EventResult: ...
