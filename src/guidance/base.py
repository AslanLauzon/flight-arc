from abc import ABC, abstractmethod
from src.propagator.state import SimState


class GuidanceBase(ABC):
    """All guidance modes implement this interface."""

    #: Guidance-commanded engine cutoffs and relight.
    #: Events poll these flags to fire non-terminal MECO, relight, and final SECO.
    cutoff_commanded: bool = False        # final SECO (terminal)
    stage1_meco_commanded: bool = False   # early stage-1 cutoff (staging fires early)
    burn1_meco_commanded: bool = False    # mid-flight MECO before coast arc
    relight_commanded: bool = False       # stage-2 relight after coast

    @abstractmethod
    def pitch_angle_deg(self, state: SimState) -> float:
        """
        Return pitch angle [deg] at current state.
        0° = horizontal, 90° = vertical.
        """
        ...
