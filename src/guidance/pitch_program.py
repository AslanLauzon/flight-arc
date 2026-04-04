import numpy as np
from src.guidance.base import GuidanceBase
from src.propagator.state import SimState


class PitchProgram(GuidanceBase):
    """
    Open-loop pitch schedule: interpolates pitch angle from a
    time-indexed table defined in mission.yaml.
    """

    def __init__(self, points: list[list[float]]) -> None:
        self.times = [p[0] for p in points]
        self.angles = [p[1] for p in points]

    def pitch_angle_deg(self, state: SimState) -> float:
        return float(np.interp(state.t, self.times, self.angles,
                               left=self.angles[0], right=self.angles[-1]))
