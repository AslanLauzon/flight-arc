from src.guidance.base import GuidanceBase
from src.propagator.state import SimState


class GravityTurn(GuidanceBase):
    """
    Closed-loop guidance: after a small pitch kick, the vehicle
    follows its own velocity vector (flight path angle).
    """

    def __init__(self, kick_time_s: float, kick_angle_deg: float) -> None:
        self.kick_time_s = kick_time_s
        self.kick_angle_deg = kick_angle_deg

    def pitch_angle_deg(self, state: SimState) -> float:
        if state.t < self.kick_time_s:
            return 90.0
        return state.flight_path_angle_deg
