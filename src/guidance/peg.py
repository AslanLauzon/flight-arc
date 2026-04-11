from __future__ import annotations

import math

from src.atmosphere.us_standard_1976 import pressure
from src.config import TargetOrbitConfig
from src.guidance.base import GuidanceBase
from src.propagator.state import SimState
from src.vehicle.vehicle import Vehicle

G0 = 9.80665
R_EARTH = 6_371_000.0
MU = 3.986004418e14


class PEG(GuidanceBase):
    def __init__(
        self,
        vehicle: Vehicle,
        target_orbit: TargetOrbitConfig,
        kick_time_s: float = 10.0,
        kick_angle_deg: float = 20.0,
        update_interval_s: float = 2.0,
        allow_two_burn: bool = False,
    ) -> None:
        self.vehicle = vehicle
        self.target_orbit = target_orbit
        self.update_interval_s = update_interval_s
        self.allow_two_burn = allow_two_burn

        self._kick_time_s = kick_time_s
        self._kick_pitch_deg = 90.0 - kick_angle_deg
        self._cutoff_tolerance_m = 3_000.0

        self.A = 0.0
        self.B = 0.0
        self.t_last_update: float | None = None
        self._last_stage_index = 0
        self._phase = "burn1"
        self._t_last_flags: float | None = None

        self.stage1_meco_commanded = False
        self.burn1_meco_commanded = False
        self.relight_commanded = False
        self.cutoff_commanded = False

    def pitch_angle_deg(self, state: SimState) -> float:
        # pitch_angle_deg is called 4 times per timestep by the RK4 substeps.
        # Guard flag checks (MECO, relight, cutoff) to run only once per timestep
        # so their side-effects (setting commanded flags) don't fire multiple times.
        do_checks = self._t_last_flags is None or state.t != self._t_last_flags
        if do_checks:
            self._t_last_flags = state.t

        stage_index = self.vehicle.mass_model.current_stage_index

        if stage_index == 0:
            if self.allow_two_burn and do_checks and not self.stage1_meco_commanded:
                self._check_stage1_early_meco(state)
            if state.t < self._kick_time_s:
                return 90.0
            return min(state.flight_path_angle_deg, self._kick_pitch_deg)

        if stage_index != self._last_stage_index:
            self._last_stage_index = stage_index
            self.t_last_update = None
            if self.allow_two_burn and self.burn1_meco_commanded:
                self._phase = "coast"

        if self.allow_two_burn and self._phase == "coast":
            if do_checks and not self.relight_commanded:
                self._check_relight(state)
            if self.relight_commanded:
                self._phase = "burn2"
            else:
                return state.flight_path_angle_deg

        if self.allow_two_burn and self._phase == "burn2":
            if do_checks and not self.cutoff_commanded:
                self._check_burn2_seco(state)
            return 0.0

        if do_checks and not self.cutoff_commanded:
            self._check_cutoff(state)

        if self.allow_two_burn and do_checks and not self.cutoff_commanded and not self.burn1_meco_commanded:
            self._check_burn1_meco(state)

        if self.allow_two_burn and self.burn1_meco_commanded:
            self._phase = "coast"
            state.engine_on = False
            return state.flight_path_angle_deg

        if self.t_last_update is None or (state.t - self.t_last_update) >= self.update_interval_s:
            self._update(state)
            self.t_last_update = state.t

        # PEG steering law: pitch = atan(A + B*tau), where tau is time elapsed
        # since the last guidance update. A is the current pitch bias and B is
        # the pitch rate, both computed from the velocity-to-go vector in _update().
        tau = state.t - (self.t_last_update or state.t)
        return math.degrees(math.atan(self.A + self.B * tau))

    def _check_burn1_meco(self, state: SimState) -> None:
        if state.vy <= 0:
            return

        r = R_EARTH + state.y
        v_sq = state.vx ** 2 + state.vy ** 2
        eps = v_sq / 2.0 - MU / r
        if eps >= 0.0:
            return

        h = r * state.vx
        a_orb = -MU / (2.0 * eps)
        e = math.sqrt(max(0.0, 1.0 + 2.0 * eps * h ** 2 / MU ** 2))
        alt_a = a_orb * (1.0 + e) - R_EARTH

        if abs(alt_a - self.target_orbit.apogee_km * 1e3) <= 5_000.0:
            self.burn1_meco_commanded = True

    def _check_relight(self, state: SimState) -> None:
        r = R_EARTH + state.y
        v_sq = state.vx ** 2 + state.vy ** 2
        eps = v_sq / 2.0 - MU / r
        if eps >= 0.0:
            return

        h = r * state.vx
        a_orb = -MU / (2.0 * eps)
        e = math.sqrt(max(0.0, 1.0 + 2.0 * eps * h ** 2 / MU ** 2))
        alt_a = a_orb * (1.0 + e) - R_EARTH
        target_alt = self.target_orbit.apogee_km * 1e3

        if abs(state.y - alt_a) <= 3_000.0 and abs(alt_a - target_alt) <= 50_000.0:
            self.relight_commanded = True

    def _check_burn2_seco(self, state: SimState) -> None:
        v_circ = math.sqrt(MU / (R_EARTH + state.y))
        if state.vx >= v_circ and abs(state.vy) < 20.0:
            self.cutoff_commanded = True

    def _check_cutoff(self, state: SimState) -> None:
        r = R_EARTH + state.y
        v_sq = state.vx ** 2 + state.vy ** 2
        eps = v_sq / 2.0 - MU / r
        if eps >= 0.0:
            return

        h = r * state.vx
        a = -MU / (2.0 * eps)
        e = math.sqrt(max(0.0, 1.0 + 2.0 * eps * h ** 2 / MU ** 2))

        alt_p = a * (1.0 - e) - R_EARTH
        alt_a = a * (1.0 + e) - R_EARTH
        target_p = self.target_orbit.perigee_km * 1e3
        target_a = self.target_orbit.apogee_km * 1e3
        target_span = abs(target_a - target_p)
        # Don't command cutoff if the vehicle hasn't climbed high enough yet —
        # avoids cutting off during the early ascent when orbit elements briefly
        # look correct but the vehicle is still on a suborbital arc.
        altitude_floor = target_p - min(4_000.0, 2_000.0 + 0.25 * target_span)

        if state.y < altitude_floor:
            return

        if (
            abs(alt_p - target_p) < self._cutoff_tolerance_m
            and abs(alt_a - target_a) < self._cutoff_tolerance_m
        ):
            self.cutoff_commanded = True

    def _update(self, state: SimState) -> None:
        # Recompute PEG A/B coefficients from current state and velocity-to-go.
        ambient_pressure = pressure(state.y)
        stage = self.vehicle.mass_model.current_stage
        thrust = self.vehicle.thrust(state.y)
        mass = self.vehicle.mass

        if mass <= 0 or thrust <= 0:
            return

        mdot = stage.mass_flow(ambient_pressure)
        if mdot <= 0:
            return

        tgo = self._total_tgo(ambient_pressure)
        if tgo <= 1.0:
            return

        r_now = R_EARTH + state.y
        r_target = R_EARTH + (self.target_orbit.perigee_km + self.target_orbit.apogee_km) / 2.0 * 1e3
        vx_target, vy_target = self._target_velocity(r_target)

        # Gravity loss correction: effective gravity at the midpoint radius,
        # reduced by centrifugal acceleration from current tangential velocity.
        r_mid = (r_now + r_target) / 2.0
        g_avg = G0 * (R_EARTH / r_mid) ** 2
        g_eff = max(0.0, g_avg - state.vx ** 2 / r_mid)

        vgo_x = vx_target - state.vx
        vgo_y = (vy_target - state.vy) + g_eff * tgo  # compensate for gravity over tgo
        if abs(vgo_x) < 1.0:
            return

        # A = initial pitch tangent; B = pitch rate so that pitch reaches the
        # target flight path angle exactly at cutoff (t_last_update + tgo).
        self.A = vgo_y / vgo_x
        tan_theta_f = vy_target / vx_target if abs(vx_target) > 1.0 else 0.0
        self.B = (tan_theta_f - self.A) / tgo

    def _total_tgo(self, ambient_pressure: float) -> float:
        mass_model = self.vehicle.mass_model
        current_mdot = mass_model.current_stage.mass_flow(ambient_pressure)
        tgo = mass_model.propellant_remaining_kg / current_mdot if current_mdot > 0 else 0.0
        for stage in mass_model.stages[mass_model.current_stage_index + 1:]:
            mdot_vac = stage.thrust_vac_N / (stage.isp_vac_s * G0)
            tgo += stage.propellant_mass_kg / mdot_vac
        return tgo

    def _target_velocity(self, r_target: float) -> tuple[float, float]:
        r_p = R_EARTH + self.target_orbit.perigee_km * 1e3
        r_a = R_EARTH + self.target_orbit.apogee_km * 1e3
        semi_major_axis = (r_p + r_a) / 2.0
        eccentricity = (r_a - r_p) / (r_a + r_p)
        p = semi_major_axis * (1.0 - eccentricity ** 2)

        r_target = max(r_p, min(r_target, r_a))
        v_target = math.sqrt(MU * (2.0 / r_target - 1.0 / semi_major_axis))
        h_orb = math.sqrt(MU * p)
        v_t = h_orb / r_target
        v_r = math.sqrt(max(0.0, v_target ** 2 - v_t ** 2))

        if r_target >= r_a:
            v_r = 0.0

        return v_t, v_r

    def _check_stage1_early_meco(self, state: SimState) -> None:
        if state.t < self._kick_time_s + 10.0 or state.vy <= 0:
            return

        mass_model = self.vehicle.mass_model
        if mass_model.current_stage_index != 0:
            return

        r = R_EARTH + state.y
        v_sq = state.vx ** 2 + state.vy ** 2
        eps = v_sq / 2.0 - MU / r
        if eps >= 0.0:
            return

        h = r * state.vx
        a_orb = -MU / (2.0 * eps)
        e = math.sqrt(max(0.0, 1.0 + 2.0 * eps * h ** 2 / MU ** 2))
        alt_a = a_orb * (1.0 + e) - R_EARTH
        target_alt = (self.target_orbit.perigee_km + self.target_orbit.apogee_km) / 2.0 * 1e3

        if abs(alt_a - target_alt) <= 5_000.0:
            self.stage1_meco_commanded = True
            self.burn1_meco_commanded = True
