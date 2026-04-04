"""
sweep_orbits.py — Test PEG two-burn insertion across 160–250 km circular orbits.
"""

from __future__ import annotations

import copy

from src.config import load_config
from src.mission_runner import run_nominal_mission

TARGETS_KM = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
TOL_KM = 15.0  # insertion tolerance


def main() -> None:
    base_cfg = load_config()

    print(f"{'Alt':>6}  {'Mode':<12}  {'Ap(km)':>8}  {'Pe(km)':>8}  {'dAp':>7}  {'dPe':>7}  {'Pass'}")
    print("-" * 70)

    for alt_km in TARGETS_KM:
        cfg = copy.deepcopy(base_cfg)
        cfg.mission.target_orbit.apogee_km  = alt_km
        cfg.mission.target_orbit.perigee_km = alt_km

        try:
            result = run_nominal_mission(cfg=cfg)
            el = result.elements

            if el is None:
                print(f"{alt_km:>6}  {'--':<12}  {'N/A':>8}  {'N/A':>8}  {'--':>7}  {'--':>7}  FAIL (no orbit)")
                continue

            ap_km  = el.apogee_alt_km
            pe_km  = el.perigee_alt_km
            d_ap   = ap_km  - alt_km
            d_pe   = pe_km  - alt_km
            passed = abs(d_ap) <= TOL_KM and abs(d_pe) <= TOL_KM

            # Determine which phase sequence fired
            event_names = [e.result.name for e in result.events if e.result]
            if "burn1_meco" in event_names and "relight" in event_names:
                mode = "2-burn"
            elif "burn1_meco" not in event_names:
                mode = "1-burn"
            else:
                mode = "?"

            status = "PASS" if passed else "FAIL"
            print(
                f"{alt_km:>6}  {mode:<12}  {ap_km:>8.1f}  {pe_km:>8.1f}  "
                f"{d_ap:>+7.1f}  {d_pe:>+7.1f}  {status}"
            )

        except Exception as exc:
            print(f"{alt_km:>6}  {'ERROR':<12}  {'':>8}  {'':>8}  {'':>7}  {'':>7}  {exc}")


if __name__ == "__main__":
    main()
