"""Does the regimen result survive human-constrained pharmacodynamics?

Two layers that the published analysis declared unmeasured can in fact be
constrained by data from the very volunteers whose arterial concentrations
calibrate layer 1 (Kamp 2020, the calibration set):

  bbb   Blood-brain transfer SPEED. Its ratios are fixed by Kp,uu = 0.6
        (Moaddel 2023); its magnitude sets the plasma-to-brain equilibration
        half-time, which the model leaves at 2.35 min. Olofsen et al.
        (Anesthesiology 2022;136:792-801, PMID 35188952) measured a lumped
        blood-effect-site half-life of 8.3 min (95% CI 5.1 to 13.0), so the
        model equilibrates about 3.5x too fast. Matching it scales all four
        transfer clearances by 0.283 and leaves Kp,uu unchanged.

  psych Psychotomimetic overlay. The model uses an illustrative clipped linear
        ramp over 0.02 to 0.10 mg/L brain. Olofsen fitted a sigmoid Emax to
        Bowdle external perception with C50 0.51 nmol/ml effect-site and a Hill
        coefficient of 5.33, i.e. C50 = 0.0727 mg/L brain at Kp,uu = 0.6.

The headline claim is that at a matched dose a bolus produces a far higher
brain peak, and more modelled injury, than a slow infusion. Slower brain
equilibration blunts a bolus peak more than an infusion peak, so this is a
genuine test of that claim rather than a cosmetic refit.

Four conditions are run: baseline (as published), each constraint alone, and
both together.

Run with:  PYTHONPATH=. python scripts/run_human_constrained.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from src.l1_pk.config import L1Parameters
from src.l4_l5.injury import ClinicalParams
from scripts.regenerate_all import (
    bolus_regimen, infusion_regimen, run_full_stack, window_from_curve,
)

BBB_FACTOR = 0.283          # -> t1/2 8.3 min, from scripts/calibrate_bbb_speed.py
BBB_FACTOR_CI = (0.1807, 0.4606)   # -> 13.0 and 5.1 min
OUT = Path("results/human_constrained.json")

DOSES = np.round(np.concatenate([np.arange(0.05, 1.0, 0.05), np.arange(1.0, 3.01, 0.25)]), 3)
REGIMENS = {
    "bolus": lambda: bolus_regimen(0.5),
    "infusion_40min": lambda: infusion_regimen(0.5, 40.0 / 60.0),
    "infusion_2h": lambda: infusion_regimen(0.5, 2.0),
    "infusion_4h": lambda: infusion_regimen(0.5, 4.0),
    "infusion_8h": lambda: infusion_regimen(0.5, 8.0),
}
KEYS = ("I_final", "C_brain_peak", "B_int_peak", "SD_burden", "T_NRHypo", "psych_max")


def params_for(bbb: float) -> L1Parameters:
    base = L1Parameters()
    return dataclasses.replace(
        base, clearances=dataclasses.replace(base.clearances, bbb_speed_factor=bbb))


def run_condition(bbb: float, psych: str) -> dict:
    p = params_for(bbb)
    clin = ClinicalParams(psych_model=psych)
    out: dict = {"bbb_speed_factor": bbb, "psych_model": psych}

    regs = {}
    for name, make in REGIMENS.items():
        r = run_full_stack(make(), params=p, clinical_params=clin)
        regs[name] = None if r is None else {k: float(r[k]) for k in KEYS}
    out["regimens"] = regs

    no_drug = run_full_stack(infusion_regimen(0.0, 1.0), params=p, clinical_params=clin)
    out["no_drug_I_final"] = None if no_drug is None else float(no_drug["I_final"])

    injury = []
    for d in DOSES:
        r = run_full_stack(infusion_regimen(float(d), 40.0 / 60.0), params=p, clinical_params=clin)
        injury.append(None if r is None else float(r["I_final"]))
    out["dose_response"] = {"doses": DOSES.tolist(), "I_final": injury}
    if all(v is not None for v in injury):
        arr = np.array(injury)
        w = dict(window_from_curve(DOSES, arr))
        w["u_shaped"] = bool(int(np.argmin(arr)) not in (0, len(arr) - 1))
        out["window"] = w

    b, i8 = regs.get("bolus"), regs.get("infusion_8h")
    i40 = regs.get("infusion_40min")
    if b and i8:
        out["bolus_vs_8h"] = {
            "peak_fold": b["C_brain_peak"] / i8["C_brain_peak"],
            "injury_reduction_pct": (1 - i8["I_final"] / b["I_final"]) * 100,
            "SD_burden_change_pct": (1 - i8["SD_burden"] / b["SD_burden"]) * 100,
        }
    if b and i40:
        out["bolus_worse_than_40min"] = bool(b["I_final"] > i40["I_final"])
    return out


def main() -> None:
    conditions = {
        "baseline": (1.0, "linear"),
        "slow_bbb": (BBB_FACTOR, "linear"),
        "human_psych": (1.0, "hill"),
        "human_constrained": (BBB_FACTOR, "hill"),
        "slow_bbb_ci_fast": (BBB_FACTOR_CI[1], "hill"),
        "slow_bbb_ci_slow": (BBB_FACTOR_CI[0], "hill"),
    }
    results = {}
    for name, (bbb, psych) in conditions.items():
        print(f"[{name}] bbb_speed_factor={bbb}, psych={psych}")
        results[name] = run_condition(bbb, psych)
        bv = results[name].get("bolus_vs_8h")
        w = results[name].get("window", {})
        if bv:
            print(f"    peak fold {bv['peak_fold']:.1f}x | injury -{bv['injury_reduction_pct']:.0f}%"
                  f" | bolus worse than 40 min: {results[name]['bolus_worse_than_40min']}")
        if w:
            print(f"    optimum {w['optimal_dose']} mg/kg, window "
                  f"{w['window_lower']} to {w['window_upper']}, U-shaped {w['u_shaped']}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
