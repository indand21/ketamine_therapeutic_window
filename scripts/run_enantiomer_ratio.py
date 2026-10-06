"""How much does the assumed S:R potency ratio drive the enantiomer result?

The occupancy layer takes its S:R ratio from PCP-site binding constants
(Temme 2018: Ki 419 nM for S, 497 nM for R), giving k_on_S/k_on_R = 1.33 in
pyramidal and 1.36 in interneuron populations. Because k_off is shared between
enantiomers, the modelled potency ratio equals the k_on ratio exactly.

Human functional estimates from the trial that supplies the calibration set are
larger, and disagree with each other:

  Olofsen et al., Anesthesiology 2022;136:792-801 (PMID 35188952) found that
  R-ketamine did not contribute to Bowdle external perception or to the pain
  pressure threshold at all, i.e. an effectively unbounded S:R ratio.

  Dahan et al., ACS Pharmacol Transl Sci 2024;7:2044-2053 (PMID 39022368), a
  later analysis of the same trial including the nitroprusside arms, estimated
  R-ketamine potency at 0.52 of S-ketamine (95% CI 0.18 to 1.01), i.e. S:R
  = 1.9 (1.0 to 5.6). The interval includes equipotency.

This script re-runs the enantiomer comparison, the racemic dose response and
the regimen comparison across that range, to separate conclusions that depend
on the assumed ratio from those that do not.

Run with:  PYTHONPATH=. python scripts/run_enantiomer_ratio.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from src.l1_pk.config import L1Parameters
from src.l2_occupancy.model import DEFAULT_L2_PARAMS, NMDARParams
from src.l4_l5.injury import ClinicalParams
from scripts.regenerate_all import (
    bolus_regimen, infusion_regimen, run_full_stack, window_from_curve,
)

BBB_FACTOR = 0.283          # human-constrained brain kinetics (Olofsen t1/2ke0)
DOSES = np.round(np.concatenate([np.arange(0.05, 1.0, 0.05), np.arange(1.0, 3.01, 0.25)]), 3)
OUT = Path("results/enantiomer_ratio.json")

# S:R potency ratios to test. None means R-ketamine is inert (k_on_R = 0).
RATIOS = {
    "model_default_temme": None,   # filled in below from the shipped parameters
    "dahan2024_point_1.9": 1.923,
    "dahan2024_ci_equipotent": 0.990,
    "dahan2024_ci_weak_R": 5.556,
    "olofsen2022_R_inert": "inert",
}


def l2_with_ratio(ratio) -> object:
    """Return L2 params with k_on_R set from k_on_S and the requested S:R ratio."""
    nm = DEFAULT_L2_PARAMS.nmdar
    if ratio == "inert":
        k_on_R = {p: 0.0 for p in nm.k_on_R}
    else:
        k_on_R = {p: nm.k_on_S[p] / float(ratio) for p in nm.k_on_R}
    return dataclasses.replace(
        DEFAULT_L2_PARAMS, nmdar=dataclasses.replace(nm, k_on_R=k_on_R))


def params_slow_bbb() -> L1Parameters:
    base = L1Parameters()
    return dataclasses.replace(
        base, clearances=dataclasses.replace(base.clearances, bbb_speed_factor=BBB_FACTOR))


def run_one(ratio, l1p, clin) -> dict:
    l2 = l2_with_ratio(ratio)
    out: dict = {}

    # Enantiomer comparison: racemic vs S-only vs R-only
    ena = {}
    for name, sf in (("racemic", 0.5), ("s_only", 1.0), ("r_only", 0.0)):
        inj = []
        for d in DOSES:
            r = run_full_stack(infusion_regimen(float(d), 40.0 / 60.0, s_fraction=sf),
                               params=l1p, clinical_params=clin, l2_params=l2)
            inj.append(None if r is None else float(r["I_final"]))
        if all(v is not None for v in inj):
            arr = np.array(inj)
            w = dict(window_from_curve(DOSES, arr))
            w["u_shaped"] = bool(int(np.argmin(arr)) not in (0, len(arr) - 1))
            w["I_final"] = inj
            ena[name] = w
    out["enantiomer"] = ena

    # Regimen comparison at a matched racemic dose
    regs = {}
    for name, make in (("bolus", lambda: bolus_regimen(0.5)),
                       ("infusion_40min", lambda: infusion_regimen(0.5, 40.0 / 60.0)),
                       ("infusion_8h", lambda: infusion_regimen(0.5, 8.0))):
        r = run_full_stack(make(), params=l1p, clinical_params=clin, l2_params=l2)
        regs[name] = None if r is None else {
            k: float(r[k]) for k in ("I_final", "C_brain_peak", "B_int_peak", "SD_burden")}
    out["regimens"] = regs
    b, i40, i8 = regs["bolus"], regs["infusion_40min"], regs["infusion_8h"]
    if b and i40 and i8:
        out["bolus_worse_than_40min"] = bool(b["I_final"] > i40["I_final"])
        out["bolus_vs_8h_peak_fold"] = b["C_brain_peak"] / i8["C_brain_peak"]
        out["bolus_vs_8h_injury_reduction_pct"] = (1 - i8["I_final"] / b["I_final"]) * 100
    return out


def main() -> None:
    nm = DEFAULT_L2_PARAMS.nmdar
    RATIOS["model_default_temme"] = round(nm.k_on_S["int"] / nm.k_on_R["int"], 4)
    print("model S:R ratio  pyr %.3f  int %.3f"
          % (nm.k_on_S["pyr"] / nm.k_on_R["pyr"], nm.k_on_S["int"] / nm.k_on_R["int"]))

    l1p, clin = params_slow_bbb(), ClinicalParams(psych_model="hill")
    results = {"bbb_speed_factor": BBB_FACTOR, "psych_model": "hill", "conditions": {}}
    for name, ratio in RATIOS.items():
        print(f"\n[{name}] S:R = {ratio}")
        res = run_one(ratio, l1p, clin)
        results["conditions"][name] = {"s_to_r_ratio": ratio, **res}
        for k, v in res["enantiomer"].items():
            print(f"    {k:8} optimum {v['optimal_dose']:.2f} mg/kg, "
                  f"min injury {v['optimal_injury']:.1f}, U-shaped {v['u_shaped']}")
        print(f"    bolus worse than 40 min: {res.get('bolus_worse_than_40min')}  "
              f"| peak {res.get('bolus_vs_8h_peak_fold', float('nan')):.1f}x")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
