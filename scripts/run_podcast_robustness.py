"""Is the PODCAST conclusion robust to the enantiomer-potency assumption?

The published analysis makes two separate claims about the trial:

  A. Dose claim. Both trial doses (0.5 and 1.0 mg/kg) lie above the upper bound
     of the modelled window, so both were overdoses.

  B. Regimen claim. Both trial arms were boluses, and a bolus produces more
     modelled injury than the same dose infused slowly, so the delivery, not
     the dose alone, explains the null result.

Claim A depends on where the window's upper bound falls, which scripts/
run_enantiomer_ratio.py shows moves with the assumed S:R potency ratio: the
bound is 0.45 mg/kg at the ratio taken from in vitro binding (Temme 2018) but
0.55 to 0.85 mg/kg at the human functional estimates from the trial that
supplies the calibration set (Dahan 2024, Olofsen 2022).

This script evaluates both claims across that range, under human-constrained
brain kinetics and the human-calibrated psychotomimetic overlay, by running the
actual trial arms (boluses) against a dose-matched slow infusion and the
no-drug baseline.

Run with:  PYTHONPATH=. python scripts/run_podcast_robustness.py
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from src.l1_pk.config import L1Parameters
from src.l4_l5.injury import ClinicalParams
from scripts.regenerate_all import bolus_regimen, infusion_regimen, run_full_stack
from scripts.run_enantiomer_ratio import RATIOS, l2_with_ratio

BBB_FACTOR = 0.283
OUT = Path("results/podcast_robustness.json")
TRIAL_DOSES = (0.5, 1.0)


def main() -> None:
    base = L1Parameters()
    l1p = dataclasses.replace(
        base, clearances=dataclasses.replace(base.clearances, bbb_speed_factor=BBB_FACTOR))
    clin = ClinicalParams(psych_model="hill")

    from src.l2_occupancy.model import DEFAULT_L2_PARAMS
    nm = DEFAULT_L2_PARAMS.nmdar
    ratios = dict(RATIOS)
    ratios["model_default_temme"] = round(nm.k_on_S["int"] / nm.k_on_R["int"], 4)

    windows = {}
    ep = Path("results/enantiomer_ratio.json")
    if ep.exists():
        for k, v in json.loads(ep.read_text(encoding="utf-8"))["conditions"].items():
            windows[k] = v["enantiomer"]["racemic"]["window_upper"]

    results = {"bbb_speed_factor": BBB_FACTOR, "psych_model": "hill", "conditions": {}}
    print(f"{'condition':26} {'dose':>5} {'bolus I':>8} {'infus I':>8} {'baseline':>9} "
          f"{'bolus>base':>11} {'bolus>infus':>12} {'above window':>13}")
    for name, ratio in ratios.items():
        l2 = l2_with_ratio(ratio)
        nodrug = run_full_stack(infusion_regimen(0.0, 1.0), params=l1p,
                                clinical_params=clin, l2_params=l2)
        base_I = float(nodrug["I_final"])
        arms = {}
        for d in TRIAL_DOSES:
            b = run_full_stack(bolus_regimen(d), params=l1p, clinical_params=clin, l2_params=l2)
            inf = run_full_stack(infusion_regimen(d, 8.0), params=l1p,
                                 clinical_params=clin, l2_params=l2)
            w_up = windows.get(name)
            arms[str(d)] = {
                "bolus_I": float(b["I_final"]),
                "infusion_8h_I": float(inf["I_final"]),
                "no_drug_I": base_I,
                "bolus_pct_above_baseline": (b["I_final"] / base_I - 1) * 100,
                "infusion_pct_above_baseline": (inf["I_final"] / base_I - 1) * 100,
                "bolus_net_harmful": bool(b["I_final"] > base_I),
                "infusion_net_harmful": bool(inf["I_final"] > base_I),
                "bolus_worse_than_infusion": bool(b["I_final"] > inf["I_final"]),
                "dose_above_window_upper": None if w_up is None else bool(d > w_up),
                "window_upper": w_up,
            }
            a = arms[str(d)]
            print(f"{name:26} {d:>5.1f} {a['bolus_I']:>8.1f} {a['infusion_8h_I']:>8.1f} "
                  f"{base_I:>9.1f} {str(a['bolus_net_harmful']):>11} "
                  f"{str(a['bolus_worse_than_infusion']):>12} "
                  f"{str(a['dose_above_window_upper']):>13}")
        results["conditions"][name] = {"s_to_r_ratio": ratio, "arms": arms}

    # Summarise which claim survives across every condition tested.
    claim_a, claim_b = [], []
    for c in results["conditions"].values():
        for d, a in c["arms"].items():
            claim_a.append(a["dose_above_window_upper"])
            claim_b.append(a["bolus_worse_than_infusion"] and a["bolus_net_harmful"])
    results["claim_A_dose_above_window_always"] = all(x is True for x in claim_a)
    results["claim_A_holds_fraction"] = sum(1 for x in claim_a if x) / len(claim_a)
    results["claim_B_bolus_harmful_and_worse_always"] = all(claim_b)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nclaim A (dose above window) holds in "
          f"{results['claim_A_holds_fraction']:.0%} of arm-condition pairs")
    print(f"claim B (bolus net harmful and worse than slow infusion) holds always: "
          f"{results['claim_B_bolus_harmful_and_worse_always']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
