"""Design of the prospective study that would calibrate the model.

The identifiability analysis (scripts/run_identifiability.py) shows that the
injury layer has four shape-determining quantities: the spreading depolarisation
threshold gain kappa, the interneuron toxic threshold theta, the protective to
excitotoxic ratio beta/alpha, and the toxic weight Gamma/alpha. This script
works out what a prospective study would have to measure to pin each of them
down, and quantifies how well competing designs separate them.

Three things are computed.

1. Exposure separation. For each candidate design arm, the model's predicted
   peak brain concentration, peak occupancy in each neuronal population, and
   the time spent above the interneuron toxic threshold. A design can only
   identify the threshold if some arms sit below it and others above it, and it
   can only identify the toxic weight if some arms generate appreciable
   supra-threshold time.

2. Design comparison. The PODCAST design (placebo plus two boluses) is compared
   with a design that adds a dose-matched slow infusion, on the spread of
   supra-threshold exposure the two designs generate. The wider that spread,
   the more precisely the toxic arm can be estimated from the same number of
   patients.

3. Sample size. The primary clinical contrast the model predicts is between a
   bolus and a dose-matched slow infusion in the psychotomimetic readout. Since
   the mapping from the model's dimensionless burden to a clinical scale is
   exactly what the study would establish, the sample size is reported as a
   curve against the standardised effect size rather than as a single number
   resting on an assumed mapping.

Run with:  PYTHONPATH=. python scripts/run_prospective_design.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from src.l1_pk import L1Model, DosingRegimen, DoseEvent, InfusionSegment
from src.l1_pk.config import default_parameters
from src.l2_occupancy import simulate_occupancy, DEFAULT_L2_PARAMS
from src.l3b_nrhypo import DEFAULT_NRHYPPO_PARAMS
from src.l4_l5.injury import compute_clinical_constraints
from src.numeric_compat import trapezoid

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

WEIGHT = 70.0
T_END = 24.0
N_POINTS = 400
THETA = DEFAULT_NRHYPPO_PARAMS.B_int_thresh

# Candidate study arms. The first three reproduce PODCAST; the remainder are
# the additions the model implies would be informative.
ARMS = [
    ("Placebo", None),
    ("Bolus 0.5 mg/kg", ("bolus", 0.5)),
    ("Bolus 1.0 mg/kg", ("bolus", 1.0)),
    ("Infusion 0.5 mg/kg over 40 min", ("infusion", 0.5, 40.0 / 60.0)),
    ("Infusion 0.5 mg/kg over 4 h", ("infusion", 0.5, 4.0)),
    ("Infusion 0.5 mg/kg over 8 h", ("infusion", 0.5, 8.0)),
    ("Infusion 0.25 mg/kg over 4 h", ("infusion", 0.25, 4.0)),
    ("Infusion 1.0 mg/kg over 4 h", ("infusion", 1.0, 4.0)),
]


def regimen_for(spec):
    if spec is None:
        return DosingRegimen(infusions=[InfusionSegment(0.0, 0.1, 0.0)],
                             s_fraction=0.5)
    if spec[0] == "bolus":
        return DosingRegimen(boluses=[DoseEvent(0.0, spec[1] * WEIGHT)],
                             s_fraction=0.5)
    _, dose, dur = spec
    return DosingRegimen(
        infusions=[InfusionSegment(0.0, dur, dose * WEIGHT / dur)],
        s_fraction=0.5)


def arm_exposure(spec):
    t = np.linspace(0.0, T_END, N_POINTS)
    r1 = L1Model(default_parameters()).simulate(regimen_for(spec), T_END,
                                                t_eval=t)
    if not r1.success:
        raise RuntimeError("L1 integration failed")
    c_s, c_r = r1.brain_ecf("KET_S"), r1.brain_ecf("KET_R")
    c_brain = c_s + c_r
    r2 = simulate_occupancy(t, c_s, c_r, DEFAULT_L2_PARAMS)
    b_int = r2["int"]
    excess = np.maximum(0.0, b_int - THETA)
    r5 = compute_clinical_constraints(c_brain)
    return {
        "peak_brain_conc_mgL": float(np.max(c_brain)),
        "peak_B_pyr": float(np.max(r2["pyr"])),
        "peak_B_int": float(np.max(b_int)),
        "crosses_toxic_threshold": bool(np.max(b_int) > THETA),
        "hours_above_threshold": float(trapezoid((b_int > THETA).astype(float), t)),
        "supra_threshold_exposure": float(trapezoid(excess ** 2, t)),
        "peak_psychotomimetic_burden": float(np.max(r5["psych_burden"])),
    }


def spread(values):
    """Log-scale spread of the informative (non-zero) exposures."""
    v = np.array([x for x in values if x > 0], dtype=float)
    if v.size < 2:
        return 0.0
    return float(np.log10(v.max() / v.min()))


def sample_size_per_arm(effect_d, power=0.90, alpha=0.05):
    """Two-sample comparison of means, two-sided."""
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    return float(np.ceil(2 * ((z_a + z_b) / effect_d) ** 2))


def main():
    print("Exposure profile of candidate study arms")
    print(f"  interneuron toxic threshold theta = {THETA}")
    print(f"{'arm':34s} {'Cmax,brain':>11s} {'B_int':>7s} {'>thr':>6s} "
          f"{'h>thr':>7s} {'suprathr':>9s}")
    arms = []
    for label, spec in ARMS:
        e = arm_exposure(spec)
        e["arm"] = label
        arms.append(e)
        print(f"{label:34s} {e['peak_brain_conc_mgL']:11.4f} "
              f"{e['peak_B_int']:7.3f} "
              f"{str(e['crosses_toxic_threshold']):>6s} "
              f"{e['hours_above_threshold']:7.2f} "
              f"{e['supra_threshold_exposure']:9.4f}")

    by_label = {a["arm"]: a for a in arms}
    podcast = ["Placebo", "Bolus 0.5 mg/kg", "Bolus 1.0 mg/kg"]
    proposed = ["Placebo", "Bolus 0.5 mg/kg", "Infusion 0.5 mg/kg over 4 h",
                "Infusion 1.0 mg/kg over 4 h", "Bolus 1.0 mg/kg"]

    designs = {}
    for name, labels in (("PODCAST-like", podcast), ("Proposed", proposed)):
        supra = [by_label[l]["supra_threshold_exposure"] for l in labels]
        peaks = [by_label[l]["peak_brain_conc_mgL"] for l in labels]
        n_cross = sum(by_label[l]["crosses_toxic_threshold"] for l in labels)
        # A placebo arm carries no exposure information, so what matters for
        # locating the threshold is the number of drug-treated arms below it.
        active = [l for l in labels if by_label[l]["peak_brain_conc_mgL"] > 0]
        n_active_below = sum(
            not by_label[l]["crosses_toxic_threshold"] for l in active)
        designs[name] = {
            "arms": labels,
            "n_arms": len(labels),
            "n_arms_crossing_threshold": int(n_cross),
            "n_active_arms_below_threshold": int(n_active_below),
            "log10_spread_peak_concentration": spread(peaks),
            "log10_spread_supra_threshold_exposure": spread(supra),
            "brackets_threshold_with_active_arms": bool(
                n_cross > 0 and n_active_below > 0),
        }
        print(f"\n{name} design ({len(labels)} arms):")
        print(f"  arms crossing the toxic threshold: {n_cross}/{len(labels)}")
        print(f"  drug-treated arms below the threshold: {n_active_below}")
        print(f"  spread in peak brain concentration: "
              f"{designs[name]['log10_spread_peak_concentration']:.2f} log10 units")
        print(f"  spread in supra-threshold exposure: "
              f"{designs[name]['log10_spread_supra_threshold_exposure']:.2f} "
              f"log10 units")

    # What each estimable quantity needs.
    requirements = [
        {"quantity": "pharmacokinetic disposition",
         "measurement": "plasma S- and R-ketamine and norketamine "
                        "concentrations, sparse sampling",
         "design_requirement": "at least two regimens differing in infusion "
                               "duration, so distribution and elimination are "
                               "separately informed",
         "identified_by_this_design": True},
        {"quantity": "spreading depolarisation threshold gain, kappa",
         "measurement": "electrocorticographic spreading depolarisation "
                        "burden during and after drug exposure",
         "design_requirement": "patients with invasive neuromonitoring "
                               "(subarachnoid haemorrhage or traumatic brain "
                               "injury), crossover on and off drug",
         "identified_by_this_design": False,
         "note": "requires the neuromonitoring sub-study; not obtainable in "
                 "elective surgery"},
        {"quantity": "interneuron toxic threshold, theta",
         "measurement": "psychotomimetic score (for example the Clinician "
                        "Administered Dissociative States Scale) time course",
         "design_requirement": "arms bracketing the threshold, that is some "
                               "arms with and some without supra-threshold "
                               "occupancy",
         "identified_by_this_design":
             designs["Proposed"]["brackets_threshold_with_active_arms"]},
        {"quantity": "toxic weight, Gamma/alpha",
         "measurement": "delirium incidence and severity, plus the "
                        "psychotomimetic score",
         "design_requirement": "a wide spread of supra-threshold exposure "
                               "across arms",
         "identified_by_this_design": True},
        {"quantity": "protective to excitotoxic ratio, beta/alpha",
         "measurement": "the outcome endpoint together with the spreading "
                        "depolarisation burden",
         "design_requirement": "both arms of the balance measured in the same "
                               "patients",
         "identified_by_this_design": False,
         "note": "requires the neuromonitoring sub-study"},
    ]

    # Sample size against standardised effect size.
    ds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
    power_curve = [{"standardised_effect_size": d,
                    "n_per_arm_90_power": sample_size_per_arm(d, 0.90),
                    "n_per_arm_80_power": sample_size_per_arm(d, 0.80)}
                   for d in ds]
    print("\nSample size for the bolus versus dose-matched infusion contrast")
    print(f"  {'effect size d':>14s} {'n/arm (90%)':>12s} {'n/arm (80%)':>12s}")
    for row in power_curve:
        print(f"  {row['standardised_effect_size']:14.1f} "
              f"{row['n_per_arm_90_power']:12.0f} "
              f"{row['n_per_arm_80_power']:12.0f}")

    out = {
        "toxic_threshold": THETA,
        "arms": arms,
        "designs": designs,
        "identifiability_requirements": requirements,
        "sample_size": {
            "contrast": "bolus versus dose-matched slow infusion, "
                        "psychotomimetic readout",
            "test": "two-sample comparison of means, two-sided, alpha 0.05",
            "curve": power_curve,
            "note": ("Reported against the standardised effect size because "
                     "the mapping from the model's dimensionless burden to a "
                     "clinical scale is precisely what the study would "
                     "establish."),
        },
    }
    (RESULTS / "prospective_design.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  saved {RESULTS / 'prospective_design.json'}")


if __name__ == "__main__":
    main()
