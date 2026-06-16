"""Robustness of the qualitative conclusions to the uncalibrated injury parameters.

Addresses the central reviewer critique: the U-shaped window, its asymmetry, and
the regimen effect are produced by uncalibrated downstream coefficients
(alpha, beta, gamma, B_int_thresh, g_gain, kappa). Here we sample those six
coefficients over wide plausible ranges and report the fraction of
parameterizations that still yield (i) a U-shaped window with an interior
optimum, and (ii) a bolus that is worse than a dose-matched infusion.

Efficiency: layers 1 and 2 (pharmacokinetics and receptor occupancy) do not
depend on the injury coefficients, so the occupancy trajectories are precomputed
once per dose and reused across all parameter samples.

Run with:  PYTHONPATH=. python scripts/run_robustness.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.l1_pk import L1Model, DosingRegimen, DoseEvent, InfusionSegment
from src.l1_pk.config import default_parameters
from src.l2_occupancy import simulate_occupancy, DEFAULT_L2_PARAMS
from src.l3a_sd import simulate_sd_dynamics, DEFAULT_SD_PARAMS
from src.l3b_nrhypo import simulate_nrhypo, DEFAULT_NRHYPPO_PARAMS
from src.l4_l5.injury import simulate_injury, DEFAULT_INJURY_PARAMS

RESULTS = Path("results")
WEIGHT = 70.0
T_END = 24.0
N_POINTS = 200
DOSES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0])

# Wide plausible ranges for the uncalibrated downstream coefficients.
RANGES = {
    "alpha": (0.5, 2.0), "beta": (0.2, 1.5), "gamma": (10.0, 150.0),
    "B_int_thresh": (0.15, 0.45), "g_gain": (10.0, 150.0), "kappa": (2.0, 10.0),
}


def occupancy_for_regimen(regimen):
    l1 = L1Model(default_parameters())
    t = np.linspace(0, T_END, N_POINTS)
    r1 = l1.simulate(regimen, T_END, t_eval=t)
    r2 = simulate_occupancy(t, r1.brain_ecf("KET_S"), r1.brain_ecf("KET_R"),
                            DEFAULT_L2_PARAMS)
    return {"t": t, "B_pyr": r2["pyr"], "B_int": r2["int"]}


def injury_for(pre, sd_p, nr_p, inj_p):
    t, B_pyr, B_int = pre["t"], pre["B_pyr"], pre["B_int"]
    r3a = simulate_sd_dynamics(t, B_pyr, sd_p)
    r3b = simulate_nrhypo(t, B_int, nr_p)
    sd_rate = r3a["lambda_SD"] * r3a["D_SD"]
    glu = np.clip((r3b["Glu"] - nr_p.Glu_0) / nr_p.Glu_max, 0, 1)
    r4 = simulate_injury(t, B_pyr, B_int, sd_rate, glu, r3b["injury_rate"], inj_p)
    return float(r4["I_final"])


def main(n_samples=300, seed=0):
    print("Precomputing occupancy (layers 1-2) once per regimen...")
    # 40-min infusion at each dose, plus a 0.5 mg/kg bolus.
    pre_dose = {}
    for d in DOSES:
        reg = DosingRegimen(
            infusions=[InfusionSegment(0.0, 40.0 / 60.0, d * WEIGHT / (40.0 / 60.0))],
            s_fraction=0.5)
        pre_dose[float(d)] = occupancy_for_regimen(reg)
    pre_bolus = occupancy_for_regimen(
        DosingRegimen(boluses=[DoseEvent(0.0, 0.5 * WEIGHT)], s_fraction=0.5))
    i_half = int(np.where(np.isclose(DOSES, 0.5))[0][0])

    rng = np.random.default_rng(seed)
    samples = {k: rng.uniform(lo, hi, n_samples) for k, (lo, hi) in RANGES.items()}

    n_ushape = 0
    n_bolus_worse = 0
    optima = []
    print(f"Sampling {n_samples} injury-parameter sets...")
    for i in range(n_samples):
        sd_p = replace(DEFAULT_SD_PARAMS, kappa=float(samples["kappa"][i]))
        nr_p = replace(DEFAULT_NRHYPPO_PARAMS,
                       B_int_thresh=float(samples["B_int_thresh"][i]),
                       g_gain=float(samples["g_gain"][i]))
        inj_p = replace(DEFAULT_INJURY_PARAMS,
                        alpha=float(samples["alpha"][i]),
                        beta=float(samples["beta"][i]),
                        gamma=float(samples["gamma"][i]))
        inj = np.array([injury_for(pre_dose[float(d)], sd_p, nr_p, inj_p) for d in DOSES])
        imin = int(np.argmin(inj))
        # U-shape: interior optimum with a clearly rising toxic limb.
        ushape = (0 < imin < len(DOSES) - 1) and (inj[-1] > inj[imin] * 1.05) \
            and (inj[0] >= inj[imin])
        if ushape:
            n_ushape += 1
            optima.append(float(DOSES[imin]))
        # Regimen effect: bolus vs matched 40-min infusion at 0.5 mg/kg.
        inj_bolus = injury_for(pre_bolus, sd_p, nr_p, inj_p)
        inj_inf = inj[i_half]
        if inj_bolus > inj_inf:
            n_bolus_worse += 1

    out = {
        "n_samples": n_samples,
        "ranges": RANGES,
        "frac_ushape": n_ushape / n_samples,
        "frac_bolus_worse_than_infusion": n_bolus_worse / n_samples,
        "optima_when_ushape": optima,
        "median_optimum": float(np.median(optima)) if optima else None,
        "optimum_iqr": [float(np.percentile(optima, 25)),
                        float(np.percentile(optima, 75))] if optima else None,
    }
    (RESULTS / "robustness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  U-shaped window in {out['frac_ushape']*100:.0f}% of samples")
    print(f"  bolus worse than infusion in {out['frac_bolus_worse_than_infusion']*100:.0f}%")
    if optima:
        print(f"  median optimum {out['median_optimum']:.2f} mg/kg "
              f"(IQR {out['optimum_iqr'][0]:.2f}-{out['optimum_iqr'][1]:.2f})")
    print(f"  saved {RESULTS / 'robustness.json'}")


if __name__ == "__main__":
    main(300)
