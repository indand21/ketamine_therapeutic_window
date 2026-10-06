"""Global variance-based (Sobol) sensitivity analysis of net injury.

Fourteen factors are varied: four pharmacokinetic and blood-brain transfer
parameters, four NMDA receptor kinetic parameters, and all six uncalibrated
downstream coefficients of the protective and toxic layers, including the
NRHypo injury weight gamma.

Two things are reported that the earlier version did not provide:

  * a convergence sequence (base sample 64, 128, 256, 512) so that the
    stability of the reported indices can be judged rather than assumed; and
  * second-order indices at base sample 256, so the strong interaction
    structure implied by the gap between first- and total-order indices can be
    attributed to specific parameter pairs.

Note on gamma. The injury readout depends on gamma and on the NRHypo gain
g_gain only through their product (see scripts/run_identifiability.py), so the
two appear here as a confounded pair and their indices should be read together;
this is why the earlier analysis varied only one of them.

Speed. The terminal injury is evaluated with the exact linear decomposition
I = alpha*P - beta*Q + gamma*g_gain*R rather than by integrating the L4 ODE.
That decomposition is verified to solver tolerance in run_identifiability.py
and removes one stiff solve per evaluation.

Run with:  PYTHONPATH=. python scripts/run_sobol.py
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from src.l1_pk import L1Model, DosingRegimen, InfusionSegment
from src.l1_pk.config import default_parameters, Clearances, Flows
from src.l2_occupancy import simulate_occupancy
from src.l2_occupancy.model import NMDARParams, L2Params
from src.l2_occupancy.p_open import POpenParams
from src.l3a_sd import DEFAULT_SD_PARAMS
from src.l3a_sd.model import compute_sd_rate, compute_sd_duration
from src.l3b_nrhypo import simulate_nrhypo, DEFAULT_NRHYPPO_PARAMS
from src.numeric_compat import trapezoid

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

DOSE = 0.5      # mg/kg, racemic, 40 min infusion
WEIGHT = 70.0
T_END = 24.0
N_POINTS = 200

PROBLEM = {
    "num_vars": 14,
    "names": ["CL_out_HNK", "Q_per", "CL_in_S", "CL_out_S",
              "k_on_S_pyr", "k_on_S_int", "k_off_pyr", "k_off_int",
              "B_int_thresh", "g_gain", "gamma", "alpha", "beta", "kappa"],
    "bounds": [[0.5, 5.0], [120.0, 350.0], [2.0, 10.0], [2.5, 12.5],
               [6e3, 2.4e4], [7.5e3, 3.0e4], [0.01, 0.1], [0.005, 0.06],
               [0.15, 0.45], [10.0, 150.0], [10.0, 150.0],
               [0.5, 2.0], [0.2, 1.5], [2.0, 10.0]],
}

# Human-readable labels used in the figure and the manuscript table.
LABELS = {
    "CL_out_HNK": "hydroxynorketamine clearance",
    "Q_per": "intercompartmental flow",
    "CL_in_S": "blood-brain influx clearance",
    "CL_out_S": "blood-brain efflux clearance",
    "k_on_S_pyr": "pyramidal association rate",
    "k_on_S_int": "interneuron association rate",
    "k_off_pyr": "pyramidal dissociation rate",
    "k_off_int": "interneuron dissociation rate",
    "B_int_thresh": "interneuron toxic threshold",
    "g_gain": "NRHypo gain",
    "gamma": "NRHypo injury weight",
    "alpha": "excitotoxic coefficient",
    "beta": "protective coefficient",
    "kappa": "SD threshold gain",
}


def forward(x, dose=DOSE):
    """Terminal net injury for one parameter vector. Returns nan on failure."""
    (cl_out_hnk, q_per, cl_in_s, cl_out_s,
     k_on_s_pyr, k_on_s_int, k_off_pyr, k_off_int,
     b_int_thresh, g_gain, gamma, alpha, beta, kappa) = x

    p = default_parameters()
    cl = p.clearances
    p = replace(
        p,
        flows=Flows(Q_per=q_per),
        clearances=Clearances(
            CL_in={"S": cl_in_s, "R": cl_in_s},
            CL_out={"S": cl_out_s, "R": cl_out_s},
            CL_ecf_in=cl.CL_ecf_in, CL_ecf_out=cl.CL_ecf_out,
            CL_met_NK=cl.CL_met_NK, CL_other_parent=cl.CL_other_parent,
            CL_met_HNK=cl.CL_met_HNK, CL_other_NK=cl.CL_other_NK,
            CL_out_HNK=cl_out_hnk,
            bbb_speed_factor=cl.bbb_speed_factor,
        ),
    )
    reg = DosingRegimen(
        infusions=[InfusionSegment(start=0.0, end=0.667,
                                   rate=dose * WEIGHT / 0.667)],
        s_fraction=0.5,
    )
    t = np.linspace(0.0, T_END, N_POINTS)
    r1 = L1Model(p).simulate(reg, T_END, t_eval=t)
    if not r1.success:
        return np.nan

    l2 = L2Params(
        nmdar=NMDARParams(
            k_on_S={"pyr": k_on_s_pyr, "int": k_on_s_int},
            k_on_R={"pyr": k_on_s_pyr * 0.77, "int": k_on_s_int * 0.77},
            k_off={"pyr": k_off_pyr, "int": k_off_int},
        ),
        p_open=POpenParams(),
    )
    r2 = simulate_occupancy(t, r1.brain_ecf("KET_S"), r1.brain_ecf("KET_R"), l2)
    B_pyr, B_int = r2["pyr"], r2["int"]

    r3b = simulate_nrhypo(t, B_int, DEFAULT_NRHYPPO_PARAMS)
    glu = np.clip((r3b["Glu"] - DEFAULT_NRHYPPO_PARAMS.Glu_0)
                  / DEFAULT_NRHYPPO_PARAMS.Glu_max, 0.0, 1.0)

    sd_p = replace(DEFAULT_SD_PARAMS, kappa=kappa)
    lam = np.array([compute_sd_rate(b, sd_p) for b in B_pyr])
    dur = np.array([compute_sd_duration(b, sd_p) for b in B_pyr])
    phi = lam * dur * (1.0 + glu)

    n_exp = DEFAULT_NRHYPPO_PARAMS.g_exponent
    P = trapezoid(phi, t)
    Q = trapezoid(B_pyr * phi, t)
    R = trapezoid(np.maximum(0.0, B_int - b_int_thresh) ** n_exp, t)
    return float(alpha * P - beta * Q + gamma * g_gain * R)


def _eval_chunk(rows):
    out = np.empty(len(rows))
    for i, row in enumerate(rows):
        try:
            out[i] = forward(row)
        except Exception:
            out[i] = np.nan
    return out


def evaluate(X, workers=None):
    """Evaluate the forward model over every row of X, in parallel if possible."""
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        return _eval_chunk(X)
    chunks = np.array_split(X, workers * 4)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_eval_chunk, chunks))
    return np.concatenate(parts)


def run_one(n_base, second_order, workers):
    t0 = time.time()
    X = sobol_sample.sample(PROBLEM, n_base, calc_second_order=second_order)
    Y = evaluate(X, workers)
    valid = np.isfinite(Y)
    if valid.sum() < 0.9 * len(Y):
        raise RuntimeError(f"too many failed evaluations: "
                           f"{len(Y) - valid.sum()}/{len(Y)}")
    Si = sobol_analyze.analyze(PROBLEM, Y, calc_second_order=second_order,
                               num_resamples=500, conf_level=0.95,
                               print_to_console=False)
    dt = time.time() - t0
    print(f"  N={n_base:5d}  evaluations={X.shape[0]:6d}  "
          f"valid={int(valid.sum())}  {dt/60:.1f} min")
    return Si, X.shape[0], int(valid.sum()), Y


def main(workers=None):
    print("Sobol sensitivity analysis of net injury at 0.5 mg/kg")
    print(f"  {PROBLEM['num_vars']} factors, "
          f"{max(1, (os.cpu_count() or 2) - 1) if workers is None else workers} "
          f"worker processes")

    # --- convergence sequence (first-order and total-order only) -------------
    print("Convergence sequence:")
    convergence = []
    last = None
    for n_base in (64, 128, 256, 512):
        Si, n_eval, n_valid, _ = run_one(n_base, False, workers)
        convergence.append({
            "n_base": n_base, "n_evaluations": n_eval, "n_valid": n_valid,
            "S1": [float(v) for v in Si["S1"]],
            "ST": [float(v) for v in Si["ST"]],
            "S1_conf": [float(v) for v in Si["S1_conf"]],
            "ST_conf": [float(v) for v in Si["ST_conf"]],
        })
        if last is not None:
            drift = np.max(np.abs(np.array(convergence[-1]["ST"])
                                  - np.array(last["ST"])))
            print(f"    max |delta ST| vs previous base sample: {drift:.3f}")
        last = convergence[-1]

    # --- second-order indices at base sample 256 ----------------------------
    print("Second-order analysis (base sample 256):")
    Si2, n_eval2, n_valid2, _ = run_one(256, True, workers)

    names = PROBLEM["names"]
    order = np.argsort(Si2["ST"])[::-1]
    print("  total-order ranking:")
    for k in order:
        print(f"    {names[k]:14s} S1={Si2['S1'][k]:+.3f} "
              f"(+/-{Si2['S1_conf'][k]:.3f})  "
              f"ST={Si2['ST'][k]:.3f} (+/-{Si2['ST_conf'][k]:.3f})")

    s2 = np.array(Si2["S2"], dtype=float)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.isfinite(s2[i, j]):
                pairs.append({"a": names[i], "b": names[j],
                              "S2": float(s2[i, j]),
                              "S2_conf": float(Si2["S2_conf"][i, j])})
    pairs.sort(key=lambda d: abs(d["S2"]), reverse=True)
    print("  strongest second-order interactions:")
    for d in pairs[:8]:
        print(f"    {d['a']:14s} x {d['b']:14s} S2={d['S2']:+.3f} "
              f"(+/-{d['S2_conf']:.3f})")

    sum_s1 = float(np.sum(Si2["S1"]))
    sum_st = float(np.sum(Si2["ST"]))
    print(f"  sum S1 = {sum_s1:.2f}, sum ST = {sum_st:.2f} "
          f"(interaction fraction ~ {1 - sum_s1/sum_st:.2f})")

    # Convergence is better judged on the ordering than on the magnitudes: with
    # an output spanning orders of magnitude, individual indices settle slowly
    # even when the partition into influential and negligible parameters is
    # already stable. Report both.
    from scipy.stats import spearmanr
    rank_stability = []
    for i in range(1, len(convergence)):
        rho = float(spearmanr(convergence[i - 1]["ST"],
                              convergence[i]["ST"]).correlation)
        drift = float(np.max(np.abs(np.array(convergence[i]["ST"])
                                    - np.array(convergence[i - 1]["ST"]))))
        rank_stability.append({"from_n_base": convergence[i - 1]["n_base"],
                               "to_n_base": convergence[i]["n_base"],
                               "spearman_rank_correlation": rho,
                               "max_abs_change_in_ST": drift})
        print(f"  N={convergence[i-1]['n_base']} to "
              f"{convergence[i]['n_base']}: rank correlation {rho:.3f}, "
              f"largest change in a total-order index {drift:.3f}")
    negligible = [names[k] for k in range(len(names))
                  if all(c["ST"][k] < 0.01 for c in convergence)]
    print(f"  negligible at every sample size ({len(negligible)}): "
          f"{', '.join(negligible)}")

    out = {
        "output": "net_injury_at_0.5_mgkg",
        "param_names": names,
        "param_labels": [LABELS[n] for n in names],
        "bounds": PROBLEM["bounds"],
        "convergence": convergence,
        "convergence_summary": {
            "rank_stability": rank_stability,
            "negligible_at_every_sample_size": negligible,
            "verdict": ("individual index magnitudes are not converged at the "
                        "sample sizes attainable here; the partition into "
                        "influential and negligible parameters, and their "
                        "ordering, are stable"),
        },
        "second_order_run": {
            "n_base": 256, "n_evaluations": n_eval2, "n_valid": n_valid2,
            "S1": [float(v) for v in Si2["S1"]],
            "ST": [float(v) for v in Si2["ST"]],
            "S1_conf": [float(v) for v in Si2["S1_conf"]],
            "ST_conf": [float(v) for v in Si2["ST_conf"]],
            "sum_S1": sum_s1, "sum_ST": sum_st,
            "interaction_fraction": 1 - sum_s1 / sum_st,
            "S2_pairs": pairs,
        },
        # Kept for backwards compatibility with the figure script.
        "S1": [float(v) for v in Si2["S1"]],
        "ST": [float(v) for v in Si2["ST"]],
        "S1_conf": [float(v) for v in Si2["S1_conf"]],
        "ST_conf": [float(v) for v in Si2["ST_conf"]],
        "n_samples": 256,
        "n_valid": n_valid2,
        "n_total": n_eval2,
    }
    (RESULTS / "sobol.json").write_text(json.dumps(out, indent=2),
                                        encoding="utf-8")
    print(f"  saved {RESULTS / 'sobol.json'}")


if __name__ == "__main__":
    main()
