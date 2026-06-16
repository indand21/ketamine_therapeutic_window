"""Corrected global (Sobol) sensitivity analysis of net injury at 0.5 mg/kg.

The repository's src/phase6/sensitivity.py forward model imports NMDARParams from
the wrong module, so every evaluation throws and SALib reports 0 valid samples.
This script reimplements the forward model with correct imports and parameter
ranges centred on the corrected nominal values, then writes sobol.json.

Run with:  PYTHONPATH=. python scripts/run_sobol.py
"""

from __future__ import annotations

import json
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
from src.l3a_sd import simulate_sd_dynamics, DEFAULT_SD_PARAMS
from src.l3b_nrhypo import simulate_nrhypo, NRHypoParams, DEFAULT_NRHYPPO_PARAMS
from src.l4_l5.injury import simulate_injury

RESULTS = Path("results")

PROBLEM = {
    "num_vars": 10,
    "names": ["CL_out_HNK", "Q_per", "CL_in_S", "CL_out_S",
              "k_on_S_pyr", "k_on_S_int", "k_off_pyr", "k_off_int",
              "B_int_thresh", "g_gain"],
    "bounds": [[0.5, 5.0], [120.0, 350.0], [2.0, 10.0], [2.5, 12.5],
               [6e3, 2.4e4], [7.5e3, 3.0e4], [0.01, 0.1], [0.005, 0.06],
               [0.1, 0.5], [10.0, 100.0]],
}


def forward(x, dose=0.5):
    cl_out_hnk, q_per, cl_in_s, cl_out_s = x[0:4]
    k_on_s_pyr, k_on_s_int, k_off_pyr, k_off_int = x[4:8]
    b_int_thresh, g_gain = x[8:10]

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
        ),
    )
    reg = DosingRegimen(
        infusions=[InfusionSegment(start=0.0, end=0.667, rate=dose * 70.0 / 0.667)],
        s_fraction=0.5,
    )
    t = np.linspace(0, 24, 200)
    r1 = L1Model(p).simulate(reg, 24.0, t_eval=t)
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
    r3a = simulate_sd_dynamics(t, r2["pyr"], DEFAULT_SD_PARAMS)
    nr = NRHypoParams(B_int_thresh=b_int_thresh, g_gain=g_gain)
    r3b = simulate_nrhypo(t, r2["int"], nr)
    glu = np.clip((r3b["Glu"] - nr.Glu_0) / nr.Glu_max, 0, 1)
    r4 = simulate_injury(t, r2["pyr"], r2["int"],
                         r3a["lambda_SD"] * r3a["D_SD"], glu, r3b["injury_rate"])
    return float(r4["I_final"])


def main(n_samples=64):
    print(f"Sobol sampling (n_samples={n_samples})...")
    X = sobol_sample.sample(PROBLEM, n_samples, calc_second_order=False)
    print(f"  {X.shape[0]} evaluations")
    Y = np.full(X.shape[0], np.nan)
    for i in range(X.shape[0]):
        try:
            Y[i] = forward(X[i])
        except Exception as e:
            if i < 3:
                print(f"  eval {i} failed: {e}")
    valid = np.isfinite(Y)
    print(f"  valid: {valid.sum()}/{len(Y)}")
    if valid.sum() < 0.5 * len(Y):
        raise RuntimeError("too many failed evaluations")
    Si = sobol_analyze.analyze(PROBLEM, Y, calc_second_order=False,
                               num_resamples=200, conf_level=0.95,
                               print_to_console=False)
    out = {"param_names": PROBLEM["names"], "S1": list(Si["S1"]),
           "ST": list(Si["ST"]), "S1_conf": list(Si["S1_conf"]),
           "ST_conf": list(Si["ST_conf"]), "output": "net_injury",
           "n_samples": n_samples, "n_valid": int(valid.sum()),
           "n_total": int(len(Y))}
    order = np.argsort(Si["ST"])[::-1]
    for k in order:
        print(f"  {PROBLEM['names'][k]:14s} S1={Si['S1'][k]:+.3f} ST={Si['ST'][k]:.3f}")
    (RESULTS / "sobol.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  saved {RESULTS / 'sobol.json'}")


if __name__ == "__main__":
    main(256)
