"""Regenerate ALL manuscript numerical results from the CURRENT (corrected) codebase.

This is the single source of truth for every number in the BJA manuscript. It
re-runs each analysis (PK validation, dose-response window, PODCAST, regimen,
enantiomer, virtual population, CYP2B6, emergence, Sobol) against the corrected
L1 parameters (Q_per=250, CL_out_HNK=2.0) and writes raw outputs to
results/ as CSV/JSON.

Run with:  PYTHONPATH=. python scripts/regenerate_all.py

NOTE on units: downstream "net injury" is in arbitrary units (illustrative
L3b/L4 parameters). The manuscript frames these qualitatively.
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.l1_pk import L1Model, DosingRegimen, DoseEvent, InfusionSegment
from src.l1_pk.config import default_parameters, Clearances, Flows, CompartmentVolumes
from src.l2_occupancy import simulate_occupancy, DEFAULT_L2_PARAMS
from src.l3a_sd import simulate_sd_dynamics, DEFAULT_SD_PARAMS
from src.l3b_nrhypo import simulate_nrhypo, DEFAULT_NRHYPPO_PARAMS
from src.l4_l5.injury import (
    simulate_injury, compute_clinical_constraints,
    DEFAULT_INJURY_PARAMS, DEFAULT_CLINICAL_PARAMS,
)
from src.validation.l1_pk_validation import (
    run_all_standard_regimens, check_pk_metrics, simulate_regimen,
    zhao_2012_regimen, kamp_2020_escalating_regimen, hasan_2021_regimen,
    DEFAULT_REF, REF_WEIGHT_KG,
)
from src.phase6.virtual_population import (
    generate_virtual_population, CYP2B6_ALLELE_FREQ, CYP2B6_CL_SCALE,
)

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

WEIGHT = REF_WEIGHT_KG  # 70 kg
T_END = 24.0
N_POINTS = 200

# Fine dose grid for window/dose-response (mg/kg), dense near the optimum.
DOSE_GRID = np.unique(np.concatenate([
    np.round(np.arange(0.05, 1.001, 0.05), 3),
    np.array([1.25, 1.5, 2.0, 2.5, 3.0]),
]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsonify(obj):
    """Recursively convert numpy types to JSON-native types."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonify(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(name: str, data) -> None:
    path = RESULTS / name
    path.write_text(json.dumps(_jsonify(data), indent=2), encoding="utf-8")
    print(f"  saved {path}")


def save_csv(name: str, header: list[str], rows: list[list]) -> None:
    path = RESULTS / name
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(
            f"{v:.6g}" if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
            for v in r
        ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  saved {path}")


def make_l1_params(cl_scale: float = 1.0, pk_cl: float = 1.0, pk_vd: float = 1.0):
    """Return L1 params with CYP2B6 CL scaling and PK variability applied.

    cl_scale scales the N-demethylation CL (CYP2B6 effect); pk_cl scales all
    parent/metabolite CLs; pk_vd scales central+peripheral volumes.
    """
    p = default_parameters()
    cl = p.clearances
    new_cl = Clearances(
        CL_in=cl.CL_in,
        CL_out=cl.CL_out,
        CL_ecf_in=cl.CL_ecf_in,
        CL_ecf_out=cl.CL_ecf_out,
        CL_met_NK={e: cl.CL_met_NK[e] * cl_scale * pk_cl for e in ("S", "R")},
        CL_other_parent={e: cl.CL_other_parent[e] * pk_cl for e in ("S", "R")},
        CL_met_HNK=cl.CL_met_HNK,
        CL_other_NK=cl.CL_other_NK,
        CL_out_HNK=cl.CL_out_HNK,
    )
    new_vol = CompartmentVolumes(
        V_cen=p.volumes.V_cen * pk_vd,
        V_per=p.volumes.V_per * pk_vd,
        V_vasc=p.volumes.V_vasc,
        V_ecf=p.volumes.V_ecf,
    )
    return replace(p, clearances=new_cl, volumes=new_vol)


def run_full_stack(regimen, params=None, t_end=T_END, n_points=N_POINTS,
                   ablate_l3b=False):
    """Run L0->L5 for a given regimen. Returns summary metrics + time courses."""
    if params is None:
        params = default_parameters()
    l1 = L1Model(params)
    t = np.linspace(0, t_end, n_points)
    r1 = l1.simulate(regimen, t_end, t_eval=t)
    if not r1.success:
        return None

    r2 = simulate_occupancy(t, r1.brain_ecf("KET_S"), r1.brain_ecf("KET_R"),
                            DEFAULT_L2_PARAMS)
    r3a = simulate_sd_dynamics(t, r2["pyr"], DEFAULT_SD_PARAMS)
    r3b = simulate_nrhypo(t, r2["int"], DEFAULT_NRHYPPO_PARAMS)

    sd_rate = r3a["lambda_SD"] * r3a["D_SD"]
    glu_excess = np.clip(
        (r3b["Glu"] - DEFAULT_NRHYPPO_PARAMS.Glu_0) / DEFAULT_NRHYPPO_PARAMS.Glu_max,
        0, 1,
    )
    nrhypo_rate = r3b["injury_rate"]
    if ablate_l3b:
        # Toxic arm removed: no NRHypo injury, no glutamate surge.
        glu_excess = np.zeros_like(glu_excess)
        nrhypo_rate = np.zeros_like(nrhypo_rate)

    r4 = simulate_injury(t, r2["pyr"], r2["int"], sd_rate, glu_excess, nrhypo_rate)
    c_brain = r1.brain_ecf("KET_S") + r1.brain_ecf("KET_R")
    r5 = compute_clinical_constraints(c_brain)

    return {
        "t": t, "r1": r1, "r2": r2, "r3a": r3a, "r3b": r3b, "r4": r4, "r5": r5,
        "I_final": float(r4["I_final"]),
        "protection_total": float(r4["protection_total"]),
        "toxicity_total": float(r4["toxicity_total"]),
        "SD_burden": float(r3a["burden"]),
        "T_NRHypo": float(r3b["T_NRHypo_final"]),
        "B_pyr_peak": float(np.max(r2["pyr"])),
        "B_int_peak": float(np.max(r2["int"])),
        "psych_max": float(np.max(r5["psych_burden"])),
        "C_brain_peak": float(np.max(c_brain)),
    }


def infusion_regimen(dose_mg_kg, duration_h, s_fraction=0.5, weight=WEIGHT):
    total = dose_mg_kg * weight
    return DosingRegimen(
        infusions=[InfusionSegment(start=0.0, end=duration_h, rate=total / duration_h)],
        s_fraction=s_fraction,
    )


def bolus_regimen(dose_mg_kg, s_fraction=0.5, weight=WEIGHT):
    return DosingRegimen(
        boluses=[DoseEvent(time=0.0, amount=dose_mg_kg * weight)],
        s_fraction=s_fraction,
    )


def window_from_curve(doses, injury, rel_tol=0.10):
    """Optimal dose + window (contiguous region within rel_tol of the minimum)."""
    doses = np.asarray(doses)
    injury = np.asarray(injury)
    opt_idx = int(np.argmin(injury))
    opt_dose = float(doses[opt_idx])
    opt_inj = float(injury[opt_idx])
    thr = opt_inj * (1.0 + rel_tol)
    feasible = injury <= thr
    lower = float(doses[feasible][0]) if np.any(feasible) else opt_dose
    upper = float(doses[feasible][-1]) if np.any(feasible) else opt_dose
    return {"optimal_dose": opt_dose, "optimal_injury": opt_inj,
            "window_lower": lower, "window_upper": upper,
            "window_width": upper - lower}


# ---------------------------------------------------------------------------
# 1. L1 PK validation
# ---------------------------------------------------------------------------

def analysis_l1_validation():
    print("[1] L1 PK validation")
    regs = run_all_standard_regimens()
    checks = check_pk_metrics(regs, DEFAULT_REF)
    out = []
    for c in checks:
        out.append({
            "metric": c.metric,
            "value": None if c.value is None else float(c.value),
            "ref_low": None if c.ref_range is None else float(c.ref_range[0]),
            "ref_high": None if c.ref_range is None else float(c.ref_range[1]),
            "passed": bool(c.passed),
        })
        rng = "" if c.ref_range is None else f" (ref {c.ref_range[0]}-{c.ref_range[1]})"
        print(f"    {'PASS' if c.passed else 'FAIL'}  {c.metric} = "
              f"{c.value:.4g}{rng}" if c.value is not None else f"    {c.metric}: no value")
    n_pass = sum(1 for c in checks if c.passed)
    save_json("l1_validation.json", {"n_pass": n_pass, "n_total": len(checks),
                                     "checks": out})

    # Zhao 0.5 mg/kg IV40 reference profile (central compartment) for the PK figure.
    rr = regs["Zhao_2012_0.5mg_kg_IV40"]
    res = rr.result
    rows = []
    for i, tt in enumerate(res.t):
        rows.append([
            tt,
            res.concentration("KET_S", "cen")[i],
            res.concentration("KET_R", "cen")[i],
            res.concentration("NK_S", "cen")[i],
            res.concentration("NK_R", "cen")[i],
            res.concentration("HNK", "cen")[i],
        ])
    save_csv("pk_zhao_profile.csv",
             ["t_h", "KET_S", "KET_R", "NK_S", "NK_R", "HNK"], rows)
    return n_pass, len(checks)


# ---------------------------------------------------------------------------
# 2. Digitized-data overlays (Hasan IV 5 mg; Kamp escalating)
# ---------------------------------------------------------------------------

def analysis_digitized_overlays():
    print("[2] Digitized overlays")
    try:
        from src.ingestion.digitized_data import (
            HASAN_2021_S_KET_PLASMA, HASAN_2021_R_KET_PLASMA,
            KAMP_2020_S_KET_ESKETAMINE,
        )
    except Exception as e:
        print(f"    skipped (import failed: {e})")
        return

    params = default_parameters()

    # Hasan IV 5 mg racemic bolus -> model central S/R.
    hasan_reg = DosingRegimen(boluses=[DoseEvent(time=0.0, amount=5.0)], s_fraction=0.5)
    l1 = L1Model(params)
    t = np.linspace(0, 12, 400)
    rh = l1.simulate(hasan_reg, 12.0, t_eval=t)
    save_csv("overlay_hasan_model.csv", ["t_h", "KET_S", "KET_R"],
             [[t[i], rh.concentration("KET_S", "cen")[i],
               rh.concentration("KET_R", "cen")[i]] for i in range(len(t))])
    for curve, tag in [(HASAN_2021_S_KET_PLASMA, "S"), (HASAN_2021_R_KET_PLASMA, "R")]:
        save_csv(f"overlay_hasan_digitized_{tag}.csv", ["t_h", "conc_mgL"],
                 [[th, cc] for th, cc in zip(curve.times_h, curve.concentrations)])

    # Kamp escalating esketamine -> model central S.
    kamp_reg = kamp_2020_escalating_regimen(WEIGHT, "esketamine")
    l1k = L1Model(params)
    tk = np.linspace(0, 8, 400)
    rk = l1k.simulate(kamp_reg, 8.0, t_eval=tk)
    save_csv("overlay_kamp_model.csv", ["t_h", "KET_S"],
             [[tk[i], rk.concentration("KET_S", "cen")[i]] for i in range(len(tk))])
    save_csv("overlay_kamp_digitized_S.csv", ["t_h", "conc_mgL"],
             [[th, cc] for th, cc in zip(KAMP_2020_S_KET_ESKETAMINE.times_h,
                                         KAMP_2020_S_KET_ESKETAMINE.concentrations)])


# ---------------------------------------------------------------------------
# 3. Dose-response therapeutic window (racemic, 40-min IV)
# ---------------------------------------------------------------------------

def analysis_window():
    print("[3] Dose-response window (racemic IV 40 min)")
    rows = []
    for dose in DOSE_GRID:
        r = run_full_stack(infusion_regimen(dose, 40.0 / 60.0))
        if r is None:
            continue
        rows.append([dose, r["I_final"], r["SD_burden"], r["T_NRHypo"],
                     r["B_pyr_peak"], r["B_int_peak"], r["psych_max"],
                     r["C_brain_peak"], r["protection_total"], r["toxicity_total"]])
        print(f"    {dose:.2f} mg/kg: I={r['I_final']:.3f} "
              f"B_pyr={r['B_pyr_peak']:.3f} B_int={r['B_int_peak']:.3f}")
    save_csv("window_doseresponse.csv",
             ["dose_mgkg", "I_final", "SD_burden", "T_NRHypo", "B_pyr_peak",
              "B_int_peak", "psych_max", "C_brain_peak", "protection_total",
              "toxicity_total"], rows)
    doses = [r[0] for r in rows]
    injury = [r[1] for r in rows]
    win = window_from_curve(doses, injury)
    win["B_int_thresh"] = float(DEFAULT_NRHYPPO_PARAMS.B_int_thresh)
    save_json("window_summary.json", win)
    print(f"    optimal={win['optimal_dose']:.2f} window="
          f"[{win['window_lower']:.2f},{win['window_upper']:.2f}] mg/kg")
    return win


# ---------------------------------------------------------------------------
# 4. PODCAST arms
# ---------------------------------------------------------------------------

def analysis_podcast():
    print("[4] PODCAST arms")
    arms = {
        "placebo": DosingRegimen(boluses=[], s_fraction=0.5),
        "ket_0.5_bolus": bolus_regimen(0.5),
        "ket_1.0_bolus": bolus_regimen(1.0),
    }
    out = {}
    for name, reg in arms.items():
        r = run_full_stack(reg)
        if r is None:
            out[name] = None
            continue
        out[name] = {k: r[k] for k in ("I_final", "SD_burden", "T_NRHypo",
                                       "B_pyr_peak", "B_int_peak", "psych_max",
                                       "C_brain_peak")}
        print(f"    {name}: I={r['I_final']:.2f} psych={r['psych_max']:.2f} "
              f"C_peak={r['C_brain_peak']:.4f}")
    save_json("podcast.json", out)
    return out


# ---------------------------------------------------------------------------
# 5. Regimen comparison (DOSE-MATCHED at 0.5 mg/kg total)
# ---------------------------------------------------------------------------

def analysis_regimen():
    print("[5] Regimen comparison (dose-matched 0.5 mg/kg total)")
    regs = {
        "bolus": bolus_regimen(0.5),
        "infusion_40min": infusion_regimen(0.5, 40.0 / 60.0),
        "infusion_2h": infusion_regimen(0.5, 2.0),
        "infusion_4h": infusion_regimen(0.5, 4.0),
        "infusion_8h": infusion_regimen(0.5, 8.0),
    }
    out = {}
    for name, reg in regs.items():
        r = run_full_stack(reg)
        out[name] = None if r is None else {
            k: r[k] for k in ("I_final", "SD_burden", "T_NRHypo", "B_pyr_peak",
                              "B_int_peak", "psych_max", "C_brain_peak")}
        if r is not None:
            print(f"    {name}: I={r['I_final']:.2f} C_peak={r['C_brain_peak']:.4f} "
                  f"B_int={r['B_int_peak']:.3f} psych={r['psych_max']:.2f}")
    save_json("regimen.json", out)
    return out


# ---------------------------------------------------------------------------
# 6. Enantiomer comparison (dose sweep + at 0.5 mg/kg)
# ---------------------------------------------------------------------------

def analysis_enantiomer():
    print("[6] Enantiomer comparison")
    fracs = {"racemic": 0.5, "s_only": 1.0, "r_only": 0.0}
    sweep = {}
    for name, sf in fracs.items():
        doses, injury, bpyr, bint = [], [], [], []
        for dose in DOSE_GRID:
            r = run_full_stack(infusion_regimen(dose, 40.0 / 60.0, s_fraction=sf))
            if r is None:
                continue
            doses.append(dose); injury.append(r["I_final"])
            bpyr.append(r["B_pyr_peak"]); bint.append(r["B_int_peak"])
        win = window_from_curve(doses, injury)
        sweep[name] = {"doses": doses, "I_final": injury, "B_pyr": bpyr,
                       "B_int": bint, **win}
        print(f"    {name}: optimal={win['optimal_dose']:.2f} mg/kg "
              f"min_I={win['optimal_injury']:.2f}")
    save_json("enantiomer.json", sweep)
    return sweep


# ---------------------------------------------------------------------------
# 7. Virtual population (N=100)
# ---------------------------------------------------------------------------

def analysis_vpop(n_subjects=100):
    print(f"[7] Virtual population (N={n_subjects})")
    pop = generate_virtual_population(n_subjects=n_subjects, seed=42)
    key_doses = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    injuries = np.full((n_subjects, len(key_doses)), np.nan)
    for i, subj in enumerate(pop.subjects):
        params = make_l1_params(cl_scale=subj.cyp2b6_cl_scale,
                                pk_cl=subj.pk_cl_variability,
                                pk_vd=subj.pk_vd_variability)
        for j, dose in enumerate(key_doses):
            reg = infusion_regimen(dose, 40.0 / 60.0, weight=subj.weight)
            r = run_full_stack(reg, params=params, n_points=100)
            if r is not None:
                injuries[i, j] = r["I_final"]
    median = np.nanmedian(injuries, axis=0)
    p5 = np.nanpercentile(injuries, 5, axis=0)
    p95 = np.nanpercentile(injuries, 95, axis=0)
    # Per-subject optimal dose.
    opt_doses = []
    for i in range(n_subjects):
        row = injuries[i, :]
        if np.any(np.isfinite(row)):
            opt_doses.append(key_doses[int(np.nanargmin(row))])
    pop_opt = float(key_doses[int(np.argmin(median))])
    geno_counts = {}
    for s in pop.subjects:
        geno_counts[s.cyp2b6_genotype] = geno_counts.get(s.cyp2b6_genotype, 0) + 1

    save_csv("vpop_injuries.csv",
             ["subject"] + [f"dose_{d}" for d in key_doses],
             [[i] + list(injuries[i, :]) for i in range(n_subjects)])
    summary = {
        "n_subjects": n_subjects,
        "key_doses": key_doses,
        "median": list(median),
        "p5": list(p5),
        "p95": list(p95),
        "population_optimal_dose": pop_opt,
        "optimal_dose_distribution": opt_doses,
        "genotype_counts": geno_counts,
    }
    save_json("vpop_summary.json", summary)
    print(f"    population optimal={pop_opt:.2f} mg/kg; genotypes={geno_counts}")
    return summary


# ---------------------------------------------------------------------------
# 8. CYP2B6 genotype windows
# ---------------------------------------------------------------------------

def analysis_cyp2b6():
    print("[8] CYP2B6 genotype windows")
    out = {}
    for geno, scale in CYP2B6_CL_SCALE.items():
        params = make_l1_params(cl_scale=scale)
        doses, injury = [], []
        for dose in DOSE_GRID:
            r = run_full_stack(infusion_regimen(dose, 40.0 / 60.0), params=params)
            if r is None:
                continue
            doses.append(dose); injury.append(r["I_final"])
        win = window_from_curve(doses, injury)
        out[geno] = {"cl_scale": scale, "doses": doses, "I_final": injury, **win}
        print(f"    {geno} (CL x{scale}): optimal={win['optimal_dose']:.2f} mg/kg")
    out["allele_freq"] = CYP2B6_ALLELE_FREQ
    save_json("cyp2b6.json", out)
    return out


# ---------------------------------------------------------------------------
# 9. Emergence test (ablate toxic arm)
# ---------------------------------------------------------------------------

def analysis_emergence():
    print("[9] Emergence test (ablate L3b toxic arm)")
    doses, intact, ablated = [], [], []
    for dose in DOSE_GRID:
        r_full = run_full_stack(infusion_regimen(dose, 40.0 / 60.0))
        r_abl = run_full_stack(infusion_regimen(dose, 40.0 / 60.0), ablate_l3b=True)
        if r_full is None or r_abl is None:
            continue
        doses.append(dose); intact.append(r_full["I_final"]); ablated.append(r_abl["I_final"])
    # Is the ablated curve monotonically non-increasing (no upper bound)?
    abl = np.asarray(ablated)
    monotonic = bool(np.all(np.diff(abl) <= 1e-6))
    intact_win = window_from_curve(doses, intact)
    save_json("emergence.json", {
        "doses": doses, "injury_intact": intact, "injury_ablated": ablated,
        "ablated_monotonic_decreasing": monotonic,
        "intact_has_upper_bound": intact_win["window_upper"] < max(doses),
        "intact_optimal_dose": intact_win["optimal_dose"],
    })
    print(f"    ablated monotonic-decreasing: {monotonic}; "
          f"intact optimal={intact_win['optimal_dose']:.2f}")


# ---------------------------------------------------------------------------
# 10. Sobol global sensitivity
# ---------------------------------------------------------------------------

def analysis_sobol(n_samples=64):
    print(f"[10] Sobol sensitivity (n_samples={n_samples})")
    try:
        from src.phase6.sensitivity import run_sobol_analysis, SENSITIVITY_PROBLEM
    except Exception as e:
        print(f"    skipped ({e})")
        return
    try:
        res = run_sobol_analysis(n_samples=n_samples, output="net_injury", dose=0.5)
    except Exception as e:
        print(f"    Sobol failed: {e}")
        return
    out = {
        "param_names": res.param_names,
        "S1": list(res.S1),
        "ST": list(res.ST),
        "output": res.output_name,
        "n_samples": n_samples,
    }
    order = np.argsort(res.ST)[::-1]
    for k in order:
        print(f"    {res.param_names[k]:14s} S1={res.S1[k]:.3f} ST={res.ST[k]:.3f}")
    save_json("sobol.json", out)


# ---------------------------------------------------------------------------
# Parameter snapshot for Table 1
# ---------------------------------------------------------------------------

def dump_param_snapshot():
    print("[*] Parameter snapshot")
    p = default_parameters()
    l2 = DEFAULT_L2_PARAMS
    snap = {
        "L1": {
            "V_cen": p.volumes.V_cen, "V_per": p.volumes.V_per,
            "V_vasc": p.volumes.V_vasc, "V_ecf": p.volumes.V_ecf,
            "Q_per": p.flows.Q_per,
            "CL_met_NK": p.clearances.CL_met_NK,
            "CL_other_parent": p.clearances.CL_other_parent,
            "CL_met_HNK": p.clearances.CL_met_HNK,
            "CL_other_NK": p.clearances.CL_other_NK,
            "CL_out_HNK": p.clearances.CL_out_HNK,
            "CL_in": p.clearances.CL_in, "CL_out": p.clearances.CL_out,
            "CL_ecf_in": p.clearances.CL_ecf_in, "CL_ecf_out": p.clearances.CL_ecf_out,
            "f_m": p.fractions.f_m, "f_m_HNK": p.fractions.f_m_HNK,
            "Kp_uu_brain": p.Kp_uu_brain,
        },
        "L4_injury": {"alpha": DEFAULT_INJURY_PARAMS.alpha,
                      "beta": DEFAULT_INJURY_PARAMS.beta,
                      "gamma": DEFAULT_INJURY_PARAMS.gamma},
        "L3b_nrhypo": {"B_int_thresh": DEFAULT_NRHYPPO_PARAMS.B_int_thresh,
                       "g_gain": DEFAULT_NRHYPPO_PARAMS.g_gain,
                       "g_exponent": DEFAULT_NRHYPPO_PARAMS.g_exponent},
        "L3a_sd": {"K_thr_0": DEFAULT_SD_PARAMS.K_thr_0,
                   "kappa": DEFAULT_SD_PARAMS.kappa},
        "CYP2B6": {"allele_freq": CYP2B6_ALLELE_FREQ, "cl_scale": CYP2B6_CL_SCALE},
    }
    # L2 NMDAR kinetics (best-effort; structure may vary).
    try:
        snap["L2"] = {
            "k_on_S": l2.nmdar.k_on_S, "k_on_R": l2.nmdar.k_on_R,
            "k_off": l2.nmdar.k_off,
        }
    except Exception as e:
        snap["L2"] = f"unavailable: {e}"
    save_json("param_snapshot.json", snap)


def main():
    print("=" * 70)
    print("REGENERATING ALL MANUSCRIPT RESULTS (corrected codebase)")
    print("=" * 70)
    dump_param_snapshot()
    n_pass, n_total = analysis_l1_validation()
    analysis_digitized_overlays()
    analysis_window()
    analysis_podcast()
    analysis_regimen()
    analysis_enantiomer()
    analysis_vpop(100)
    analysis_cyp2b6()
    analysis_emergence()
    analysis_sobol(64)
    print("=" * 70)
    print(f"DONE. L1 validation: {n_pass}/{n_total}. Results in {RESULTS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
