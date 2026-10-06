"""Regenerate ALL manuscript numerical results from the CURRENT (corrected) codebase.

This is the single source of truth for every number in the manuscript. It
re-runs each analysis (pharmacokinetic prediction error, dose-response window,
PODCAST arms, regimen, enantiomer, virtual population, CYP2B6, emergence)
against the calibrated L1 parameters and writes raw outputs to results/ as
CSV/JSON.

The variance-based sensitivity analysis, the identifiability analysis and the
L1 calibration itself live in their own scripts (run_sobol.py,
run_identifiability.py, run_l1_calibration.py) because they are slower and are
run less often.

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
    compute_hnk_ket_ratio,
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
        bbb_speed_factor=cl.bbb_speed_factor,
    )
    new_vol = CompartmentVolumes(
        V_cen=p.volumes.V_cen * pk_vd,
        V_per=p.volumes.V_per * pk_vd,
        V_vasc=p.volumes.V_vasc,
        V_ecf=p.volumes.V_ecf,
    )
    return replace(p, clearances=new_cl, volumes=new_vol)


def run_full_stack(regimen, params=None, t_end=T_END, n_points=N_POINTS,
                   ablate_l3b=False, clinical_params=None):
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
    r5 = (compute_clinical_constraints(c_brain) if clinical_params is None
          else compute_clinical_constraints(c_brain, clinical_params))

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
    """Secondary pharmacokinetic parameters against published intravenous data.

    The primary evidence that the pharmacokinetic layer is adequate is the
    prediction error against the digitized concentration-time curves, produced
    by scripts/run_l1_calibration.py. This function reports the complementary
    check: whether the secondary parameters implied by the calibrated model
    agree with the intravenous estimates published for the same drug, expressed
    as a standardised difference in units of the published standard deviation
    rather than as an arbitrary pass or fail band.
    """
    print("[1] L1 secondary parameters vs published intravenous estimates")
    from src.ingestion.digitized_data import (
        HASAN_2021_SUMMARY, WEISS_2022_SUMMARY,
    )
    regs = run_all_standard_regimens()
    rr = regs["Zhao_2012_0.5mg_kg_IV40"]
    res = rr.result
    p_l1 = res.params
    m_s = rr.metrics.get("KET_S")
    m_r = rr.metrics.get("KET_R")

    h = HASAN_2021_SUMMARY
    ref = [
        ("Terminal half-life, S-ketamine (h)",
         None if m_s is None else m_s.t_half,
         h["IV_thalf_S_h"], h["IV_thalf_S_SD_h"],
         "Hasan 2021, intravenous arm"),
        ("Total clearance, S-ketamine (l/h)",
         None if m_s is None else m_s.cl_total,
         h["IV_CL_S_mL_per_min"] * 60.0 / 1000.0,
         h["IV_CL_S_SD_mL_per_min"] * 60.0 / 1000.0,
         "Hasan 2021, intravenous arm"),
        ("Total clearance, R-ketamine (l/h)",
         None if m_r is None else m_r.cl_total,
         h["IV_CL_R_mL_per_min"] * 60.0 / 1000.0,
         h["IV_CL_R_SD_mL_per_min"] * 60.0 / 1000.0,
         "Hasan 2021, intravenous arm"),
        ("Steady-state volume of distribution (l)",
         p_l1.volumes.V_cen + p_l1.volumes.V_per,
         h["IV_Vdss_S_L_per_kg"] * WEIGHT, h["IV_Vdss_S_SD_L_per_kg"] * WEIGHT,
         "Hasan 2021, intravenous arm"),
    ]

    out = []
    for label, value, mu, sd, source in ref:
        if value is None or not np.isfinite(value):
            continue
        z = (float(value) - mu) / sd if sd else None
        out.append({"metric": label, "model": float(value),
                    "published_mean": float(mu), "published_sd": float(sd),
                    "standardised_difference": None if z is None else float(z),
                    "within_1_sd": bool(abs(z) <= 1.0) if z is not None else None,
                    "source": source})
        print(f"    {label:42s} model {float(value):8.1f}  published "
              f"{mu:.1f} +/- {sd:.1f}  ({z:+.2f} SD)")

    # Enantiomer clearance ratio and the metabolite ratio are reported against
    # published point estimates without a standard deviation.
    if m_s and m_r and m_s.cl_total and m_r.cl_total:
        ratio = m_s.cl_total / m_r.cl_total
        pub = h["IV_CL_S_mL_per_min"] / h["IV_CL_R_mL_per_min"]
        out.append({"metric": "Clearance ratio, S to R", "model": float(ratio),
                    "published_mean": float(pub), "published_sd": None,
                    "standardised_difference": None, "within_1_sd": None,
                    "source": "Hasan 2021, intravenous arm"})
        print(f"    {'Clearance ratio, S to R':42s} model {ratio:8.2f}  "
              f"published {pub:.2f}")

    ratio_hnk = compute_hnk_ket_ratio(res, 48.0, "cen")
    out.append({"metric": "Hydroxynorketamine to ketamine ratio at 48 h",
                "model": float(ratio_hnk),
                "published_mean": None, "published_sd": None,
                "published_range": [WEISS_2022_SUMMARY["HNK_KET_ratio_S_steady_state"],
                                    WEISS_2022_SUMMARY["HNK_KET_ratio_R_steady_state"]],
                "standardised_difference": None, "within_1_sd": None,
                "source": "Weiss and Siegmund 2022, steady-state ratios",
                "note": ("descriptive only: hydroxynorketamine was excluded "
                         "from the calibration and feeds no downstream layer")})
    print(f"    {'HNK:ketamine ratio at 48 h':42s} model {ratio_hnk:8.1f}  "
          f"published steady-state 14 to 46 (descriptive)")

    n_within = sum(1 for o in out if o.get("within_1_sd"))
    n_scored = sum(1 for o in out if o.get("within_1_sd") is not None)
    save_json("l1_validation.json", {
        "n_within_1_sd": n_within, "n_scored": n_scored,
        "metrics": out,
        "note": ("The primary pharmacokinetic evidence is the prediction error "
                 "in results/l1_calibration.json; this file reports secondary "
                 "parameters against published intravenous estimates."),
    })

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
    return n_within, n_scored


# ---------------------------------------------------------------------------
# 2. Digitized-data overlays (Hasan IV 5 mg; Kamp escalating)
# ---------------------------------------------------------------------------

def analysis_digitized_overlays():
    """Model predictions alongside every digitized curve, in one long table.

    Nine curves are covered: both parent enantiomers after 5 mg intravenous
    racemate (Hasan 2021, venous), and parent, norketamine and
    hydroxynorketamine during an escalating intravenous infusion of esketamine
    and of racemate (Kamp 2020, arterial).
    """
    print("[2] Digitized overlays")
    try:
        from src.ingestion.digitized_data import (
            HASAN_2021_S_KET_PLASMA, HASAN_2021_R_KET_PLASMA,
            KAMP_2020_S_KET_ESKETAMINE, KAMP_2020_S_KET_RACEMIC,
            KAMP_2020_R_KET_RACEMIC, KAMP_2020_S_NK_ESKETAMINE,
            KAMP_2020_S_NK_RACEMIC, KAMP_2020_R_NK_RACEMIC,
            KAMP_2020_HNK_RACEMIC,
        )
    except Exception as e:
        print(f"    skipped (import failed: {e})")
        return

    params = default_parameters()
    # 5 mg racemate intravenously over 30 min (Hasan 2021), not a bolus.
    hasan_reg = DosingRegimen(
        infusions=[InfusionSegment(start=0.0, end=0.5, rate=5.0 / 0.5)],
        s_fraction=0.5)
    kamp_esk = kamp_2020_escalating_regimen(WEIGHT, "esketamine")
    kamp_rac = kamp_2020_escalating_regimen(WEIGHT, "racemic")

    panels = [
        ("hasan_S_ket", HASAN_2021_S_KET_PLASMA, "KET_S", hasan_reg, "venous",
         "Hasan 2021", "external"),
        ("hasan_R_ket", HASAN_2021_R_KET_PLASMA, "KET_R", hasan_reg, "venous",
         "Hasan 2021", "external"),
        ("kamp_S_ket_esk", KAMP_2020_S_KET_ESKETAMINE, "KET_S", kamp_esk,
         "arterial", "Kamp 2020", "calibration"),
        ("kamp_S_ket_rac", KAMP_2020_S_KET_RACEMIC, "KET_S", kamp_rac,
         "arterial", "Kamp 2020", "calibration"),
        ("kamp_R_ket_rac", KAMP_2020_R_KET_RACEMIC, "KET_R", kamp_rac,
         "arterial", "Kamp 2020", "calibration"),
        ("kamp_S_nk_esk", KAMP_2020_S_NK_ESKETAMINE, "NK_S", kamp_esk,
         "arterial", "Kamp 2020", "calibration"),
        ("kamp_S_nk_rac", KAMP_2020_S_NK_RACEMIC, "NK_S", kamp_rac,
         "arterial", "Kamp 2020", "calibration"),
        ("kamp_R_nk_rac", KAMP_2020_R_NK_RACEMIC, "NK_R", kamp_rac,
         "arterial", "Kamp 2020", "calibration"),
        ("kamp_hnk_rac", KAMP_2020_HNK_RACEMIC, "HNK", kamp_rac,
         "arterial", "Kamp 2020", "descriptive"),
    ]

    model_rows, obs_rows = [], []
    sims = {}
    for panel, curve, species, reg, site, study, role in panels:
        t_end = 12.0
        key = id(reg)
        if key not in sims:
            grid = np.linspace(0.0, t_end, 600)
            sims[key] = (grid, L1Model(params).simulate(reg, t_end, t_eval=grid))
        grid, r = sims[key]
        conc = r.concentration(species, "cen")
        for i in range(len(grid)):
            model_rows.append([panel, species, study, site, role,
                               grid[i], conc[i]])
        pred = np.interp(curve.times_h, grid, conc)
        for th, cc, pp in zip(curve.times_h, curve.concentrations, pred):
            obs_rows.append([panel, species, study, site, role, th, cc, pp])
        print(f"    {panel:16s} {role:11s} n={len(curve.times_h):3d}")

    save_csv("overlay_model.csv",
             ["panel", "species", "study", "sampling_site", "role",
              "t_h", "conc_mgL"], model_rows)
    save_csv("overlay_observed.csv",
             ["panel", "species", "study", "sampling_site", "role",
              "t_h", "observed_mgL", "predicted_mgL"], obs_rows)


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
    print("REGENERATING ALL MANUSCRIPT RESULTS (calibrated codebase)")
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
    print("=" * 70)
    print(f"DONE. L1 secondary parameters within 1 SD: {n_pass}/{n_total}. Results in {RESULTS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
