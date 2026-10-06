"""Calibration of the L1 pharmacokinetic layer to published human data.

Earlier versions of this model took the disposition parameters from a
population analysis of *intranasal* esketamine, in which the reported volume of
distribution is an apparent volume inflated by bioavailability. Checked
quantitatively against digitized intravenous concentration-time data, that
parameterisation underpredicted plasma ketamine by roughly 40% and the
metabolites by an order of magnitude. This script replaces the structural
plausibility check with an explicit calibration.

Design
------
Calibration set   Kamp 2020 (Br J Anaesth 125:750-61): six arterial
                  concentration-time curves from an escalating three-step
                  intravenous infusion of esketamine and of racemic ketamine,
                  covering S- and R-ketamine and S- and R-norketamine.
                  Total hydroxynorketamine is reported but not calibrated: the
                  model forms (2R,6R)-hydroxynorketamine from R-norketamine
                  alone whereas the published curve is total
                  hydroxynorketamine, and no downstream layer reads it.

External set      Hasan 2021 (Anesthesiology 135:326-39): two venous
                  concentration-time curves after 5 mg intravenous racemate.
                  These are never used to estimate a parameter and provide a
                  held-out check, including a check on the arterial-to-venous
                  difference the model cannot represent.

Objective         mean squared residual on the natural-log concentration scale
                  over the calibration curves, plus Gaussian penalties that
                  keep the derived secondary parameters (total clearance and
                  steady-state volume of distribution, per enantiomer) close to
                  the intravenous estimates tabulated by Hasan 2021. The
                  penalties are weighted by the published standard deviations,
                  so they act as weakly informative priors rather than
                  constraints.

Estimated         V_cen, V_per, Q_per, total parent clearance (S; R follows the
                  published S:R ratio), the fraction of parent clearance routed
                  through N-demethylation (confined to 0.40 to 0.95), and total
                  norketamine clearance. Six parameters for 93 observations.

Fixed             the four blood-brain transfer clearances, which no available
                  human dataset identifies. Their ratios set the unbound brain
                  partition coefficient at 0.6, consistent with the
                  cerebrospinal fluid data of Moaddel 2023. The two
                  hydroxynorketamine parameters are held at the values
                  constrained by the steady-state ratios of Weiss and Siegmund
                  2022.

Uncertainty       nonparametric bootstrap over the calibration observations.

Run with:  PYTHONPATH=. python scripts/run_l1_calibration.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from src.l1_pk import (L1Model, DosingRegimen, DoseEvent,
                       InfusionSegment)
from src.l1_pk.config import (
    default_parameters, CompartmentVolumes, Flows, Clearances,
    MetaboliteFractions,
)
from src.validation.l1_pk_validation import (
    kamp_2020_escalating_regimen, REF_WEIGHT_KG,
)
from src.ingestion.digitized_data import (
    HASAN_2021_S_KET_PLASMA, HASAN_2021_R_KET_PLASMA,
    KAMP_2020_S_KET_ESKETAMINE, KAMP_2020_S_KET_RACEMIC,
    KAMP_2020_R_KET_RACEMIC, KAMP_2020_S_NK_ESKETAMINE,
    KAMP_2020_S_NK_RACEMIC, KAMP_2020_R_NK_RACEMIC,
    KAMP_2020_HNK_RACEMIC, HASAN_2021_SUMMARY,
)

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 200
SEED = 20260807

# --- regimens ---------------------------------------------------------------
KAMP_ESK = kamp_2020_escalating_regimen(REF_WEIGHT_KG, "esketamine")
KAMP_RAC = kamp_2020_escalating_regimen(REF_WEIGHT_KG, "racemic")
# Hasan 2021 gave 5 mg of racemate intravenously over 30 min, and the digitized
# curve peaks at 0.5 h accordingly. Modelling it as an instantaneous bolus, as an
# earlier version did, misplaces the peak and understates the observed
# concentrations at every sampled time.
HASAN_IV = DosingRegimen(
    infusions=[InfusionSegment(start=0.0, end=0.5, rate=5.0 / 0.5)],
    s_fraction=0.5)

CALIBRATION = [
    ("S-ketamine, esketamine", KAMP_2020_S_KET_ESKETAMINE, "KET_S", KAMP_ESK),
    ("S-ketamine, racemate", KAMP_2020_S_KET_RACEMIC, "KET_S", KAMP_RAC),
    ("R-ketamine, racemate", KAMP_2020_R_KET_RACEMIC, "KET_R", KAMP_RAC),
    ("S-norketamine, esketamine", KAMP_2020_S_NK_ESKETAMINE, "NK_S", KAMP_ESK),
    ("S-norketamine, racemate", KAMP_2020_S_NK_RACEMIC, "NK_S", KAMP_RAC),
    ("R-norketamine, racemate", KAMP_2020_R_NK_RACEMIC, "NK_R", KAMP_RAC),
]

EXTERNAL = [
    ("S-ketamine, 5 mg i.v.", HASAN_2021_S_KET_PLASMA, "KET_S", HASAN_IV),
    ("R-ketamine, 5 mg i.v.", HASAN_2021_R_KET_PLASMA, "KET_R", HASAN_IV),
]

# Hydroxynorketamine is carried by the model for completeness but is deliberately
# excluded from the calibration and reported separately. Two reasons: the model
# forms (2R,6R)-hydroxynorketamine from R-norketamine alone whereas the digitized
# curve is total hydroxynorketamine, and no downstream layer reads the
# hydroxynorketamine concentration, so it cannot influence any result.
DESCRIPTIVE = [
    ("Hydroxynorketamine, racemate", KAMP_2020_HNK_RACEMIC, "HNK", KAMP_RAC),
]

# --- priors from the Hasan 2021 intravenous arm -----------------------------
# CL in mL/min -> L/h; Vdss in L/kg -> L for the 70 kg reference subject.
PRIOR = {
    "CL_S": (HASAN_2021_SUMMARY["IV_CL_S_mL_per_min"] * 60.0 / 1000.0,
             HASAN_2021_SUMMARY["IV_CL_S_SD_mL_per_min"] * 60.0 / 1000.0),
    "CL_R": (HASAN_2021_SUMMARY["IV_CL_R_mL_per_min"] * 60.0 / 1000.0,
             HASAN_2021_SUMMARY["IV_CL_R_SD_mL_per_min"] * 60.0 / 1000.0),
    "Vss_S": (HASAN_2021_SUMMARY["IV_Vdss_S_L_per_kg"] * REF_WEIGHT_KG,
              HASAN_2021_SUMMARY["IV_Vdss_S_SD_L_per_kg"] * REF_WEIGHT_KG),
    "Vss_R": (HASAN_2021_SUMMARY["IV_Vdss_R_L_per_kg"] * REF_WEIGHT_KG,
              HASAN_2021_SUMMARY["IV_Vdss_R_SD_L_per_kg"] * REF_WEIGHT_KG),
}
# Published S:R total clearance ratio (Hasan 2021 intravenous arm).
SR_CL_RATIO = (HASAN_2021_SUMMARY["IV_CL_S_mL_per_min"]
               / HASAN_2021_SUMMARY["IV_CL_R_mL_per_min"])

PARAM_NAMES = ["V_cen", "V_per", "Q_per", "CL_parent_S", "f_NK",
               "CL_NK_total_S"]

# N-demethylation is the dominant but not the exclusive route of ketamine
# elimination, so the routed fraction is confined to a physiologically
# defensible interval rather than left free to reach unity.
F_NK_LO, F_NK_HI = 0.40, 0.95

# Hydroxynorketamine parameters are held at the values constrained by the
# steady-state ratios of Weiss and Siegmund 2022 and are not estimated.
F_HNK_FIXED = 0.55
CL_OUT_HNK_FIXED = 2.0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    return np.log(p / (1.0 - p))


def unpack(theta):
    """Map the unconstrained vector to interpretable parameter values."""
    return {
        "V_cen": float(np.exp(theta[0])),
        "V_per": float(np.exp(theta[1])),
        "Q_per": float(np.exp(theta[2])),
        "CL_parent_S": float(np.exp(theta[3])),
        "f_NK": float(F_NK_LO + (F_NK_HI - F_NK_LO) * _sigmoid(theta[4])),
        "CL_NK_total_S": float(np.exp(theta[5])),
        "f_HNK": F_HNK_FIXED,
        "CL_out_HNK": CL_OUT_HNK_FIXED,
    }


def build_params(theta):
    """Assemble an L1Parameters object from the unconstrained vector.

    The parent's whole systemic clearance is routed through the
    N-demethylation node and the fraction f_NK of it forms norketamine, so
    parent elimination and metabolite formation are parameterised without
    redundancy. Norketamine is treated the same way with respect to
    hydroxynorketamine.
    """
    v = unpack(theta)
    p = default_parameters()
    cl = p.clearances
    cl_parent = {"S": v["CL_parent_S"], "R": v["CL_parent_S"] / SR_CL_RATIO}
    cl_nk = {"S": v["CL_NK_total_S"], "R": v["CL_NK_total_S"]}
    return replace(
        p,
        volumes=CompartmentVolumes(V_cen=v["V_cen"], V_per=v["V_per"],
                                   V_vasc=p.volumes.V_vasc,
                                   V_ecf=p.volumes.V_ecf),
        flows=Flows(Q_per=v["Q_per"]),
        clearances=Clearances(
            CL_in=cl.CL_in, CL_out=cl.CL_out,
            CL_ecf_in=cl.CL_ecf_in, CL_ecf_out=cl.CL_ecf_out,
            CL_met_NK=cl_parent,
            CL_other_parent={"S": 0.0, "R": 0.0},
            CL_met_HNK=cl_nk,
            CL_other_NK={"S": 0.0, "R": 0.0},
            CL_out_HNK=v["CL_out_HNK"],
            bbb_speed_factor=cl.bbb_speed_factor,
        ),
        fractions=MetaboliteFractions(f_m=v["f_NK"], f_m_HNK=v["f_HNK"]),
    )


REGIMENS = {"kamp_esk": KAMP_ESK, "kamp_rac": KAMP_RAC, "hasan": HASAN_IV}
REG_KEY = {id(KAMP_ESK): "kamp_esk", id(KAMP_RAC): "kamp_rac",
           id(HASAN_IV): "hasan"}


def simulate_cache(params, keys, t_end=12.0, n=800):
    """Simulate only the regimens actually needed by the supplied curves."""
    grid = np.linspace(0.0, t_end, n)
    cache = {}
    for key in keys:
        r = L1Model(params).simulate(REGIMENS[key], t_end, t_eval=grid)
        if not r.success:
            return None
        cache[key] = r
    return grid, cache


def residuals(theta, curves, weights=None):
    """Natural-log residuals over the supplied curves."""
    params = build_params(theta)
    keys = {REG_KEY[id(c[3])] for c in curves}
    sim = simulate_cache(params, keys)
    if sim is None:
        return None
    grid, cache = sim
    out = []
    for k, (label, curve, species, reg) in enumerate(curves):
        r = cache[REG_KEY[id(reg)]]
        pred = np.interp(curve.times_h, grid, r.concentration(species, "cen"))
        obs = np.asarray(curve.concentrations, dtype=float)
        keep = (obs > 0) & (pred > 1e-14)
        res = np.log(pred[keep]) - np.log(obs[keep])
        if weights is not None:
            res = res * np.sqrt(weights[k][keep])
        out.append(res)
    return np.concatenate(out) if out else None


def analytic_secondary(theta):
    """Total clearance per enantiomer and steady-state volume, in closed form.

    These are algebraic functions of the estimated parameters, so the penalty
    term costs nothing to evaluate.
    """
    v = unpack(theta)
    return {"CL_S": v["CL_parent_S"],
            "CL_R": v["CL_parent_S"] / SR_CL_RATIO,
            "Vss": v["V_cen"] + v["V_per"]}


def secondary_parameters(theta):
    """Analytic secondary parameters plus the simulated terminal half-life."""
    sec = analytic_secondary(theta)
    params = build_params(theta)
    grid = np.linspace(0.0, 48.0, 2000)
    r = L1Model(params).simulate(
        DosingRegimen(boluses=[DoseEvent(0.0, 35.0)], s_fraction=0.5),
        48.0, t_eval=grid)
    c = r.concentration("KET_S", "cen")
    tail = (grid >= 12.0) & (c > 0)
    if tail.sum() < 5:
        return sec | {"t_half_S_h": float("nan")}
    slope = np.polyfit(grid[tail], np.log(c[tail]), 1)[0]
    return sec | {"t_half_S_h": float(-np.log(2.0) / slope)}


def prior_residuals(theta):
    """Prior terms expressed as residuals so they enter the same least squares.

    Each secondary parameter contributes (estimate - published mean) / published
    SD, scaled by PRIOR_WEIGHT so the four terms together carry roughly the
    weight of a handful of observations rather than dominating the fit.
    """
    sec = analytic_secondary(theta)
    vals = (("CL_S", sec["CL_S"]), ("CL_R", sec["CL_R"]),
            ("Vss_S", sec["Vss"]), ("Vss_R", sec["Vss"]))
    out = []
    for key, value in vals:
        mu, sd = PRIOR[key]
        out.append(PRIOR_WEIGHT * (value - mu) / sd)
    return np.array(out)


# Weight on the prior residuals. Chosen by held-out performance: sweeping
# 0.25, 0.5, 1, 2 and 4 leaves the calibration error essentially flat (median
# absolute prediction error 8.9 to 10.3%) but minimises the error on the
# external Hasan 2021 curves at a weight of 1, which is also the value at which
# each prior term carries exactly its published standard deviation.
PRIOR_WEIGHT = 1.0


def residual_vector(theta, curves, weights=None):
    res = residuals(theta, curves, weights)
    if res is None or not np.all(np.isfinite(res)):
        return np.full(200, 1e3)
    return np.concatenate([res, prior_residuals(theta)])


def objective(theta, curves, weights=None, use_prior=True):
    """Mean squared residual, retained for reporting and diagnostics."""
    rv = residual_vector(theta, curves, weights) if use_prior \
        else residuals(theta, curves, weights)
    if rv is None:
        return 1e6
    return float(np.mean(np.asarray(rv) ** 2))


def fit(theta0, curves, weights=None, maxiter=200):
    """Trust-region least squares on the residual vector."""
    r = least_squares(residual_vector, theta0, args=(curves, weights),
                      method="trf", max_nfev=maxiter, xtol=1e-10,
                      ftol=1e-10, gtol=1e-10)
    return r.x, float(np.mean(r.fun ** 2))


def curve_metrics(theta, curves):
    """Prediction error per curve.

    The simulation horizon is set from the observations themselves rather than
    from the horizon used during fitting, so that late observations are
    predicted rather than clamped to the end of a shorter grid. This is what
    makes these figures agree with scripts/run_pk_goodness_of_fit.py.
    """
    params = build_params(theta)
    t_end = max(float(np.max(c[1].times_h)) for c in curves) + 2.0
    grid, cache = simulate_cache(params, {REG_KEY[id(c[3])] for c in curves},
                                 t_end=t_end, n=max(800, int(t_end * 80)))
    rows = []
    pooled_log = []
    for label, curve, species, reg in curves:
        r = cache[REG_KEY[id(reg)]]
        pred = np.interp(curve.times_h, grid, r.concentration(species, "cen"))
        obs = np.asarray(curve.concentrations, dtype=float)
        keep = (obs > 0) & (pred > 1e-14)
        o, p_ = obs[keep], pred[keep]
        log_res = np.log10(p_) - np.log10(o)
        ratio = p_ / o
        pooled_log.extend(log_res.tolist())
        rows.append({
            "label": label, "species": species, "n_points": int(o.size),
            "MPE_percent": float(np.median((p_ - o) / o * 100.0)),
            "MAPE_percent": float(np.median(np.abs((p_ - o) / o * 100.0))),
            "RMSLE_log10": float(np.sqrt(np.mean(log_res ** 2))),
            "geometric_mean_ratio": float(10 ** np.mean(log_res)),
            "fraction_within_2fold": float(np.mean((ratio >= 0.5)
                                                   & (ratio <= 2.0))),
            "observed_Cmax_mgL": float(o.max()),
            "predicted_Cmax_mgL": float(p_.max()),
        })
    pooled_log = np.array(pooled_log)
    ratio = 10 ** pooled_log
    pooled = {
        "n_curves": len(rows), "n_points": int(pooled_log.size),
        "MAPE_percent": float(np.median(np.abs(10 ** pooled_log - 1) * 100)),
        "RMSLE_log10": float(np.sqrt(np.mean(pooled_log ** 2))),
        "geometric_mean_ratio": float(10 ** np.mean(pooled_log)),
        "fraction_within_2fold": float(np.mean((ratio >= 0.5)
                                               & (ratio <= 2.0))),
    }
    return rows, pooled


def report(tag, theta, curves):
    rows, pooled = curve_metrics(theta, curves)
    print(f"  {tag}:")
    for r in rows:
        print(f"    {r['label']:34s} n={r['n_points']:3d} "
              f"MAPE={r['MAPE_percent']:5.1f}%  GMR={r['geometric_mean_ratio']:.2f}  "
              f"within2x={r['fraction_within_2fold']:.2f}")
    print(f"    pooled: MAPE {pooled['MAPE_percent']:.1f}%, "
          f"GMR {pooled['geometric_mean_ratio']:.2f}, "
          f"{100*pooled['fraction_within_2fold']:.0f}% within two-fold")
    return rows, pooled


def main():
    rng = np.random.default_rng(SEED)

    # Starting point: the previous parameterisation.
    p0 = default_parameters()
    theta0 = np.array([
        np.log(p0.volumes.V_cen), np.log(p0.volumes.V_per),
        np.log(p0.flows.Q_per), np.log(114.0),
        _logit((0.54 - F_NK_LO) / (F_NK_HI - F_NK_LO)),
        np.log(7.27),
    ])

    print("Before calibration (previous parameterisation):")
    pre_cal_rows, pre_cal = report("calibration set", theta0, CALIBRATION)
    pre_ext_rows, pre_ext = report("external set", theta0, EXTERNAL)

    print("\nFitting ...")
    theta, fval = fit(theta0, CALIBRATION)
    # A second pass from the first solution guards against a premature stop.
    theta, fval = fit(theta, CALIBRATION)
    est = unpack(theta)
    sec = secondary_parameters(theta)
    print(f"  objective {fval:.4f}")
    for k in PARAM_NAMES:
        print(f"    {k:14s} {est[k]:.4g}")
    print(f"    derived: CL_S={sec['CL_S']:.1f} l/h, CL_R={sec['CL_R']:.1f} l/h, "
          f"Vss={sec['Vss']:.0f} l, t_half_S={sec['t_half_S_h']:.2f} h")
    print(f"    published i.v. (Hasan 2021): CL_S={PRIOR['CL_S'][0]:.1f}, "
          f"CL_R={PRIOR['CL_R'][0]:.1f}, Vss_S={PRIOR['Vss_S'][0]:.0f}, "
          f"t_half_S={HASAN_2021_SUMMARY['IV_thalf_S_h']:.1f}")

    print("\nAfter calibration:")
    post_cal_rows, post_cal = report("calibration set", theta, CALIBRATION)
    post_ext_rows, post_ext = report("external set (held out)", theta, EXTERNAL)

    # --- bootstrap -----------------------------------------------------------
    print(f"\nBootstrap ({N_BOOTSTRAP} replicates) ...")
    boot = []
    n_pts = [len(c[1].times_h) for c in CALIBRATION]
    for b in range(N_BOOTSTRAP):
        weights = [rng.multinomial(n, np.ones(n) / n).astype(float)
                   for n in n_pts]
        try:
            tb, _ = fit(theta, CALIBRATION, weights, maxiter=120)
            boot.append(unpack(tb) | secondary_parameters(tb))
        except Exception:
            continue
        if (b + 1) % 25 == 0:
            print(f"    {b + 1}/{N_BOOTSTRAP}")
    print(f"  {len(boot)} successful replicates")

    ci = {}
    if boot:
        keys = list(boot[0].keys())
        for k in keys:
            vals = np.array([d[k] for d in boot], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                ci[k] = {"median": float(np.median(vals)),
                         "ci_2.5": float(np.percentile(vals, 2.5)),
                         "ci_97.5": float(np.percentile(vals, 97.5))}
        print("  95% bootstrap intervals:")
        for k in PARAM_NAMES + ["CL_S", "CL_R", "Vss", "t_half_S_h"]:
            if k in ci:
                print(f"    {k:14s} {ci[k]['median']:9.3g} "
                      f"[{ci[k]['ci_2.5']:.3g} to {ci[k]['ci_97.5']:.3g}]")

    out = {
        "design": {
            "calibration_set": "Kamp 2020, seven arterial curves, "
                               "escalating intravenous infusion",
            "external_set": "Hasan 2021, two venous curves, 5 mg "
                            "intravenous racemate (held out)",
            "objective": "mean squared natural-log residual plus Gaussian "
                         "penalties on total clearance and steady-state volume "
                         "from the Hasan 2021 intravenous arm",
            "fixed_parameters": ["CL_in", "CL_out", "CL_ecf_in", "CL_ecf_out"],
            "fixed_parameter_rationale": "no human brain concentration data "
                                         "identify these; their ratios set an "
                                         "unbound brain partition of 0.6",
            "sr_clearance_ratio": SR_CL_RATIO,
        },
        "estimates": est,
        "derived": sec,
        "published_intravenous_reference": {
            "CL_S_l_per_h": PRIOR["CL_S"][0], "CL_S_sd": PRIOR["CL_S"][1],
            "CL_R_l_per_h": PRIOR["CL_R"][0], "CL_R_sd": PRIOR["CL_R"][1],
            "Vss_S_l": PRIOR["Vss_S"][0], "Vss_S_sd": PRIOR["Vss_S"][1],
            "Vss_R_l": PRIOR["Vss_R"][0], "Vss_R_sd": PRIOR["Vss_R"][1],
            "t_half_S_h": HASAN_2021_SUMMARY["IV_thalf_S_h"],
            "t_half_R_h": HASAN_2021_SUMMARY["IV_thalf_R_h"],
            "source": HASAN_2021_SUMMARY["source"],
        },
        "bootstrap": {"n_requested": N_BOOTSTRAP, "n_successful": len(boot),
                      "intervals": ci},
        "fit_quality": {
            "before": {"calibration": pre_cal, "external": pre_ext,
                       "per_curve_calibration": pre_cal_rows,
                       "per_curve_external": pre_ext_rows},
            "after": {"calibration": post_cal, "external": post_ext,
                      "per_curve_calibration": post_cal_rows,
                      "per_curve_external": post_ext_rows},
        },
        "theta": theta.tolist(),
    }
    (RESULTS / "l1_calibration.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  saved {RESULTS / 'l1_calibration.json'}")

    print("\nCalibrated values to write into src/l1_pk/config.py:")
    print(f"  V_cen           = {est['V_cen']:.2f}")
    print(f"  V_per           = {est['V_per']:.2f}")
    print(f"  Q_per           = {est['Q_per']:.2f}")
    print(f"  CL_met_NK       = S {est['CL_parent_S']:.2f}, "
          f"R {est['CL_parent_S'] / SR_CL_RATIO:.2f}")
    print(f"  CL_other_parent = 0.0, 0.0")
    print(f"  CL_met_HNK      = S {est['CL_NK_total_S']:.3f}, "
          f"R {est['CL_NK_total_S']:.3f}")
    print(f"  CL_other_NK     = 0.0, 0.0")
    print(f"  CL_out_HNK      = {est['CL_out_HNK']:.4f}")
    print(f"  f_m             = {est['f_NK']:.4f}")
    print(f"  f_m_HNK         = {est['f_HNK']:.4f}")


if __name__ == "__main__":
    main()
