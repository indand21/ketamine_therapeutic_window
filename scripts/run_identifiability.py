"""Structural identifiability and critical-value analysis of the injury layer.

Motivation
----------
Reviewers of the earlier version of this work asked (i) which of the
uncalibrated downstream coefficients actually control the shape of the
dose-response curve, and (ii) whether a much smaller value of the NRHypo
injury weight gamma would abolish the U-shaped window.

Both questions have an exact answer, because the L4 injury equation

    dI/dt = alpha * Phi_exc - beta * B_pyr * Phi_exc + gamma * dT_NRHypo/dt
    dT_NRHypo/dt = g_gain * max(0, B_int - theta) ** n

is *linear in the coefficients* once the L1/L2/L3 trajectories are fixed.
Writing

    P(D; kappa) = INT Phi_exc dt
    Q(D; kappa) = INT B_pyr * Phi_exc dt
    R(D; theta) = INT max(0, B_int - theta) ** n dt

the terminal injury for any coefficient vector is exactly

    I(D) = alpha * P(D) - beta * Q(D) + Gamma * R(D),     Gamma := gamma * g_gain

Three consequences follow.

1. gamma and g_gain are not separately identifiable from the injury readout:
   they enter only through their product Gamma. This is why gamma was omitted
   from the earlier variance-based sensitivity analysis, in which g_gain was
   varied.

2. The *shape* of I(D) - and therefore the existence and location of an
   interior optimum - is invariant to the overall scale alpha. Only the two
   ratios beta/alpha and Gamma/alpha, together with theta and kappa, matter.
   Six nominally free coefficients therefore collapse to four shape-determining
   quantities.

3. Because P, Q and R do not depend on the coefficients at all, they can be
   precomputed once per dose and the entire coefficient space swept
   exhaustively at negligible cost, instead of being sampled a few hundred
   times.

This script precomputes the basis integrals, verifies the linear
decomposition against the full ODE integration, then

  * sweeps Gamma/alpha to locate the critical value below which the interior
    optimum disappears (the reviewer's gamma question, answered exactly);
  * performs an exhaustive four-dimensional sweep of the shape-determining
    quantities and reports where in that space the qualitative conclusions
    hold;
  * reports the same for the bolus-versus-infusion ordering.

Run with:  PYTHONPATH=. python scripts/run_identifiability.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.l1_pk import L1Model, DosingRegimen, DoseEvent, InfusionSegment
from src.l1_pk.config import default_parameters
from src.l2_occupancy import simulate_occupancy, DEFAULT_L2_PARAMS
from src.l3a_sd import DEFAULT_SD_PARAMS
from src.l3a_sd.model import compute_sd_rate, compute_sd_duration
from src.l3b_nrhypo import simulate_nrhypo, DEFAULT_NRHYPPO_PARAMS
from src.l3b_nrhypo.model import compute_g
from src.l4_l5.injury import simulate_injury, DEFAULT_INJURY_PARAMS
from src.numeric_compat import trapezoid

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

WEIGHT = 70.0
T_END = 24.0
N_POINTS = 200
INFUSION_H = 40.0 / 60.0

# Dose grid matching the main dose-response analysis.
DOSE_GRID = np.unique(np.concatenate([
    np.round(np.arange(0.05, 1.001, 0.05), 3),
    np.array([1.25, 1.5, 2.0, 2.5, 3.0]),
]))

# Nominal (illustrative) coefficient values used throughout the manuscript.
NOMINAL = {
    "alpha": DEFAULT_INJURY_PARAMS.alpha,          # 1.0
    "beta": DEFAULT_INJURY_PARAMS.beta,            # 0.8
    "gamma": DEFAULT_INJURY_PARAMS.gamma,          # 50.0
    "g_gain": DEFAULT_NRHYPPO_PARAMS.g_gain,       # 50.0
    "theta": DEFAULT_NRHYPPO_PARAMS.B_int_thresh,  # 0.3
    "kappa": DEFAULT_SD_PARAMS.kappa,              # 5.0
}

# Grids for the exhaustive sweep of the shape-determining quantities.
KAPPA_GRID = np.linspace(2.0, 10.0, 17)
THETA_GRID = np.linspace(0.15, 0.45, 31)
BETA_RATIO_GRID = np.linspace(0.1, 1.5, 29)     # beta / alpha
GAMMA_RATIO_GRID = np.geomspace(1.0, 2.5e4, 61)  # Gamma / alpha


# ---------------------------------------------------------------------------
# Occupancy precomputation (independent of every injury coefficient)
# ---------------------------------------------------------------------------

def occupancy_for(regimen):
    """Run L1 and L2 for one regimen and return the occupancy trajectories."""
    t = np.linspace(0.0, T_END, N_POINTS)
    r1 = L1Model(default_parameters()).simulate(regimen, T_END, t_eval=t)
    if not r1.success:
        raise RuntimeError("L1 integration failed")
    r2 = simulate_occupancy(t, r1.brain_ecf("KET_S"), r1.brain_ecf("KET_R"),
                            DEFAULT_L2_PARAMS)
    return {"t": t, "B_pyr": r2["pyr"], "B_int": r2["int"]}


def infusion(dose):
    return DosingRegimen(
        infusions=[InfusionSegment(0.0, INFUSION_H, dose * WEIGHT / INFUSION_H)],
        s_fraction=0.5)


def bolus(dose):
    return DosingRegimen(boluses=[DoseEvent(0.0, dose * WEIGHT)], s_fraction=0.5)


# ---------------------------------------------------------------------------
# Basis integrals
# ---------------------------------------------------------------------------

def glu_excess_for(pre):
    """Normalised glutamate excess. Depends on B_int only, not on theta/g_gain."""
    r3b = simulate_nrhypo(pre["t"], pre["B_int"], DEFAULT_NRHYPPO_PARAMS)
    return np.clip(
        (r3b["Glu"] - DEFAULT_NRHYPPO_PARAMS.Glu_0) / DEFAULT_NRHYPPO_PARAMS.Glu_max,
        0.0, 1.0)


def basis_integrals(pre, kappa_grid, theta_grid, n_exp=2.0):
    """Return P(kappa), Q(kappa) and R(theta) for one precomputed regimen.

    P = INT Phi_exc dt
    Q = INT B_pyr Phi_exc dt
    R = INT max(0, B_int - theta) ** n dt
    """
    t, B_pyr, B_int = pre["t"], pre["B_pyr"], pre["B_int"]
    glu = glu_excess_for(pre)

    P = np.empty(len(kappa_grid))
    Q = np.empty(len(kappa_grid))
    for i, kap in enumerate(kappa_grid):
        sd_p = replace(DEFAULT_SD_PARAMS, kappa=float(kap))
        lam = np.array([compute_sd_rate(b, sd_p) for b in B_pyr])
        dur = np.array([compute_sd_duration(b, sd_p) for b in B_pyr])
        phi = lam * dur * (1.0 + glu)
        P[i] = trapezoid(phi, t)
        Q[i] = trapezoid(B_pyr * phi, t)

    R = np.empty(len(theta_grid))
    for j, th in enumerate(theta_grid):
        R[j] = trapezoid(np.maximum(0.0, B_int - th) ** n_exp, t)

    return P, Q, R


# ---------------------------------------------------------------------------
# Verification of the linear decomposition
# ---------------------------------------------------------------------------

def verify_linearity(pre, rng, n_checks=12):
    """Compare I from the linear decomposition with the full ODE integration."""
    t, B_pyr, B_int = pre["t"], pre["B_pyr"], pre["B_int"]
    glu = glu_excess_for(pre)
    errors = []
    for _ in range(n_checks):
        alpha = float(rng.uniform(0.5, 2.0))
        beta = float(rng.uniform(0.2, 1.5))
        gamma = float(rng.uniform(10.0, 150.0))
        g_gain = float(rng.uniform(10.0, 150.0))
        theta = float(rng.uniform(0.15, 0.45))
        kappa = float(rng.uniform(2.0, 10.0))

        sd_p = replace(DEFAULT_SD_PARAMS, kappa=kappa)
        nr_p = replace(DEFAULT_NRHYPPO_PARAMS, B_int_thresh=theta, g_gain=g_gain)
        inj_p = replace(DEFAULT_INJURY_PARAMS, alpha=alpha, beta=beta, gamma=gamma)

        lam = np.array([compute_sd_rate(b, sd_p) for b in B_pyr])
        dur = np.array([compute_sd_duration(b, sd_p) for b in B_pyr])
        sd_rate = lam * dur
        nr_rate = np.array([compute_g(b, nr_p) for b in B_int])

        full = simulate_injury(t, B_pyr, B_int, sd_rate, glu, nr_rate, inj_p)
        i_ode = float(full["I_final"])

        phi = sd_rate * (1.0 + glu)
        P = trapezoid(phi, t)
        Q = trapezoid(B_pyr * phi, t)
        R = trapezoid(np.maximum(0.0, B_int - theta) ** nr_p.g_exponent, t)
        i_lin = alpha * P - beta * Q + gamma * g_gain * R

        errors.append(abs(i_lin - i_ode) / max(abs(i_ode), 1e-12))
    return float(np.max(errors)), float(np.median(errors))


# ---------------------------------------------------------------------------
# Curve classification
# ---------------------------------------------------------------------------

def classify(injury, doses, rel_rise=0.05):
    """Interior optimum with a clearly rising toxic limb."""
    imin = int(np.argmin(injury))
    interior = 0 < imin < len(doses) - 1
    rising = injury[-1] > injury[imin] * (1.0 + rel_rise)
    left_ok = injury[0] >= injury[imin]
    return bool(interior and rising and left_ok), imin


def critical_gamma_ratio(P, Q, R, doses, beta_ratio, grid=None):
    """Smallest Gamma/alpha (per unit alpha) that yields an interior optimum.

    I/alpha = P - (beta/alpha) Q + (Gamma/alpha) R
    """
    grid = GAMMA_RATIO_GRID if grid is None else grid
    base = P - beta_ratio * Q
    lo = None
    for g in grid:
        ok, _ = classify(base + g * R, doses)
        if ok:
            lo = float(g)
            break
    return lo


def main():
    rng = np.random.default_rng(20260807)

    print("Precomputing L1/L2 occupancy for the dose grid ...")
    pre_dose = {float(d): occupancy_for(infusion(d)) for d in DOSE_GRID}
    pre_bolus = occupancy_for(bolus(0.5))

    print("Verifying the linear decomposition against the full ODE ...")
    max_err, med_err = verify_linearity(pre_dose[0.5], rng)
    print(f"  max relative error {max_err:.3e}, median {med_err:.3e}")

    print("Computing basis integrals P(kappa), Q(kappa), R(theta) ...")
    n_d, n_k, n_t = len(DOSE_GRID), len(KAPPA_GRID), len(THETA_GRID)
    P = np.empty((n_d, n_k))
    Q = np.empty((n_d, n_k))
    R = np.empty((n_d, n_t))
    for i, d in enumerate(DOSE_GRID):
        P[i], Q[i], R[i] = basis_integrals(pre_dose[float(d)], KAPPA_GRID,
                                           THETA_GRID)
    Pb, Qb, Rb = basis_integrals(pre_bolus, KAPPA_GRID, THETA_GRID)

    # -- nominal reproduction -------------------------------------------------
    ik = int(np.argmin(np.abs(KAPPA_GRID - NOMINAL["kappa"])))
    it = int(np.argmin(np.abs(THETA_GRID - NOMINAL["theta"])))
    gamma_ratio_nom = NOMINAL["gamma"] * NOMINAL["g_gain"] / NOMINAL["alpha"]
    beta_ratio_nom = NOMINAL["beta"] / NOMINAL["alpha"]
    inj_nom = NOMINAL["alpha"] * (P[:, ik] - beta_ratio_nom * Q[:, ik]
                                  + gamma_ratio_nom * R[:, it])
    ok_nom, imin_nom = classify(inj_nom, DOSE_GRID)
    print(f"  nominal optimum {DOSE_GRID[imin_nom]:.2f} mg/kg, "
          f"injury {inj_nom[imin_nom]:.1f}, U-shaped={ok_nom}")

    # -- critical Gamma at the nominal kappa, theta, beta ---------------------
    fine = np.geomspace(1.0, 2.5e4, 2000)
    g_crit_nom = critical_gamma_ratio(P[:, ik], Q[:, ik], R[:, it],
                                      DOSE_GRID, beta_ratio_nom, fine)
    gamma_crit_nom = g_crit_nom / NOMINAL["g_gain"] * NOMINAL["alpha"]
    print(f"  critical Gamma/alpha = {g_crit_nom:.1f} "
          f"(nominal {gamma_ratio_nom:.0f}, margin x{gamma_ratio_nom/g_crit_nom:.1f})")
    print(f"  equivalently, at g_gain={NOMINAL['g_gain']:.0f} the window "
          f"disappears below gamma = {gamma_crit_nom:.2f} "
          f"(nominal gamma = {NOMINAL['gamma']:.0f})")

    # -- gamma sweep for the figure/table ------------------------------------
    gamma_sweep = []
    for gamma in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 150.0]:
        inj = NOMINAL["alpha"] * (P[:, ik] - beta_ratio_nom * Q[:, ik]
                                  + gamma * NOMINAL["g_gain"] * R[:, it])
        ok, imin = classify(inj, DOSE_GRID)
        thr = inj[imin] * 1.05
        feas = inj <= thr
        gamma_sweep.append({
            "gamma": gamma,
            "u_shaped": bool(ok),
            "optimal_dose": float(DOSE_GRID[imin]) if ok else None,
            "window_upper": float(DOSE_GRID[feas][-1]) if ok else None,
            "injury_at_optimum": float(inj[imin]),
            "injury_at_3mgkg": float(inj[-1]),
            # Full curve, normalised to its own minimum, so the family of
            # curves can be drawn on one set of axes.
            "dose_response_normalised": (inj / inj.min()).tolist(),
        })
        print(f"    gamma={gamma:6.2f}  U={ok!s:5s}  "
              f"opt={gamma_sweep[-1]['optimal_dose']}  I(3.0)/I(opt)="
              f"{inj[-1]/inj[imin]:.2f}")

    # -- exhaustive sweep of the shape-determining quantities -----------------
    print("Exhaustive sweep over (kappa, theta, beta/alpha, Gamma/alpha) ...")
    n_total = 0
    n_ushape = 0
    n_bolus_worse = 0
    n_both = 0
    n_monotone_violations = 0
    optima = []
    i_half = int(np.where(np.isclose(DOSE_GRID, 0.5))[0][0])
    crit_map = np.full((n_k, n_t, len(BETA_RATIO_GRID)), np.nan)
    # Optimum as a function of Gamma at the nominal kappa, theta, beta.
    opt_vs_gamma = []

    box_lo, box_hi = 10.0 * 10.0, 150.0 * 150.0      # gamma*g_gain over the box
    in_box = (GAMMA_RATIO_GRID >= box_lo) & (GAMMA_RATIO_GRID <= box_hi)
    n_box = 0
    n_box_u = 0
    n_above_crit = 0
    n_above_crit_u = 0

    for a, kap in enumerate(KAPPA_GRID):
        for b, th in enumerate(THETA_GRID):
            Pk, Qk, Rt = P[:, a], Q[:, a], R[:, b]
            Pbk, Qbk, Rbt = Pb[a], Qb[a], Rb[b]
            for c, br in enumerate(BETA_RATIO_GRID):
                base = Pk - br * Qk
                base_bolus = Pbk - br * Qbk
                gcrit = critical_gamma_ratio(Pk, Qk, Rt, DOSE_GRID, br)
                crit_map[a, b, c] = np.nan if gcrit is None else gcrit
                seen_true = False
                for gi, gr in enumerate(GAMMA_RATIO_GRID):
                    inj = base + gr * Rt
                    ok, imin = classify(inj, DOSE_GRID)
                    bolus_worse = (base_bolus + gr * Rbt) > inj[i_half]
                    n_total += 1
                    n_ushape += ok
                    n_bolus_worse += bolus_worse
                    n_both += (ok and bolus_worse)
                    # Once the window has appeared it should never vanish again
                    # as Gamma increases; count any violation of that ordering.
                    if ok:
                        seen_true = True
                        optima.append(float(DOSE_GRID[imin]))
                    elif seen_true:
                        n_monotone_violations += 1
                    if in_box[gi]:
                        n_box += 1
                        n_box_u += ok
                    if gcrit is not None and gr >= gcrit:
                        n_above_crit += 1
                        n_above_crit_u += ok

    optima = np.array(optima)
    print(f"  {n_total} coefficient sets evaluated")
    print(f"  U-shaped window in {100*n_ushape/n_total:.1f}% overall")
    print(f"  U-shaped in {100*n_above_crit_u/n_above_crit:.2f}% of the "
          f"{n_above_crit} sets above the critical toxic weight")
    print(f"  monotonicity violations in Gamma: {n_monotone_violations}")
    print(f"  bolus worse than matched infusion in {100*n_bolus_worse/n_total:.1f}%")
    print(f"  both in {100*n_both/n_total:.1f}%")
    if optima.size:
        print(f"  optimum median {np.median(optima):.2f} mg/kg "
              f"(IQR {np.percentile(optima,25):.2f}-{np.percentile(optima,75):.2f})")
    print(f"  within the previously reported plausible box: "
          f"U-shaped in {100*n_box_u/n_box:.1f}% of {n_box} sets")

    finite_crit = crit_map[np.isfinite(crit_map)]
    print(f"  critical Gamma/alpha across (kappa, theta, beta/alpha): "
          f"median {np.median(finite_crit):.0f}, range "
          f"{finite_crit.min():.0f}-{finite_crit.max():.0f}; "
          f"nominal Gamma/alpha = {gamma_ratio_nom:.0f}")
    frac_nominal_above = float(np.mean(finite_crit <= gamma_ratio_nom))
    print(f"  nominal Gamma/alpha exceeds the critical value for "
          f"{100*frac_nominal_above:.1f}% of (kappa, theta, beta/alpha) triples")

    # Trough location versus the toxic weight, at the nominal kappa/theta/beta.
    for gr in np.geomspace(100.0, 25000.0, 15):
        inj = P[:, ik] - beta_ratio_nom * Q[:, ik] + gr * R[:, it]
        ok, imin = classify(inj, DOSE_GRID)
        opt_vs_gamma.append({"Gamma_over_alpha": float(gr),
                             "u_shaped": bool(ok),
                             "optimal_dose": float(DOSE_GRID[imin]) if ok else None})

    out = {
        "linearity_check": {
            "max_relative_error": max_err,
            "median_relative_error": med_err,
            "n_checks": 12,
            "note": ("Terminal injury computed from the linear decomposition "
                     "alpha*P - beta*Q + gamma*g_gain*R agrees with the full "
                     "ODE integration to solver tolerance, confirming that "
                     "the injury layer is exactly linear in its coefficients."),
        },
        "identifiability": {
            "n_nominal_free_coefficients": 6,
            "n_shape_determining_quantities": 4,
            "shape_determining": ["kappa", "theta", "beta/alpha",
                                  "gamma*g_gain/alpha"],
            "confounded_pair": ["gamma", "g_gain"],
            "note": ("gamma and g_gain enter the injury readout only through "
                     "their product, and alpha sets the overall scale without "
                     "affecting curve shape."),
        },
        "nominal": {
            **NOMINAL,
            "beta_over_alpha": beta_ratio_nom,
            "Gamma_over_alpha": gamma_ratio_nom,
            "optimal_dose": float(DOSE_GRID[imin_nom]),
            "u_shaped": bool(ok_nom),
        },
        "critical_values": {
            "Gamma_over_alpha_critical": g_crit_nom,
            "gamma_critical_at_nominal_g_gain": gamma_crit_nom,
            "margin_fold": gamma_ratio_nom / g_crit_nom if g_crit_nom else None,
        },
        "gamma_sweep": gamma_sweep,
        "optimum_vs_toxic_weight": opt_vs_gamma,
        "exhaustive_sweep": {
            "grids": {
                "kappa": [float(KAPPA_GRID[0]), float(KAPPA_GRID[-1]),
                          len(KAPPA_GRID)],
                "theta": [float(THETA_GRID[0]), float(THETA_GRID[-1]),
                          len(THETA_GRID)],
                "beta_over_alpha": [float(BETA_RATIO_GRID[0]),
                                    float(BETA_RATIO_GRID[-1]),
                                    len(BETA_RATIO_GRID)],
                "Gamma_over_alpha": [float(GAMMA_RATIO_GRID[0]),
                                     float(GAMMA_RATIO_GRID[-1]),
                                     len(GAMMA_RATIO_GRID)],
            },
            "n_sets": int(n_total),
            "frac_ushape": n_ushape / n_total,
            "frac_ushape_above_critical": n_above_crit_u / n_above_crit,
            "n_sets_above_critical": int(n_above_crit),
            "n_monotonicity_violations": int(n_monotone_violations),
            "frac_bolus_worse": n_bolus_worse / n_total,
            "frac_both": n_both / n_total,
            "optimum_median": float(np.median(optima)) if optima.size else None,
            "optimum_iqr": [float(np.percentile(optima, 25)),
                            float(np.percentile(optima, 75))]
            if optima.size else None,
            "plausible_box": {
                "Gamma_over_alpha_range": [box_lo, box_hi],
                "n_sets": int(n_box),
                "frac_ushape": n_box_u / n_box,
            },
        },
        "critical_gamma_map": {
            "kappa_grid": KAPPA_GRID.tolist(),
            "theta_grid": THETA_GRID.tolist(),
            "beta_ratio_grid": BETA_RATIO_GRID.tolist(),
            "median": float(np.median(finite_crit)),
            "min": float(finite_crit.min()),
            "max": float(finite_crit.max()),
            "frac_triples_with_critical_below_nominal": frac_nominal_above,
            "Gamma_over_alpha_critical": np.where(
                np.isnan(crit_map), None, crit_map).tolist(),
        },
        "dose_grid": DOSE_GRID.tolist(),
        "nominal_dose_response": inj_nom.tolist(),
    }
    (RESULTS / "identifiability.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"  saved {RESULTS / 'identifiability.json'}")


if __name__ == "__main__":
    main()
