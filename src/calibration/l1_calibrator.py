"""L1 PK calibration engine — fit to digitized published curves.

Uses scipy.optimize.minimize to adjust uncertain L1 parameters so the
model output matches digitized concentration-time data from published
figures. Supports:
  - Weighted least-squares objective (chi²)
  - Parameter bounds from literature uncertainty ranges
  - Multi-curve fitting (multiple species/regimens simultaneously)
  - Result reporting with residuals and fit statistics

This is the non-Bayesian calibration path when raw patient data is
unavailable. For Bayesian NLME calibration, see docs/Calibration_Strategy.md.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, replace
from scipy.optimize import minimize, differential_evolution

from src.l1_pk.config import (
    L1Parameters, Clearances, Flows, CompartmentVolumes, default_parameters,
)
from src.l1_pk.model import L1Model
from src.l1_pk.dosing import DosingRegimen, DoseEvent, InfusionSegment
from src.ingestion.digitized_data import (
    DigitizedCurve,
    ALL_DIGITIZED_CURVES,
)
from src.validation.l1_pk_validation import (
    kamp_2020_escalating_regimen,
    zhao_2012_regimen,
    estimate_half_life,
    compute_hnk_ket_ratio,
    DEFAULT_REF,
    REF_WEIGHT_KG,
)


# ===========================================================================
# Parameter space for calibration
# ===========================================================================

@dataclass(frozen=True)
class ParamBound:
    """Lower and upper bounds for a single parameter."""
    name: str
    lower: float
    upper: float
    default: float


# Parameters eligible for calibration with their bounds.
# Bounds derived from literature uncertainty (see docs/L1_Parameter_Provenance.md).
# Every default below equals the corresponding value in src/l1_pk/config.py, so
# the identity point of this calibrator IS the shipped model. The disposition
# values come from scripts/run_l1_calibration.py, which estimates them from
# digitized human intravenous concentration-time data; bounds bracket the 95%
# bootstrap intervals reported there, widened where the interval is narrow.
# See docs/L1_Calibration.md.
CALIBRATION_PARAMS: list[ParamBound] = [
    # Hydroxynorketamine parameters are FIXED in the shipped model, not
    # estimated: hydroxynorketamine was excluded from the calibration and no
    # downstream layer reads it. They are exposed here only so that the
    # calibrator can be pointed at them deliberately.
    ParamBound("CL_out_HNK",   0.5,   5.0,   2.0),
    # Estimated: 75.54 l/h [56.5 to 86.1].
    ParamBound("Q_per",       40.0,  350.0,  75.54),
    # Blood-brain transfer: FIXED in the shipped model. No human dataset
    # identifies these; their ratios set Kp,uu = 0.6.
    ParamBound("CL_in_S",      2.0,   10.0,   5.0),
    ParamBound("CL_out_S",     2.5,   12.5,   6.25),
    ParamBound("CL_ecf_in_S",  1.0,    8.0,   4.0),
    ParamBound("CL_ecf_out_S", 1.5,   10.0,   5.33),
    # Total norketamine clearance, estimated: 6.836 l/h [5.11 to 7.33],
    # applied to both enantiomers.
    ParamBound("CL_met_HNK_S", 1.0,   12.0,   6.836),
    ParamBound("CL_met_HNK_R", 1.0,   12.0,   6.836),
    # Absorbed into CL_met_HNK by the shipped parameterisation.
    ParamBound("CL_other_NK_S", 0.0,   6.0,   0.0),
    ParamBound("CL_other_NK_R", 0.0,   6.0,   0.0),
    # Estimated: V_cen 49.72 l [45.5 to 57.3], V_per 234.37 l [220 to 316].
    ParamBound("V_cen",       20.0,  100.0,  49.72),
    ParamBound("V_per",      100.0,  800.0, 234.37),
]

# Default values array for initial guess
DEFAULT_VALUES = np.array([p.default for p in CALIBRATION_PARAMS])
BOUNDS = [(p.lower, p.upper) for p in CALIBRATION_PARAMS]
PARAM_NAMES = [p.name for p in CALIBRATION_PARAMS]


def apply_calibration_vector(
    params: L1Parameters, x: np.ndarray
) -> L1Parameters:
    """Apply a calibration vector to produce modified L1Parameters.

    Maps the flat optimization vector back to the parameter structure.
    R-enantiomer BBB clearances are scaled proportionally from S.
    """
    cl = params.clearances
    fl = params.flows

    # Unpack
    cl_out_hnk = x[0]
    q_per = x[1]
    cl_in_s = x[2]
    cl_out_s = x[3]
    cl_ecf_in_s = x[4]
    cl_ecf_out_s = x[5]
    cl_met_hnk_s = x[6]
    cl_met_hnk_r = x[7]
    cl_other_nk_s = x[8]
    cl_other_nk_r = x[9]
    # Central volumes (optional tail of the vector; absent → keep base values).
    v_cen = x[10] if len(x) > 10 else params.volumes.V_cen
    v_per = x[11] if len(x) > 11 else params.volumes.V_per

    # Maintain Kp,uu ratio for R-enantiomer (same as S)
    kp_ratio_in_out = cl_in_s / cl_out_s if cl_out_s > 0 else 0.8
    kp_ratio_ecf = cl_ecf_in_s / cl_ecf_out_s if cl_ecf_out_s > 0 else 0.75

    new_cl = Clearances(
        CL_in={"S": cl_in_s, "R": cl_in_s},
        CL_out={"S": cl_out_s, "R": cl_out_s},
        CL_ecf_in={"S": cl_ecf_in_s, "R": cl_ecf_in_s},
        CL_ecf_out={"S": cl_ecf_out_s, "R": cl_ecf_out_s},
        CL_met_NK=dict(cl.CL_met_NK),  # Keep fixed from literature
        CL_other_parent=dict(cl.CL_other_parent),  # Keep fixed
        CL_met_HNK={"S": cl_met_hnk_s, "R": cl_met_hnk_r},
        CL_other_NK={"S": cl_other_nk_s, "R": cl_other_nk_r},
        CL_out_HNK=cl_out_hnk,
        bbb_speed_factor=cl.bbb_speed_factor,
    )

    new_volumes = CompartmentVolumes(
        V_cen=v_cen,
        V_per=v_per,
        V_vasc=params.volumes.V_vasc,
        V_ecf=params.volumes.V_ecf,
    )

    return L1Parameters(
        volumes=new_volumes,
        flows=Flows(Q_per=q_per),
        clearances=new_cl,
        fractions=params.fractions,
        Kp_uu_brain=params.Kp_uu_brain,
    )


# ===========================================================================
# Objective function
# ===========================================================================

def _build_regimen_for_curve(curve: DigitizedCurve) -> DosingRegimen:
    """Construct a DosingRegimen from a DigitizedCurve's metadata."""
    reg = curve.regimen.lower()
    if "escalating" in reg:
        # Kamp 2020 / Olofsen 2021 escalating 3-step infusion over 180 min.
        formulation = "esketamine" if "esketamine" in reg else "racemic"
        return kamp_2020_escalating_regimen(curve.weight_kg, formulation)
    if "bolus" in curve.regimen.lower() and "infusion" not in curve.regimen.lower():
        # Pure bolus
        return DosingRegimen(
            boluses=[DoseEvent(time=0.0, amount=curve.dose_mg)],
            s_fraction=1.0 if curve.species.endswith("S") else 0.0,
        )
    elif "bolus" in curve.regimen.lower() and "infusion" in curve.regimen.lower():
        # Bolus + infusion (e.g., Hasan 2021: 5 mg bolus + 5 mg over 1 h)
        bolus_dose = curve.dose_mg
        infusion_dose = curve.dose_mg  # Same amount as bolus
        return DosingRegimen(
            boluses=[DoseEvent(time=0.0, amount=bolus_dose)],
            infusions=[InfusionSegment(start=0.0, end=1.0, rate=infusion_dose)],
            s_fraction=0.5,  # Racemate
        )
    elif "40 min" in curve.regimen.lower() or "over 40" in curve.regimen.lower():
        # Zhao 2012 style
        dose = curve.dose_mg
        duration = 40.0 / 60.0
        return DosingRegimen(
            infusions=[InfusionSegment(start=0.0, end=duration, rate=dose / duration)],
            s_fraction=0.5,
        )
    else:
        # Default: bolus
        return DosingRegimen(
            boluses=[DoseEvent(time=0.0, amount=curve.dose_mg)],
            s_fraction=0.5,
        )


def simulate_curve(
    curve: DigitizedCurve, params: L1Parameters, t_end: float | None = None
) -> np.ndarray | None:
    """Simulate a curve with given params and return interpolated concentrations.

    Integrates only to the curve's last data point (+0.5 h) instead of a fixed
    48 h. For short curves (Kamp ends at 5 h) this is a large speedup with no
    effect on the concentrations at the data times (the IVP is deterministic).
    """
    regimen = _build_regimen_for_curve(curve)
    model = L1Model(params)
    if t_end is None:
        t_end = float(np.max(curve.times_h)) + 0.5
    n_points = max(400, int(t_end * 50))
    t_eval = np.linspace(0, t_end, n_points)
    result = model.simulate(regimen, t_end, t_eval=t_eval)

    if not result.success:
        return None

    # Interpolate model output at the digitized time points
    species = curve.species
    comp = "cen" if curve.compartment == "plasma" else "ecf"

    if species == "HNK":
        c_model = np.interp(curve.times_h, result.t, result.concentration("HNK", "cen"))
    else:
        c_model = np.interp(curve.times_h, result.t, result.concentration(species, comp))

    return c_model


# Fraction of a curve's Cmax used as the residual-normalization floor. Points
# are weighted by 1/(FLOOR·Cmax + |C_obs|), so each residual is a fractional
# error relative to the curve's own peak (well-conditioned across curves whose
# absolute scales differ by ~25×, e.g. Hasan ~0.01 vs Kamp ~0.25 mg/L) while
# near-baseline points cannot blow up the way a raw relative error would.
_RESIDUAL_FLOOR_FRAC = 0.05


def _scaled_residuals(c_model: np.ndarray, c_obs: np.ndarray) -> np.ndarray:
    """Fractional-of-Cmax residuals for one curve (dimensionless, O(1))."""
    cmax = float(np.max(np.abs(c_obs)))
    if cmax <= 0.0:
        return np.zeros_like(c_model)
    denom = _RESIDUAL_FLOOR_FRAC * cmax + np.abs(c_obs)
    return (c_model - c_obs) / denom


def _range_penalty(
    value: float | None, lo: float, hi: float, miss: float = 1.0
) -> float:
    """Squared *relative* violation of a [lo, hi] range.

    0 inside the range; grows quadratically with the fractional overshoot
    outside it (1.0 at a 2× violation); ``miss`` if the value is unavailable.
    """
    if value is None:
        return miss
    if value < lo:
        return ((lo - value) / lo) ** 2
    if value > hi:
        return ((value - hi) / hi) ** 2
    return 0.0


def _regularization_penalty(params: L1Parameters, ref=DEFAULT_REF) -> float:
    """Penalty keeping key PK metrics within their published ranges.

    Vdss, terminal t½, and the HNK:KET ratio are INDEPENDENT references
    (venous popPK / steady state) that the digitized arterial Kamp curves pull
    against. Adding them as soft constraints lets calibration balance
    curve fit against physiological plausibility instead of overfitting the
    curves at the references' expense. Vdss is read directly from the volumes;
    t½ and HNK:KET come from one reference simulation (Zhao 0.5 mg/kg IV40),
    matching :func:`check_pk_metrics`.
    """
    pen = 0.0
    # Anatomical Vdss = V_cen + V_per — no simulation needed.
    vdss = params.volumes.V_cen + params.volumes.V_per
    pen += _range_penalty(vdss, *ref.vdss_s)

    model = L1Model(params)
    res = model.simulate(
        zhao_2012_regimen(REF_WEIGHT_KG), 48.0, t_eval=np.linspace(0.0, 48.0, 600)
    )
    if not res.success:
        return pen + 100.0  # unstable parameters → heavy penalty
    c_s = res.concentration("KET_S", "cen")
    pen += _range_penalty(estimate_half_life(res.t, c_s), *ref.t_half_s_ket)
    pen += _range_penalty(compute_hnk_ket_ratio(res, 48.0, "cen"), *ref.hnk_ket_ratio_ss)
    return pen


def objective(
    x: np.ndarray,
    curves: list[DigitizedCurve],
    base_params: L1Parameters,
    weights: dict[str, float] | None = None,
    reg_weight: float = 0.0,
) -> float:
    """Rescaled least-squares objective with optional physiological regularization.

    Each curve contributes the MEAN squared fractional-of-Cmax residual (see
    :func:`_scaled_residuals`). Using the mean (not the sum) removes the
    point-count bias (Kamp curves have 17 points, Hasan 10), and the Cmax
    normalization removes the absolute-scale bias — so every curve is an O(1)
    term and the optimizer is not dominated by the largest/longest curve.

    When ``reg_weight > 0`` a physiological penalty (:func:`_regularization_penalty`)
    is added so the fit cannot satisfy the digitized curves by driving Vdss / t½ /
    HNK:KET out of their published ranges.

    Args:
        x: Parameter vector (same order as CALIBRATION_PARAMS)
        curves: Digitized curves to fit against
        base_params: Base L1Parameters to modify
        weights: Optional per-curve weights (by source)
        reg_weight: Weight on the physiological regularization term (0 = off)

    Returns:
        Weighted sum of mean squared fractional residuals, plus the weighted
        regularization penalty.
    """
    params = apply_calibration_vector(base_params, x)
    total = 0.0

    for curve in curves:
        c_model = simulate_curve(curve, params)
        if c_model is None:
            return 1e12  # Penalize failed simulations

        res = _scaled_residuals(c_model, curve.concentrations)
        mse = float(np.mean(res ** 2))

        w = weights.get(curve.source, 1.0) if weights else 1.0
        total += w * mse

    if reg_weight > 0.0:
        total += reg_weight * _regularization_penalty(params)

    return total


# ===========================================================================
# Calibration result
# ===========================================================================

@dataclass
class CalibrationResult:
    """Result of a calibration run."""
    success: bool
    x_optimal: np.ndarray
    params_optimal: L1Parameters
    objective_value: float
    param_names: list[str]
    curves_used: list[str]
    residuals_per_curve: dict[str, float]  # source -> SSE
    method: str


def calibrate(
    curves: list[DigitizedCurve] | None = None,
    base_params: L1Parameters | None = None,
    method: str = "differential_evolution",
    maxiter: int = 1000,
    seed: int = 42,
    weights: dict[str, float] | None = None,
    popsize: int = 15,
    workers: int = 1,
    reg_weight: float = 0.0,
) -> CalibrationResult:
    """Run calibration against digitized curves.

    Args:
        curves: Curves to fit. If None, uses all available.
        base_params: Base parameters. If None, uses defaults.
        method: "differential_evolution" for global (default), "L-BFGS-B" local.
        maxiter: Maximum iterations / generations.
        seed: Random seed for reproducibility.
        weights: Per-curve weights (keyed by source).
        popsize: DE population multiplier (population = popsize × n_params).
        workers: DE parallelism (-1 = all cores). >1 requires a __main__ guard
            in the calling script (Windows spawn) and uses deferred updating.
        reg_weight: Weight on physiological regularization (Vdss / t½ / HNK:KET
            range penalties). 0 = pure curve fit; >0 balances against the
            published reference ranges.

    Returns:
        CalibrationResult with optimal parameters.
    """
    if curves is None:
        curves = ALL_DIGITIZED_CURVES
    if base_params is None:
        base_params = default_parameters()

    # Filter out curves with zero dose (pure S-ketamine regimen has no R-KET)
    active_curves = [c for c in curves if c.dose_mg > 0 or c.species == "HNK"]

    print(f"Calibrating against {len(active_curves)} curves "
          f"(method={method}, maxiter={maxiter}"
          + (f", popsize={popsize}, workers={workers}" if method == "differential_evolution" else "")
          + ")...")
    for c in active_curves:
        print(f"  {c.source} — {c.species} ({len(c.times_h)} points)")

    if method == "differential_evolution":
        result = differential_evolution(
            objective,
            bounds=BOUNDS,
            args=(active_curves, base_params, weights, reg_weight),
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            tol=1e-6,
            polish=True,
            disp=True,
            workers=workers,
            updating="deferred" if workers != 1 else "immediate",
        )
        x_opt = result.x
        obj_val = result.fun
        success = result.success
    else:
        x0 = DEFAULT_VALUES.copy()
        result = minimize(
            objective,
            x0,
            args=(active_curves, base_params, weights, reg_weight),
            method="L-BFGS-B",
            bounds=BOUNDS,
            options={"maxiter": maxiter, "ftol": 1e-10, "eps": 1e-3},
        )
        x_opt = result.x
        obj_val = result.fun
        success = result.success

    params_opt = apply_calibration_vector(base_params, x_opt)

    # Per-curve fit quality, using the same rescaled metric as the objective.
    # Keyed by species + source so same-source curves (all Kamp panels) stay
    # distinct in the report.
    residuals = {}
    for curve in active_curves:
        c_model = simulate_curve(curve, params_opt)
        if c_model is not None:
            mse = float(np.mean(_scaled_residuals(c_model, curve.concentrations) ** 2))
            residuals[f"{curve.species} | {curve.source}"] = mse

    return CalibrationResult(
        success=success,
        x_optimal=x_opt,
        params_optimal=params_opt,
        objective_value=obj_val,
        param_names=PARAM_NAMES,
        curves_used=[c.source for c in active_curves],
        residuals_per_curve=residuals,
        method=method,
    )


# ===========================================================================
# Report generation
# ===========================================================================

def format_calibration_report(result: CalibrationResult) -> str:
    """Generate a markdown calibration report."""
    lines = [
        "# L1 PK Calibration Report",
        "",
        f"**Method:** {result.method}",
        f"**Converged:** {'Yes' if result.success else 'No'}",
        f"**Objective (χ²):** {result.objective_value:.4f}",
        "",
        "## Calibrated Parameters",
        "",
        "| Parameter | Default | Calibrated | Range |",
        "|-----------|---------|------------|-------|",
    ]

    for i, name in enumerate(result.param_names):
        default_val = DEFAULT_VALUES[i]
        cal_val = result.x_optimal[i]
        lo, hi = BOUNDS[i]
        flag = ""
        if abs(cal_val - lo) / (hi - lo) < 0.05:
            flag = " ⚠️ near lower bound"
        elif abs(cal_val - hi) / (hi - lo) < 0.05:
            flag = " ⚠️ near upper bound"
        lines.append(
            f"| {name} | {default_val:.3f} | {cal_val:.3f} | [{lo}, {hi}]{flag} |"
        )

    lines.extend([
        "",
        "## Fit Quality (per curve)",
        "",
        "| Source | MSE (relative) |",
        "|--------|----------------|",
    ])
    for source, mse in result.residuals_per_curve.items():
        lines.append(f"| {source} | {mse:.6f} |")

    lines.extend([
        "",
        "## Curves Used",
        "",
    ])
    for source in result.curves_used:
        lines.append(f"- {source}")

    return "\n".join(lines)
