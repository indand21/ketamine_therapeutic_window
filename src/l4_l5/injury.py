"""L4 Injury Balance Engine + L5 Clinical Overlay.

Implements the net injury equation balancing protective (SD suppression)
and toxic (NRHypo) arms (Protocol §3.6, WP3).

L4 Injury equation:
    dI/dt = α · Φ_exc(t) − β · B_pyr(t) · Φ_exc(t) + γ · dT_NRHypo/dt

Where:
    Φ_exc = excitotoxic flux (SD burden rate)
    B_pyr = pyramidal NMDAR occupancy (protective)
    T_NRHypo = cumulative NRHypo injury index (toxic)
    α, β, γ = weighting coefficients

L5 Clinical overlay:
    - CPP/MAP: sympathomimetic effect (ketamine raises BP)
    - ICP: neutral per Carlson 2018/2019
    - Psychotomimetic burden: dose-dependent (B_pyr proxy)

Therapeutic window = argmax protection s.t. toxicity + clinical constraints.

References:
    - Protocol §3.6, §10
    - Carlson 2018/2019: ICP neutral, SD suppression >1.15 mg/kg/h
    - Avidan 2017 (PODCAST): dose-dependent psychotomimetic harm
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp
from src.numeric_compat import trapezoid


@dataclass(frozen=True)
class InjuryParams:
    """Parameters for the L4 injury balance equation.

    Units:
    - alpha: h^-1 (excitotoxic injury rate)
    - beta: h^-1 (protective modulation rate)
    - gamma: dimensionless (NRHypo injury weight)
    """

    # Excitotoxic injury coefficient
    alpha: float = 1.0  # h^-1

    # Protective modulation coefficient
    # B_pyr reduces excitotoxic injury
    beta: float = 0.8  # h^-1

    # NRHypo injury weight
    # Must be large enough that NRHypo toxicity outweighs SD protection
    # at high doses to create the U-shaped window
    gamma: float = 50.0  # dimensionless


@dataclass(frozen=True)
class ClinicalParams:
    """Parameters for L5 clinical constraints.

    Units:
    - CPP_min: mmHg (minimum acceptable CPP)
    - MAP_sympatho: mmHg (ketamine sympathomimetic effect on MAP)
    - ICP_neutral: boolean (ICP neutral per Carlson)
    - psych_threshold: mg/L (brain concentration for psychotomimetic onset)
    - psych_max: mg/L (brain concentration for severe psychotomimetic)
    """

    # CPP/MAP constraints
    CPP_min: float = 60.0  # mmHg
    MAP_baseline: float = 90.0  # mmHg
    MAP_sympatho_gain: float = 20.0  # mmHg max increase from ketamine
    MAP_sympatho_EC50: float = 0.05  # mg/L brain ECF for half-max effect

    # ICP: neutral per Carlson 2018/2019
    ICP_neutral: bool = True

    # Psychotomimetic burden
    psych_threshold: float = 0.02  # mg/L brain ECF (onset)
    psych_max: float = 0.10  # mg/L brain ECF (severe)
    psych_weight: float = 1.0  # weight in objective


DEFAULT_INJURY_PARAMS = InjuryParams()
DEFAULT_CLINICAL_PARAMS = ClinicalParams()


def compute_excitotoxic_flux(
    SD_burden_rate: float,
    Glu_excess: float,
) -> float:
    """Compute excitotoxic flux Φ_exc.

    Φ_exc = SD_burden_rate * (1 + Glu_excess/Glu_max)

    SD events cause excitotoxic injury; elevated glutamate amplifies it.

    Args:
        SD_burden_rate: instantaneous SD burden rate [events·min/h]
        Glu_excess: normalized glutamate excess above resting [0-1]

    Returns:
        Excitotoxic flux [dimensionless].
    """
    return SD_burden_rate * (1.0 + Glu_excess)


def compute_injury_rate(
    B_pyr: float,
    SD_burden_rate: float,
    Glu_excess: float,
    nrhypo_rate: float,
    params: InjuryParams = DEFAULT_INJURY_PARAMS,
) -> float:
    """Compute instantaneous net injury rate.

    dI/dt = α · Φ_exc − β · B_pyr · Φ_exc + γ · dT_NRHypo/dt

    Positive = injury, negative = protection.

    Args:
        B_pyr: pyramidal NMDAR occupancy [0-1]
        SD_burden_rate: SD burden rate [events·min/h]
        Glu_excess: normalized glutamate excess [0-1]
        nrhypo_rate: NRHypo injury rate [h^-1]
        params: injury parameters

    Returns:
        Net injury rate [h^-1].
    """
    phi_exc = compute_excitotoxic_flux(SD_burden_rate, Glu_excess)
    injury = params.alpha * phi_exc
    protection = params.beta * B_pyr * phi_exc
    toxicity = params.gamma * nrhypo_rate
    return injury - protection + toxicity


def simulate_injury(
    t_h: np.ndarray,
    B_pyr: np.ndarray,
    B_int: np.ndarray,
    SD_burden_rate: np.ndarray,
    Glu_excess: np.ndarray,
    nrhypo_rate: np.ndarray,
    params: InjuryParams = DEFAULT_INJURY_PARAMS,
    I0: float = 0.0,
) -> dict:
    """Simulate net injury over time.

    Integrates dI/dt with time-varying inputs from L2/L3a/L3b.

    Args:
        t_h: time points [h]
        B_pyr: pyramidal occupancy at each time [0-1]
        B_int: interneuron occupancy at each time [0-1]
        SD_burden_rate: SD burden rate at each time [events·min/h]
        Glu_excess: normalized glutamate excess at each time [0-1]
        nrhypo_rate: NRHypo injury rate at each time [h^-1]
        params: injury parameters
        I0: initial injury [dimensionless]

    Returns:
        Dict with I(t), protection(t), toxicity(t), and summary stats.
    """
    def interp(arr):
        return lambda t: np.interp(t, t_h, arr)

    B_pyr_i = interp(B_pyr)
    SD_i = interp(SD_burden_rate)
    Glu_i = interp(Glu_excess)
    nrhypo_i = interp(nrhypo_rate)

    def rhs(t, y):
        I = y[0]
        bp = B_pyr_i(t)
        sd = SD_i(t)
        glu = Glu_i(t)
        nr = nrhypo_i(t)
        return [compute_injury_rate(bp, sd, glu, nr, params)]

    sol = solve_ivp(rhs, (t_h[0], t_h[-1]), [I0], method="LSODA",
                    t_eval=t_h, rtol=1e-8, atol=1e-10)

    I_t = sol.y[0]

    # Compute components
    protection = np.array([
        params.beta * B_pyr[i] * compute_excitotoxic_flux(SD_burden_rate[i], Glu_excess[i])
        for i in range(len(t_h))
    ])
    toxicity = np.array([params.gamma * nr for nr in nrhypo_rate])

    return {
        "I": I_t,
        "protection": protection,
        "toxicity": toxicity,
        "t": t_h,
        "I_final": float(I_t[-1]),
        "protection_total": float(trapezoid(protection, t_h)),
        "toxicity_total": float(trapezoid(toxicity, t_h)),
    }


def compute_clinical_constraints(
    C_brain: np.ndarray,
    params: ClinicalParams = DEFAULT_CLINICAL_PARAMS,
) -> dict:
    """Compute L5 clinical constraint violations.

    Args:
        C_brain: brain ECF concentration at each time [mg/L]
        params: clinical parameters

    Returns:
        Dict with constraint time courses and violation flags.
    """
    # MAP: sympathomimetic effect (increases with concentration)
    MAP = params.MAP_baseline + params.MAP_sympatho_gain * (
        C_brain / (C_brain + params.MAP_sympatho_EC50)
    )
    CPP = MAP  # simplified: CPP ≈ MAP (ICP neutral)
    CPP_violated = CPP < params.CPP_min

    # Psychotomimetic burden: Emax-like function of C_brain
    psych = np.clip(
        (C_brain - params.psych_threshold) / (params.psych_max - params.psych_threshold),
        0.0, 1.0,
    )

    return {
        "MAP": MAP,
        "CPP": CPP,
        "CPP_violated": CPP_violated,
        "psych_burden": psych,
        "psych_integral": float(trapezoid(psych, np.linspace(0, 1, len(psych)))),
    }


def compute_therapeutic_window(
    dose_range: np.ndarray,
    injury_results: list[dict],
    clinical_results: list[dict],
    params: ClinicalParams = DEFAULT_CLINICAL_PARAMS,
) -> dict:
    """Compute the therapeutic window from dose-response results.

    The window is the dose range where:
    - Net injury I is minimized (protection > toxicity)
    - Clinical constraints are satisfied (CPP > min, psych < threshold)

    Args:
        dose_range: array of doses tested
        injury_results: list of injury simulation results per dose
        clinical_results: list of clinical constraint results per dose

    Returns:
        Dict with window boundaries and optimal dose.
    """
    n_doses = len(dose_range)
    I_final = np.array([r["I_final"] for r in injury_results])
    psych_max = np.array([np.max(r["psych_burden"]) for r in clinical_results])
    cpp_ok = np.array([not np.any(r["CPP_violated"]) for r in clinical_results])

    # Feasible doses: all clinical constraints satisfied
    feasible = cpp_ok

    if not np.any(feasible):
        return {
            "window_found": False,
            "optimal_dose": None,
            "dose_range": dose_range,
            "I_final": I_final,
        }

    # Among feasible doses, find minimum injury
    feasible_idx = np.where(feasible)[0]
    best_idx = feasible_idx[np.argmin(I_final[feasible])]

    # Window boundaries: contiguous feasible region around optimum
    # Lower bound: first feasible dose
    # Upper bound: last feasible dose (or psychotomimetic limit)
    lower = dose_range[feasible_idx[0]]
    upper = dose_range[feasible_idx[-1]]

    return {
        "window_found": True,
        "optimal_dose": float(dose_range[best_idx]),
        "optimal_injury": float(I_final[best_idx]),
        "window_lower": float(lower),
        "window_upper": float(upper),
        "dose_range": dose_range,
        "I_final": I_final,
        "psych_max": psych_max,
        "feasible": feasible,
    }
