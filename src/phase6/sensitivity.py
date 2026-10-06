"""Global sensitivity analysis for the QSP framework.

Uses SALib to perform Sobol and Morris sensitivity analysis, attributing
therapeutic window location and width to model parameters (Task #22).

The forward model maps parameter vectors to window-relevant outputs:
    - SD burden (protective arm)
    - T_NRHypo (toxic arm)
    - Net injury I (L4)
    - Window boundaries
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    HAS_SALIB = True
except ImportError:
    HAS_SALIB = False


# Parameters eligible for sensitivity analysis
# Bounds from literature uncertainty (docs/*_Parameter_Provenance.md)
SENSITIVITY_PROBLEM = {
    "num_vars": 10,
    "names": [
        "CL_out_HNK", "Q_per", "CL_in_S", "CL_out_S",
        "k_on_S_pyr", "k_on_S_int", "k_off_pyr", "k_off_int",
        "B_int_thresh", "g_gain",
    ],
    "bounds": [
        [0.05, 1.0],      # CL_out_HNK [L/h]
        [100.0, 300.0],    # Q_per [L/h]
        [2.0, 10.0],       # CL_in_S [L/h]
        [2.5, 12.5],       # CL_out_S [L/h]
        [6e3, 2.4e4],      # k_on_S_pyr [M^-1 s^-1]
        [7.5e3, 3.0e4],    # k_on_S_int [M^-1 s^-1]
        [0.01, 0.1],       # k_off_pyr [s^-1]
        [0.005, 0.06],     # k_off_int [s^-1]
        [0.1, 0.5],        # B_int_thresh
        [5.0, 20.0],       # g_gain [h^-1]
    ],
}


@dataclass
class SensitivityResult:
    """Result of a sensitivity analysis."""
    method: str
    param_names: list[str]
    S1: np.ndarray | None = None       # First-order indices
    ST: np.ndarray | None = None       # Total-order indices
    mu_star: np.ndarray | None = None  # Morris mu*
    sigma: np.ndarray | None = None    # Morris sigma
    output_name: str = ""


def qsp_forward(params: np.ndarray, dose: float = 0.5) -> dict:
    """Forward model for sensitivity analysis.

    Takes a parameter vector and returns window-relevant outputs.

    Args:
        params: parameter vector (same order as SENSITIVITY_PROBLEM)
        dose: ketamine dose [mg/kg]

    Returns:
        Dict with SD_burden, T_NRHypo, net_injury, etc.
    """
    from src.l1_pk import L1Model, DosingRegimen, InfusionSegment
    from src.l1_pk.config import (
        default_parameters, Clearances, Flows, NMDARParams,
    )
    from src.l2_occupancy import simulate_occupancy, L2Params, POpenParams
    from src.l3a_sd import simulate_sd_dynamics, SDParams
    from src.l3b_nrhypo import simulate_nrhypo, NRHypoParams
    from src.l4_l5.injury import simulate_injury

    # Unpack parameters
    cl_out_hnk, q_per, cl_in_s, cl_out_s = params[0:4]
    k_on_s_pyr, k_on_s_int, k_off_pyr, k_off_int = params[4:8]
    b_int_thresh, g_gain = params[8:10]

    # L1: PK
    l1_params = default_parameters()
    from dataclasses import replace
    l1_params = replace(
        l1_params,
        flows=Flows(Q_per=q_per),
        clearances=Clearances(
            CL_in={"S": cl_in_s, "R": cl_in_s},
            CL_out={"S": cl_out_s, "R": cl_out_s},
            CL_ecf_in=l1_params.clearances.CL_ecf_in,
            CL_ecf_out=l1_params.clearances.CL_ecf_out,
            CL_met_NK=l1_params.clearances.CL_met_NK,
            CL_other_parent=l1_params.clearances.CL_other_parent,
            CL_met_HNK=l1_params.clearances.CL_met_HNK,
            CL_other_NK=l1_params.clearances.CL_other_NK,
            CL_out_HNK=cl_out_hnk,
            bbb_speed_factor=cl.bbb_speed_factor,
        ),
    )

    total_dose = dose * 70.0
    duration = 0.667
    regimen = DosingRegimen(
        infusions=[InfusionSegment(start=0.0, end=duration, rate=total_dose / duration)],
        s_fraction=0.5,
    )
    l1_model = L1Model(l1_params)
    t = np.linspace(0, 24, 200)
    r1 = l1_model.simulate(regimen, 24.0, t_eval=t)

    if not r1.success:
        return {"SD_burden": np.nan, "T_NRHypo": np.nan, "net_injury": np.nan}

    # L2: Occupancy
    l2_params = L2Params(
        nmdar=NMDARParams(
            k_on_S={"pyr": k_on_s_pyr, "int": k_on_s_int},
            k_on_R={"pyr": k_on_s_pyr * 0.77, "int": k_on_s_int * 0.77},
            k_off={"pyr": k_off_pyr, "int": k_off_int},
        ),
        p_open=POpenParams(),
    )
    r2 = simulate_occupancy(t, r1.brain_ecf("KET_S"), r1.brain_ecf("KET_R"), l2_params)

    # L3a: SD
    r3a = simulate_sd_dynamics(t, r2["pyr"], DEFAULT_SD_PARAMS)

    # L3b: NRHypo
    nrhypo_params = NRHypoParams(B_int_thresh=b_int_thresh, g_gain=g_gain)
    r3b = simulate_nrhypo(t, r2["int"], nrhypo_params)

    # L4: Injury
    r4 = simulate_injury(
        t, r2["pyr"], r2["int"],
        r3a["lambda_SD"] * r3a["D_SD"],
        np.clip((r3b["Glu"] - nrhypo_params.Glu_0) / nrhypo_params.Glu_max, 0, 1),
        r3b["injury_rate"],
    )

    return {
        "SD_burden": r3a["burden"],
        "T_NRHypo": r3b["T_NRHypo_final"],
        "net_injury": r4["I_final"],
    }


# Import default params for forward model
from src.l3a_sd import DEFAULT_SD_PARAMS


def run_sobol_analysis(
    n_samples: int = 256,
    output: str = "net_injury",
    dose: float = 0.5,
) -> SensitivityResult:
    """Run Sobol sensitivity analysis.

    Args:
        n_samples: base sample size (total = n_samples * (2D + 2))
        output: which output to analyze
        dose: ketamine dose [mg/kg]

    Returns:
        SensitivityResult with first-order and total-order indices.
    """
    if not HAS_SALIB:
        raise ImportError("SALib is required for sensitivity analysis")

    problem = SENSITIVITY_PROBLEM

    # Generate samples
    X = saltelli.sample(problem, n_samples, calc_second_order=False)
    n_evals = X.shape[0]

    # Evaluate forward model
    Y = np.zeros(n_evals)
    for i in range(n_evals):
        try:
            result = qsp_forward(X[i], dose=dose)
            Y[i] = result.get(output, np.nan)
        except Exception:
            Y[i] = np.nan

    # Remove failed evaluations
    valid = np.isfinite(Y)
    if valid.sum() < n_evals * 0.5:
        raise RuntimeError(f"Too many failed evaluations: {valid.sum()}/{n_evals}")

    # Analyze
    Si = sobol.analyze(problem, Y[valid], calc_second_order=False, print_to_console=False)

    return SensitivityResult(
        method="Sobol",
        param_names=problem["names"],
        S1=Si["S1"],
        ST=Si["ST"],
        output_name=output,
    )


def run_morris_analysis(
    n_trajectories: int = 10,
    output: str = "net_injury",
    dose: float = 0.5,
) -> SensitivityResult:
    """Run Morris elementary effects screening.

    Args:
        n_trajectories: number of Morris trajectories
        output: which output to analyze
        dose: ketamine dose [mg/kg]

    Returns:
        SensitivityResult with mu* and sigma.
    """
    if not HAS_SALIB:
        raise ImportError("SALib is required for sensitivity analysis")

    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze

    problem = SENSITIVITY_PROBLEM

    X = morris_sample.sample(problem, n_trajectories, num_levels=4)
    Y = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        try:
            result = qsp_forward(X[i], dose=dose)
            Y[i] = result.get(output, np.nan)
        except Exception:
            Y[i] = np.nan

    valid = np.isfinite(Y)
    Si = morris_analyze.analyze(problem, X[valid], Y[valid], print_to_console=False)

    return SensitivityResult(
        method="Morris",
        param_names=problem["names"],
        mu_star=Si["mu_star"],
        sigma=Si["sigma"],
        output_name=output,
    )
