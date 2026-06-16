"""L3b NRHypo Engine — Toxic Arm.

Implements the NMDA receptor hypofunction (NRHypo) injury pathway
via interneuron disinhibition (WP3, Protocol §3.5).

Mechanism (Olney 1995, Li 2002):
    B_int → reduced GABA inhibition → pyramidal disinhibition →
    sustained glutamate release → AMPA/kainate excitotoxicity

The E-I neural mass model tracks:
    - Inhibitory drive: I_GABA(t) = I_GABA_0 * (1 - B_int(t))
    - Excitatory drive: E_pyr(t) = f(I_GABA, baseline excitation)
    - Glutamate accumulation: dGlu/dt = production - clearance
    - NRHypo injury index: T_NRHypo = ∫ g(B_int(τ)) dτ

Where g(B_int) is a convex gain function above a disinhibition threshold.

References:
    - Olney & Farber 1995: NRHypo mechanism
    - Li et al. 2002: MK-801 disinhibition in PCC/RSC
    - Morgan et al. 2014: interneuron 10× sensitivity
    - Yan & Rein 2021: PFC disinhibition review
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class NRHypoParams:
    """Parameters for the NRHypo toxic arm model.

    Units:
    - I_GABA_0: normalized inhibitory drive [dimensionless, 0-1]
    - E_pyr_0: normalized excitatory drive [dimensionless, 0-1]
    - B_int_thresh: disinhibition threshold for injury [dimensionless]
    - g_gain: convex gain above threshold [1/h]
    - g_exponent: convexity exponent [dimensionless]
    - tau_Glu: glutamate clearance time constant [h]
    - Glu_0: resting glutamate [µM]
    - Glu_max: maximum pathological glutamate [µM]
    """

    # Baseline inhibitory drive (normalized)
    I_GABA_0: float = 0.4  # 40% of pyramidal activity is GABA-inhibited

    # Disinhibition threshold
    # Below this B_int, disinhibition is minimal
    # Above this, injury accumulates (convex)
    B_int_thresh: float = 0.3  # 30% interneuron occupancy → disinhibition starts

    # Convex gain function: g(B_int) = g_gain * max(0, B_int - thresh)^g_exponent
    g_gain: float = 50.0     # injury rate gain [h^-1]
    g_exponent: float = 2.0  # convexity (2 = quadratic, >2 = steeper)

    # Glutamate dynamics
    tau_Glu: float = 0.5     # glutamate clearance time constant [h]
    Glu_0: float = 2.0       # resting extracellular glutamate [µM]
    Glu_max: float = 100.0   # maximum pathological glutamate [µM]

    # Excitatory drive scaling
    E_pyr_0: float = 0.3     # baseline excitatory drive
    E_pyr_max: float = 1.0   # maximum excitatory drive (full disinhibition)

    # Injury dynamics
    T_NRHypo_0: float = 0.0  # initial injury index


DEFAULT_NRHYPPO_PARAMS = NRHypoParams()


def compute_disinhibition(B_int: float, params: NRHypoParams) -> float:
    """Compute disinhibition level from interneuron occupancy.

    Disinhibition = I_GABA_0 * B_int (fraction of inhibition removed)

    Args:
        B_int: interneuron NMDAR occupancy [0-1]
        params: NRHypo parameters

    Returns:
        Disinhibition level [0-1]. Higher = more disinhibition.
    """
    return params.I_GABA_0 * B_int


def compute_excitatory_drive(
    B_int: float,
    params: NRHypoParams,
) -> float:
    """Compute excitatory pyramidal drive as function of interneuron block.

    E_pyr = E_pyr_0 + (E_pyr_max - E_pyr_0) * disinhibition

    When B_int is high → disinhibition is high → E_pyr increases.

    Args:
        B_int: interneuron NMDAR occupancy [0-1]
        params: NRHypo parameters

    Returns:
        Normalized excitatory drive [E_pyr_0, E_pyr_max].
    """
    disinhibition = compute_disinhibition(B_int, params)
    return params.E_pyr_0 + (params.E_pyr_max - params.E_pyr_0) * disinhibition


def compute_g(B_int: float, params: NRHypoParams) -> float:
    """Convex gain function for NRHypo injury.

    g(B_int) = g_gain * max(0, B_int - B_int_thresh)^g_exponent

    Below threshold: g = 0 (no injury)
    Above threshold: convex increase (supralinear injury)

    Args:
        B_int: interneuron NMDAR occupancy [0-1]
        params: NRHypo parameters

    Returns:
        Injury rate [h^-1].
    """
    excess = max(0.0, B_int - params.B_int_thresh)
    return params.g_gain * excess ** params.g_exponent


def compute_nrhypo_injury_rate(
    B_int: float,
    params: NRHypoParams,
) -> float:
    """Compute instantaneous NRHypo injury rate.

    Injury rate = g(B_int) * Glu_excess / Glu_max

    Where Glu_excess is the pathological glutamate above resting.

    Args:
        B_int: interneuron NMDAR occupancy [0-1]
        params: NRHypo parameters

    Returns:
        Instantaneous injury rate [h^-1].
    """
    return compute_g(B_int, params)


def simulate_nrhypo(
    t_h: np.ndarray,
    B_int: np.ndarray,
    params: NRHypoParams = DEFAULT_NRHYPPO_PARAMS,
) -> dict:
    """Simulate NRHypo injury dynamics over time.

    Integrates:
        dGlu/dt = (E_pyr * Glu_max - Glu) / tau_Glu  (disinhibition-driven)
        dT/dt = g(B_int) * (Glu - Glu_0) / Glu_max   (injury accumulation)

    Args:
        t_h: time points [h]
        B_int: interneuron occupancy at each time [0-1]
        params: NRHypo parameters

    Returns:
        Dict with Glu(t), T_NRHypo(t), injury rate, and summary stats.
    """
    def B_interp(t):
        return np.interp(t, t_h, B_int)

    def rhs(t, y):
        Glu, T = y
        B = B_interp(t)

        # Glutamate dynamics: driven by excitatory drive
        E_pyr = compute_excitatory_drive(B, params)
        dGlu = (E_pyr * params.Glu_max - Glu) / params.tau_Glu

        # Injury accumulation: convex function of B_int
        # Scaled by glutamate excess above resting
        glu_excess = max(0.0, Glu - params.Glu_0) / params.Glu_max
        dT = compute_g(B, params) * glu_excess

        return [dGlu, dT]

    y0 = [params.Glu_0, params.T_NRHypo_0]
    t_span = (t_h[0], t_h[-1])
    sol = solve_ivp(rhs, t_span, y0, method="LSODA", t_eval=t_h,
                    rtol=1e-8, atol=1e-10)

    Glu_t = sol.y[0]
    T_t = sol.y[1]

    # Compute injury rate at each time point
    injury_rate = np.array([compute_g(b, params) for b in B_int])
    disinhibition = np.array([compute_disinhibition(b, params) for b in B_int])

    return {
        "Glu": Glu_t,
        "T_NRHypo": T_t,
        "injury_rate": injury_rate,
        "disinhibition": disinhibition,
        "t": t_h,
        "T_NRHypo_final": float(T_t[-1]),
        "Glu_max_achieved": float(np.max(Glu_t)),
    }


def compute_nrhypo_burden(
    t_h: np.ndarray,
    B_int: np.ndarray,
    params: NRHypoParams = DEFAULT_NRHYPPO_PARAMS,
) -> float:
    """Compute total NRHypo injury burden (integral of injury rate).

    T_NRHypo = ∫ g(B_int(τ)) dτ

    This is the cumulative injury index — the toxic arm readout.

    Args:
        t_h: time points [h]
        B_int: interneuron occupancy at each time [0-1]
        params: NRHypo parameters

    Returns:
        Total NRHypo injury burden [dimensionless, cumulative].
    """
    injury_rate = np.array([compute_g(b, params) for b in B_int])
    return float(np.trapz(injury_rate, t_h))
