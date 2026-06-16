"""L3a Spreading Depolarization (SD) Engine — Protective Arm.

Implements the reduced bistable reaction-diffusion formulation for SD
susceptibility and propagation (WP2 TechSpec §3, Protocol §3.4).

The model tracks extracellular potassium [K+]_o dynamics with a bistable
switch that captures the SD bifurcation structure. NMDAR block (B_pyr)
modulates:
    - SD initiation threshold: K_thr(B_pyr) = K_thr^0 * (1 + κ * B_pyr)
    - SD duration: D_SD(B_pyr) = D_SD^0 / (1 + κ_D * B_pyr)
    - Recovery time: τ_rec(B_pyr) = τ_rec^0 / (1 + κ_rec * B_pyr)

References:
    - Carlson 2018/2019: SD suppression at >1.15 mg/kg/h ketamine
    - Sánchez-Porras 2022: swine SD electrophysiology
    - Hübel & Dahlem 2014: ion-based SD models
    - Dreier 2011: SD in acute brain injury
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class SDParams:
    """Parameters for the reduced bistable SD model.

    Units:
    - K_thr: mM (extracellular K+ threshold for SD initiation)
    - K_rest: mM (resting extracellular K+)
    - K_peak: mM (peak K+ during SD)
    - D_SD: min (SD depolarization duration)
    - τ_rec: min (recovery time constant)
    - λ_SD: events/h (baseline SD event rate in injury)
    - κ: dimensionless (B_pyr gain on threshold)
    - κ_D: dimensionless (B_pyr gain on duration)
    - κ_rec: dimensionless (B_pyr gain on recovery)
    - D_K: mm²/min (K+ diffusion coefficient)
    """

    # SD initiation threshold parameters
    K_thr_0: float = 12.0      # baseline SD threshold [mM K+]
    K_rest: float = 3.0        # resting extracellular K+ [mM]
    K_peak: float = 60.0       # peak K+ during SD [mM]

    # SD timing parameters
    D_SD_0: float = 1.0        # baseline SD depolarization duration [min]
    tau_rec_0: float = 5.0     # baseline recovery time constant [min]

    # SD rate parameters (injury context)
    lambda_SD_0: float = 2.0   # baseline SD event rate in injury [events/h]

    # B_pyr coupling gains
    kappa: float = 5.0         # threshold elevation gain (K_thr modulation)
    kappa_D: float = 2.0       # duration shortening gain
    kappa_rec: float = 1.0     # recovery acceleration gain

    # Spatial diffusion (for future 1D/2D extension)
    D_K: float = 2.5           # K+ diffusion coefficient [mm²/min]

    # Bistable switch parameters
    # dK/dt = f(K) where f is a bistable function with stable states at
    # K_rest and K_peak, modulated by B_pyr
    k_prod: float = 0.1        # K+ production rate at rest [mM/min]
    k_clear: float = 0.05      # K+ clearance rate [1/min]
    k_leak: float = 0.02       # K+ leak rate [mM/min]


DEFAULT_SD_PARAMS = SDParams()


def compute_sd_threshold(B_pyr: float, params: SDParams) -> float:
    """SD initiation threshold as function of pyramidal occupancy.

    K_thr(B_pyr) = K_thr_0 * (1 + κ * B_pyr)

    NMDAR block RAISES the threshold → harder to initiate SD → protective.

    Args:
        B_pyr: pyramidal NMDAR occupancy [0-1]
        params: SD parameters

    Returns:
        SD threshold [mM K+]
    """
    return params.K_thr_0 * (1.0 + params.kappa * B_pyr)


def compute_sd_duration(B_pyr: float, params: SDParams) -> float:
    """SD depolarization duration as function of B_pyr.

    D_SD(B_pyr) = D_SD_0 / (1 + κ_D * B_pyr)

    NMDAR block SHORTENS SD duration → less tissue damage.

    Args:
        B_pyr: pyramidal NMDAR occupancy [0-1]
        params: SD parameters

    Returns:
        SD duration [min]
    """
    return params.D_SD_0 / (1.0 + params.kappa_D * B_pyr)


def compute_sd_recovery(B_pyr: float, params: SDParams) -> float:
    """Recovery time constant as function of B_pyr.

    τ_rec(B_pyr) = τ_rec_0 / (1 + κ_rec * B_pyr)

    NMDAR block ACCELERATES recovery → faster return to baseline.

    Args:
        B_pyr: pyramidal NMDAR occupancy [0-1]
        params: SD parameters

    Returns:
        Recovery time constant [min]
    """
    return params.tau_rec_0 / (1.0 + params.kappa_rec * B_pyr)


def compute_sd_rate(B_pyr: float, params: SDParams) -> float:
    """SD event rate as function of B_pyr.

    λ_SD(B_pyr) = λ_SD_0 * max(0, 1 - B_pyr / B_pyr_crit)

    Where B_pyr_crit is the occupancy needed to fully suppress SD.
    Derived from: K_thr(B_pyr_crit) = K_peak (threshold exceeds peak possible K+).

    Args:
        B_pyr: pyramidal NMDAR occupancy [0-1]
        params: SD parameters

    Returns:
        SD event rate [events/h]
    """
    # Critical occupancy: threshold exceeds peak possible K+
    B_pyr_crit = (params.K_peak / params.K_thr_0 - 1.0) / params.kappa
    B_pyr_crit = np.clip(B_pyr_crit, 0.0, 1.0)

    if B_pyr_crit <= 0:
        return params.lambda_SD_0

    suppression = max(0.0, 1.0 - B_pyr / B_pyr_crit)
    return params.lambda_SD_0 * suppression


def compute_sd_burden(
    t_h: np.ndarray,
    B_pyr: np.ndarray,
    params: SDParams,
) -> dict:
    """Compute SD burden integral over time.

    SD burden = ∫ λ_SD(B_pyr(t)) * D_SD(B_pyr(t)) dt

    Also returns:
    - λ_SD(t): SD event rate time course
    - D_SD(t): SD duration time course
    - τ_rec(t): recovery time time course
    - K_thr(t): threshold time course

    Args:
        t_h: time points [h]
        B_pyr: pyramidal occupancy at each time [0-1]
        params: SD parameters

    Returns:
        Dict with burden integral and time courses.
    """
    lambda_t = np.array([compute_sd_rate(b, params) for b in B_pyr])
    D_t = np.array([compute_sd_duration(b, params) for b in B_pyr])
    tau_t = np.array([compute_sd_recovery(b, params) for b in B_pyr])
    K_thr_t = np.array([compute_sd_threshold(b, params) for b in B_pyr])

    # Burden integrand: λ_SD * D_SD [events·min/h]
    integrand = lambda_t * D_t
    burden = float(np.trapz(integrand, t_h))

    return {
        "burden": burden,
        "lambda_SD": lambda_t,
        "D_SD": D_t,
        "tau_rec": tau_t,
        "K_thr": K_thr_t,
        "t": t_h,
    }


def simulate_sd_dynamics(
    t_h: np.ndarray,
    B_pyr: np.ndarray,
    params: SDParams,
    K0: float | None = None,
) -> dict:
    """Simulate extracellular K+ dynamics with bistable SD model.

    The ODE: dK/dt = f(K, B_pyr) where f captures:
    - Basal K+ production and clearance
    - Bistable switch for SD events (K+ avalanche)
    - B_pyr modulation of the threshold

    Args:
        t_h: time points [h]
        B_pyr: pyramidal occupancy at each time [0-1]
        params: SD parameters
        K0: initial K+ [mM] (default: K_rest)

    Returns:
        Dict with K+(t), SD events, and SD descriptors.
    """
    if K0 is None:
        K0 = params.K_rest

    # Interpolate B_pyr
    def B_interp(t):
        return np.interp(t, t_h, B_pyr)

    def rhs(t, y):
        K = y[0]
        B = B_interp(t)
        K_thr = compute_sd_threshold(B, params)

        # Bistable dynamics:
        # When K < K_thr: slow production, normal clearance
        # When K > K_thr: rapid K+ avalanche (SD event)
        # Recovery: slow return to K_rest after SD

        if K < K_thr:
            # Pre-SD: slow dynamics
            dK = params.k_prod - params.k_clear * (K - params.K_rest)
        else:
            # SD event: rapid K+ rise toward K_peak
            # Bistable switch: positive feedback
            dK = params.k_prod + 0.5 * (params.K_peak - K) * (K - K_thr) / params.K_peak
            # B_pyr reduces the SD amplitude
            dK *= (1.0 - 0.5 * B)

        return [dK]

    t_span = (t_h[0], t_h[-1])
    sol = solve_ivp(rhs, t_span, [K0], method="LSODA", t_eval=t_h,
                    rtol=1e-8, atol=1e-10)

    K_t = sol.y[0]

    # Detect SD events: K+ crossings above threshold
    sd_events = []
    in_sd = False
    for i in range(len(t_h)):
        B = B_interp(t_h[i])
        K_thr = compute_sd_threshold(B, params)
        if K_t[i] > K_thr and not in_sd:
            in_sd = True
            sd_events.append({"onset": t_h[i], "peak_K": K_t[i]})
        elif K_t[i] < K_thr and in_sd:
            in_sd = False
            if sd_events:
                sd_events[-1]["offset"] = t_h[i]
                sd_events[-1]["duration"] = t_h[i] - sd_events[-1]["onset"]

    # Compute SD descriptors
    burden = compute_sd_burden(t_h, B_pyr, params)

    return {
        "K": K_t,
        "sd_events": sd_events,
        "n_sd_events": len(sd_events),
        **burden,
    }
