"""L2 NMDAR occupancy engine — state-dependent open-channel block.

Implements the use-dependent, voltage-dependent, trapping open-channel
block kinetics per neuronal population (Protocol §3.3, WP2 TechSpec §2).

Governing equation per population p:
    dB_p/dt = k_on_p * C_brain * P_open_p(V,[Glu]) * (1 - B_p) - k_off_p * B_p

where:
    B_p   = fractional NMDAR block (occupancy) in population p
    C_brain = brain ECF concentration from L1 (sum of enantiomer contributions)
    P_open_p = state-dependent open probability (voltage + glutamate gate)
    k_on_p, k_off_p = population-specific association/dissociation constants

Enantiomer-specific affinity (S ≈ 3-4× R) enters via k_on:
    k_on_eff = k_on_S * C_ecf_S + k_on_R * C_ecf_R

Population segregation: pyramidal (GluN2A-dominant) vs interneuron
(GluN2B-dominant) have different k_on, k_off, and P_open profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp

from src.l2_occupancy.p_open import POpenParams, compute_p_open


# Neuronal populations tracked by L2
POPULATIONS = ("pyr", "int")

# Unit conversion: k_on [M^-1 s^-1], k_off [s^-1] → per hour
# ODE solver operates in hours; rate constants must be converted
_S_PER_H = 3600.0  # seconds per hour

# Ketamine molecular weight [g/mol] for concentration conversion
_KETAMINE_MW = 237.73


@dataclass(frozen=True)
class NMDARParams:
    """NMDAR block kinetics parameters per population.

    Literature-derived values (see docs/L2_Parameter_Provenance.md):
    - Temme 2018: Ki S-ketamine ≈ 419 nM, R-ketamine ≈ 497 nM at PCP site
    - MacDonald 1991: k_on ≈ 10^4 M^-1 s^-1 for trapping open-channel blockers
    - Orser 1997: k_off ≈ 0.01-0.1 s^-1 for ketamine
    - S:R affinity ratio ≈ 1.2-1.5 (not 3-4×; protocol notes this for future refinement)
    - GluN2A vs GluN2B: different P_open profiles (Banke & Traynelis 2003)

    Units:
    - k_on: M^-1 s^-1 (association rate)
    - k_off: s^-1 (dissociation rate)
    - KD: M (equilibrium dissociation constant = k_off / k_on)
    """

    # Association rate constants [M^-1 s^-1]
    # S-ketamine has ~1.3× higher affinity than R at PCP site
    k_on_S: dict[str, float] = field(
        default_factory=lambda: {"pyr": 1.2e4, "int": 1.5e4}
    )
    k_on_R: dict[str, float] = field(
        default_factory=lambda: {"pyr": 0.9e4, "int": 1.1e4}
    )

    # Dissociation rate constants [s^-1]
    # Slower k_off = longer trapping = more potent block
    # Interneurons: slightly slower k_off (GluN2B trapping is stronger)
    k_off: dict[str, float] = field(
        default_factory=lambda: {"pyr": 0.05, "int": 0.03}
    )

    # Initial occupancy (resting state, no drug)
    B0: dict[str, float] = field(
        default_factory=lambda: {"pyr": 0.0, "int": 0.0}
    )


# Literature-derived defaults
DEFAULT_NMDAR_PARAMS = NMDARParams()


@dataclass(frozen=True)
class L2Params:
    """Complete L2 parameter set."""
    nmdar: NMDARParams = field(default_factory=NMDARParams)
    p_open: POpenParams = field(default_factory=POpenParams)

    # Resting membrane potential [mV] and glutamate [µM] per population
    # These are default values; L3a/L3b supply dynamic values
    V_rest: dict[str, float] = field(
        default_factory=lambda: {"pyr": -65.0, "int": -65.0}
    )
    Glu_rest: dict[str, float] = field(
        default_factory=lambda: {"pyr": 0.5, "int": 0.5}
    )


DEFAULT_L2_PARAMS = L2Params()


def compute_occupancy(
    c_ecf_s: float,
    c_ecf_r: float,
    params: L2Params,
    V: dict[str, float] | None = None,
    Glu: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute steady-state occupancy for given concentrations.

    For constant C_brain, V, Glu, the steady state is:
        B_ss = (k_on * C_brain * P_open) / (k_on * C_brain * P_open + k_off)

    Args:
        c_ecf_s: S-ketamine brain ECF concentration [mg/L]
        c_ecf_r: R-ketamine brain ECF concentration [mg/L]
        params: L2 parameters
        V: membrane potential per population [mV] (default: resting)
        Glu: glutamate concentration per population [µM] (default: resting)

    Returns:
        Dict mapping population name to steady-state occupancy [0-1].
    """
    if V is None:
        V = params.V_rest
    if Glu is None:
        Glu = params.Glu_rest

    # Convert mg/L to M (ketamine MW = 237.73 g/mol)
    MW = 237.73  # g/mol
    c_s_M = c_ecf_s * 1e3 / MW * 1e-3  # mg/L -> g/L -> mol/L = M
    c_r_M = c_ecf_r * 1e3 / MW * 1e-3

    result = {}
    for pop in POPULATIONS:
        p_open = compute_p_open(V[pop], Glu[pop], params.p_open, pop)

        # Effective association: enantiomer sum
        k_on_eff = (
            params.nmdar.k_on_S[pop] * c_s_M
            + params.nmdar.k_on_R[pop] * c_r_M
        )

        k_off = params.nmdar.k_off[pop]

        # Steady-state occupancy
        if k_on_eff * p_open + k_off > 0:
            B_ss = (k_on_eff * p_open) / (k_on_eff * p_open + k_off)
        else:
            B_ss = 0.0

        result[pop] = np.clip(B_ss, 0.0, 1.0)

    return result


def simulate_occupancy(
    t_h: np.ndarray,
    c_ecf_s: np.ndarray,
    c_ecf_r: np.ndarray,
    params: L2Params,
    V: dict[str, np.ndarray] | None = None,
    Glu: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Simulate dynamic NMDAR occupancy over time.

    Integrates dB_p/dt with time-varying C_brain, V, and Glu inputs.

    Args:
        t_h: time points [h]
        c_ecf_s: S-ketamine brain ECF concentration at each time [mg/L]
        c_ecf_r: R-ketamine brain ECF concentration at each time [mg/L]
        params: L2 parameters
        V: membrane potential per population at each time [mV]
        Glu: glutamate per population at each time [µM]

    Returns:
        Dict mapping population name to occupancy time course [0-1].
    """
    n_t = len(t_h)

    # Default: constant resting values
    if V is None:
        V = {pop: np.full(n_t, params.V_rest[pop]) for pop in POPULATIONS}
    if Glu is None:
        Glu = {pop: np.full(n_t, params.Glu_rest[pop]) for pop in POPULATIONS}

    # Convert mg/L to M
    MW = 237.73
    c_s_M = c_ecf_s * 1e3 / MW * 1e-3
    c_r_M = c_ecf_r * 1e3 / MW * 1e-3

    # Interpolation functions
    def c_s_interp(t):
        return np.interp(t, t_h, c_s_M)

    def c_r_interp(t):
        return np.interp(t, t_h, c_r_M)

    V_interp = {pop: lambda t, p=pop: np.interp(t, t_h, V[p]) for pop in POPULATIONS}
    Glu_interp = {pop: lambda t, p=pop: np.interp(t, t_h, Glu[p]) for pop in POPULATIONS}

    # Initial conditions
    y0 = np.array([params.nmdar.B0[pop] for pop in POPULATIONS])

    def rhs(t, y):
        dy = np.zeros(2)
        c_s = c_s_interp(t)
        c_r = c_r_interp(t)

        for i, pop in enumerate(POPULATIONS):
            p_open = compute_p_open(
                V_interp[pop](t), Glu_interp[pop](t), params.p_open, pop
            )
            k_on_eff = (
                params.nmdar.k_on_S[pop] * c_s
                + params.nmdar.k_on_R[pop] * c_r
            )
            # Convert from s^-1 to h^-1 for ODE solver
            k_on_eff_h = k_on_eff * _S_PER_H
            k_off_h = params.nmdar.k_off[pop] * _S_PER_H
            B = y[i]

            dy[i] = k_on_eff_h * p_open * (1 - B) - k_off_h * B

        return dy

    t_span = (t_h[0], t_h[-1])
    sol = solve_ivp(rhs, t_span, y0, method="LSODA", t_eval=t_h,
                    rtol=1e-8, atol=1e-10)

    return {pop: sol.y[i] for i, pop in enumerate(POPULATIONS)}
