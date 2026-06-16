"""L2 P_open submodel — voltage and glutamate dependence.

Implements the separable saturating gate:
    P_open,p(V,[Glu]) = σ_p(V) · [Glu]/(K_Glu,p + [Glu])

where:
    σ_p(V) = sigmoidal voltage-relief term (Mg²⁺ block surrogate)
    K_Glu,p = population-specific glutamate EC50

The voltage-relief term σ(V) models Mg²⁺ unblock as a function of
membrane depolarization:
    σ(V) = 1 / (1 + exp(−(V − V_half) / k))

Population differences:
    - Pyramidal: GluN2A-dominant → lower K_Glu, faster kinetics
    - Interneuron: GluN2B-dominant → higher K_Glu, slower kinetics

References:
    - Mayer 1984, Nowak 1984: Mg²⁺ voltage-dependent block
    - Vargas-Caballero & Robinson 2004: asymmetric trapping block model
    - Banke & Traynelis 2003: NMDA receptor gating kinetics
    - Clarke & Johnson 2008: inherent voltage dependence of NMDAR
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class POpenParams:
    """Parameters for the P_open submodel.

    Units:
    - V_half: mV (voltage at half-maximal Mg²⁺ relief)
    - k: mV (slope factor for σ(V))
    - K_Glu: µM (glutamate EC50)
    - P_open_max: dimensionless (maximum open probability)
    """

    # Mg²⁺ block voltage-relief parameters
    # σ(V) = 1 / (1 + exp(-(V - V_half) / k))
    # At rest (-65 mV): σ ≈ 0.01 (strong Mg²⁺ block)
    # At depolarized (-20 mV): σ ≈ 0.8 (relief)
    V_half: dict[str, float] = field(
        default_factory=lambda: {"pyr": -20.0, "int": -25.0}
    )
    k: dict[str, float] = field(
        default_factory=lambda: {"pyr": 12.0, "int": 14.0}
    )

    # Glutamate sensitivity [µM]
    # GluN2A (pyr): K_Glu ≈ 2-3 µM (higher affinity)
    # GluN2B (int): K_Glu ≈ 3-5 µM (lower affinity)
    K_Glu: dict[str, float] = field(
        default_factory=lambda: {"pyr": 2.5, "int": 4.0}
    )

    # Maximum open probability (channel intrinsic)
    # GluN2A: ~0.5, GluN2B: ~0.4 (Banke & Traynelis 2003)
    P_open_max: dict[str, float] = field(
        default_factory=lambda: {"pyr": 0.5, "int": 0.4}
    )


DEFAULT_P_OPEN_PARAMS = POpenParams()


def compute_sigma(V: float, V_half: float, k: float) -> float:
    """Voltage-relief term σ(V) — Mg²⁺ block surrogate.

    Args:
        V: membrane potential [mV]
        V_half: voltage at half-maximal relief [mV]
        k: slope factor [mV]

    Returns:
        σ(V) in [0, 1]. Low at rest (strong Mg²⁺ block),
        high when depolarized (Mg²⁺ expelled).
    """
    return 1.0 / (1.0 + np.exp(-(V - V_half) / k))


def compute_glu_gate(Glu: float, K_Glu: float) -> float:
    """Glutamate gating term — saturating Hill function (n=1).

    Args:
        Glu: extracellular glutamate concentration [µM]
        K_Glu: glutamate EC50 [µM]

    Returns:
        Glu/(K_Glu + Glu) in [0, 1].
    """
    if Glu <= 0:
        return 0.0
    return Glu / (K_Glu + Glu)


def compute_p_open(
    V: float,
    Glu: float,
    params: POpenParams,
    population: str,
) -> float:
    """Compute state-dependent open probability for a population.

    P_open,p(V,[Glu]) = P_open_max_p * σ_p(V) * [Glu]/(K_Glu,p + [Glu])

    Args:
        V: membrane potential [mV]
        Glu: extracellular glutamate [µM]
        params: P_open parameters
        population: "pyr" or "int"

    Returns:
        P_open in [0, P_open_max].
    """
    sigma = compute_sigma(V, params.V_half[population], params.k[population])
    glu_gate = compute_glu_gate(Glu, params.K_Glu[population])
    return params.P_open_max[population] * sigma * glu_gate
