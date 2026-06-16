"""L3b NRHypo Engine — Toxic Arm.

Provides:
    - Disinhibition computation from B_int
    - Excitatory drive modulation
    - Convex injury gain function g(B_int)
    - NRHypo injury index accumulation
    - Glutamate dynamics simulation
"""

from src.l3b_nrhypo.model import (
    NRHypoParams,
    DEFAULT_NRHYPPO_PARAMS,
    compute_disinhibition,
    compute_excitatory_drive,
    compute_g,
    compute_nrhypo_injury_rate,
    compute_nrhypo_burden,
    simulate_nrhypo,
)
