"""L2 NMDAR Occupancy Engine — state-dependent open-channel block.

Provides:
    - P_open submodel (voltage + glutamate dependence)
    - NMDAR block kinetics (dB/dt per population)
    - Steady-state and dynamic occupancy computation
"""

from src.l2_occupancy.p_open import (
    POpenParams,
    DEFAULT_P_OPEN_PARAMS,
    compute_p_open,
    compute_sigma,
    compute_glu_gate,
)
from src.l2_occupancy.model import (
    NMDARParams,
    L2Params,
    DEFAULT_NMDAR_PARAMS,
    DEFAULT_L2_PARAMS,
    POPULATIONS,
    compute_occupancy,
    simulate_occupancy,
)
