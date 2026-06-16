"""L3a SD Engine — Protective Arm.

Provides:
    - SD threshold, duration, recovery modulation by B_pyr
    - SD event rate computation
    - SD burden integral
    - Bistable K+ dynamics simulation
"""

from src.l3a_sd.model import (
    SDParams,
    DEFAULT_SD_PARAMS,
    compute_sd_threshold,
    compute_sd_duration,
    compute_sd_recovery,
    compute_sd_rate,
    compute_sd_burden,
    simulate_sd_dynamics,
)
