"""L4 Injury Balance + L5 Clinical Overlay.

Provides:
    - Net injury equation (protection vs toxicity)
    - Clinical constraints (CPP/MAP, ICP, psychotomimetic)
    - Therapeutic window computation
"""

from src.l4_l5.injury import (
    InjuryParams,
    ClinicalParams,
    DEFAULT_INJURY_PARAMS,
    DEFAULT_CLINICAL_PARAMS,
    compute_excitotoxic_flux,
    compute_injury_rate,
    simulate_injury,
    compute_clinical_constraints,
    compute_therapeutic_window,
)
