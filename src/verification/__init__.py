"""Verification utilities for the L1 engine.

Mass-balance, unit-consistency, and solver-convergence helpers per
``docs/WP1_TechSpec_PBPK_PK_PGx.md`` §7 and Protocol §10.
"""

from src.verification.checks import (
    total_drug_mass,
    mass_conservation_residual,
    check_unit_consistency,
)

__all__ = [
    "total_drug_mass",
    "mass_conservation_residual",
    "check_unit_consistency",
]
