"""Phase 6 — Sensitivity Analysis, Virtual Population, Emergence Test.

Provides:
    - Global sensitivity analysis (Sobol/Morris) via SALib
    - Virtual population generation (CYP2B6, injury severity, age)
    - Emergence test (ablate L3b, perturb L2, U-shape audit)
    - Structural/practical identifiability checks
"""

from src.phase6.sensitivity import (
    SensitivityResult,
    run_sobol_analysis,
    run_morris_analysis,
    qsp_forward,
)
from src.phase6.virtual_population import (
    VirtualSubject,
    VirtualPopulation,
    generate_virtual_population,
    CYP2B6_ALLELE_FREQ,
)
from src.phase6.emergence import (
    EmergenceResult,
    run_emergence_test,
    run_u_shape_audit,
)
from src.phase6.identifiability import (
    IdentifiabilityResult,
    check_structural_identifiability,
    compute_fim,
)
