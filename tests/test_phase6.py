"""Tests for Phase 6 — Sensitivity, Virtual Population, Emergence, Identifiability."""

import numpy as np
import pytest

from src.phase6.virtual_population import (
    VirtualSubject, VirtualPopulation,
    generate_virtual_population, CYP2B6_ALLELE_FREQ, CYP2B6_CL_SCALE,
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
    analyze_fim,
)


# ===========================================================================
# Virtual population tests
# ===========================================================================

class TestVirtualPopulation:
    """Virtual population generation."""

    def test_generate_population(self):
        """Should generate the requested number of subjects."""
        pop = generate_virtual_population(n_subjects=100, seed=42)
        assert pop.n_subjects == 100
        assert len(pop.subjects) == 100

    def test_subjects_have_required_fields(self):
        """Each subject should have all required fields."""
        pop = generate_virtual_population(n_subjects=10, seed=42)
        for s in pop.subjects:
            assert isinstance(s, VirtualSubject)
            assert 40 <= s.weight <= 120
            assert s.cyp2b6_genotype in CYP2B6_CL_SCALE
            assert 0.0 <= s.injury_severity <= 1.0
            assert 18 <= s.age <= 80
            assert 0.0 <= s.gaba_reserve <= 1.0
            assert 0.0 <= s.sd_susceptibility <= 1.0
            assert 0.0 <= s.nrhypo_sensitivity <= 1.0
            assert s.pk_cl_variability > 0
            assert s.pk_vd_variability > 0

    def test_cyp2b6_frequency(self):
        """CYP2B6*6 frequency should be ~38% in the population."""
        pop = generate_virtual_population(n_subjects=10000, seed=42)
        genotypes = [s.cyp2b6_genotype for s in pop.subjects]
        n_66 = sum(1 for g in genotypes if g == "*6/*6")
        freq_66 = n_66 / len(genotypes)
        # *6/*6 frequency should be ~0.38^2 = 0.1444
        assert 0.10 < freq_66 < 0.20, f"*6/*6 freq = {freq_66:.3f}"

    def test_age_gaba_correlation(self):
        """GABAergic reserve should decline with age."""
        pop = generate_virtual_population(n_subjects=1000, seed=42)
        ages = np.array([s.age for s in pop.subjects])
        gabas = np.array([s.gaba_reserve for s in pop.subjects])
        correlation = np.corrcoef(ages, gabas)[0, 1]
        assert correlation < 0, f"Age-GABA correlation = {correlation:.2f}, expected < 0"

    def test_injury_sd_correlation(self):
        """SD susceptibility should increase with injury severity."""
        pop = generate_virtual_population(n_subjects=1000, seed=42)
        injuries = np.array([s.injury_severity for s in pop.subjects])
        sd_susc = np.array([s.sd_susceptibility for s in pop.subjects])
        correlation = np.corrcoef(injuries, sd_susc)[0, 1]
        assert correlation > 0, f"Injury-SD correlation = {correlation:.2f}, expected > 0"

    def test_reproducibility(self):
        """Same seed should produce same population."""
        pop1 = generate_virtual_population(n_subjects=50, seed=42)
        pop2 = generate_virtual_population(n_subjects=50, seed=42)
        for s1, s2 in zip(pop1.subjects, pop2.subjects):
            assert s1.weight == s2.weight
            assert s1.cyp2b6_genotype == s2.cyp2b6_genotype


# ===========================================================================
# Emergence tests
# ===========================================================================

class TestEmergence:
    """Emergence test suite."""

    def test_emergence_tests_run(self):
        """All emergence tests should complete."""
        results = run_emergence_test()
        assert len(results) == 3
        for r in results:
            assert isinstance(r, EmergenceResult)
            assert r.test_name

    def test_independence_audit(self):
        """B_int should be > B_pyr at same concentration (interneurons more sensitive)."""
        results = run_emergence_test()
        indep = [r for r in results if "Independence" in r.test_name][0]
        assert indep.passed, indep.details

    def test_u_shape_audit(self):
        """U-shape audit should complete and report results."""
        result = run_u_shape_audit()
        assert "u_shape_found" in result
        assert "optimal_dose" in result
        assert "injuries" in result


# ===========================================================================
# Identifiability tests
# ===========================================================================

class TestIdentifiability:
    """Structural and practical identifiability."""

    def test_structural_identifiability(self):
        """Should classify parameters by observability."""
        param_names = [
            "CL_met_NK_S", "V_cen", "k_on_S_pyr", "B_int_thresh",
        ]
        results = check_structural_identifiability(param_names)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, IdentifiabilityResult)

    def test_well_observed_identifiable(self):
        """PK parameters should be structurally identifiable."""
        results = check_structural_identifiability(["CL_met_NK_S", "V_per"])
        for r in results:
            assert r.structurally_identifiable

    def test_fim_computation(self):
        """FIM should be a positive semi-definite matrix."""
        # Simple test: linear model y = a*x + b
        def linear_model(params):
            x = np.linspace(0, 1, 10)
            return params[0] * x + params[1]

        params = np.array([2.0, 1.0])
        FIM = compute_fim(params, ["a", "b"], linear_model)
        assert FIM.shape == (2, 2)
        # Should be positive semi-definite
        eigenvalues = np.linalg.eigvalsh(FIM)
        assert np.all(eigenvalues >= -1e-10)

    def test_fim_analysis(self):
        """FIM analysis should return condition number and eigenvalues."""
        FIM = np.array([[2.0, 0.5], [0.5, 1.0]])
        result = analyze_fim(FIM, ["a", "b"])
        assert "condition_number" in result
        assert "eigenvalues" in result
        assert result["identifiable"]  # 2x2 well-conditioned


# ===========================================================================
# Sensitivity analysis (SALib) tests
# ===========================================================================

class TestSensitivity:
    """Sensitivity analysis (requires SALib)."""

    def test_salib_available(self):
        """SALib should be importable."""
        try:
            from SALib.sample import saltelli
            from SALib.analyze import sobol
            assert True
        except ImportError:
            pytest.skip("SALib not installed")
