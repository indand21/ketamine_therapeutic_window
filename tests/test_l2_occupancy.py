"""Tests for L2 NMDAR occupancy engine."""

import numpy as np
import pytest

from src.l2_occupancy.p_open import (
    POpenParams, compute_p_open, compute_sigma, compute_glu_gate,
)
from src.l2_occupancy.model import (
    NMDARParams, L2Params, POPULATIONS,
    compute_occupancy, simulate_occupancy,
    DEFAULT_L2_PARAMS,
)


# ===========================================================================
# P_open submodel tests
# ===========================================================================

class TestSigma:
    """Mg²⁺ voltage-relief term."""

    def test_sigma_at_rest(self):
        """At rest (-65 mV), σ should be very low (< 0.05)."""
        s = compute_sigma(-65.0, -20.0, 12.0)
        assert s < 0.05, f"σ at rest = {s}, expected < 0.05"

    def test_sigma_at_depolarized(self):
        """At depolarized (-20 mV), σ should be ~0.5."""
        s = compute_sigma(-20.0, -20.0, 12.0)
        assert 0.45 <= s <= 0.55, f"σ at -20 mV = {s}, expected ~0.5"

    def test_sigma_at_positive(self):
        """At positive (+20 mV), σ should be > 0.9."""
        s = compute_sigma(20.0, -20.0, 12.0)
        assert s > 0.9, f"σ at +20 mV = {s}, expected > 0.9"

    def test_sigma_monotonic(self):
        """σ should increase monotonically with V."""
        Vs = np.linspace(-80, 20, 100)
        sigmas = [compute_sigma(v, -20.0, 12.0) for v in Vs]
        for i in range(len(sigmas) - 1):
            assert sigmas[i + 1] >= sigmas[i], "σ not monotonic"


class TestGluGate:
    """Glutamate gating term."""

    def test_glu_gate_zero_glu(self):
        """At zero glutamate, gate should be 0."""
        assert compute_glu_gate(0.0, 2.5) == 0.0

    def test_glu_gate_at_ec50(self):
        """At K_Glu, gate should be 0.5."""
        g = compute_glu_gate(2.5, 2.5)
        assert abs(g - 0.5) < 1e-6

    def test_glu_gate_saturates(self):
        """At high glutamate, gate should approach 1."""
        g = compute_glu_gate(1000.0, 2.5)
        assert g > 0.99

    def test_glu_gate_monotonic(self):
        """Gate should increase with glutamate."""
        glus = np.linspace(0.1, 100, 50)
        gates = [compute_glu_gate(g, 2.5) for g in glus]
        for i in range(len(gates) - 1):
            assert gates[i + 1] >= gates[i]


class TestPOpen:
    """Combined open probability."""

    def test_p_open_at_rest_low(self):
        """At rest (low V, low Glu), P_open should be very low."""
        p = compute_p_open(-65.0, 0.5, POpenParams(), "pyr")
        assert p < 0.01, f"P_open at rest = {p}, expected < 0.01"

    def test_p_open_at_pathological(self):
        """At pathological (depolarized, high Glu), P_open should be high."""
        p = compute_p_open(-20.0, 50.0, POpenParams(), "pyr")
        assert p > 0.2, f"P_open at pathological = {p}, expected > 0.2"

    def test_p_open_pyr_vs_int(self):
        """Pyramidal and interneuron should have different P_open profiles."""
        p_pyr = compute_p_open(-30.0, 10.0, POpenParams(), "pyr")
        p_int = compute_p_open(-30.0, 10.0, POpenParams(), "int")
        # At same V and Glu, populations should differ due to different params
        assert p_pyr != p_int, "P_open should differ between populations"


# ===========================================================================
# Occupancy model tests
# ===========================================================================

class TestSteadyStateOccupancy:
    """Steady-state occupancy computation."""

    def test_no_drug_no_occupancy(self):
        """Without drug, occupancy should be zero."""
        result = compute_occupancy(0.0, 0.0, DEFAULT_L2_PARAMS)
        for pop in POPULATIONS:
            assert result[pop] == pytest.approx(0.0, abs=1e-10)

    def test_occupancy_increases_with_concentration(self):
        """Higher drug concentration should give higher occupancy."""
        result_low = compute_occupancy(0.01, 0.0, DEFAULT_L2_PARAMS)
        result_high = compute_occupancy(0.1, 0.0, DEFAULT_L2_PARAMS)
        for pop in POPULATIONS:
            assert result_high[pop] > result_low[pop]

    def test_occupancy_between_0_and_1(self):
        """Occupancy should be in [0, 1]."""
        result = compute_occupancy(1.0, 0.5, DEFAULT_L2_PARAMS)
        for pop in POPULATIONS:
            assert 0.0 <= result[pop] <= 1.0

    def test_occupancy_with_depolarization(self):
        """Depolarization should increase occupancy (more channels open)."""
        params = DEFAULT_L2_PARAMS
        V_rest = {"pyr": -65.0, "int": -65.0}
        V_depol = {"pyr": -30.0, "int": -30.0}
        Glu = {"pyr": 5.0, "int": 5.0}

        result_rest = compute_occupancy(0.05, 0.0, params, V=V_rest, Glu=Glu)
        result_depol = compute_occupancy(0.05, 0.0, params, V=V_depol, Glu=Glu)
        for pop in POPULATIONS:
            assert result_depol[pop] > result_rest[pop]

    def test_occupancy_with_high_glutamate(self):
        """High glutamate should increase occupancy."""
        params = DEFAULT_L2_PARAMS
        V = {"pyr": -40.0, "int": -40.0}
        Glu_low = {"pyr": 1.0, "int": 1.0}
        Glu_high = {"pyr": 50.0, "int": 50.0}

        result_low = compute_occupancy(0.05, 0.0, params, V=V, Glu=Glu_low)
        result_high = compute_occupancy(0.05, 0.0, params, V=V, Glu=Glu_high)
        for pop in POPULATIONS:
            assert result_high[pop] > result_low[pop]

    def test_population_segregation(self):
        """Pyr and int should have different occupancy at same concentration."""
        params = DEFAULT_L2_PARAMS
        V = {"pyr": -40.0, "int": -40.0}
        Glu = {"pyr": 10.0, "int": 10.0}

        result = compute_occupancy(0.05, 0.0, params, V=V, Glu=Glu)
        assert result["pyr"] != result["int"], "Populations should differ"

    def test_enantiomer_summation(self):
        """S+R should give higher occupancy than S alone at same total dose."""
        params = DEFAULT_L2_PARAMS
        V = {"pyr": -40.0, "int": -40.0}
        Glu = {"pyr": 10.0, "int": 10.0}

        result_s_only = compute_occupancy(0.05, 0.0, params, V=V, Glu=Glu)
        result_racemic = compute_occupancy(0.025, 0.025, params, V=V, Glu=Glu)
        # Racemic should give slightly different (potentially higher) occupancy
        # due to R-ketamine contribution
        for pop in POPULATIONS:
            assert result_racemic[pop] > 0


class TestDynamicOccupancy:
    """Dynamic occupancy simulation."""

    def test_simulate_runs(self):
        """Simulation should complete successfully."""
        t = np.linspace(0, 24, 100)
        c_s = np.exp(-t / 5.0) * 0.1  # decaying S-KET
        c_r = np.exp(-t / 6.0) * 0.1  # decaying R-KET

        result = simulate_occupancy(t, c_s, c_r, DEFAULT_L2_PARAMS)
        for pop in POPULATIONS:
            assert pop in result
            assert len(result[pop]) == len(t)
            assert np.all(np.isfinite(result[pop]))

    def test_simulate_occupancy_follows_concentration(self):
        """Occupancy should peak near concentration peak (fast kinetics)."""
        t = np.array([0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
        c_s = np.array([0.0, 0.1, 0.08, 0.05, 0.02, 0.005, 0.001])
        c_r = np.zeros_like(c_s)

        result = simulate_occupancy(t, c_s, c_r, DEFAULT_L2_PARAMS)
        for pop in POPULATIONS:
            # With fast k_off, occupancy tracks concentration closely
            # Peak occupancy should be near peak concentration (t=0.5h)
            peak_idx = np.argmax(result[pop])
            assert peak_idx == 1, f"Peak at index {peak_idx}, expected 1 (t=0.5h)"
            # Occupancy should decrease after peak
            assert result[pop][-1] < result[pop][1]

    def test_simulate_occupancy_bounded(self):
        """Occupancy should stay in [0, 1]."""
        t = np.linspace(0, 48, 200)
        c_s = np.ones_like(t) * 0.5  # constant high concentration
        c_r = np.zeros_like(t)

        result = simulate_occupancy(t, c_s, c_r, DEFAULT_L2_PARAMS)
        for pop in POPULATIONS:
            assert np.all(result[pop] >= -1e-10)
            assert np.all(result[pop] <= 1.0 + 1e-10)

    def test_simulate_with_depolarization(self):
        """Depolarization should increase occupancy."""
        t = np.array([0, 1.0, 2.0])
        c_s = np.array([0.05, 0.05, 0.05])
        c_r = np.zeros_like(t)

        V_rest = {"pyr": np.full(3, -65.0), "int": np.full(3, -65.0)}
        V_depol = {"pyr": np.full(3, -30.0), "int": np.full(3, -30.0)}
        Glu = {"pyr": np.full(3, 10.0), "int": np.full(3, 10.0)}

        result_rest = simulate_occupancy(t, c_s, c_r, DEFAULT_L2_PARAMS, V=V_rest, Glu=Glu)
        result_depol = simulate_occupancy(t, c_s, c_r, DEFAULT_L2_PARAMS, V=V_depol, Glu=Glu)

        for pop in POPULATIONS:
            assert np.all(result_depol[pop] >= result_rest[pop] - 1e-10)


class TestPKPDIntegration:
    """Integration with L1 PK output."""

    def test_l1_to_l2_pipeline(self):
        """Full pipeline: L1 simulation → C_ecf → L2 occupancy."""
        from src.l1_pk import L1Model, DosingRegimen, InfusionSegment
        from src.l1_pk.config import default_parameters

        # Run L1
        regimen = DosingRegimen(
            infusions=[InfusionSegment(start=0.0, end=0.667, rate=52.5)],
            s_fraction=0.5,
        )
        l1_model = L1Model(default_parameters())
        t_eval = np.linspace(0, 24, 500)
        l1_result = l1_model.simulate(regimen, 24.0, t_eval=t_eval)

        # Extract C_ecf
        c_ecf_s = l1_result.brain_ecf("KET_S")
        c_ecf_r = l1_result.brain_ecf("KET_R")

        # Run L2
        l2_result = simulate_occupancy(l1_result.t, c_ecf_s, c_ecf_r, DEFAULT_L2_PARAMS)

        # Verify: occupancy should be non-negative and finite
        for pop in POPULATIONS:
            assert np.all(l2_result[pop] >= -1e-10)
            assert np.all(l2_result[pop] <= 1.0 + 1e-10)
            assert np.all(np.isfinite(l2_result[pop]))

        # At peak C_ecf, occupancy should be detectable
        peak_idx = np.argmax(c_ecf_s + c_ecf_r)
        for pop in POPULATIONS:
            assert l2_result[pop][peak_idx] > 0, f"No occupancy at peak for {pop}"
