"""Tests for L3b NRHypo Engine — Toxic Arm."""

import numpy as np
import pytest

from src.l3b_nrhypo.model import (
    NRHypoParams, DEFAULT_NRHYPPO_PARAMS,
    compute_disinhibition, compute_excitatory_drive,
    compute_g, compute_nrhypo_injury_rate,
    compute_nrhypo_burden, simulate_nrhypo,
)


# ===========================================================================
# Disinhibition tests
# ===========================================================================

class TestDisinhibition:
    """Disinhibition from interneuron block."""

    def test_no_block_no_disinhibition(self):
        """Without interneuron block, disinhibition should be zero."""
        d = compute_disinhibition(0.0, DEFAULT_NRHYPPO_PARAMS)
        assert d == 0.0

    def test_disinhibition_increases_with_B_int(self):
        """Higher B_int should give more disinhibition."""
        d_low = compute_disinhibition(0.2, DEFAULT_NRHYPPO_PARAMS)
        d_high = compute_disinhibition(0.8, DEFAULT_NRHYPPO_PARAMS)
        assert d_high > d_low

    def test_disinhibition_bounded(self):
        """Disinhibition should be in [0, I_GABA_0]."""
        for B in np.linspace(0, 1, 50):
            d = compute_disinhibition(B, DEFAULT_NRHYPPO_PARAMS)
            assert 0.0 <= d <= DEFAULT_NRHYPPO_PARAMS.I_GABA_0


class TestExcitatoryDrive:
    """Excitatory drive modulation."""

    def test_baseline_drive(self):
        """At zero B_int, drive should be at baseline."""
        E = compute_excitatory_drive(0.0, DEFAULT_NRHYPPO_PARAMS)
        assert E == pytest.approx(DEFAULT_NRHYPPO_PARAMS.E_pyr_0)

    def test_drive_increases_with_B_int(self):
        """Higher B_int should increase excitatory drive."""
        E_low = compute_excitatory_drive(0.0, DEFAULT_NRHYPPO_PARAMS)
        E_high = compute_excitatory_drive(0.8, DEFAULT_NRHYPPO_PARAMS)
        assert E_high > E_low


# ===========================================================================
# Convex gain function tests
# ===========================================================================

class TestGainFunction:
    """Convex injury gain function g(B_int)."""

    def test_g_zero_below_threshold(self):
        """Below threshold, g should be zero."""
        for B in np.linspace(0, DEFAULT_NRHYPPO_PARAMS.B_int_thresh, 10):
            g = compute_g(B, DEFAULT_NRHYPPO_PARAMS)
            assert g == 0.0

    def test_g_positive_above_threshold(self):
        """Above threshold, g should be positive."""
        B = DEFAULT_NRHYPPO_PARAMS.B_int_thresh + 0.1
        g = compute_g(B, DEFAULT_NRHYPPO_PARAMS)
        assert g > 0.0

    def test_g_convex(self):
        """g should be convex (accelerating) above threshold."""
        B_vals = np.linspace(
            DEFAULT_NRHYPPO_PARAMS.B_int_thresh,
            1.0, 50
        )
        g_vals = [compute_g(b, DEFAULT_NRHYPPO_PARAMS) for b in B_vals]
        # Second derivative should be positive (convex)
        for i in range(1, len(g_vals) - 1):
            d2g = g_vals[i+1] - 2*g_vals[i] + g_vals[i-1]
            assert d2g >= -1e-10, "g should be convex"

    def test_g_non_negative(self):
        """g should never be negative."""
        for B in np.linspace(0, 1, 100):
            g = compute_g(B, DEFAULT_NRHYPPO_PARAMS)
            assert g >= 0.0


# ===========================================================================
# NRHypo simulation tests
# ===========================================================================

class TestNRHypoSimulation:
    """NRHypo injury simulation."""

    def test_simulation_runs(self):
        """Simulation should complete successfully."""
        t = np.linspace(0, 24, 500)
        B_int = np.ones_like(t) * 0.5
        result = simulate_nrhypo(t, B_int, DEFAULT_NRHYPPO_PARAMS)
        assert "Glu" in result
        assert "T_NRHypo" in result
        assert len(result["Glu"]) == len(t)
        assert np.all(np.isfinite(result["Glu"]))
        assert np.all(np.isfinite(result["T_NRHypo"]))

    def test_glu_starts_at_rest(self):
        """Glutamate should start at resting level."""
        t = np.linspace(0, 1, 100)
        B_int = np.zeros_like(t)
        result = simulate_nrhypo(t, B_int, DEFAULT_NRHYPPO_PARAMS)
        assert result["Glu"][0] == pytest.approx(DEFAULT_NRHYPPO_PARAMS.Glu_0)

    def test_nrhypo_zero_without_drug(self):
        """Without interneuron block, NRHypo injury should be zero."""
        t = np.linspace(0, 24, 500)
        B_int = np.zeros_like(t)
        result = simulate_nrhypo(t, B_int, DEFAULT_NRHYPPO_PARAMS)
        assert result["T_NRHypo_final"] == pytest.approx(0.0, abs=1e-6)

    def test_nrhypo_increases_with_sustained_block(self):
        """Sustained interneuron block should accumulate injury."""
        t = np.linspace(0, 24, 500)
        B_int = np.ones_like(t) * 0.5  # sustained 50% block
        result = simulate_nrhypo(t, B_int, DEFAULT_NRHYPPO_PARAMS)
        assert result["T_NRHypo_final"] > 0.0

    def test_nrhypo_dose_response(self):
        """Higher B_int should give more injury."""
        t = np.linspace(0, 24, 500)
        burdens = []
        for B_level in [0.0, 0.2, 0.4, 0.6, 0.8]:
            B_int = np.ones_like(t) * B_level
            burden = compute_nrhypo_burden(t, B_int, DEFAULT_NRHYPPO_PARAMS)
            burdens.append(burden)
        # Should be monotonically increasing
        for i in range(len(burdens) - 1):
            assert burdens[i + 1] >= burdens[i]

    def test_glu_elevated_with_block(self):
        """Glutamate should be elevated with sustained block."""
        t = np.linspace(0, 24, 500)
        B_int = np.ones_like(t) * 0.5
        result = simulate_nrhypo(t, B_int, DEFAULT_NRHYPPO_PARAMS)
        assert result["Glu_max_achieved"] > DEFAULT_NRHYPPO_PARAMS.Glu_0

    def test_emergence_test(self):
        """Ablating L3b (B_int=0) should eliminate toxic arm."""
        t = np.linspace(0, 24, 500)
        B_int_zero = np.zeros_like(t)
        B_int_high = np.ones_like(t) * 0.8

        result_zero = simulate_nrhypo(t, B_int_zero, DEFAULT_NRHYPPO_PARAMS)
        result_high = simulate_nrhypo(t, B_int_high, DEFAULT_NRHYPPO_PARAMS)

        # With B_int=0, no injury
        assert result_zero["T_NRHypo_final"] == pytest.approx(0.0, abs=1e-6)
        # With B_int=0.8, significant injury
        assert result_high["T_NRHypo_final"] > 0.0


# ===========================================================================
# Integration tests
# ===========================================================================

class TestL2L3bIntegration:
    """Integration with L2 occupancy engine."""

    def test_pk_to_nrhypo_pipeline(self):
        """Full pipeline: L1 PK → L2 occupancy → L3b NRHypo."""
        from src.l1_pk import L1Model, DosingRegimen, InfusionSegment
        from src.l1_pk.config import default_parameters
        from src.l2_occupancy import simulate_occupancy, DEFAULT_L2_PARAMS

        # Run L1
        regimen = DosingRegimen(
            infusions=[InfusionSegment(start=0.0, end=0.667, rate=52.5)],
            s_fraction=0.5,
        )
        l1_model = L1Model(default_parameters())
        t_eval = np.linspace(0, 24, 500)
        l1_result = l1_model.simulate(regimen, 24.0, t_eval=t_eval)

        # Run L2
        c_ecf_s = l1_result.brain_ecf("KET_S")
        c_ecf_r = l1_result.brain_ecf("KET_R")
        l2_result = simulate_occupancy(
            l1_result.t, c_ecf_s, c_ecf_r, DEFAULT_L2_PARAMS
        )

        # Run L3b
        B_int = l2_result["int"]
        l3b_result = simulate_nrhypo(l1_result.t, B_int, DEFAULT_NRHYPPO_PARAMS)

        # Verify pipeline runs
        assert np.all(np.isfinite(l3b_result["Glu"]))
        assert np.all(np.isfinite(l3b_result["T_NRHypo"]))

    def test_higher_dose_more_nrhypo(self):
        """Higher dose should give more NRHypo injury (toxic arm)."""
        from src.l1_pk import L1Model, DosingRegimen, InfusionSegment
        from src.l1_pk.config import default_parameters
        from src.l2_occupancy import simulate_occupancy, DEFAULT_L2_PARAMS

        doses = [0.25, 0.5, 1.0, 2.0]
        injuries = []

        for dose in doses:
            total_dose = dose * 70.0
            rate = total_dose / 0.667
            regimen = DosingRegimen(
                infusions=[InfusionSegment(start=0.0, end=0.667, rate=rate)],
                s_fraction=0.5,
            )
            l1_model = L1Model(default_parameters())
            t_eval = np.linspace(0, 24, 500)
            l1_result = l1_model.simulate(regimen, 24.0, t_eval=t_eval)

            c_ecf_s = l1_result.brain_ecf("KET_S")
            c_ecf_r = l1_result.brain_ecf("KET_R")
            l2_result = simulate_occupancy(
                l1_result.t, c_ecf_s, c_ecf_r, DEFAULT_L2_PARAMS
            )

            l3b_result = simulate_nrhypo(
                l1_result.t, l2_result["int"], DEFAULT_NRHYPPO_PARAMS
            )
            injuries.append(l3b_result["T_NRHypo_final"])

        # Higher dose should give more injury (toxic arm)
        assert injuries[-1] >= injuries[0]
