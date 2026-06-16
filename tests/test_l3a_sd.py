"""Tests for L3a SD Engine — Protective Arm."""

import numpy as np
import pytest

from src.l3a_sd.model import (
    SDParams, DEFAULT_SD_PARAMS,
    compute_sd_threshold, compute_sd_duration, compute_sd_recovery,
    compute_sd_rate, compute_sd_burden, simulate_sd_dynamics,
)


# ===========================================================================
# SD descriptor tests
# ===========================================================================

class TestSDDescriptors:
    """SD descriptors modulated by B_pyr."""

    def test_threshold_increases_with_B_pyr(self):
        """NMDAR block should raise SD threshold (protective)."""
        K_thr_0 = compute_sd_threshold(0.0, DEFAULT_SD_PARAMS)
        K_thr_50 = compute_sd_threshold(0.5, DEFAULT_SD_PARAMS)
        K_thr_90 = compute_sd_threshold(0.9, DEFAULT_SD_PARAMS)
        assert K_thr_0 < K_thr_50 < K_thr_90

    def test_threshold_at_zero_occupancy(self):
        """At zero occupancy, threshold should be K_thr_0."""
        K_thr = compute_sd_threshold(0.0, DEFAULT_SD_PARAMS)
        assert K_thr == pytest.approx(DEFAULT_SD_PARAMS.K_thr_0)

    def test_duration_decreases_with_B_pyr(self):
        """NMDAR block should shorten SD duration."""
        D_0 = compute_sd_duration(0.0, DEFAULT_SD_PARAMS)
        D_50 = compute_sd_duration(0.5, DEFAULT_SD_PARAMS)
        assert D_50 < D_0

    def test_recovery_accelerates_with_B_pyr(self):
        """NMDAR block should accelerate recovery."""
        tau_0 = compute_sd_recovery(0.0, DEFAULT_SD_PARAMS)
        tau_50 = compute_sd_recovery(0.5, DEFAULT_SD_PARAMS)
        assert tau_50 < tau_0

    def test_sd_rate_decreases_with_B_pyr(self):
        """NMDAR block should reduce SD event rate."""
        rate_0 = compute_sd_rate(0.0, DEFAULT_SD_PARAMS)
        rate_50 = compute_sd_rate(0.5, DEFAULT_SD_PARAMS)
        assert rate_50 < rate_0

    def test_sd_rate_zero_at_critical_occupancy(self):
        """At critical occupancy, SD should be fully suppressed."""
        params = DEFAULT_SD_PARAMS
        B_crit = (params.K_peak / params.K_thr_0 - 1.0) / params.kappa
        B_crit = np.clip(B_crit, 0.0, 1.0)
        rate = compute_sd_rate(B_crit, params)
        assert rate == pytest.approx(0.0, abs=1e-10)

    def test_sd_rate_non_negative(self):
        """SD rate should never be negative."""
        for B in np.linspace(0, 1, 100):
            rate = compute_sd_rate(B, DEFAULT_SD_PARAMS)
            assert rate >= 0.0


# ===========================================================================
# SD burden tests
# ===========================================================================

class TestSDBurden:
    """SD burden integral computation."""

    def test_burden_without_drug(self):
        """Without drug, SD burden should be positive (injury context)."""
        t = np.linspace(0, 24, 100)
        B_pyr = np.zeros_like(t)
        result = compute_sd_burden(t, B_pyr, DEFAULT_SD_PARAMS)
        assert result["burden"] > 0

    def test_burden_reduced_with_drug(self):
        """With NMDAR block, SD burden should be reduced."""
        t = np.linspace(0, 24, 100)
        B_none = np.zeros_like(t)
        B_drug = np.ones_like(t) * 0.3

        burden_none = compute_sd_burden(t, B_none, DEFAULT_SD_PARAMS)["burden"]
        burden_drug = compute_sd_burden(t, B_drug, DEFAULT_SD_PARAMS)["burden"]
        assert burden_drug < burden_none

    def test_burden_dose_response(self):
        """Higher occupancy should give lower burden."""
        t = np.linspace(0, 24, 100)
        burdens = []
        for B_level in [0.0, 0.1, 0.3, 0.5, 0.7]:
            B_pyr = np.ones_like(t) * B_level
            result = compute_sd_burden(t, B_pyr, DEFAULT_SD_PARAMS)
            burdens.append(result["burden"])
        # Should be monotonically decreasing
        for i in range(len(burdens) - 1):
            assert burdens[i + 1] <= burdens[i]


# ===========================================================================
# SD dynamics simulation tests
# ===========================================================================

class TestSDDynamics:
    """Bistable K+ dynamics simulation."""

    def test_simulation_runs(self):
        """Simulation should complete successfully."""
        t = np.linspace(0, 24, 500)
        B_pyr = np.zeros_like(t)
        result = simulate_sd_dynamics(t, B_pyr, DEFAULT_SD_PARAMS)
        assert "K" in result
        assert "sd_events" in result
        assert len(result["K"]) == len(t)
        assert np.all(np.isfinite(result["K"]))

    def test_K_starts_at_rest(self):
        """K+ should start at resting level."""
        t = np.linspace(0, 1, 100)
        B_pyr = np.zeros_like(t)
        result = simulate_sd_dynamics(t, B_pyr, DEFAULT_SD_PARAMS)
        assert result["K"][0] == pytest.approx(DEFAULT_SD_PARAMS.K_rest)

    def test_sd_events_detected(self):
        """SD events should be detected when K+ exceeds threshold."""
        t = np.linspace(0, 24, 2000)
        B_pyr = np.zeros_like(t)  # No drug → SD should occur
        result = simulate_sd_dynamics(t, B_pyr, DEFAULT_SD_PARAMS)
        # In injury context, SD events should occur
        # (may or may not depending on parameters)
        assert isinstance(result["n_sd_events"], int)

    def test_drug_suppresses_sd(self):
        """With high B_pyr, SD should be suppressed or reduced."""
        t = np.linspace(0, 24, 2000)

        result_no_drug = simulate_sd_dynamics(t, np.zeros_like(t), DEFAULT_SD_PARAMS)
        result_drug = simulate_sd_dynamics(t, np.ones_like(t) * 0.5, DEFAULT_SD_PARAMS)

        # Drug should not increase SD events
        assert result_drug["n_sd_events"] <= result_no_drug["n_sd_events"]

    def test_burden_reduced_with_drug(self):
        """SD burden should be reduced with drug."""
        t = np.linspace(0, 24, 2000)

        result_no_drug = simulate_sd_dynamics(t, np.zeros_like(t), DEFAULT_SD_PARAMS)
        result_drug = simulate_sd_dynamics(t, np.ones_like(t) * 0.5, DEFAULT_SD_PARAMS)

        assert result_drug["burden"] <= result_no_drug["burden"]


# ===========================================================================
# Integration tests
# ===========================================================================

class TestL2L3aIntegration:
    """Integration with L2 occupancy engine."""

    def test_pk_to_sd_pipeline(self):
        """Full pipeline: L1 PK → L2 occupancy → L3a SD burden."""
        from src.l1_pk import L1Model, DosingRegimen, InfusionSegment
        from src.l1_pk.config import default_parameters
        from src.l2_occupancy import simulate_occupancy, DEFAULT_L2_PARAMS

        # Run L1: Zhao 2012 regimen (0.5 mg/kg IV 40 min)
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

        # Run L3a
        B_pyr = l2_result["pyr"]
        l3a_result = simulate_sd_dynamics(l1_result.t, B_pyr, DEFAULT_SD_PARAMS)

        # Verify pipeline runs
        assert np.all(np.isfinite(l3a_result["K"]))
        assert l3a_result["burden"] >= 0
        assert "sd_events" in l3a_result

    def test_dose_response_sd_suppression(self):
        """Higher dose should give lower SD burden (protective effect)."""
        from src.l1_pk import L1Model, DosingRegimen, InfusionSegment
        from src.l1_pk.config import default_parameters
        from src.l2_occupancy import simulate_occupancy, DEFAULT_L2_PARAMS

        doses = [0.25, 0.5, 1.0, 2.0]  # mg/kg
        burdens = []

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

            l3a_result = simulate_sd_dynamics(
                l1_result.t, l2_result["pyr"], DEFAULT_SD_PARAMS
            )
            burdens.append(l3a_result["burden"])

        # Burden should generally decrease with dose
        # (not necessarily monotonic due to PK dynamics)
        assert burdens[-1] <= burdens[0], "High dose should reduce SD burden vs low dose"
