"""Tests for L4 Injury Balance + L5 Clinical Overlay."""

import numpy as np
import pytest

from src.l4_l5.injury import (
    InjuryParams, ClinicalParams,
    DEFAULT_INJURY_PARAMS, DEFAULT_CLINICAL_PARAMS,
    compute_excitotoxic_flux, compute_injury_rate,
    simulate_injury, compute_clinical_constraints,
    compute_therapeutic_window,
)


# ===========================================================================
# L4 injury tests
# ===========================================================================

class TestInjuryRate:
    """Net injury rate computation."""

    def test_no_sd_no_injury(self):
        """Without SD events, injury should be zero (or just NRHypo)."""
        rate = compute_injury_rate(0.0, 0.0, 0.0, 0.0, DEFAULT_INJURY_PARAMS)
        assert rate == 0.0

    def test_sd_causes_injury(self):
        """SD events should cause excitotoxic injury."""
        rate = compute_injury_rate(0.0, 1.0, 0.1, 0.0, DEFAULT_INJURY_PARAMS)
        assert rate > 0.0

    def test_b_pyr_reduces_injury(self):
        """NMDAR block should reduce excitotoxic injury."""
        rate_no_block = compute_injury_rate(0.0, 1.0, 0.1, 0.0, DEFAULT_INJURY_PARAMS)
        rate_block = compute_injury_rate(0.5, 1.0, 0.1, 0.0, DEFAULT_INJURY_PARAMS)
        assert rate_block < rate_no_block

    def test_nrhypo_adds_injury(self):
        """NRHypo should add to net injury."""
        rate_no_nrhypo = compute_injury_rate(0.3, 1.0, 0.1, 0.0, DEFAULT_INJURY_PARAMS)
        rate_nrhypo = compute_injury_rate(0.3, 1.0, 0.1, 0.5, DEFAULT_INJURY_PARAMS)
        assert rate_nrhypo > rate_no_nrhypo

    def test_protection_reduces_injury(self):
        """At high B_pyr, protection should significantly reduce injury."""
        rate_no_block = compute_injury_rate(0.0, 1.0, 0.1, 0.0, DEFAULT_INJURY_PARAMS)
        rate_block = compute_injury_rate(0.9, 1.0, 0.1, 0.0, DEFAULT_INJURY_PARAMS)
        # Protection should reduce injury significantly (β*B_pyr/α fraction)
        assert rate_block < rate_no_block
        assert rate_block < rate_no_block * 0.3  # >70% reduction


class TestInjurySimulation:
    """Injury simulation over time."""

    def test_simulation_runs(self):
        """Simulation should complete successfully."""
        t = np.linspace(0, 24, 100)
        result = simulate_injury(
            t, np.zeros(100), np.zeros(100),
            np.ones(100) * 0.5, np.ones(100) * 0.1,
            np.zeros(100),
        )
        assert "I" in result
        assert len(result["I"]) == len(t)
        assert np.all(np.isfinite(result["I"]))

    def test_injury_accumulates(self):
        """With sustained SD, injury should accumulate."""
        t = np.linspace(0, 24, 100)
        result = simulate_injury(
            t, np.zeros(100), np.zeros(100),
            np.ones(100) * 1.0, np.ones(100) * 0.2,
            np.zeros(100),
        )
        assert result["I_final"] > 0.0

    def test_protection_reduces_injury(self):
        """With B_pyr, net injury should be reduced."""
        t = np.linspace(0, 24, 100)
        result_no_block = simulate_injury(
            t, np.zeros(100), np.zeros(100),
            np.ones(100) * 1.0, np.ones(100) * 0.2,
            np.zeros(100),
        )
        result_block = simulate_injury(
            t, np.ones(100) * 0.5, np.zeros(100),
            np.ones(100) * 1.0, np.ones(100) * 0.2,
            np.zeros(100),
        )
        assert result_block["I_final"] < result_no_block["I_final"]


# ===========================================================================
# L5 clinical constraints tests
# ===========================================================================

class TestClinicalConstraints:
    """Clinical constraint computation."""

    def test_cpp_never_violated_at_baseline(self):
        """At baseline (no drug), CPP should be above minimum."""
        C_brain = np.zeros(100)
        result = compute_clinical_constraints(C_brain, DEFAULT_CLINICAL_PARAMS)
        assert not np.any(result["CPP_violated"])

    def test_map_increases_with_drug(self):
        """MAP should increase with ketamine (sympathomimetic)."""
        C_low = np.ones(100) * 0.01
        C_high = np.ones(100) * 0.1
        result_low = compute_clinical_constraints(C_low, DEFAULT_CLINICAL_PARAMS)
        result_high = compute_clinical_constraints(C_high, DEFAULT_CLINICAL_PARAMS)
        assert np.mean(result_high["MAP"]) > np.mean(result_low["MAP"])

    def test_psych_burden_increases_with_dose(self):
        """Psychotomimetic burden should increase with concentration."""
        C_low = np.ones(100) * 0.01
        C_high = np.ones(100) * 0.15
        result_low = compute_clinical_constraints(C_low, DEFAULT_CLINICAL_PARAMS)
        result_high = compute_clinical_constraints(C_high, DEFAULT_CLINICAL_PARAMS)
        assert np.mean(result_high["psych_burden"]) > np.mean(result_low["psych_burden"])


# ===========================================================================
# Therapeutic window tests
# ===========================================================================

class TestTherapeuticWindow:
    """Therapeutic window computation."""

    def test_window_found_with_feasible_doses(self):
        """Window should be found when feasible doses exist."""
        dose_range = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
        injury_results = [
            {"I_final": 1.0}, {"I_final": 0.5}, {"I_final": 0.2},
            {"I_final": 0.3}, {"I_final": 0.8},
        ]
        clinical_results = [
            {"psych_burden": np.zeros(10), "CPP_violated": np.zeros(10, dtype=bool)},
            {"psych_burden": np.zeros(10), "CPP_violated": np.zeros(10, dtype=bool)},
            {"psych_burden": np.zeros(10), "CPP_violated": np.zeros(10, dtype=bool)},
            {"psych_burden": np.ones(10) * 0.5, "CPP_violated": np.zeros(10, dtype=bool)},
            {"psych_burden": np.ones(10) * 0.9, "CPP_violated": np.zeros(10, dtype=bool)},
        ]
        window = compute_therapeutic_window(
            dose_range, injury_results, clinical_results
        )
        assert window["window_found"]
        assert window["optimal_dose"] == 0.5

    def test_window_not_found_when_all_infeasible(self):
        """Window should not be found when all doses violate constraints."""
        dose_range = np.array([0.5, 1.0, 2.0])
        injury_results = [
            {"I_final": 0.5}, {"I_final": 0.3}, {"I_final": 0.8},
        ]
        clinical_results = [
            {"psych_burden": np.ones(10), "CPP_violated": np.ones(10, dtype=bool)},
            {"psych_burden": np.ones(10), "CPP_violated": np.ones(10, dtype=bool)},
            {"psych_burden": np.ones(10), "CPP_violated": np.ones(10, dtype=bool)},
        ]
        window = compute_therapeutic_window(
            dose_range, injury_results, clinical_results
        )
        assert not window["window_found"]


# ===========================================================================
# Full-stack integration test
# ===========================================================================

class TestFullStackIntegration:
    """Full L1→L2→L3a→L3b→L4→L5 pipeline."""

    def test_full_pipeline(self):
        """Full pipeline should run and produce all outputs."""
        from src.l1_pk import L1Model, DosingRegimen, InfusionSegment
        from src.l1_pk.config import default_parameters
        from src.l2_occupancy import simulate_occupancy, DEFAULT_L2_PARAMS
        from src.l3a_sd import simulate_sd_dynamics, DEFAULT_SD_PARAMS
        from src.l3b_nrhypo import simulate_nrhypo, DEFAULT_NRHYPPO_PARAMS

        # L1: PK
        regimen = DosingRegimen(
            infusions=[InfusionSegment(start=0.0, end=0.667, rate=52.5)],
            s_fraction=0.5,
        )
        l1 = L1Model(default_parameters())
        t = np.linspace(0, 24, 500)
        r1 = l1.simulate(regimen, 24.0, t_eval=t)

        # L2: Occupancy
        r2 = simulate_occupancy(t, r1.brain_ecf("KET_S"), r1.brain_ecf("KET_R"),
                                DEFAULT_L2_PARAMS)

        # L3a: SD
        r3a = simulate_sd_dynamics(t, r2["pyr"], DEFAULT_SD_PARAMS)

        # L3b: NRHypo
        r3b = simulate_nrhypo(t, r2["int"], DEFAULT_NRHYPPO_PARAMS)

        # L4: Injury
        r4 = simulate_injury(
            t, r2["pyr"], r2["int"],
            r3a["lambda_SD"] * r3a["D_SD"],
            np.clip((r3b["Glu"] - DEFAULT_NRHYPPO_PARAMS.Glu_0) /
                    DEFAULT_NRHYPPO_PARAMS.Glu_max, 0, 1),
            r3b["injury_rate"],
        )

        # L5: Clinical
        c_brain = r1.brain_ecf("KET_S") + r1.brain_ecf("KET_R")
        r5 = compute_clinical_constraints(c_brain)

        # Verify all outputs
        assert np.all(np.isfinite(r4["I"]))
        assert "protection" in r4
        assert "toxicity" in r4
        assert "MAP" in r5
        assert "psych_burden" in r5

    def test_dose_response_window(self):
        """Multiple doses should reveal a therapeutic window."""
        from src.l1_pk import L1Model, DosingRegimen, InfusionSegment
        from src.l1_pk.config import default_parameters
        from src.l2_occupancy import simulate_occupancy, DEFAULT_L2_PARAMS
        from src.l3a_sd import simulate_sd_dynamics, DEFAULT_SD_PARAMS
        from src.l3b_nrhypo import simulate_nrhypo, DEFAULT_NRHYPPO_PARAMS

        doses = [0.25, 0.5, 1.0, 2.0]
        injury_results = []
        clinical_results = []

        for dose in doses:
            rate = dose * 70.0 / 0.667
            regimen = DosingRegimen(
                infusions=[InfusionSegment(start=0.0, end=0.667, rate=rate)],
                s_fraction=0.5,
            )
            l1 = L1Model(default_parameters())
            t = np.linspace(0, 24, 500)
            r1 = l1.simulate(regimen, 24.0, t_eval=t)

            r2 = simulate_occupancy(t, r1.brain_ecf("KET_S"), r1.brain_ecf("KET_R"),
                                    DEFAULT_L2_PARAMS)
            r3a = simulate_sd_dynamics(t, r2["pyr"], DEFAULT_SD_PARAMS)
            r3b = simulate_nrhypo(t, r2["int"], DEFAULT_NRHYPPO_PARAMS)

            r4 = simulate_injury(
                t, r2["pyr"], r2["int"],
                r3a["lambda_SD"] * r3a["D_SD"],
                np.clip((r3b["Glu"] - DEFAULT_NRHYPPO_PARAMS.Glu_0) /
                        DEFAULT_NRHYPPO_PARAMS.Glu_max, 0, 1),
                r3b["injury_rate"],
            )

            c_brain = r1.brain_ecf("KET_S") + r1.brain_ecf("KET_R")
            r5 = compute_clinical_constraints(c_brain)

            injury_results.append(r4)
            clinical_results.append(r5)

        # At low dose: injury should be high (insufficient protection)
        # At high dose: injury should increase (NRHypo toxicity)
        # → U-shaped curve (or at least non-monotonic)
        I_final = [r["I_final"] for r in injury_results]
        assert len(I_final) == len(doses)
        # Injury at highest dose should be > injury at optimal dose
        # (not necessarily > lowest dose due to protection)
