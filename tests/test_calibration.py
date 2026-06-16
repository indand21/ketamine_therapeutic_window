"""Tests for the L1 PK calibration engine."""

import numpy as np
import pytest

from src.l1_pk.config import default_parameters
from src.ingestion.digitized_data import (
    ALL_DIGITIZED_CURVES,
    DigitizedCurve,
    PEREZ_RUIXO_2021_SUMMARY,
    WEISS_2022_SUMMARY,
)
from src.calibration.l1_calibrator import (
    CALIBRATION_PARAMS,
    DEFAULT_VALUES,
    BOUNDS,
    PARAM_NAMES,
    apply_calibration_vector,
    simulate_curve,
    calibrate,
    CalibrationResult,
)


class TestDigitizedData:
    """Verify digitized data module loads correctly."""

    def test_all_curves_available(self):
        assert len(ALL_DIGITIZED_CURVES) > 0

    def test_curves_have_required_fields(self):
        for curve in ALL_DIGITIZED_CURVES:
            assert curve.source
            assert curve.species
            assert curve.compartment
            assert len(curve.times_h) > 0
            assert len(curve.concentrations) > 0
            assert len(curve.times_h) == len(curve.concentrations)

    def test_summary_stats_present(self):
        assert PEREZ_RUIXO_2021_SUMMARY["CL_S_L_per_h"] == 114.0
        assert WEISS_2022_SUMMARY["HNK_KET_ratio_R_steady_state"] == 46.0


class TestCalibrationParams:
    """Verify calibration parameter space is well-defined."""

    def test_param_count(self):
        assert len(CALIBRATION_PARAMS) == len(DEFAULT_VALUES)
        assert len(CALIBRATION_PARAMS) == len(BOUNDS)
        assert len(CALIBRATION_PARAMS) == len(PARAM_NAMES)

    def test_bounds_valid(self):
        for i, (lo, hi) in enumerate(BOUNDS):
            assert lo < hi, f"Bound {PARAM_NAMES[i]}: {lo} >= {hi}"
            assert DEFAULT_VALUES[i] >= lo, f"Default {PARAM_NAMES[i]} below lower bound"
            assert DEFAULT_VALUES[i] <= hi, f"Default {PARAM_NAMES[i]} above upper bound"

    def test_apply_calibration_vector_identity(self):
        """The default calibration vector should reproduce the config defaults.

        The calibration-param defaults are aligned with the literature/config
        values (default_parameters()), so the identity point IS the validated
        model — calibration refines from a 5/5 baseline rather than a broken one.
        """
        params = default_parameters()
        result = apply_calibration_vector(params, DEFAULT_VALUES)
        # Calibrated fields must equal the config defaults.
        assert abs(result.clearances.CL_out_HNK - params.clearances.CL_out_HNK) < 1e-6
        assert abs(result.flows.Q_per - params.flows.Q_per) < 1e-6
        assert abs(result.volumes.V_cen - params.volumes.V_cen) < 1e-6
        assert abs(result.volumes.V_per - params.volumes.V_per) < 1e-6

    def test_apply_calibration_vector_modifies(self):
        """Changing the vector should modify parameters."""
        params = default_parameters()
        x = DEFAULT_VALUES.copy()
        x[0] = 0.05  # CL_out_HNK
        x[1] = 150.0  # Q_per
        result = apply_calibration_vector(params, x)
        assert abs(result.clearances.CL_out_HNK - 0.05) < 1e-6
        assert abs(result.flows.Q_per - 150.0) < 1e-6


class TestSimulateCurve:
    """Verify curve simulation works for all digitized data."""

    @pytest.mark.parametrize("curve", ALL_DIGITIZED_CURVES, ids=lambda c: f"{c.source}_{c.species}")
    def test_simulate_runs(self, curve):
        params = default_parameters()
        c_model = simulate_curve(curve, params)
        if curve.dose_mg > 0:
            assert c_model is not None, f"Simulation failed for {curve.source} {curve.species}"
            assert len(c_model) == len(curve.times_h)

    def test_s_ketamine_produces_positive_concentrations(self):
        from src.ingestion.digitized_data import KAMP_2020_S_KET_ESKETAMINE
        params = default_parameters()
        c_model = simulate_curve(KAMP_2020_S_KET_ESKETAMINE, params)
        assert c_model is not None
        assert np.max(c_model) > 0


class TestCalibration:
    """Smoke test for the calibration engine."""

    @pytest.mark.slow
    def test_calibrate_runs_with_lbfgsb(self):
        """L-BFGS-B should complete (marked slow — ~60s)."""
        from src.ingestion.digitized_data import KAMP_2020_S_KET_ESKETAMINE
        result = calibrate(
            curves=[KAMP_2020_S_KET_ESKETAMINE],
            method="L-BFGS-B",
            maxiter=5,
        )
        assert isinstance(result, CalibrationResult)
        assert result.x_optimal is not None
        assert len(result.x_optimal) == len(PARAM_NAMES)
        # All parameters should be within bounds
        for i, (lo, hi) in enumerate(BOUNDS):
            assert lo <= result.x_optimal[i] <= hi + 1e-6

    def test_calibrate_report_generation(self):
        """Report should be a valid markdown string (fast — uses default params)."""
        from src.calibration.l1_calibrator import format_calibration_report
        # Create a mock result without running optimizer
        mock_result = CalibrationResult(
            success=True,
            x_optimal=DEFAULT_VALUES.copy(),
            params_optimal=default_parameters(),
            objective_value=10.0,
            param_names=PARAM_NAMES,
            curves_used=["test"],
            residuals_per_curve={"test": 0.5},
            method="mock",
        )
        report = format_calibration_report(mock_result)
        assert "# L1 PK Calibration Report" in report
        assert "Calibrated Parameters" in report
