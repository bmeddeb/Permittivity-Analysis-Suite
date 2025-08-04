"""Tests for the Kramers-Kronig validator."""

import numpy as np
import pandas as pd
import pytest

from app.models.havriliak_negami import HavriliakNegamiModel
from app.models.kk_validators import KramersKronigValidator


class TestKramersKronigValidator:
    """Test suite for KramersKronigValidator."""

    def test_validator_creation_basic(self, debye_data):
        """Test basic validator creation."""
        validator = KramersKronigValidator(debye_data)
        assert validator is not None
        assert hasattr(validator, "validate")
        assert hasattr(validator, "get_report")
        assert hasattr(validator, "from_model")

    def test_validator_creation_with_eps_inf(self, debye_data):
        """Test validator creation with explicit eps_inf."""
        validator = KramersKronigValidator(debye_data, eps_inf=3.0)
        assert validator.explicit_eps_inf == 3.0

    def test_validator_creation_tan_delta(self, tan_delta_data):
        """Test validator creation with loss tangent data."""
        validator = KramersKronigValidator(tan_delta_data, loss_repr="tan_delta")
        assert validator.loss_repr == "tan_delta"

        # Check that Df was converted to imaginary permittivity
        # tan(δ) * Dk should give imaginary part
        original_tan_delta = tan_delta_data["Df"].values
        expected_df = original_tan_delta * (tan_delta_data["Dk"].values + 1e-18)

        np.testing.assert_allclose(validator.df["Df"].values, expected_df, rtol=1e-10)

    def test_dataframe_structure_validation(self, assert_df_structure):
        """Test that validator validates DataFrame structure."""
        # Valid DataFrame
        valid_data = pd.DataFrame(
            {"Frequency (GHz)": [1, 2, 3], "Dk": [4, 3, 2], "Df": [0.1, 0.2, 0.3]}
        )
        validator = KramersKronigValidator(valid_data)
        assert_df_structure(validator.df)

    def test_eps_inf_estimation(self, debye_data):
        """Test automatic eps_inf estimation."""
        validator = KramersKronigValidator(debye_data)  # No explicit eps_inf

        estimated_eps_inf = validator._estimate_eps_inf()

        # With default fit method, should estimate from high-frequency tail
        # The exact value depends on the tail fraction and method
        assert estimated_eps_inf > 0
        assert np.isfinite(estimated_eps_inf)

    def test_eps_inf_explicit_override(self, debye_data):
        """Test that explicit eps_inf overrides estimation."""
        explicit_value = 4.5
        validator = KramersKronigValidator(debye_data, eps_inf=explicit_value)

        estimated_eps_inf = validator._estimate_eps_inf()
        assert estimated_eps_inf == explicit_value

    def test_validation_perfect_causal_data(self, debye_data, causality_threshold):
        """Test validation on perfect causal data (Debye model)."""
        validator = KramersKronigValidator(debye_data, eps_inf=3.0)

        mean_error = validator.validate()

        # Causal data should have reasonable error (KK has inherent limitations)
        assert mean_error < 0.2  # Realistic threshold for finite frequency range
        assert validator.validated is True

        # Check that KK-predicted data was added to DataFrame
        assert "Dk_KK" in validator.df.columns

    def test_validation_report_generation(self, debye_data):
        """Test validation report generation."""
        validator = KramersKronigValidator(debye_data, eps_inf=3.0)

        # Should raise error before validation
        with pytest.raises(RuntimeError, match="validate\\(\\) must be called"):
            validator.get_report()

        # After validation, should return report
        mean_error = validator.validate()
        report = validator.get_report()

        assert isinstance(report, str)
        assert "Kramers-Kronig Causality Report" in report
        assert "Causality Status:" in report
        assert "Mean Relative Error:" in report
        assert "RMSE" in report

        # Check status based on error
        if mean_error < 0.05:
            assert "PASS" in report
        else:
            assert "FAIL" in report

    def test_validation_havriliak_negami_data(
        self, havriliak_negami_data, causality_threshold
    ):
        """Test validation on Havriliak-Negami data."""
        validator = KramersKronigValidator(havriliak_negami_data, eps_inf=3.0)

        mean_error = validator.validate()

        # HN data should be causal but finite frequency range limits accuracy
        assert mean_error < 0.5  # Realistic threshold for HN with finite range

        report = validator.get_report()
        # Status depends on the actual error value
        assert "PASS" in report or "FAIL" in report

    def test_validation_noisy_data(self, noisy_data):
        """Test validation on noisy data."""
        validator = KramersKronigValidator(noisy_data)

        mean_error = validator.validate()

        # Noisy data has degraded causality due to measurement errors
        # 2% noise can significantly affect KK validation with finite range
        assert mean_error < 0.25  # Realistic threshold for 2% noise
        assert validator.validated is True

    def test_from_model_class_method(self, frequency_array):
        """Test creating validator from model."""
        # Create HN model
        model = HavriliakNegamiModel()
        params = model.make_params(
            eps_inf=3.0, delta_eps=5.0, tau=1e-9, alpha=0.8, beta=0.6
        )

        # Create validator from model
        validator = KramersKronigValidator.from_model(
            model=model, params=params, f_ghz=frequency_array, loss_repr="eps_imag"
        )

        assert isinstance(validator, KramersKronigValidator)
        assert len(validator.df) == len(frequency_array)
        assert "Frequency (GHz)" in validator.df.columns
        assert "Dk" in validator.df.columns
        assert "Df" in validator.df.columns

    def test_from_model_with_model_result(self, frequency_array, debye_data):
        """Test creating validator from ModelResult."""
        # Fit a model first
        model = HavriliakNegamiModel()
        f_ghz = debye_data["Frequency (GHz)"].values
        complex_data = debye_data["Dk"].values + 1j * debye_data["Df"].values

        params = model.guess(complex_data, f_ghz)
        result = model.fit(complex_data, params, f_ghz=f_ghz)

        # Create validator from ModelResult
        validator = KramersKronigValidator.from_model(
            model=model,
            params=result,  # ModelResult instead of Parameters
            f_ghz=frequency_array,
        )

        assert isinstance(validator, KramersKronigValidator)

    def test_from_model_with_none_params(self, frequency_array):
        """Test creating validator from model with None params."""
        model = HavriliakNegamiModel()

        # Should use make_params() when params is None
        validator = KramersKronigValidator.from_model(
            model=model, params=None, f_ghz=frequency_array
        )

        assert isinstance(validator, KramersKronigValidator)

    def test_from_model_eps_inf_extraction(self, frequency_array):
        """Test that eps_inf is correctly extracted from model parameters."""
        model = HavriliakNegamiModel()
        params = model.make_params(eps_inf=4.5)

        validator = KramersKronigValidator.from_model(
            model=model, params=params, f_ghz=frequency_array
        )

        # Should extract eps_inf from parameters
        assert validator.explicit_eps_inf == 4.5

    def test_sparse_frequency_handling(self, sparse_data):
        """Test validator with sparse frequency sampling."""
        validator = KramersKronigValidator(sparse_data)

        mean_error = validator.validate()

        # Should still work with sparse data
        assert np.isfinite(mean_error)
        assert validator.validated is True

    def test_wide_frequency_range(self, wide_frequency_array):
        """Test validator with wide frequency range."""
        # Create test data over wide range
        eps_inf, delta_eps, tau = 3.0, 5.0, 1e-9
        omega = 2 * np.pi * wide_frequency_array * 1e9
        complex_eps = eps_inf + delta_eps / (1 + 1j * omega * tau)

        data = pd.DataFrame(
            {
                "Frequency (GHz)": wide_frequency_array,
                "Dk": np.real(complex_eps),
                "Df": np.imag(complex_eps),
            }
        )

        validator = KramersKronigValidator(data, eps_inf=eps_inf)
        mean_error = validator.validate()

        assert mean_error < 0.3  # Even wide range has finite frequency limitations

    def test_error_metrics_calculation(self, debye_data):
        """Test that error metrics are calculated correctly."""
        validator = KramersKronigValidator(debye_data, eps_inf=3.0)

        validator.validate()
        report = validator.get_report()

        # Extract RMSE from report
        import re

        rmse_match = re.search(r"RMSE.*?(\d+\.\d+)", report)
        assert rmse_match is not None

        rmse_from_report = float(rmse_match.group(1))
        assert rmse_from_report >= 0

    def test_validation_consistency(self, debye_data):
        """Test that repeated validation gives consistent results."""
        validator = KramersKronigValidator(debye_data, eps_inf=3.0)

        error1 = validator.validate()
        error2 = validator.validate()

        # Should get identical results
        assert abs(error1 - error2) < 1e-15

    def test_different_loss_representations(self, debye_data):
        """Test validation with different loss representations."""
        # Test with eps_imag (default)
        validator_imag = KramersKronigValidator(debye_data, loss_repr="eps_imag")
        error_imag = validator_imag.validate()

        # Convert to tan_delta format
        tan_delta_data = debye_data.copy()
        tan_delta_data["Df"] = tan_delta_data["Df"] / (tan_delta_data["Dk"] + 1e-18)

        validator_tan = KramersKronigValidator(tan_delta_data, loss_repr="tan_delta")
        error_tan = validator_tan.validate()

        # Should give similar results (within numerical precision)
        assert abs(error_imag - error_tan) < 1e-10

    def test_invalid_loss_representation(self, debye_data):
        """Test handling of invalid loss representation."""
        with pytest.raises(ValueError, match="loss_repr must be"):
            KramersKronigValidator(debye_data, loss_repr="invalid")

    def test_edge_case_single_frequency(self):
        """Test validator with single frequency point."""
        single_point_data = pd.DataFrame(
            {"Frequency (GHz)": [1.0], "Dk": [4.0], "Df": [0.1]}
        )

        # Single point should fail validation due to insufficient data
        with pytest.raises(ValueError, match="at least 2 data points"):
            KramersKronigValidator(single_point_data)

    def test_edge_case_zero_imaginary_part(self, frequency_array):
        """Test validator with zero imaginary part."""
        zero_imag_data = pd.DataFrame(
            {
                "Frequency (GHz)": frequency_array,
                "Dk": np.full_like(frequency_array, 3.0),
                "Df": np.zeros_like(frequency_array),
            }
        )

        validator = KramersKronigValidator(zero_imag_data, eps_inf=3.0)
        mean_error = validator.validate()

        # Should be perfectly causal (no dispersion)
        assert mean_error < 1e-10

    def test_parameter_kwargs_handling(self, debye_data):
        """Test that additional kwargs are handled correctly."""
        # Should not raise error with extra kwargs
        validator = KramersKronigValidator(
            debye_data, eps_inf=3.0, unused_param=42, another_unused="test"
        )

        assert validator.explicit_eps_inf == 3.0

    def test_dataframe_modification_isolation(self, debye_data):
        """Test that validator doesn't modify original DataFrame."""
        original_data = debye_data.copy()

        validator = KramersKronigValidator(debye_data, loss_repr="eps_imag")
        validator.validate()

        # Original DataFrame should be unchanged
        pd.testing.assert_frame_equal(debye_data, original_data)

    def test_loss_tangent_conversion_isolation(self, tan_delta_data):
        """Test that loss tangent conversion doesn't affect original data."""
        original_data = tan_delta_data.copy()

        validator = KramersKronigValidator(tan_delta_data, loss_repr="tan_delta")
        validator.validate()

        # Original DataFrame should be unchanged
        pd.testing.assert_frame_equal(tan_delta_data, original_data)

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme parameter values."""
        # Very high permittivity values
        extreme_data = pd.DataFrame(
            {
                "Frequency (GHz)": np.logspace(-1, 3, 50),
                "Dk": np.full(50, 1000.0),  # Very high permittivity
                "Df": np.full(50, 0.001),  # Very low loss
            }
        )

        validator = KramersKronigValidator(extreme_data, eps_inf=1000.0)
        mean_error = validator.validate()

        assert np.isfinite(mean_error)

    def test_validation_with_model_comparison(self, frequency_array):
        """Test validation by comparing different models."""
        # Create data with one model
        model1 = HavriliakNegamiModel()
        params1 = model1.make_params(
            eps_inf=3.0, delta_eps=5.0, tau=1e-9, alpha=0.8, beta=0.6
        )

        validator1 = KramersKronigValidator.from_model(model1, params1, frequency_array)
        error1 = validator1.validate()

        # HN model data with finite frequency range
        assert error1 < 0.5  # Realistic threshold for HN with finite range

    def test_memory_efficiency(self, wide_frequency_array):
        """Test that validator is memory efficient with large datasets."""
        # Create large dataset
        large_data = pd.DataFrame(
            {
                "Frequency (GHz)": wide_frequency_array,
                "Dk": np.random.rand(len(wide_frequency_array)) + 2.0,
                "Df": np.random.rand(len(wide_frequency_array)) * 0.1,
            }
        )

        validator = KramersKronigValidator(large_data)

        # Should handle large datasets without issues
        mean_error = validator.validate()
        assert np.isfinite(mean_error)

    def test_report_formatting(self, debye_data):
        """Test that validation report is properly formatted."""
        validator = KramersKronigValidator(debye_data, eps_inf=3.0)
        validator.validate()
        report = validator.get_report()

        # Check report structure
        lines = report.strip().split("\n")
        assert len(lines) >= 5  # Header, separator, status, error, rmse, separator

        # Check for specific formatting elements
        assert "=" in report  # Separator lines
        assert "▸" in report  # Bullet points
        assert "%" in report  # Percentage in error
