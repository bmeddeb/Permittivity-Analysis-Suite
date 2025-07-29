"""
Test suite for KramersKronigValidator class.

Tests cover:
- Causality validation for various dielectric models
- Error handling and edge cases
- Different KK transform methods
- Parameter variations
- Output format validation
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any
import warnings

from app.models import KramersKronigValidator


# Import the validator class (adjust import path as needed)
# from your_module import KramersKronigValidator, NUMBA_AVAILABLE


class TestKramersKronigValidator:
    """Test suite for KramersKronigValidator."""

    # ========== Causality Validation Tests ==========

    def test_single_debye_causality(self, single_debye_data_uniform):
        """Test that ideal single Debye data on uniform grid passes causality check."""
        validator = KramersKronigValidator(single_debye_data_uniform)
        results = validator.validate(causality_threshold=0.05)

        assert results['causality_status'] == 'PASS'
        assert results['mean_relative_error'] < 0.02  # Reasonable threshold
        assert validator.is_causal
        assert 'dk_kk' in results
        assert len(results['dk_kk']) == len(single_debye_data_uniform)

    def test_double_debye_causality(self, double_debye_data):
        """Test double Debye data validation."""
        validator = KramersKronigValidator(double_debye_data)
        results = validator.validate(causality_threshold=0.1)  # More lenient for complex data

        # Should still pass, as double Debye is causal
        assert results['causality_status'] == 'PASS'
        assert results['mean_relative_error'] < 0.1

        # Check diagnostics show multiple peaks
        diagnostics = validator.get_diagnostics()
        assert diagnostics['num_peaks'] >= 1

    def test_cole_cole_causality(self, cole_cole_data):
        """Test Cole-Cole (distributed relaxation) data validation."""
        validator = KramersKronigValidator(cole_cole_data)
        results = validator.validate(causality_threshold=0.1)  # Cole-Cole can have larger errors

        # Cole-Cole is causal, should pass
        assert results['causality_status'] == 'PASS'
        assert results['mean_relative_error'] < 0.1

    def test_noisy_data_tolerance(self, noisy_single_debye_data):
        """Test that validator handles noisy data appropriately."""
        validator = KramersKronigValidator(noisy_single_debye_data)
        results = validator.validate(causality_threshold=0.1)  # More lenient for noisy data

        # With 1% noise, should still pass with reasonable threshold
        assert results['mean_relative_error'] < 0.3  # Noisy data can have larger errors
        assert 'rmse' in results
        assert results['rmse'] < 1.0  # Reasonable RMSE for noisy data

    def test_causality_violation_detection(self, causality_violating_data):
        """Test that validator detects causality violations."""
        validator = KramersKronigValidator(causality_violating_data)
        results = validator.validate(causality_threshold=0.05)

        # Should fail due to intentional violations
        assert results['causality_status'] == 'FAIL'
        assert results['mean_relative_error'] > 0.05
        assert not validator.is_causal

    def test_low_loss_material(self, low_loss_material_data):
        """Test validation of low-loss materials with small Df."""
        validator = KramersKronigValidator(low_loss_material_data)
        results = validator.validate(causality_threshold=0.1)

        # Low-loss materials can be tricky due to small signal
        assert results['causality_status'] in ['PASS', 'FAIL']
        assert 'dk_kk' in results

    # ========== Method Selection Tests ==========

    def test_auto_method_selection(self, single_debye_data_uniform, non_uniform_grid_data):
        """Test automatic method selection based on grid uniformity."""
        # Uniform grid should select Hilbert
        validator_uniform = KramersKronigValidator(single_debye_data_uniform, method='auto')
        assert validator_uniform._is_grid_uniform()

        # Non-uniform grid should select trapz
        validator_nonuniform = KramersKronigValidator(non_uniform_grid_data, method='auto')
        assert not validator_nonuniform._is_grid_uniform()

    def test_hilbert_method_uniform_grid(self, single_debye_data_uniform):
        """Test explicit Hilbert method on uniform grid."""
        validator = KramersKronigValidator(single_debye_data_uniform, method='hilbert')
        results = validator.validate()

        assert results['causality_status'] == 'PASS'
        assert results['mean_relative_error'] < 0.02

    def test_hilbert_method_nonuniform_grid(self, non_uniform_grid_data):
        """Test Hilbert method on non-uniform grid (should resample)."""
        validator = KramersKronigValidator(non_uniform_grid_data, method='hilbert')
        results = validator.validate()

        # Should still work via resampling
        assert 'dk_kk' in results
        assert len(results['dk_kk']) == len(non_uniform_grid_data)

    def test_trapz_method(self, single_debye_data):
        """Test explicit trapezoidal method."""
        validator = KramersKronigValidator(single_debye_data, method='trapz')
        results = validator.validate(causality_threshold=0.1)

        # Trapz on log-spaced grid may have larger errors
        assert results['causality_status'] == 'PASS'
        assert results['mean_relative_error'] < 0.1

    # ========== Parameter Tests ==========

    def test_eps_inf_estimation_methods(self, single_debye_data_uniform):
        """Test different eps_inf estimation methods."""
        # Test 'mean' method
        validator_mean = KramersKronigValidator(
            single_debye_data_uniform,
            eps_inf_method='mean'
        )
        eps_inf_mean = validator_mean._estimate_eps_inf()

        # Test 'fit' method
        validator_fit = KramersKronigValidator(
            single_debye_data_uniform,
            eps_inf_method='fit'
        )
        eps_inf_fit = validator_fit._estimate_eps_inf()

        # Both should be close to theoretical value (3.0)
        assert abs(eps_inf_mean - 3.0) < 0.5
        assert abs(eps_inf_fit - 3.0) < 0.5

        # Fit method should generally be more accurate
        assert abs(eps_inf_fit - 3.0) <= abs(eps_inf_mean - 3.0) + 0.1  # Allow small tolerance

    def test_explicit_eps_inf(self, single_debye_data_uniform):
        """Test using explicit eps_inf value."""
        explicit_value = 3.0
        validator = KramersKronigValidator(
            single_debye_data_uniform,
            eps_inf=explicit_value
        )

        assert validator._estimate_eps_inf() == explicit_value

        # Should get excellent results with correct eps_inf
        results = validator.validate()
        assert results['causality_status'] == 'PASS'
        assert results['mean_relative_error'] < 0.01

    def test_window_functions(self, single_debye_data_uniform):
        """Test different window functions."""
        windows = ['hamming', 'hann', 'blackman', 'bartlett']

        for window in windows:
            validator = KramersKronigValidator(
                single_debye_data_uniform,
                method='hilbert',
                window=window
            )
            results = validator.validate()
            assert 'dk_kk' in results
            assert results['causality_status'] == 'PASS'

    def test_resample_points_option(self, non_uniform_grid_data):
        """Test custom resample points for Hilbert method."""
        resample_values = [512, 1024, 2048, 4096]

        for n_points in resample_values:
            validator = KramersKronigValidator(
                non_uniform_grid_data,
                method='hilbert',
                resample_points=n_points
            )
            results = validator.validate()

            assert 'dk_kk' in results
            assert len(results['dk_kk']) == len(non_uniform_grid_data)

    def test_tail_fraction_parameter(self, single_debye_data_uniform):
        """Test different tail fraction values for eps_inf estimation."""
        tail_fractions = [0.05, 0.1, 0.2, 0.3]
        eps_inf_values = []

        for frac in tail_fractions:
            validator = KramersKronigValidator(
                single_debye_data_uniform,
                tail_fraction=frac,
                eps_inf_method='mean'
            )
            eps_inf_values.append(validator._estimate_eps_inf())

        # All should be reasonably close to true value (3.0)
        assert all(abs(e - 3.0) < 1.0 for e in eps_inf_values)

        # Larger tail fractions should give more stable estimates
        assert np.std(eps_inf_values[-2:]) <= np.std(eps_inf_values[:2]) + 0.1  # Allow small tolerance

    def test_min_tail_points_parameter(self, minimal_data):
        """Test minimum tail points requirement."""
        # Should work with default min_tail_points=3
        validator = KramersKronigValidator(minimal_data)
        results = validator.validate()
        assert 'dk_kk' in results

        # Should fail if we require more points than available
        validator_strict = KramersKronigValidator(
            minimal_data,
            min_tail_points=5
        )
        with pytest.raises(ValueError, match="Insufficient data points"):
            validator_strict.validate()

    # ========== Error Handling Tests ==========

    def test_invalid_method_raises_error(self, single_debye_data):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="method must be"):
            KramersKronigValidator(single_debye_data, method='invalid')

    def test_invalid_eps_inf_method_raises_error(self, single_debye_data):
        """Test that invalid eps_inf_method raises error."""
        with pytest.raises(ValueError, match="eps_inf_method must be"):
            KramersKronigValidator(single_debye_data, eps_inf_method='invalid')

    def test_invalid_window_raises_error(self, single_debye_data):
        """Test that invalid window name raises error."""
        with pytest.raises(ValueError, match="window must be one of"):
            KramersKronigValidator(single_debye_data, window='invalid_window')

    def test_missing_column_raises_error(self, invalid_data_missing_column):
        """Test that missing column raises appropriate error."""
        with pytest.raises(ValueError, match="Missing required columns"):
            KramersKronigValidator(invalid_data_missing_column)

    def test_non_numeric_data_raises_error(self, invalid_data_non_numeric):
        """Test that non-numeric data raises error."""
        with pytest.raises(ValueError, match="non-numeric values or NaNs"):
            KramersKronigValidator(invalid_data_non_numeric)

    def test_negative_frequency_raises_error(self, invalid_data_negative_freq):
        """Test that negative frequency raises error."""
        with pytest.raises(ValueError, match="Negative frequencies detected"):
            KramersKronigValidator(invalid_data_negative_freq)

    def test_non_monotonic_frequencies_error(self, invalid_data_non_monotonic):
        """Test that non-monotonic frequencies raise error."""
        with pytest.raises(ValueError, match="strictly increasing"):
            KramersKronigValidator(invalid_data_non_monotonic)

    def test_insufficient_data_points(self, invalid_data_single_point):
        """Test that insufficient data points raises error."""
        with pytest.raises(ValueError, match="at least 2 data points"):
            KramersKronigValidator(invalid_data_single_point)

    def test_data_with_nans_raises_error(self, invalid_data_with_nans):
        """Test that data with NaNs raises error."""
        with pytest.raises(ValueError, match="non-numeric values or NaNs"):
            KramersKronigValidator(invalid_data_with_nans)

    # ========== Feature Tests ==========

    def test_context_manager(self, single_debye_data):
        """Test context manager functionality."""
        with KramersKronigValidator(single_debye_data) as validator:
            results = validator.validate()
            assert validator.results  # Results should exist
            assert validator._eps_inf_cache is not None

        # After exiting context, results should be cleared
        assert not validator.results
        assert validator._eps_inf_cache is None

    def test_properties_before_validation(self, single_debye_data):
        """Test that properties raise error before validation."""
        validator = KramersKronigValidator(single_debye_data)

        with pytest.raises(RuntimeError, match="Must call validate"):
            _ = validator.is_causal

        with pytest.raises(RuntimeError, match="Must call validate"):
            _ = validator.relative_error

    def test_properties_after_validation(self, single_debye_data):
        """Test properties work correctly after validation."""
        validator = KramersKronigValidator(single_debye_data)
        validator.validate()

        # Properties should now work
        assert isinstance(validator.is_causal, bool)
        assert isinstance(validator.relative_error, float)
        assert validator.relative_error >= 0

    def test_get_diagnostics(self, double_debye_data):
        """Test diagnostic information retrieval."""
        validator = KramersKronigValidator(double_debye_data)
        diagnostics = validator.get_diagnostics()

        # Check all expected keys
        expected_keys = [
            'grid_uniform', 'num_points', 'freq_range_ghz',
            'eps_inf', 'method_used', 'num_peaks', 'max_df_freq_ghz',
            'mean_relative_error', 'rmse', 'causality_status'
        ]
        for key in expected_keys:
            assert key in diagnostics

        # Validate diagnostic values
        assert isinstance(diagnostics['grid_uniform'], bool)
        assert diagnostics['num_points'] == len(double_debye_data)
        assert len(diagnostics['freq_range_ghz']) == 2
        assert diagnostics['freq_range_ghz'][0] < diagnostics['freq_range_ghz'][1]

    def test_get_report(self, single_debye_data):
        """Test report generation."""
        validator = KramersKronigValidator(single_debye_data)

        # Before validation
        report_before = validator.get_report()
        assert "not been run" in report_before

        # After validation
        validator.validate()
        report_after = validator.get_report()
        assert "PASS" in report_after or "FAIL" in report_after
        assert "Mean Relative Error" in report_after
        assert "RMSE" in report_after
        assert "=" * 50 in report_after  # Check formatting

    def test_to_dict_format(self, single_debye_data):
        """Test standardized dictionary output format."""
        validator = KramersKronigValidator(single_debye_data)
        result_dict = validator.to_dict()

        # Check required keys
        required_keys = [
            'model_name', 'freq_ghz', 'eps_fit', 'success',
            'rmse', 'aic', 'bic', 'params', 'causality_report'
        ]
        for key in required_keys:
            assert key in result_dict

        # Validate specific values
        assert result_dict['model_name'] == 'Kramers-Kronig'
        assert isinstance(result_dict['eps_fit'], np.ndarray)
        assert result_dict['eps_fit'].dtype == complex
        assert len(result_dict['eps_fit']) == len(single_debye_data)
        assert result_dict['aic'] == np.inf
        assert result_dict['bic'] == np.inf
        assert 'eps_inf' in result_dict['params']
        assert isinstance(result_dict['success'], bool)

    def test_peak_detection(self, single_debye_data, double_debye_data):
        """Test peak detection functionality."""
        # Single Debye should have 1 peak
        validator_single = KramersKronigValidator(single_debye_data)
        df_np = validator_single.df_exp.to_numpy()
        peaks_single = validator_single._detect_peaks(df_np)
        assert peaks_single == 1

        # Double Debye should have 2 peaks (or possibly 1 if merged)
        validator_double = KramersKronigValidator(double_debye_data)
        df_np = validator_double.df_exp.to_numpy()
        peaks_double = validator_double._detect_peaks(df_np)
        assert peaks_double >= 1
        # For well-separated relaxations, should detect 2 peaks
        if validator_double.freq_ghz.max() / validator_double.freq_ghz.min() > 1000:
            assert peaks_double == 2

    def test_eps_inf_caching(self, single_debye_data):
        """Test that eps_inf is properly cached."""
        validator = KramersKronigValidator(single_debye_data)

        # First call should compute and cache
        eps_inf_1 = validator._estimate_eps_inf()
        assert validator._eps_inf_cache is not None

        # Second call should return cached value
        eps_inf_2 = validator._estimate_eps_inf()
        assert eps_inf_1 == eps_inf_2
        assert validator._eps_inf_cache == eps_inf_1

    # ========== Edge Case Tests ==========

    def test_edge_case_wide_frequency_range(self, edge_case_data):
        """Test validation over very wide frequency range (6 decades)."""
        validator = KramersKronigValidator(edge_case_data)
        results = validator.validate()

        assert 'dk_kk' in results
        assert results['causality_status'] in ['PASS', 'FAIL']
        # Wide frequency range might have larger errors
        assert results['mean_relative_error'] < 0.1

    def test_minimal_data_validation(self, minimal_data):
        """Test validation with minimal dataset."""
        validator = KramersKronigValidator(minimal_data)

        # Should work but with limitations
        results = validator.validate()
        assert 'dk_kk' in results

        # Check that fit method falls back to mean for small datasets
        validator_fit = KramersKronigValidator(
            minimal_data,
            eps_inf_method='fit',
            min_tail_points=2
        )
        with warnings.catch_warnings(record=True) as w:
            eps_inf = validator_fit._estimate_eps_inf()
            # Should have warned about falling back to mean
            assert any("Falling back to 'mean'" in str(warning.message) for warning in w)

    def test_numba_availability(self, single_debye_data):
        """Test that code works with or without Numba."""
        validator = KramersKronigValidator(single_debye_data, method='trapz')
        results = validator.validate()

        # Should work regardless of Numba availability
        assert 'dk_kk' in results
        assert results['causality_status'] in ['PASS', 'FAIL']

        # If testing both conditions, can mock NUMBA_AVAILABLE
        # This would require modifying the import structure


# ========== Parametrized Tests ==========

@pytest.mark.parametrize("method", ['auto', 'hilbert', 'trapz'])
def test_all_methods_consistency(single_debye_data_uniform, method):
    """Test that all methods give consistent results for good data."""
    validator = KramersKronigValidator(single_debye_data_uniform, method=method)
    results = validator.validate()

    # All methods should pass for ideal Debye data on uniform grid
    assert results['causality_status'] == 'PASS'
    assert results['mean_relative_error'] < 0.05


@pytest.mark.parametrize("window", ['hamming', 'hann', 'blackman', 'bartlett', 'kaiser', 'tukey'])
def test_all_window_functions(single_debye_data_uniform, window):
    """Test all supported window functions."""
    if window in ['kaiser', 'tukey']:
        # These windows need additional parameters in scipy
        # Skip or handle appropriately
        pytest.skip(f"Window {window} requires additional parameters")

    validator = KramersKronigValidator(
        single_debye_data_uniform,
        method='hilbert',
        window=window
    )
    results = validator.validate()
    assert results['causality_status'] == 'PASS'


@pytest.mark.parametrize("tail_fraction,expected_accuracy", [
    (0.05, 1.0),   # Small tail, less accurate
    (0.1, 0.5),    # Default
    (0.2, 0.3),    # Larger tail
    (0.3, 0.2),   # Even larger tail, more accurate
])
def test_tail_fraction_accuracy(single_debye_data_uniform, tail_fraction, expected_accuracy):
    """Test that larger tail fractions give more accurate eps_inf."""
    validator = KramersKronigValidator(
        single_debye_data_uniform,
        tail_fraction=tail_fraction,
        eps_inf_method='mean'
    )
    eps_inf = validator._estimate_eps_inf()

    # True value is 3.0
    assert abs(eps_inf - 3.0) < expected_accuracy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])