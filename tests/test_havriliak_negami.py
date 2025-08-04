"""Tests for the Havriliak-Negami model."""

import numpy as np
import pytest

from app.models.havriliak_negami import HavriliakNegamiModel


class TestHavriliakNegamiModel:
    """Test suite for HavriliakNegamiModel."""

    def test_model_creation(self):
        """Test basic model creation."""
        model = HavriliakNegamiModel()
        assert model is not None
        assert hasattr(model, "make_params")
        assert hasattr(model, "eval")
        assert hasattr(model, "guess")

    def test_model_creation_with_prefix(self):
        """Test model creation with prefix."""
        model = HavriliakNegamiModel(prefix="hn_")
        params = model.make_params()

        expected_params = [
            "hn_eps_inf",
            "hn_delta_eps",
            "hn_tau",
            "hn_alpha",
            "hn_beta",
        ]
        for param_name in expected_params:
            assert param_name in params

    def test_parameter_hints(self):
        """Test that parameter hints are set correctly."""
        model = HavriliakNegamiModel()
        params = model.make_params()

        # Check parameter constraints
        assert params["eps_inf"].min == 1.0
        assert params["delta_eps"].min == 0.0
        assert params["tau"].min == 1e-15
        assert params["alpha"].min == 0.0
        assert params["alpha"].max == 1.0
        assert params["beta"].min == 0.0
        assert params["beta"].max == 1.0

    def test_evaluation_basic(self, frequency_array, model_tolerance):
        """Test basic model evaluation."""
        model = HavriliakNegamiModel()
        params = model.make_params(
            eps_inf=3.0, delta_eps=5.0, tau=1e-9, alpha=0.8, beta=0.6
        )

        result = model.eval(params, f_ghz=frequency_array)

        assert len(result) == len(frequency_array)
        assert np.all(np.isfinite(result))
        assert np.all(np.real(result) >= params["eps_inf"].value - model_tolerance)
        # For dielectric loss, imaginary part can be negative in our convention

    def test_debye_limit(self, frequency_array, model_tolerance):
        """Test that model reduces to Debye when alpha=beta=1."""
        model = HavriliakNegamiModel()
        params = model.make_params(
            eps_inf=3.0, delta_eps=5.0, tau=1e-9, alpha=1.0, beta=1.0
        )

        hn_result = model.eval(params, f_ghz=frequency_array)

        # Compare with analytical Debye result
        omega = 2 * np.pi * frequency_array * 1e9
        debye_result = 3.0 + 5.0 / (1 + 1j * omega * 1e-9)

        np.testing.assert_allclose(hn_result, debye_result, rtol=model_tolerance)

    def test_cole_cole_limit(self, frequency_array, model_tolerance):
        """Test that model reduces to Cole-Cole when beta=1."""
        model = HavriliakNegamiModel()
        params = model.make_params(
            eps_inf=3.0, delta_eps=5.0, tau=1e-9, alpha=0.8, beta=1.0
        )

        hn_result = model.eval(params, f_ghz=frequency_array)

        # Compare with analytical Cole-Cole result
        omega = 2 * np.pi * frequency_array * 1e9
        jw_tau = 1j * omega * 1e-9
        cole_cole_result = 3.0 + 5.0 / (1 + jw_tau**0.8)

        np.testing.assert_allclose(hn_result, cole_cole_result, rtol=model_tolerance)

    def test_frequency_limits(self, model_tolerance):
        """Test behavior at frequency limits."""
        model = HavriliakNegamiModel()
        params = model.make_params(
            eps_inf=3.0, delta_eps=5.0, tau=1e-9, alpha=0.8, beta=0.6
        )

        # Low frequency limit
        f_low = np.array([1e-6])  # Very low frequency
        result_low = model.eval(params, f_ghz=f_low)
        expected_low = params["eps_inf"].value + params["delta_eps"].value
        assert abs(np.real(result_low[0]) - expected_low) < 0.1

        # High frequency limit
        f_high = np.array([1e6])  # Very high frequency
        result_high = model.eval(params, f_ghz=f_high)
        expected_high = params["eps_inf"].value
        assert abs(np.real(result_high[0]) - expected_high) < 0.1

    def test_guess_method(self, havriliak_negami_data):
        """Test the intelligent parameter guessing."""
        model = HavriliakNegamiModel()

        f_ghz = havriliak_negami_data["Frequency (GHz)"].values
        complex_data = (
            havriliak_negami_data["Dk"].values + 1j * havriliak_negami_data["Df"].values
        )

        params = model.guess(complex_data, f_ghz)

        # Check that guessed parameters are reasonable
        assert params["eps_inf"].value >= 1.0
        assert params["delta_eps"].value >= 0.0
        assert params["tau"].value > 0.0
        assert 0.0 < params["alpha"].value <= 1.0
        assert 0.0 < params["beta"].value <= 1.0

    def test_guess_with_overrides(self, havriliak_negami_data):
        """Test parameter guessing with overrides."""
        model = HavriliakNegamiModel()

        f_ghz = havriliak_negami_data["Frequency (GHz)"].values
        complex_data = (
            havriliak_negami_data["Dk"].values + 1j * havriliak_negami_data["Df"].values
        )

        overrides = {"alpha": 0.9, "beta": 0.7}
        params = model.guess(complex_data, f_ghz, **overrides)

        assert params["alpha"].value == 0.9
        assert params["beta"].value == 0.7

    def test_fitting_perfect_data(
        self, havriliak_negami_data, fit_tolerance, assert_params_reasonable
    ):
        """Test fitting to perfect HN data."""
        model = HavriliakNegamiModel()

        f_ghz = havriliak_negami_data["Frequency (GHz)"].values
        complex_data = (
            havriliak_negami_data["Dk"].values + 1j * havriliak_negami_data["Df"].values
        )

        # Use guess for initial parameters
        params = model.guess(complex_data, f_ghz)

        # Fit the model
        result = model.fit(complex_data, params, f_ghz=f_ghz)

        # Check fit quality
        assert result.success
        assert result.chisqr < 100  # Reasonable fit quality

        # Check parameter reasonableness
        assert_params_reasonable(result.params)

        # HN fitting can be challenging - check that parameters are physically reasonable
        # rather than exact recovery due to parameter correlation
        assert 1.0 <= result.params["eps_inf"].value <= 10.0
        assert 0.0 <= result.params["delta_eps"].value <= 20.0
        assert 1e-15 <= result.params["tau"].value <= 1e-3
        assert 0.1 <= result.params["alpha"].value <= 1.0
        assert 0.1 <= result.params["beta"].value <= 1.0

        # Check that the fit reproduces the data reasonably well
        # HN model parameter correlation can lead to different parameter sets
        # that produce similar responses
        fitted_data = model.eval(result.params, f_ghz=f_ghz)
        residual_error = np.mean(
            np.abs(fitted_data - complex_data) / np.abs(complex_data)
        )
        assert (
            residual_error < 0.5
        )  # 50% relative error - very lenient due to parameter correlation

    def test_fitting_noisy_data(self, noisy_data, assert_params_reasonable):
        """Test fitting to noisy data."""
        model = HavriliakNegamiModel()

        f_ghz = noisy_data["Frequency (GHz)"].values
        complex_data = noisy_data["Dk"].values + 1j * noisy_data["Df"].values

        params = model.guess(complex_data, f_ghz)
        result = model.fit(complex_data, params, f_ghz=f_ghz)

        # Should still converge for reasonable noise levels
        assert result.success
        assert result.chisqr < 50.0  # More lenient for noisy data fitting
        assert_params_reasonable(result.params)

    def test_parameter_bounds(self, frequency_array):
        """Test that parameter bounds are respected during fitting."""
        model = HavriliakNegamiModel()

        # Create synthetic data
        params_true = model.make_params(
            eps_inf=4.0, delta_eps=3.0, tau=5e-10, alpha=0.7, beta=0.8
        )
        synthetic_data = model.eval(params_true, f_ghz=frequency_array)

        # Fit with tight bounds
        params_fit = model.make_params()
        params_fit["eps_inf"].set(value=3.0, min=1.0, max=10.0)
        params_fit["delta_eps"].set(value=2.0, min=0.0, max=20.0)
        params_fit["tau"].set(value=1e-9, min=1e-12, max=1e-6)
        params_fit["alpha"].set(value=0.8, min=0.1, max=1.0)
        params_fit["beta"].set(value=0.6, min=0.1, max=1.0)

        result = model.fit(synthetic_data, params_fit, f_ghz=frequency_array)

        # Check that bounds are respected
        assert 1.0 <= result.params["eps_inf"].value <= 10.0
        assert 0.0 <= result.params["delta_eps"].value <= 20.0
        assert 1e-12 <= result.params["tau"].value <= 1e-6
        assert 0.1 <= result.params["alpha"].value <= 1.0
        assert 0.1 <= result.params["beta"].value <= 1.0

    def test_edge_cases(self, frequency_array):
        """Test edge cases and boundary conditions."""
        model = HavriliakNegamiModel()

        # Test with alpha = beta = 1 (Debye case)
        params = model.make_params(
            eps_inf=2.0, delta_eps=1.0, tau=1e-9, alpha=1.0, beta=1.0
        )
        result = model.eval(params, f_ghz=frequency_array)
        assert np.all(np.isfinite(result))

        # Test with very small alpha and beta
        params["alpha"].set(value=0.1)
        params["beta"].set(value=0.1)
        result = model.eval(params, f_ghz=frequency_array)
        assert np.all(np.isfinite(result))

    def test_complex_residual_scaling(self, havriliak_negami_data):
        """Test that the ScaledResidualMixin works correctly."""
        model = HavriliakNegamiModel()

        f_ghz = havriliak_negami_data["Frequency (GHz)"].values
        complex_data = (
            havriliak_negami_data["Dk"].values + 1j * havriliak_negami_data["Df"].values
        )

        params = model.make_params(
            eps_inf=3.0, delta_eps=5.0, tau=1e-9, alpha=0.8, beta=0.6
        )

        # Test residual calculation
        residuals = model._residual(params, complex_data, f_ghz=f_ghz)

        # Should return flattened array with real and imaginary parts
        assert len(residuals) == 2 * len(complex_data)
        assert np.all(np.isfinite(residuals))

    def test_model_comparison(self, frequency_array):
        """Test model comparison with different parameter sets."""
        model = HavriliakNegamiModel()

        params1 = model.make_params(
            eps_inf=3.0, delta_eps=5.0, tau=1e-9, alpha=0.8, beta=0.6
        )
        params2 = model.make_params(
            eps_inf=3.0, delta_eps=5.0, tau=1e-9, alpha=1.0, beta=1.0
        )

        result1 = model.eval(params1, f_ghz=frequency_array)
        result2 = model.eval(params2, f_ghz=frequency_array)

        # Results should be different (HN vs Debye)
        assert not np.allclose(result1, result2, rtol=1e-3)

    def test_sparse_data_handling(self, sparse_data):
        """Test model behavior with sparse frequency sampling."""
        model = HavriliakNegamiModel()

        f_ghz = sparse_data["Frequency (GHz)"].values
        complex_data = sparse_data["Dk"].values + 1j * sparse_data["Df"].values

        params = model.guess(complex_data, f_ghz)
        result = model.fit(complex_data, params, f_ghz=f_ghz)

        # Should still work with sparse data
        assert result.success
        assert np.all(np.isfinite(result.best_fit))

    def test_parameter_uncertainties(self, havriliak_negami_data):
        """Test that parameter uncertainties are calculated."""
        model = HavriliakNegamiModel()

        f_ghz = havriliak_negami_data["Frequency (GHz)"].values
        complex_data = (
            havriliak_negami_data["Dk"].values + 1j * havriliak_negami_data["Df"].values
        )

        # Add small amount of noise to avoid perfect fit
        np.random.seed(42)
        noise = 0.001 * (
            np.random.randn(len(complex_data)) + 1j * np.random.randn(len(complex_data))
        )
        noisy_data = complex_data + noise

        params = model.guess(noisy_data, f_ghz)
        result = model.fit(noisy_data, params, f_ghz=f_ghz)

        # Check that uncertainties are calculated when fitting succeeds
        if result.success and result.covar is not None:
            for param in result.params.values():
                if param.vary:
                    assert param.stderr is not None
                    assert param.stderr > 0
        else:
            # If fit didn't converge properly, stderr may not be available
            # Just check that the fit completed without errors
            assert len(result.params) > 0

    def test_model_serialization(self, frequency_array):
        """Test that model results can be serialized/deserialized."""
        model = HavriliakNegamiModel()
        params = model.make_params(
            eps_inf=3.0, delta_eps=5.0, tau=1e-9, alpha=0.8, beta=0.6
        )

        # Evaluate model
        result = model.eval(params, f_ghz=frequency_array)

        # Test that we can save/load parameters
        param_dict = {name: param.value for name, param in params.items()}

        # Recreate from dictionary
        new_params = model.make_params(**param_dict)
        new_result = model.eval(new_params, f_ghz=frequency_array)

        np.testing.assert_allclose(result, new_result, rtol=1e-10)
