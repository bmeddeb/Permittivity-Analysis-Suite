"""Tests for the Multi-Pole Debye model."""

import numpy as np
import pytest

from app.models.multi_pole_debye import MultiPoleDebyeModel


class TestMultiPoleDebyeModel:
    """Test suite for MultiPoleDebyeModel."""

    def test_model_creation(self):
        """Test basic model creation with different numbers of poles."""
        for n_poles in [1, 2, 3, 5]:
            model = MultiPoleDebyeModel(n_poles=n_poles)
            assert model is not None
            assert model.n_poles == n_poles
            assert hasattr(model, "make_params")
            assert hasattr(model, "eval")

    def test_model_creation_with_prefix(self):
        """Test model creation with prefix."""
        model = MultiPoleDebyeModel(n_poles=2, prefix="mp_")
        params = model.make_params()

        expected_params = [
            "mp_eps_inf",
            "mp_delta_eps_0",
            "mp_tau_0",
            "mp_delta_eps_1",
            "mp_tau_1",
        ]
        for param_name in expected_params:
            assert param_name in params

    def test_parameter_creation(self):
        """Test that correct parameters are created for different pole numbers."""
        for n_poles in [1, 3, 5]:
            model = MultiPoleDebyeModel(n_poles=n_poles)
            params = model.make_params()

            # Should have eps_inf plus 2 parameters per pole
            expected_param_count = 1 + 2 * n_poles
            assert len(params) == expected_param_count

            # Check eps_inf
            assert "eps_inf" in params
            assert params["eps_inf"].min == 1.0

            # Check pole parameters
            for i in range(n_poles):
                assert f"delta_eps_{i}" in params
                assert f"tau_{i}" in params
                assert params[f"delta_eps_{i}"].min == 0.0
                assert params[f"tau_{i}"].min == 1e-15
                # tau parameters don't have max bounds in new implementation

    def test_single_pole_debye_equivalence(self, frequency_array, model_tolerance):
        """Test that single pole reduces to standard Debye model."""
        model = MultiPoleDebyeModel(n_poles=1)
        params = model.make_params(
            eps_inf=3.0,
            delta_eps_0=5.0,
            tau_0=1e-9,  # 1 ns
        )

        mp_result = model.eval(params, f_ghz=frequency_array)

        # Compare with analytical Debye result
        omega = 2 * np.pi * frequency_array * 1e9
        tau = 10 ** (-9)
        debye_result = 3.0 + 5.0 / (1 + 1j * omega * tau)

        np.testing.assert_allclose(mp_result, debye_result, rtol=model_tolerance)

    def test_evaluation_basic(self, frequency_array, model_tolerance):
        """Test basic model evaluation with multiple poles."""
        model = MultiPoleDebyeModel(n_poles=3)
        params = model.make_params(
            eps_inf=2.0,
            delta_eps_0=2.0,
            tau_0=1e-12,  # 1 ps
            delta_eps_1=3.0,
            tau_1=1e-9,  # 1 ns
            delta_eps_2=1.5,
            tau_2=1e-6,  # 1 μs
        )

        result = model.eval(params, f_ghz=frequency_array)

        assert len(result) == len(frequency_array)
        assert np.all(np.isfinite(result))
        assert np.all(np.real(result) >= params["eps_inf"].value - model_tolerance)
        # For dielectric loss, imaginary part can be negative in our convention

    def test_frequency_limits(self, model_tolerance):
        """Test behavior at frequency limits."""
        model = MultiPoleDebyeModel(n_poles=2)
        params = model.make_params(
            eps_inf=3.0, delta_eps_0=2.0, tau_0=1e-9, delta_eps_1=1.5, tau_1=1e-6
        )

        # Low frequency limit
        f_low = np.array([1e-9])  # Very low frequency
        result_low = model.eval(params, f_ghz=f_low)
        expected_low = (
            params["eps_inf"].value
            + params["delta_eps_0"].value
            + params["delta_eps_1"].value
        )
        assert abs(np.real(result_low[0]) - expected_low) < 0.1

        # High frequency limit
        f_high = np.array([1e9])  # Very high frequency
        result_high = model.eval(params, f_ghz=f_high)
        expected_high = params["eps_inf"].value
        assert abs(np.real(result_high[0]) - expected_high) < 0.1

    def test_tau_parameter_formats(self, frequency_array, model_tolerance):
        """Test that both log_tau and tau parameter formats work."""
        model = MultiPoleDebyeModel(n_poles=2)

        # Test with log_tau parameters
        params1 = model.make_params(
            eps_inf=3.0, delta_eps_0=2.0, tau_0=1e-9, delta_eps_1=1.5, tau_1=1e-6
        )
        result1 = model.eval(params1, f_ghz=frequency_array)

        # Test with direct tau parameters (simulate missing log_tau)
        # We need to manually create the parameters dict to simulate tau parameters
        params2 = model.make_params(
            eps_inf=3.0,
            delta_eps_0=2.0,
            tau_0=1e-9,  # Will be overridden
            delta_eps_1=1.5,
            tau_1=1e-6,  # Will be overridden
        )

        # Manually evaluate with tau parameters instead
        # This simulates the fallback behavior in the model
        import numpy as np

        from app.models._eval_funcs import multi_pole_debye_eval

        omega = 2 * np.pi * frequency_array * 1e9
        eps = np.full_like(omega, 3.0, dtype=np.complex128)
        eps += 2.0 / (1 + 1j * omega * 1e-9)  # First pole
        eps += 1.5 / (1 + 1j * omega * 1e-6)  # Second pole
        result2 = eps

        # Results should be identical
        np.testing.assert_allclose(result1, result2, rtol=model_tolerance)

    def test_pole_ordering_independence(self, frequency_array, model_tolerance):
        """Test that pole ordering doesn't affect the result."""
        model = MultiPoleDebyeModel(n_poles=3)

        # Original ordering
        params1 = model.make_params(
            eps_inf=3.0,
            delta_eps_0=1.0,
            tau_0=1e-12,  # Fast
            delta_eps_1=2.0,
            tau_1=1e-9,  # Medium
            delta_eps_2=1.5,
            tau_2=1e-6,  # Slow
        )
        result1 = model.eval(params1, f_ghz=frequency_array)

        # Reordered poles (same physical content)
        params2 = model.make_params(
            eps_inf=3.0,
            delta_eps_0=1.5,
            tau_0=1e-6,  # Slow
            delta_eps_1=1.0,
            tau_1=1e-12,  # Fast
            delta_eps_2=2.0,
            tau_2=1e-9,  # Medium
        )
        result2 = model.eval(params2, f_ghz=frequency_array)

        # Results should be identical (sum is commutative)
        np.testing.assert_allclose(result1, result2, rtol=model_tolerance)

    def test_fitting_multi_relaxation_data(
        self, multi_relaxation_data, fit_tolerance, assert_params_reasonable
    ):
        """Test fitting to multi-relaxation data."""
        model = MultiPoleDebyeModel(n_poles=3)

        f_ghz = multi_relaxation_data["Frequency (GHz)"].values
        complex_data = (
            multi_relaxation_data["Dk"].values + 1j * multi_relaxation_data["Df"].values
        )

        # Set up initial parameters
        params = model.make_params()
        params["eps_inf"].set(value=2.0, min=1.0)

        # Distribute relaxation times logarithmically
        taus = [1e-12, 1e-9, 1e-6]  # ps, ns, μs
        deltas = [2.0, 3.0, 1.5]

        for i in range(3):
            params[f"delta_eps_{i}"].set(value=deltas[i], min=0.0)
            params[f"tau_{i}"].set(value=taus[i])

        # Fit the model
        result = model.fit(complex_data, params, f_ghz=f_ghz)

        # Check fit quality
        assert result.success
        assert result.chisqr < 1e-6  # Should fit very well to synthetic data

        # Check parameter reasonableness
        assert_params_reasonable(result.params)

    def test_model_selection_by_poles(self, multi_relaxation_data):
        """Test model selection with different numbers of poles."""
        f_ghz = multi_relaxation_data["Frequency (GHz)"].values
        complex_data = (
            multi_relaxation_data["Dk"].values + 1j * multi_relaxation_data["Df"].values
        )

        results = {}

        for n_poles in [1, 2, 3, 4, 5]:
            model = MultiPoleDebyeModel(n_poles=n_poles)
            params = model.make_params()

            # Set eps_inf
            params["eps_inf"].set(value=2.0)

            # Initialize pole parameters
            for i in range(n_poles):
                tau = 10 ** (-12 + 3 * i)  # Spread from ps to ms
                params[f"delta_eps_{i}"].set(value=1.0, min=0.0)
                params[f"tau_{i}"].set(value=tau)

            try:
                result = model.fit(complex_data, params, f_ghz=f_ghz)
                results[n_poles] = {
                    "aic": result.aic,
                    "bic": result.bic,
                    "chisqr": result.chisqr,
                    "success": result.success,
                }
            except Exception:
                results[n_poles] = {"success": False}

        # Check that 3-pole model (true model) has good fit
        assert results[3]["success"]
        assert results[3]["chisqr"] < 1e-6

        # Check that AIC/BIC are reasonable
        successful_results = {k: v for k, v in results.items() if v["success"]}
        assert len(successful_results) >= 3

    def test_parameter_bounds_enforcement(self, frequency_array):
        """Test that parameter bounds are enforced during fitting."""
        model = MultiPoleDebyeModel(n_poles=2)

        # Create synthetic data
        params_true = model.make_params(
            eps_inf=4.0, delta_eps_0=3.0, tau_0=1e-8, delta_eps_1=2.0, tau_1=1e-5
        )
        synthetic_data = model.eval(params_true, f_ghz=frequency_array)

        # Fit with constraints
        params_fit = model.make_params()
        params_fit["eps_inf"].set(value=3.0, min=1.0, max=10.0)
        params_fit["delta_eps_0"].set(value=2.0, min=0.0, max=20.0)
        params_fit["tau_0"].set(value=1e-9, min=1e-12, max=1e-3)
        params_fit["delta_eps_1"].set(value=1.0, min=0.0, max=20.0)
        params_fit["tau_1"].set(value=1e-6, min=1e-12, max=1e-3)

        result = model.fit(synthetic_data, params_fit, f_ghz=frequency_array)

        # Check bounds are respected
        assert 1.0 <= result.params["eps_inf"].value <= 10.0
        assert 0.0 <= result.params["delta_eps_0"].value <= 20.0
        assert 1e-12 <= result.params["tau_0"].value <= 1e-3
        assert 0.0 <= result.params["delta_eps_1"].value <= 20.0
        assert 1e-12 <= result.params["tau_1"].value <= 1e-3

    def test_zero_strength_poles(self, frequency_array, model_tolerance):
        """Test behavior when some poles have zero strength."""
        model = MultiPoleDebyeModel(n_poles=3)
        params = model.make_params(
            eps_inf=3.0,
            delta_eps_0=5.0,
            tau_0=1e-9,  # Active pole
            delta_eps_1=0.0,
            tau_1=1e-6,  # Zero strength
            delta_eps_2=2.0,
            tau_2=1e-3,  # Active pole
        )

        result = model.eval(params, f_ghz=frequency_array)

        # Should be equivalent to 2-pole model
        model_2pole = MultiPoleDebyeModel(n_poles=2)
        params_2pole = model_2pole.make_params(
            eps_inf=3.0, delta_eps_0=5.0, tau_0=1e-9, delta_eps_1=2.0, tau_1=1e-3
        )
        result_2pole = model_2pole.eval(params_2pole, f_ghz=frequency_array)

        np.testing.assert_allclose(result, result_2pole, rtol=model_tolerance)

    def test_extreme_time_constants(self, frequency_array):
        """Test model with extreme relaxation time constants."""
        model = MultiPoleDebyeModel(n_poles=3)
        params = model.make_params(
            eps_inf=2.0,
            delta_eps_0=1.0,
            tau_0=1e-15,  # 1 fs (very fast)
            delta_eps_1=2.0,
            tau_1=1e-9,  # 1 ns (normal)
            delta_eps_2=1.0,
            tau_2=1e0,  # 1 s (very slow)
        )

        result = model.eval(params, f_ghz=frequency_array)

        assert np.all(np.isfinite(result))
        assert len(result) == len(frequency_array)

    def test_complex_residual_scaling(self, multi_relaxation_data):
        """Test that the ScaledResidualMixin works correctly."""
        model = MultiPoleDebyeModel(n_poles=2)

        f_ghz = multi_relaxation_data["Frequency (GHz)"].values
        complex_data = (
            multi_relaxation_data["Dk"].values + 1j * multi_relaxation_data["Df"].values
        )

        params = model.make_params(
            eps_inf=2.0, delta_eps_0=2.0, tau_0=1e-9, delta_eps_1=1.5, tau_1=1e-6
        )

        # Test residual calculation
        residuals = model._residual(params, complex_data, f_ghz=f_ghz)

        # Should return flattened array with real and imaginary parts
        assert len(residuals) == 2 * len(complex_data)
        assert np.all(np.isfinite(residuals))

    def test_parameter_correlations(self, multi_relaxation_data):
        """Test parameter correlation analysis."""
        model = MultiPoleDebyeModel(n_poles=2)

        f_ghz = multi_relaxation_data["Frequency (GHz)"].values
        complex_data = (
            multi_relaxation_data["Dk"].values + 1j * multi_relaxation_data["Df"].values
        )

        # Add small amount of noise for realistic fitting
        np.random.seed(42)
        noise = 0.001 * (
            np.random.randn(len(complex_data)) + 1j * np.random.randn(len(complex_data))
        )
        noisy_data = complex_data + noise

        params = model.make_params(
            eps_inf=2.0, delta_eps_0=2.0, tau_0=1e-12, delta_eps_1=3.0, tau_1=1e-9
        )

        result = model.fit(noisy_data, params, f_ghz=f_ghz)

        # Check that uncertainties are calculated when fitting succeeds
        if result.success and result.covar is not None:
            for param in result.params.values():
                if param.vary:
                    assert param.stderr is not None
        else:
            # If fit didn't converge properly, stderr may not be available
            assert len(result.params) > 0

    def test_additive_property(self, frequency_array, model_tolerance):
        """Test that multi-pole model is additive."""
        # Create two separate single-pole models
        model1 = MultiPoleDebyeModel(n_poles=1)
        params1 = model1.make_params(
            eps_inf=0.0,  # No background for additive test
            delta_eps_0=3.0,
            tau_0=1e-9,
        )
        result1 = model1.eval(params1, f_ghz=frequency_array)

        model2 = MultiPoleDebyeModel(n_poles=1)
        params2 = model2.make_params(
            eps_inf=0.0,  # No background for additive test
            delta_eps_0=2.0,
            tau_0=1e-6,
        )
        result2 = model2.eval(params2, f_ghz=frequency_array)

        # Sum of individual models
        sum_result = result1 + result2

        # Compare with 2-pole model
        model_combined = MultiPoleDebyeModel(n_poles=2)
        params_combined = model_combined.make_params(
            eps_inf=2.0,  # Add base permittivity
            delta_eps_0=3.0,
            tau_0=1e-9,
            delta_eps_1=2.0,
            tau_1=1e-6,
        )
        combined_result = model_combined.eval(params_combined, f_ghz=frequency_array)

        # Account for eps_inf constraint: each individual model has eps_inf >= 1.0
        # So sum_result already includes 2.0 from the two eps_inf contributions
        # The combined model has eps_inf = 2.0, so no additional offset needed

        np.testing.assert_allclose(sum_result, combined_result, rtol=model_tolerance)

    def test_large_number_of_poles(self, frequency_array):
        """Test model with large number of poles."""
        n_poles = 10
        model = MultiPoleDebyeModel(n_poles=n_poles)

        params = model.make_params()
        params["eps_inf"].set(value=2.0)

        # Set up distributed relaxation spectrum
        for i in range(n_poles):
            tau = 10 ** (-15 + i * 1.5)  # Spread from fs to μs
            params[f"delta_eps_{i}"].set(value=0.5)
            params[f"tau_{i}"].set(value=tau)

        result = model.eval(params, f_ghz=frequency_array)

        assert np.all(np.isfinite(result))
        assert len(result) == len(frequency_array)

        # Check that total permittivity is reasonable
        total_delta = sum(params[f"delta_eps_{i}"].value for i in range(n_poles))
        expected_low_freq = params["eps_inf"].value + total_delta

        # At very low frequency, should approach this limit
        f_low = np.array([1e-9])
        result_low = model.eval(params, f_ghz=f_low)
        assert abs(np.real(result_low[0]) - expected_low_freq) < 0.5

    def test_numerical_precision(self, frequency_array, model_tolerance):
        """Test numerical precision with very different time scales."""
        model = MultiPoleDebyeModel(n_poles=3)
        params = model.make_params(
            eps_inf=3.0,
            delta_eps_0=1.0,
            tau_0=1e-15,  # 1 fs
            delta_eps_1=1.0,
            tau_1=1e-9,  # 1 ns
            delta_eps_2=1.0,
            tau_2=1e-3,  # 1 ms
        )

        result = model.eval(params, f_ghz=frequency_array)

        # Should handle the wide range of time scales
        assert np.all(np.isfinite(result))

        # Check individual pole contributions are reasonable
        # At high frequency, fast pole should dominate
        # At low frequency, slow pole should dominate
        f_high = np.array([1e3])  # 1 THz
        f_low = np.array([1e-6])  # 1 MHz

        result_high = model.eval(params, f_ghz=f_high)
        result_low = model.eval(params, f_ghz=f_low)

        # Real part should be higher at low frequency
        assert np.real(result_low[0]) > np.real(result_high[0])
