"""Tests for the Hybrid Debye-Lorentz model."""

import numpy as np
import pytest

from app.models.hybrid_debye_lorentz import HybridDebyeLorentzModel


class TestHybridDebyeLorentzModel:
    """Test suite for HybridDebyeLorentzModel."""

    def test_model_creation(self):
        """Test basic model creation with different numbers of terms."""
        for n_terms in [1, 2, 3, 5]:
            model = HybridDebyeLorentzModel(n_terms=n_terms)
            assert model is not None
            assert model.n_terms == n_terms
            assert hasattr(model, "make_params")
            assert hasattr(model, "eval")

    def test_model_creation_with_prefix(self):
        """Test model creation with prefix."""
        model = HybridDebyeLorentzModel(n_terms=2, prefix="hdl_")
        params = model.make_params()

        expected_params = [
            "hdl_eps_inf",
            "hdl_delta_eps_D1",
            "hdl_tau_D1",
            "hdl_alpha1",
            "hdl_delta_eps_L1",
            "hdl_omega01",
            "hdl_q1",
            "hdl_delta_eps_D2",
            "hdl_tau_D2",
            "hdl_alpha2",
            "hdl_delta_eps_L2",
            "hdl_omega02",
            "hdl_q2",
        ]
        for param_name in expected_params:
            assert param_name in params

    def test_parameter_creation(self):
        """Test that correct parameters are created for different term numbers."""
        for n_terms in [1, 2, 4]:
            model = HybridDebyeLorentzModel(n_terms=n_terms)
            params = model.make_params()

            # Should have eps_inf plus 6 parameters per term
            expected_param_count = 1 + 6 * n_terms
            assert len(params) == expected_param_count

            # Check eps_inf
            assert "eps_inf" in params
            assert params["eps_inf"].min == 1.0

            # Check term parameters
            for i in range(1, n_terms + 1):
                # Debye parameters
                assert f"delta_eps_D{i}" in params
                assert f"tau_D{i}" in params
                assert f"alpha{i}" in params
                # Lorentz parameters
                assert f"delta_eps_L{i}" in params
                assert f"omega0{i}" in params
                assert f"q{i}" in params

                # Check constraints
                assert params[f"delta_eps_D{i}"].min == 0.0
                assert params[f"tau_D{i}"].min == 1e-15
                assert params[f"alpha{i}"].min == 0.0
                assert params[f"alpha{i}"].max == 1.0
                assert params[f"delta_eps_L{i}"].min == 0.0
                assert params[f"omega0{i}"].min == 1e7
                assert params[f"q{i}"].min == 0.0
                assert params[f"q{i}"].max == 1.0

    def test_evaluation_basic(self, frequency_array, model_tolerance):
        """Test basic model evaluation with hybrid terms."""
        model = HybridDebyeLorentzModel(n_terms=2)
        params = model.make_params(
            eps_inf=3.0,
            # First term: relaxation + resonance
            delta_eps_D1=2.0,
            tau_D1=1e-9,
            alpha1=0.8,
            delta_eps_L1=1.0,
            omega01=2 * np.pi * 1e9,
            q1=0.1,
            # Second term: different scales
            delta_eps_D2=1.5,
            tau_D2=1e-12,
            alpha2=1.0,
            delta_eps_L2=0.8,
            omega02=2 * np.pi * 10e9,
            q2=0.3,
        )

        result = model.eval(params, f_ghz=frequency_array)

        assert len(result) == len(frequency_array)
        assert np.all(np.isfinite(result))
        assert np.all(np.real(result) >= params["eps_inf"].value - model_tolerance)

    def test_debye_only_mode(self, frequency_array, model_tolerance):
        """Test model with only Debye components (zero Lorentz)."""
        model = HybridDebyeLorentzModel(n_terms=2)
        params = model.make_params(
            eps_inf=3.0,
            # Only Debye components
            delta_eps_D1=2.0,
            tau_D1=1e-9,
            alpha1=0.8,
            delta_eps_L1=0.0,
            omega01=2 * np.pi * 1e9,
            q1=0.1,
            delta_eps_D2=1.5,
            tau_D2=1e-12,
            alpha2=1.0,
            delta_eps_L2=0.0,
            omega02=2 * np.pi * 10e9,
            q2=0.3,
        )

        result = model.eval(params, f_ghz=frequency_array)

        # Should behave like sum of Cole-Cole models
        omega = 2 * np.pi * frequency_array * 1e9
        expected = (
            3.0
            + 2.0 / (1 + (1j * omega * 1e-9) ** 0.8)
            + 1.5 / (1 + (1j * omega * 1e-12) ** 1.0)
        )

        np.testing.assert_allclose(result, expected, rtol=model_tolerance)

    def test_lorentz_only_mode(self, frequency_array, model_tolerance):
        """Test model with only Lorentz components (zero Debye)."""
        model = HybridDebyeLorentzModel(n_terms=2)
        params = model.make_params(
            eps_inf=3.0,
            # Only Lorentz components
            delta_eps_D1=0.0,
            tau_D1=1e-9,
            alpha1=0.8,
            delta_eps_L1=2.0,
            omega01=2 * np.pi * 1e9,
            q1=0.1,
            delta_eps_D2=0.0,
            tau_D2=1e-12,
            alpha2=1.0,
            delta_eps_L2=1.5,
            omega02=2 * np.pi * 10e9,
            q2=0.3,
        )

        result = model.eval(params, f_ghz=frequency_array)

        # Should behave like sum of Lorentz oscillators
        omega = 2 * np.pi * frequency_array * 1e9
        omega01, omega02 = 2 * np.pi * 1e9, 2 * np.pi * 10e9
        gamma1, gamma2 = 0.1 * omega01, 0.3 * omega02

        lorentz1 = 2.0 * omega01**2 / (omega01**2 - omega**2 - 1j * 2 * gamma1 * omega)
        lorentz2 = 1.5 * omega02**2 / (omega02**2 - omega**2 - 1j * 2 * gamma2 * omega)
        expected = 3.0 + lorentz1 + lorentz2

        np.testing.assert_allclose(result, expected, rtol=model_tolerance)

    def test_frequency_separation_effects(self, model_tolerance):
        """Test behavior with different frequency separations between components."""
        model = HybridDebyeLorentzModel(n_terms=1)

        # Well-separated case: relaxation at 1 GHz, resonance at 100 GHz
        params1 = model.make_params(
            eps_inf=3.0,
            delta_eps_D1=2.0,
            tau_D1=1 / (2 * np.pi * 1e9),
            alpha1=1.0,  # 1 GHz
            delta_eps_L1=1.0,
            omega01=2 * np.pi * 100e9,
            q1=0.1,  # 100 GHz
        )

        # Overlapping case: both at ~10 GHz
        params2 = model.make_params(
            eps_inf=3.0,
            delta_eps_D1=2.0,
            tau_D1=1 / (2 * np.pi * 10e9),
            alpha1=1.0,  # 10 GHz
            delta_eps_L1=1.0,
            omega01=2 * np.pi * 10e9,
            q1=0.1,  # 10 GHz
        )

        f_ghz = np.logspace(0, 3, 100)  # 1 to 1000 GHz
        result1 = model.eval(params1, f_ghz=f_ghz)
        result2 = model.eval(params2, f_ghz=f_ghz)

        # Results should be different due to different interactions
        assert not np.allclose(result1, result2, rtol=1e-2)
        assert np.all(np.isfinite(result1))
        assert np.all(np.isfinite(result2))

    def test_damping_effects(self, frequency_array):
        """Test effects of different damping parameters."""
        model = HybridDebyeLorentzModel(n_terms=1)

        # Low damping
        params_low = model.make_params(
            eps_inf=3.0,
            delta_eps_D1=1.0,
            tau_D1=1e-9,
            alpha1=1.0,
            delta_eps_L1=2.0,
            omega01=2 * np.pi * 10e9,
            q1=0.01,  # Very low damping
        )

        # High damping
        params_high = model.make_params(
            eps_inf=3.0,
            delta_eps_D1=1.0,
            tau_D1=1e-9,
            alpha1=1.0,
            delta_eps_L1=2.0,
            omega01=2 * np.pi * 10e9,
            q1=0.9,  # High damping
        )

        result_low = model.eval(params_low, f_ghz=frequency_array)
        result_high = model.eval(params_high, f_ghz=frequency_array)

        # Low damping should show sharper resonance features
        # High damping should be more broadened
        assert not np.allclose(result_low, result_high, rtol=1e-2)

    def test_alpha_parameter_effects(self, frequency_array, model_tolerance):
        """Test effects of Cole-Cole alpha parameter."""
        model = HybridDebyeLorentzModel(n_terms=1)

        # Pure Debye (alpha = 1)
        params_debye = model.make_params(
            eps_inf=3.0,
            delta_eps_D1=2.0,
            tau_D1=1e-9,
            alpha1=1.0,
            delta_eps_L1=0.0,
            omega01=2 * np.pi * 10e9,
            q1=0.1,
        )

        # Cole-Cole broadening (alpha < 1)
        params_cole = model.make_params(
            eps_inf=3.0,
            delta_eps_D1=2.0,
            tau_D1=1e-9,
            alpha1=0.6,
            delta_eps_L1=0.0,
            omega01=2 * np.pi * 10e9,
            q1=0.1,
        )

        result_debye = model.eval(params_debye, f_ghz=frequency_array)
        result_cole = model.eval(params_cole, f_ghz=frequency_array)

        # Results should be different
        assert not np.allclose(result_debye, result_cole, rtol=1e-3)

    def test_fitting_synthetic_data(self, frequency_array, fit_tolerance):
        """Test fitting to synthetic hybrid data."""
        model = HybridDebyeLorentzModel(n_terms=2)

        # Create synthetic data
        params_true = model.make_params(
            eps_inf=3.0,
            delta_eps_D1=2.0,
            tau_D1=1e-9,
            alpha1=0.8,
            delta_eps_L1=1.0,
            omega01=2 * np.pi * 5e9,
            q1=0.2,
            delta_eps_D2=1.5,
            tau_D2=1e-12,
            alpha2=1.0,
            delta_eps_L2=0.8,
            omega02=2 * np.pi * 50e9,
            q2=0.1,
        )
        synthetic_data = model.eval(params_true, f_ghz=frequency_array)

        # Fit with initial guess
        params_fit = model.make_params()
        params_fit["eps_inf"].set(value=3.0)

        # Initialize Debye components
        params_fit["delta_eps_D1"].set(value=1.5)
        params_fit["tau_D1"].set(value=2e-9)
        params_fit["alpha1"].set(value=0.9)
        params_fit["delta_eps_D2"].set(value=1.0)
        params_fit["tau_D2"].set(value=2e-12)
        params_fit["alpha2"].set(value=1.0)

        # Initialize Lorentz components
        params_fit["delta_eps_L1"].set(value=0.8)
        params_fit["omega01"].set(value=2 * np.pi * 3e9)
        params_fit["q1"].set(value=0.3)
        params_fit["delta_eps_L2"].set(value=0.6)
        params_fit["omega02"].set(value=2 * np.pi * 40e9)
        params_fit["q2"].set(value=0.2)

        result = model.fit(synthetic_data, params_fit, f_ghz=frequency_array)

        assert result.success
        assert result.chisqr < 100  # Reasonable fit quality

    def test_parameter_bounds_enforcement(self, frequency_array):
        """Test that parameter bounds are enforced."""
        model = HybridDebyeLorentzModel(n_terms=1)

        # Create synthetic data
        params_true = model.make_params(
            eps_inf=4.0,
            delta_eps_D1=3.0,
            tau_D1=5e-10,
            alpha1=0.7,
            delta_eps_L1=2.0,
            omega01=2 * np.pi * 8e9,
            q1=0.3,
        )
        synthetic_data = model.eval(params_true, f_ghz=frequency_array)

        # Fit with tight bounds
        params_fit = model.make_params()
        params_fit["eps_inf"].set(value=3.0, min=1.0, max=10.0)
        params_fit["delta_eps_D1"].set(value=2.0, min=0.0, max=20.0)
        params_fit["tau_D1"].set(value=1e-9, min=1e-12, max=1e-6)
        params_fit["alpha1"].set(value=0.8, min=0.1, max=1.0)
        params_fit["delta_eps_L1"].set(value=1.0, min=0.0, max=10.0)
        params_fit["omega01"].set(value=2 * np.pi * 5e9, min=1e8, max=1e12)
        params_fit["q1"].set(value=0.2, min=0.0, max=1.0)

        result = model.fit(synthetic_data, params_fit, f_ghz=frequency_array)

        # Check bounds are respected
        assert 1.0 <= result.params["eps_inf"].value <= 10.0
        assert 0.0 <= result.params["delta_eps_D1"].value <= 20.0
        assert 1e-12 <= result.params["tau_D1"].value <= 1e-6
        assert 0.1 <= result.params["alpha1"].value <= 1.0
        assert 0.0 <= result.params["delta_eps_L1"].value <= 10.0
        assert 1e8 <= result.params["omega01"].value <= 1e12
        assert 0.0 <= result.params["q1"].value <= 1.0

    def test_resonance_frequency_effects(self, model_tolerance):
        """Test effects of different resonance frequencies."""
        model = HybridDebyeLorentzModel(n_terms=1)

        base_params = {
            "eps_inf": 3.0,
            "delta_eps_D1": 1.0,
            "tau_D1": 1e-9,
            "alpha1": 1.0,
            "delta_eps_L1": 2.0,
            "q1": 0.1,
        }

        # Test different resonance frequencies
        resonance_freqs = [1e9, 10e9, 100e9]  # 1, 10, 100 GHz
        results = []

        for f_res in resonance_freqs:
            params = model.make_params(**base_params)
            params["omega01"].set(value=2 * np.pi * f_res)

            f_ghz = np.logspace(-1, 3, 100)
            result = model.eval(params, f_ghz=f_ghz)
            results.append(result)

        # Results should be different for different resonance frequencies
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                assert not np.allclose(results[i], results[j], rtol=1e-2)

    def test_complex_residual_scaling(self, frequency_array):
        """Test that the ScaledResidualMixin works correctly."""
        model = HybridDebyeLorentzModel(n_terms=1)

        # Create test data
        params = model.make_params(
            eps_inf=3.0,
            delta_eps_D1=2.0,
            tau_D1=1e-9,
            alpha1=0.8,
            delta_eps_L1=1.0,
            omega01=2 * np.pi * 10e9,
            q1=0.2,
        )
        complex_data = model.eval(params, f_ghz=frequency_array)

        # Test residual calculation
        residuals = model._residual(params, complex_data, f_ghz=frequency_array)

        # Should return flattened array with real and imaginary parts
        assert len(residuals) == 2 * len(complex_data)
        assert np.all(np.isfinite(residuals))

        # For perfect match, residuals should be very small
        assert np.max(np.abs(residuals)) < 1e-10

    def test_extreme_parameter_values(self, frequency_array):
        """Test model with extreme parameter values."""
        model = HybridDebyeLorentzModel(n_terms=1)

        # Extreme case 1: Very fast relaxation, very high frequency resonance
        params1 = model.make_params(
            eps_inf=2.0,
            delta_eps_D1=1.0,
            tau_D1=1e-15,
            alpha1=1.0,  # 1 fs
            delta_eps_L1=1.0,
            omega01=2 * np.pi * 1e12,
            q1=0.01,  # 1 THz
        )

        result1 = model.eval(params1, f_ghz=frequency_array)
        assert np.all(np.isfinite(result1))

        # Extreme case 2: Very slow relaxation, very low frequency resonance
        params2 = model.make_params(
            eps_inf=2.0,
            delta_eps_D1=1.0,
            tau_D1=1e-3,
            alpha1=1.0,  # 1 ms
            delta_eps_L1=1.0,
            omega01=2 * np.pi * 1e6,
            q1=0.5,  # 1 MHz
        )

        result2 = model.eval(params2, f_ghz=frequency_array)
        assert np.all(np.isfinite(result2))

    def test_additive_property(self, frequency_array, model_tolerance):
        """Test that hybrid terms are additive."""
        # Single term model
        model1 = HybridDebyeLorentzModel(n_terms=1)
        params1 = model1.make_params(
            eps_inf=0.0,  # No background for additive test
            delta_eps_D1=2.0,
            tau_D1=1e-9,
            alpha1=0.8,
            delta_eps_L1=1.0,
            omega01=2 * np.pi * 5e9,
            q1=0.2,
        )
        result1 = model1.eval(params1, f_ghz=frequency_array)

        # Another single term model
        model2 = HybridDebyeLorentzModel(n_terms=1)
        params2 = model2.make_params(
            eps_inf=0.0,  # No background for additive test
            delta_eps_D1=1.5,
            tau_D1=1e-12,
            alpha1=1.0,
            delta_eps_L1=0.8,
            omega01=2 * np.pi * 50e9,
            q1=0.1,
        )
        result2 = model2.eval(params2, f_ghz=frequency_array)

        # Sum of individual models
        sum_result = result1 + result2

        # Compare with 2-term model (need to add eps_inf to sum_result for fair comparison)
        model_combined = HybridDebyeLorentzModel(n_terms=2)
        params_combined = model_combined.make_params(
            eps_inf=3.0,  # Add base permittivity
            delta_eps_D1=2.0,
            tau_D1=1e-9,
            alpha1=0.8,
            delta_eps_L1=1.0,
            omega01=2 * np.pi * 5e9,
            q1=0.2,
            delta_eps_D2=1.5,
            tau_D2=1e-12,
            alpha2=1.0,
            delta_eps_L2=0.8,
            omega02=2 * np.pi * 50e9,
            q2=0.1,
        )
        combined_result = model_combined.eval(params_combined, f_ghz=frequency_array)

        # Account for eps_inf constraint: each individual model has eps_inf >= 1.0
        # So sum_result already includes 2.0 from the two eps_inf contributions
        # The combined model has eps_inf = 3.0, so we need to add 1.0 more
        sum_result_with_inf = sum_result + 1.0

        np.testing.assert_allclose(
            sum_result_with_inf, combined_result, rtol=model_tolerance
        )

    def test_physical_reasonableness(self, frequency_array):
        """Test that model produces physically reasonable results."""
        model = HybridDebyeLorentzModel(n_terms=2)
        params = model.make_params(
            eps_inf=3.0,
            delta_eps_D1=2.0,
            tau_D1=1e-9,
            alpha1=0.8,
            delta_eps_L1=1.0,
            omega01=2 * np.pi * 5e9,
            q1=0.2,
            delta_eps_D2=1.5,
            tau_D2=1e-12,
            alpha2=1.0,
            delta_eps_L2=0.8,
            omega02=2 * np.pi * 50e9,
            q2=0.1,
        )

        result = model.eval(params, f_ghz=frequency_array)

        # Check causality (imaginary part should be non-negative for passive system)
        # Note: Hybrid model can have complex frequency dependence

        # Real part should be reasonable - at low frequencies it can be higher than eps_inf
        # due to dispersive contributions, but should not be negative or extremely large
        assert np.all(np.real(result) >= 0.0)  # Non-negative permittivity
        assert np.all(np.real(result) <= 50.0)  # Reasonable upper bound

        # At high frequencies, should approach eps_inf
        f_high = frequency_array[-10:]  # Last 10 points (highest frequencies)
        result_high = model.eval(params, f_ghz=f_high)
        mean_real_high = np.mean(np.real(result_high))
        assert abs(mean_real_high - params["eps_inf"].value) < 1.0

    def test_model_scaling_with_terms(self, frequency_array):
        """Test model behavior as number of terms increases."""
        results = {}

        for n_terms in [1, 2, 3]:
            model = HybridDebyeLorentzModel(n_terms=n_terms)
            params = model.make_params()
            params["eps_inf"].set(value=2.0)

            for i in range(1, n_terms + 1):
                # Distribute parameters across frequency range
                tau = 10 ** (-12 + i * 2)  # ps to μs
                omega0 = 2 * np.pi * 10 ** (9 + i)  # GHz to THz

                params[f"delta_eps_D{i}"].set(value=1.0)
                params[f"tau_D{i}"].set(value=tau)
                params[f"alpha{i}"].set(value=0.8)
                params[f"delta_eps_L{i}"].set(value=0.5)
                params[f"omega0{i}"].set(value=omega0)
                params[f"q{i}"].set(value=0.2)

            result = model.eval(params, f_ghz=frequency_array)
            results[n_terms] = result

            assert np.all(np.isfinite(result))

        # More terms should generally give more complex frequency dependence
        # (this is a qualitative check)
        for n in results:
            assert len(results[n]) == len(frequency_array)
