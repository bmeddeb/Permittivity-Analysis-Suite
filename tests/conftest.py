"""Test configuration and fixtures for the Permittivity Analysis Suite."""

import os
import sys
from pathlib import Path

# Add the project root to Python path so we can import app modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest


def debye_model(freq_ghz, eps_inf, delta_eps, tau):
    """Generate Debye model data for testing."""
    omega = 2 * np.pi * freq_ghz * 1e9
    return eps_inf + delta_eps / (1 + 1j * omega * tau)


@pytest.fixture(scope="session")
def frequency_array():
    """Standard frequency array for testing."""
    return np.logspace(-1, 3, 100)  # 0.1 to 1000 GHz


@pytest.fixture(scope="session")
def wide_frequency_array():
    """Wide frequency array for broadband testing."""
    return np.logspace(-2, 4, 200)  # 0.01 to 10,000 GHz


@pytest.fixture(scope="session")
def debye_data(frequency_array):
    """Simple Debye model test data."""
    eps_inf, delta_eps, tau = 3.0, 5.0, 1e-9
    complex_eps = debye_model(frequency_array, eps_inf, delta_eps, tau)

    return pd.DataFrame(
        {
            "Frequency (GHz)": frequency_array,
            "Dk": np.real(complex_eps),
            "Df": np.imag(complex_eps),
        }
    )


@pytest.fixture(scope="session")
def multi_relaxation_data(frequency_array):
    """Multi-relaxation test data (sum of multiple Debye processes)."""
    eps_inf = 2.0

    # Three relaxation processes
    params = [
        (2.0, 1e-12),  # Fast process
        (3.0, 1e-9),  # Medium process
        (1.5, 1e-6),  # Slow process
    ]

    complex_eps = np.full_like(frequency_array, eps_inf, dtype=complex)
    for delta_eps, tau in params:
        complex_eps += debye_model(frequency_array, 0, delta_eps, tau)

    return pd.DataFrame(
        {
            "Frequency (GHz)": frequency_array,
            "Dk": np.real(complex_eps),
            "Df": np.imag(complex_eps),
        }
    )


@pytest.fixture(scope="session")
def havriliak_negami_data(frequency_array):
    """Havriliak-Negami model test data."""
    eps_inf, delta_eps, tau, alpha, beta = 3.0, 5.0, 1e-9, 0.8, 0.6
    omega = 2 * np.pi * frequency_array * 1e9
    jw_tau = 1j * omega * tau
    complex_eps = eps_inf + delta_eps / (1 + jw_tau**alpha) ** beta

    return pd.DataFrame(
        {
            "Frequency (GHz)": frequency_array,
            "Dk": np.real(complex_eps),
            "Df": np.imag(complex_eps),
        }
    )


@pytest.fixture(scope="session")
def dsarkar_data(wide_frequency_array):
    """Djordjevic-Sarkar model test data."""
    eps_inf, delta_eps = 3.0, 5.0
    omega1, omega2 = 1e8, 1e11  # 100 MHz to 100 GHz

    omega = 2 * np.pi * wide_frequency_array * 1e9

    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log((omega2**2 + omega**2) / (omega1**2 + omega**2))
        eps_prime = eps_inf + (delta_eps / (2 * np.log(omega2 / omega1))) * log_term
        atan_term = np.arctan(omega / omega1) - np.arctan(omega / omega2)
        eps_double_prime = -(delta_eps / np.log(omega2 / omega1)) * atan_term

    complex_eps = eps_prime + 1j * eps_double_prime

    return pd.DataFrame(
        {
            "Frequency (GHz)": wide_frequency_array,
            "Dk": np.real(complex_eps),
            "Df": np.imag(complex_eps),
        }
    )


@pytest.fixture(scope="session")
def noisy_data(debye_data):
    """Add realistic noise to test data."""
    data = debye_data.copy()
    np.random.seed(42)  # Reproducible noise

    # Add 2% relative noise
    noise_level = 0.02
    dk_noise = data["Dk"] * noise_level * np.random.randn(len(data))
    df_noise = data["Df"] * noise_level * np.random.randn(len(data))

    data["Dk"] += dk_noise
    data["Df"] += df_noise

    return data


@pytest.fixture(scope="session")
def sparse_data(debye_data):
    """Sparse frequency sampling for testing robustness."""
    # Take every 5th point
    return debye_data.iloc[::5].copy()


@pytest.fixture(scope="session")
def tan_delta_data(debye_data):
    """Test data with loss tangent instead of imaginary permittivity."""
    data = debye_data.copy()
    # Convert to loss tangent: tan(δ) = ε''/ε'
    data["Df"] = data["Df"] / (data["Dk"] + 1e-18)
    return data


@pytest.fixture
def model_tolerance():
    """Standard tolerance for numerical comparisons."""
    return 1e-10


@pytest.fixture
def fit_tolerance():
    """Tolerance for fitting comparisons."""
    return 1e-3


@pytest.fixture
def causality_threshold():
    """Threshold for causality validation."""
    return 0.05  # 5%


# Helper functions for tests
def assert_complex_arrays_close(actual, expected, rtol=1e-5, atol=1e-8):
    """Assert that complex arrays are close."""
    np.testing.assert_allclose(np.real(actual), np.real(expected), rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.imag(actual), np.imag(expected), rtol=rtol, atol=atol)


def assert_dataframe_structure(df, required_columns=None):
    """Assert that DataFrame has required structure."""
    if required_columns is None:
        required_columns = ["Frequency (GHz)", "Dk", "Df"]

    assert isinstance(df, pd.DataFrame)
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert len(df) > 0, "DataFrame is empty"
    assert not df.isnull().any().any(), "DataFrame contains NaN values"


def assert_parameters_reasonable(params, model_type="general"):
    """Assert that fitted parameters are physically reasonable."""
    if "eps_inf" in params:
        assert params["eps_inf"].value >= 1.0, "eps_inf should be >= 1.0"

    # Check for delta_eps parameters
    for name, param in params.items():
        if "delta_eps" in name:
            assert param.value >= 0.0, f"{name} should be non-negative"

        if "tau" in name and "log_tau" not in name:
            assert 1e-15 <= param.value <= 1e-3, f"{name} should be in reasonable range"

        if "alpha" in name:
            assert 0.0 < param.value <= 1.0, f"{name} should be in (0, 1]"

        if "beta" in name:
            assert 0.0 < param.value <= 1.0, f"{name} should be in (0, 1]"


@pytest.fixture
def assert_complex_close():
    """Fixture providing the complex array assertion function."""
    return assert_complex_arrays_close


@pytest.fixture
def assert_df_structure():
    """Fixture providing the DataFrame structure assertion function."""
    return assert_dataframe_structure


@pytest.fixture
def assert_params_reasonable():
    """Fixture providing the parameter reasonableness assertion function."""
    return assert_parameters_reasonable
