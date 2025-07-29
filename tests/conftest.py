# tests/conftest.py
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add the project root to the Python path so pytest can find the app module
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

"""
Test fixtures for KramersKronigValidator tests.

These fixtures provide various realistic dielectric data scenarios
for testing the Kramers-Kronig validation functionality.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def single_debye_data_uniform() -> pd.DataFrame:
    """
    Generate ideal single Debye relaxation data on UNIFORM grid.
    This should have near-perfect causality and uniform spacing.
    """
    # Linear frequency spacing for uniform grid
    freq_ghz = np.linspace(0.1, 100, 100)  # 0.1 to 100 GHz
    omega = 2 * np.pi * freq_ghz * 1e9

    # Debye parameters (lower eps_s for more reasonable values)
    eps_s = 10.0  # Static permittivity
    eps_inf = 3.0  # High-frequency permittivity
    tau = 1e-11  # Relaxation time

    # Debye model
    eps_complex = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)

    return pd.DataFrame({
        'Frequency (GHz)': freq_ghz,
        'Dk': eps_complex.real,
        'Df': np.abs(eps_complex.imag)  # Positive loss factor
    })


@pytest.fixture
def single_debye_data() -> pd.DataFrame:
    """
    Generate ideal single Debye relaxation data on logarithmic grid.
    This is more realistic for broadband measurements.
    """
    # Logarithmic spacing - more realistic for measurements
    freq_ghz = np.logspace(-1, 2, 50)  # 0.1 to 100 GHz
    omega = 2 * np.pi * freq_ghz * 1e9

    # Debye parameters
    eps_s = 10.0  # Static permittivity
    eps_inf = 3.0  # High-frequency permittivity
    tau = 1e-11  # Relaxation time

    # Debye model
    eps_complex = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)

    return pd.DataFrame({
        'Frequency (GHz)': freq_ghz,
        'Dk': eps_complex.real,
        'Df': np.abs(eps_complex.imag)
    })


@pytest.fixture
def double_debye_data() -> pd.DataFrame:
    """
    Generate double Debye relaxation data.
    Common in polymers and composite materials.

    Two distinct relaxation processes at different time scales.
    """
    freq_ghz = np.logspace(-2, 3, 100)  # 0.01 to 1000 GHz
    omega = 2 * np.pi * freq_ghz * 1e9

    # First relaxation (slower)
    eps_s1, eps_inf1 = 15.0, 8.0
    tau1 = 1e-10

    # Second relaxation (faster)
    eps_s2, eps_inf2 = 8.0, 3.0
    tau2 = 1e-12

    # Calculate each relaxation
    eps1 = eps_inf1 + (eps_s1 - eps_inf1) / (1 + 1j * omega * tau1)
    eps2 = eps_inf2 + (eps_s2 - eps_inf2) / (1 + 1j * omega * tau2)

    # Combine: second process starts from where first ends
    eps_complex = eps2 + (eps1.real[0] - eps_inf1) * (eps1 - eps_inf1) / (eps_s1 - eps_inf1)

    return pd.DataFrame({
        'Frequency (GHz)': freq_ghz,
        'Dk': eps_complex.real,
        'Df': np.abs(eps_complex.imag)
    })


@pytest.fixture
def noisy_single_debye_data() -> pd.DataFrame:
    """
    Single Debye with realistic measurement noise.
    Tests robustness to experimental imperfections.

    Simulates 1% relative measurement error.
    """
    np.random.seed(42)  # Reproducibility

    freq_ghz = np.logspace(-1, 2, 50)
    omega = 2 * np.pi * freq_ghz * 1e9

    eps_s, eps_inf = 10.0, 3.0
    tau = 1e-11

    eps_complex = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)

    # Add realistic noise (1% relative error)
    dk_noise = eps_complex.real * (1 + 0.01 * np.random.randn(len(freq_ghz)))
    df_noise = np.abs(eps_complex.imag) * (1 + 0.01 * np.random.randn(len(freq_ghz)))

    # Ensure Df remains positive
    df_noise = np.abs(df_noise)

    return pd.DataFrame({
        'Frequency (GHz)': freq_ghz,
        'Dk': dk_noise,
        'Df': df_noise
    })


@pytest.fixture
def cole_cole_data() -> pd.DataFrame:
    """
    Generate Cole-Cole relaxation data.
    Represents distributed relaxation times, common in heterogeneous materials.
    """
    freq_ghz = np.logspace(-1, 2, 60)
    omega = 2 * np.pi * freq_ghz * 1e9

    # Cole-Cole parameters
    eps_s = 12.0
    eps_inf = 3.0
    tau = 1e-11
    alpha = 0.2  # Distribution parameter (0 = Debye, 1 = maximum distribution)

    # Cole-Cole model
    eps_complex = eps_inf + (eps_s - eps_inf) / (1 + (1j * omega * tau)**(1 - alpha))

    return pd.DataFrame({
        'Frequency (GHz)': freq_ghz,
        'Dk': eps_complex.real,
        'Df': np.abs(eps_complex.imag)
    })


@pytest.fixture
def non_uniform_grid_data() -> pd.DataFrame:
    """
    Data on non-uniform frequency grid.
    Common in combined measurements from multiple instruments.

    Simulates data from:
    - Low frequency analyzer (0.1-1 GHz)
    - Network analyzer (2-20 GHz)
    - Millimeter wave system (25-100 GHz)
    """
    # Simulate combined data from different frequency ranges with gaps
    freq_low = np.linspace(0.1, 1, 10)
    freq_mid = np.linspace(2, 20, 20)
    freq_high = np.logspace(np.log10(25), np.log10(100), 15)

    freq_ghz = np.concatenate([freq_low, freq_mid, freq_high])
    omega = 2 * np.pi * freq_ghz * 1e9

    eps_s, eps_inf = 8.0, 3.0
    tau = 5e-12

    eps_complex = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)

    return pd.DataFrame({
        'Frequency (GHz)': freq_ghz,
        'Dk': eps_complex.real,
        'Df': np.abs(eps_complex.imag)
    })


@pytest.fixture
def causality_violating_data() -> pd.DataFrame:
    """
    Data with intentional causality violations.
    Simulates bad measurements or processing errors.

    Violations include:
    - Dk artificially lowered in mid-frequency range
    - Non-physical spike in Df
    """
    freq_ghz = np.logspace(-1, 2, 50)
    omega = 2 * np.pi * freq_ghz * 1e9

    # Start with good Debye data
    eps_s, eps_inf = 10.0, 3.0
    tau = 1e-11
    eps_complex = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)

    dk = eps_complex.real.copy()
    df = np.abs(eps_complex.imag).copy()

    # Introduce causality violations
    # 1. Make Dk too low at some frequencies (measurement error)
    dk[10:15] *= 0.7  # 30% error

    # 2. Add non-physical spike in Df (processing artifact)
    df[25:27] *= 3.0  # Triple the loss

    return pd.DataFrame({
        'Frequency (GHz)': freq_ghz,
        'Dk': dk,
        'Df': df
    })


@pytest.fixture
def low_loss_material_data() -> pd.DataFrame:
    """
    Low-loss dielectric material data.
    Common in RF/microwave substrates (e.g., PTFE-based materials).
    """
    freq_ghz = np.linspace(1, 100, 40)  # 1 to 100 GHz, uniform grid

    # Low-loss material properties
    dk = 2.2 * np.ones_like(freq_ghz)  # Nearly constant Dk
    df = 0.001 * np.ones_like(freq_ghz)  # Very low, nearly constant Df

    # Add slight frequency dependence
    dk -= 0.01 * np.log10(freq_ghz)
    df += 0.0001 * freq_ghz / 100  # Slight increase with frequency

    return pd.DataFrame({
        'Frequency (GHz)': freq_ghz,
        'Dk': dk,
        'Df': df
    })


@pytest.fixture
def minimal_data() -> pd.DataFrame:
    """Minimal valid dataset (3 points)."""
    return pd.DataFrame({
        'Frequency (GHz)': [1.0, 10.0, 100.0],
        'Dk': [10.0, 8.0, 6.0],
        'Df': [0.5, 0.3, 0.1]
    })


@pytest.fixture
def edge_case_data() -> pd.DataFrame:
    """Edge case with very wide frequency range (6 decades)."""
    freq_ghz = np.logspace(-3, 3, 120)  # 1 MHz to 1 THz
    omega = 2 * np.pi * freq_ghz * 1e9

    eps_s, eps_inf = 20.0, 2.0
    tau = 1e-11

    eps_complex = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)

    return pd.DataFrame({
        'Frequency (GHz)': freq_ghz,
        'Dk': eps_complex.real,
        'Df': np.abs(eps_complex.imag)
    })


# Invalid data fixtures for error testing

@pytest.fixture
def invalid_data_missing_column() -> pd.DataFrame:
    """Data missing required 'Df' column."""
    return pd.DataFrame({
        'Frequency (GHz)': [1.0, 10.0, 100.0],
        'Dk': [10.0, 8.0, 6.0]
        # Missing 'Df' column
    })


@pytest.fixture
def invalid_data_non_numeric() -> pd.DataFrame:
    """Data with non-numeric values."""
    return pd.DataFrame({
        'Frequency (GHz)': [1.0, 10.0, 'invalid'],
        'Dk': [10.0, 8.0, 6.0],
        'Df': [0.5, 0.3, 0.1]
    })


@pytest.fixture
def invalid_data_negative_freq() -> pd.DataFrame:
    """Data with negative frequency."""
    return pd.DataFrame({
        'Frequency (GHz)': [-1.0, 10.0, 100.0],
        'Dk': [10.0, 8.0, 6.0],
        'Df': [0.5, 0.3, 0.1]
    })


@pytest.fixture
def invalid_data_non_monotonic() -> pd.DataFrame:
    """Data with non-monotonic frequencies."""
    return pd.DataFrame({
        'Frequency (GHz)': [1.0, 100.0, 10.0],  # Not sorted
        'Dk': [10.0, 6.0, 8.0],
        'Df': [0.5, 0.1, 0.3]
    })


@pytest.fixture
def invalid_data_single_point() -> pd.DataFrame:
    """Data with only one point (insufficient)."""
    return pd.DataFrame({
        'Frequency (GHz)': [10.0],
        'Dk': [5.0],
        'Df': [0.1]
    })


@pytest.fixture
def invalid_data_with_nans() -> pd.DataFrame:
    """Data containing NaN values."""
    return pd.DataFrame({
        'Frequency (GHz)': [1.0, 10.0, 100.0],
        'Dk': [10.0, np.nan, 6.0],
        'Df': [0.5, 0.3, 0.1]
    })