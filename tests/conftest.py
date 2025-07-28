"""
Pytest configuration and fixtures for dielectric permittivity testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class SyntheticDataGenerator:
    """Generate synthetic dielectric data with known parameters for testing"""
    
    @staticmethod
    def generate_debye_data(frequency: np.ndarray, eps_s: float = 10.0, 
                           eps_inf: float = 2.0, tau: float = 1e-9,
                           noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic Debye model data"""
        omega = 2 * np.pi * frequency * 1e9  # Convert GHz to rad/s
        omega_tau = omega * tau
        
        # Debye equations
        dk = eps_inf + (eps_s - eps_inf) / (1 + omega_tau**2)
        df = (eps_s - eps_inf) * omega_tau / (1 + omega_tau**2)
        
        # Add noise if specified
        if noise_level > 0:
            dk += np.random.normal(0, noise_level * np.std(dk), len(dk))
            df += np.random.normal(0, noise_level * np.std(df), len(df))
        
        return dk, df
    
    @staticmethod
    def generate_cole_cole_data(frequency: np.ndarray, eps_s: float = 10.0,
                               eps_inf: float = 2.0, tau: float = 1e-9,
                               alpha: float = 0.8, noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic Cole-Cole model data"""
        omega = 2 * np.pi * frequency * 1e9
        omega_tau = omega * tau
        
        # Cole-Cole equations
        denominator = 1 + (1j * omega_tau)**alpha
        epsilon_complex = eps_inf + (eps_s - eps_inf) / denominator
        
        dk = epsilon_complex.real
        df = epsilon_complex.imag
        
        # Add noise if specified
        if noise_level > 0:
            dk += np.random.normal(0, noise_level * np.std(dk), len(dk))
            df += np.random.normal(0, noise_level * np.std(df), len(df))
        
        return dk, df
    
    @staticmethod
    def generate_havriliak_negami_data(frequency: np.ndarray, eps_s: float = 10.0,
                                      eps_inf: float = 2.0, tau: float = 1e-9,
                                      alpha: float = 0.9, beta: float = 0.85,
                                      noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic Havriliak-Negami model data"""
        omega = 2 * np.pi * frequency * 1e9
        omega_tau = omega * tau
        
        # Havriliak-Negami equations
        denominator = (1 + (1j * omega_tau)**alpha)**beta
        epsilon_complex = eps_inf + (eps_s - eps_inf) / denominator
        
        dk = epsilon_complex.real
        df = epsilon_complex.imag
        
        # Add noise if specified
        if noise_level > 0:
            dk += np.random.normal(0, noise_level * np.std(dk), len(dk))
            df += np.random.normal(0, noise_level * np.std(df), len(df))
        
        return dk, df
    
    @staticmethod
    def generate_lorentz_data(frequency: np.ndarray, eps_inf: float = 2.0,
                             f0: float = 10.0, gamma: float = 2.0,
                             delta_eps: float = 5.0, noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic Lorentz oscillator data"""
        omega = 2 * np.pi * frequency
        omega0 = 2 * np.pi * f0
        gamma_omega = 2 * np.pi * gamma
        
        # Lorentz equations
        denominator = (omega0**2 - omega**2) + 1j * gamma_omega * omega
        epsilon_complex = eps_inf + delta_eps * omega0**2 / denominator
        
        dk = epsilon_complex.real
        df = epsilon_complex.imag
        
        # Add noise if specified
        if noise_level > 0:
            dk += np.random.normal(0, noise_level * np.std(dk), len(dk))
            df += np.random.normal(0, noise_level * np.std(df), len(df))
        
        return dk, df


@pytest.fixture
def frequency_range():
    """Standard frequency range for testing (1-100 GHz, 50 points)"""
    return np.logspace(0, 2, 50)  # 1 to 100 GHz


@pytest.fixture
def synthetic_data_generator():
    """Fixture providing synthetic data generator"""
    return SyntheticDataGenerator()


@pytest.fixture
def clean_debye_data(frequency_range, synthetic_data_generator):
    """Clean Debye data with known parameters"""
    params = {'eps_s': 10.0, 'eps_inf': 2.0, 'tau': 1e-9}
    dk, df = synthetic_data_generator.generate_debye_data(frequency_range, **params)
    
    data_df = pd.DataFrame({
        'Frequency_GHz': frequency_range,
        'Dk': dk,
        'Df': df
    })
    
    return data_df, params


@pytest.fixture
def noisy_debye_data(frequency_range, synthetic_data_generator):
    """Noisy Debye data with known parameters"""
    params = {'eps_s': 10.0, 'eps_inf': 2.0, 'tau': 1e-9}
    dk, df = synthetic_data_generator.generate_debye_data(
        frequency_range, noise_level=0.1, **params
    )
    
    data_df = pd.DataFrame({
        'Frequency_GHz': frequency_range,
        'Dk': dk,
        'Df': df
    })
    
    return data_df, params


@pytest.fixture
def clean_cole_cole_data(frequency_range, synthetic_data_generator):
    """Clean Cole-Cole data with known parameters"""
    params = {'eps_s': 10.0, 'eps_inf': 2.0, 'tau': 1e-9, 'alpha': 0.8}
    dk, df = synthetic_data_generator.generate_cole_cole_data(frequency_range, **params)
    
    data_df = pd.DataFrame({
        'Frequency_GHz': frequency_range,
        'Dk': dk,
        'Df': df
    })
    
    return data_df, params


@pytest.fixture
def clean_havriliak_negami_data(frequency_range, synthetic_data_generator):
    """Clean Havriliak-Negami data with known parameters"""
    params = {'eps_s': 10.0, 'eps_inf': 2.0, 'tau': 1e-9, 'alpha': 0.9, 'beta': 0.85}
    dk, df = synthetic_data_generator.generate_havriliak_negami_data(frequency_range, **params)
    
    data_df = pd.DataFrame({
        'Frequency_GHz': frequency_range,
        'Dk': dk,
        'Df': df
    })
    
    return data_df, params


@pytest.fixture
def model_test_cases():
    """Test cases for different models with their expected parameters"""
    return {
        'debye': {
            'params': {'eps_s': 10.0, 'eps_inf': 2.0, 'tau': 1e-9},
            'tolerance': {'eps_s': 0.1, 'eps_inf': 0.1, 'tau': 1e-10}
        },
        'cole_cole': {
            'params': {'eps_s': 10.0, 'eps_inf': 2.0, 'tau': 1e-9, 'alpha': 0.8},
            'tolerance': {'eps_s': 0.1, 'eps_inf': 0.1, 'tau': 1e-10, 'alpha': 0.05}
        },
        'havriliak_negami': {
            'params': {'eps_s': 10.0, 'eps_inf': 2.0, 'tau': 1e-9, 'alpha': 0.9, 'beta': 0.85},
            'tolerance': {'eps_s': 0.1, 'eps_inf': 0.1, 'tau': 1e-10, 'alpha': 0.05, 'beta': 0.05}
        }
    }


@pytest.fixture
def noise_levels():
    """Different noise levels for testing preprocessing"""
    return [0.0, 0.05, 0.1, 0.2, 0.5]