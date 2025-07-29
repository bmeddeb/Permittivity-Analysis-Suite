"""
D. Sarkar model for dielectric response with conduction.

Implements the Sarkar model combining Debye relaxation with conductivity:
ε*(ω) = ε_∞ + Δε/(1+jωτ) + σ/(jωε₀)

This model includes explicit conduction losses for broadband materials.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel

# Physical constants
EPSILON_0 = 8.854187817e-12  # F/m, vacuum permittivity


class SarkarModel(BaseModel):
    """
    D. Sarkar model with Debye relaxation and conductivity.
    
    Formula: ε*(ω) = ε_∞ + Δε/(1+jωτ) + σ/(jωε₀)
    
    Parameters:
    - eps_inf: Permittivity at infinite frequency
    - delta_eps: Relaxation strength
    - tau: Relaxation time in seconds
    - sigma: Conductivity in S/m
    
    Use cases: Broadband dielectric materials with significant conduction losses
    """
    
    def __init__(self):
        super().__init__(name="D. Sarkar Model", model_type="sarkar")
    
    @staticmethod
    def model_func(freq: np.ndarray, eps_inf: float, delta_eps: float, 
                   tau: float, sigma: float) -> np.ndarray:
        """
        Sarkar model function.
        
        Args:
            freq: Frequency array in GHz
            eps_inf: Permittivity at infinite frequency
            delta_eps: Relaxation strength
            tau: Relaxation time in seconds
            sigma: Conductivity in S/m
        
        Returns:
            Complex permittivity array
        """
        # Convert frequency to angular frequency in rad/s
        omega = BaseModel.angular_freq_from_ghz(freq)
        
        # Debye relaxation term
        debye_term = delta_eps / (1 + 1j * omega * tau)
        
        # Conductivity term
        conductivity_term = sigma / (1j * omega * EPSILON_0)
        
        # Total complex permittivity
        eps_complex = eps_inf + debye_term + conductivity_term
        
        return eps_complex
    
    def create_parameters(self, freq: np.ndarray, dk_exp: np.ndarray, 
                         df_exp: np.ndarray) -> lmfit.Parameters:
        """Create parameters with intelligent initial guesses."""
        params = lmfit.Parameters()
        
        # Basic Debye-like estimates
        eps_inf_guess = np.min(dk_exp)
        eps_s_guess = np.max(dk_exp)
        delta_eps_guess = eps_s_guess - eps_inf_guess
        
        # Relaxation time estimate
        max_df_idx = np.argmax(df_exp)
        if max_df_idx > 0 and max_df_idx < len(freq) - 1:
            peak_freq_hz = freq[max_df_idx] * 1e9
            tau_guess = 1 / (2 * np.pi * peak_freq_hz)
        else:
            mid_freq_hz = np.median(freq) * 1e9
            tau_guess = 1 / (2 * np.pi * mid_freq_hz)
        
        # Conductivity estimate from low-frequency behavior
        # At low frequencies: ε'' ≈ σ/(ωε₀)
        omega_low = BaseModel.angular_freq_from_ghz(freq[:3])  # Use first few points
        df_low = df_exp[:3]
        
        # Estimate conductivity from trend
        if len(omega_low) > 1 and omega_low[0] > 0:
            sigma_guess = np.mean(df_low * omega_low * EPSILON_0)
            sigma_guess = max(sigma_guess, 1e-6)  # Minimum conductivity
        else:
            sigma_guess = 1e-3
        
        # Add parameters
        params.add('eps_inf', value=eps_inf_guess, min=1.0, max=1000.0)
        params.add('delta_eps', value=max(delta_eps_guess, 0.1), min=0.01, max=1000.0)
        params.add('tau', value=tau_guess, min=1e-15, max=1.0)
        params.add('sigma', value=sigma_guess, min=1e-12, max=1.0)  # S/m
        
        return params
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about Sarkar model parameters."""
        return {
            'eps_inf': {
                'name': 'ε_∞',
                'description': 'Permittivity at infinite frequency',
                'units': 'dimensionless',
                'typical_range': (1.0, 100.0),
                'physical_meaning': 'High-frequency limit of permittivity'
            },
            'delta_eps': {
                'name': 'Δε',
                'description': 'Relaxation strength',
                'units': 'dimensionless',
                'typical_range': (0.1, 1000.0),
                'physical_meaning': 'Strength of the relaxation process'
            },
            'tau': {
                'name': 'τ',
                'description': 'Relaxation time',
                'units': 'seconds',
                'typical_range': (1e-15, 1.0),
                'physical_meaning': 'Characteristic time for molecular reorientation'
            },
            'sigma': {
                'name': 'σ',
                'description': 'Conductivity',
                'units': 'S/m',
                'typical_range': (1e-12, 1.0),
                'physical_meaning': 'DC conductivity of the material'
            }
        }
    
    @staticmethod
    def calculate_derived_quantities(params: Dict[str, float]) -> Dict[str, float]:
        """Calculate derived quantities."""
        eps_inf = params['eps_inf']
        delta_eps = params['delta_eps']
        tau = params['tau']
        sigma = params['sigma']
        
        derived = {
            'eps_static': eps_inf + delta_eps,
            'relaxation_freq_ghz': 1 / (2 * np.pi * tau) / 1e9,
            'conductivity_s_per_m': sigma,
            'resistivity_ohm_m': 1 / sigma if sigma > 0 else float('inf'),
        }
        
        # Frequency where conductivity contribution equals relaxation contribution
        # This occurs when σ/(ωε₀) = Δε/2 (approximately)
        if delta_eps > 0:
            omega_crossover = 2 * sigma / (delta_eps * EPSILON_0)
            derived['conductivity_crossover_freq_ghz'] = omega_crossover / (2 * np.pi) / 1e9
        
        return derived