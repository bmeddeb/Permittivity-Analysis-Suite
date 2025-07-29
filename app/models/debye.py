"""
Debye model for dielectric relaxation.

Implements the classic single-pole Debye relaxation model:
ε*(ω) = ε_∞ + Δε / (1 + jωτ)

This model describes materials with a single relaxation time.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel


class DebyeModel(BaseModel):
    """
    Single-pole Debye relaxation model.
    
    Formula: ε*(ω) = ε_∞ + Δε / (1 + jωτ)
    
    Parameters:
    - eps_inf: Permittivity at infinite frequency
    - delta_eps: Relaxation strength (ε_s - ε_∞)
    - tau: Relaxation time in seconds
    
    Use cases: Simple polar materials with single relaxation process
    """
    
    def __init__(self):
        super().__init__(name="Debye Model", model_type="debye")
    
    @staticmethod
    def model_func(freq: np.ndarray, eps_inf: float, delta_eps: float, 
                   tau: float) -> np.ndarray:
        """
        Debye model function.
        
        Args:
            freq: Frequency array in GHz
            eps_inf: Permittivity at infinite frequency
            delta_eps: Relaxation strength
            tau: Relaxation time in seconds
        
        Returns:
            Complex permittivity array
        """
        # Convert frequency to angular frequency in rad/s
        omega = BaseModel.angular_freq_from_ghz(freq)
        
        # Calculate complex permittivity
        denominator = 1 + 1j * omega * tau
        eps_complex = eps_inf + delta_eps / denominator
        
        return eps_complex
    
    def create_parameters(self, freq: np.ndarray, dk_exp: np.ndarray, 
                         df_exp: np.ndarray) -> lmfit.Parameters:
        """
        Create parameters with intelligent initial guesses.
        
        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental real permittivity
            df_exp: Experimental imaginary permittivity
        
        Returns:
            lmfit Parameters object
        """
        params = lmfit.Parameters()
        
        # Initial guesses based on data characteristics
        eps_inf_guess = np.min(dk_exp)  # Minimum of Dk as high-freq limit
        eps_s_guess = np.max(dk_exp)    # Maximum of Dk as low-freq limit
        delta_eps_guess = eps_s_guess - eps_inf_guess
        
        # Estimate relaxation time from peak of imaginary part
        max_df_idx = np.argmax(df_exp)
        if max_df_idx > 0 and max_df_idx < len(freq) - 1:
            # Peak frequency gives estimate: ωτ = 1, so τ = 1/(2πf)
            peak_freq_hz = freq[max_df_idx] * 1e9
            tau_guess = 1 / (2 * np.pi * peak_freq_hz)
        else:
            # Fallback: use middle frequency
            mid_freq_hz = np.median(freq) * 1e9
            tau_guess = 1 / (2 * np.pi * mid_freq_hz)
        
        # Add parameters with bounds
        params.add('eps_inf', value=eps_inf_guess, min=1.0, max=1000.0)
        params.add('delta_eps', value=max(delta_eps_guess, 0.1), min=0.01, max=1000.0)
        params.add('tau', value=tau_guess, min=1e-15, max=1.0)
        
        return params
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about Debye model parameters."""
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
                'physical_meaning': 'Strength of the relaxation process (ε_s - ε_∞)'
            },
            'tau': {
                'name': 'τ',
                'description': 'Relaxation time',
                'units': 'seconds',
                'typical_range': (1e-15, 1.0),
                'physical_meaning': 'Characteristic time for molecular reorientation'
            }
        }
    
    @staticmethod
    def calculate_derived_quantities(params: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate derived quantities from fitted parameters.
        
        Args:
            params: Dictionary of fitted parameters
            
        Returns:
            Dictionary of derived quantities
        """
        eps_inf = params['eps_inf']
        delta_eps = params['delta_eps']
        tau = params['tau']
        
        derived = {
            'eps_static': eps_inf + delta_eps,  # Static permittivity ε_s
            'relaxation_freq_ghz': 1 / (2 * np.pi * tau) / 1e9,  # Relaxation frequency
            'loss_factor_max': delta_eps / 2,  # Maximum loss factor
        }
        
        return derived
    
    def validate_parameters(self, params: Dict[str, float]) -> Dict[str, str]:
        """
        Validate fitted parameters for physical reasonableness.
        
        Args:
            params: Dictionary of fitted parameters
            
        Returns:
            Dictionary of validation warnings (empty if all valid)
        """
        warnings = {}
        
        eps_inf = params.get('eps_inf', 0)
        delta_eps = params.get('delta_eps', 0)
        tau = params.get('tau', 0)
        
        if eps_inf < 1.0:
            warnings['eps_inf'] = 'ε_∞ should be ≥ 1 for physical materials'
        
        if delta_eps <= 0:
            warnings['delta_eps'] = 'Δε should be positive for relaxation'
            
        if tau <= 0 or tau > 1.0:
            warnings['tau'] = 'τ should be positive and typically < 1 second'
            
        eps_static = eps_inf + delta_eps
        if eps_static > 1000:
            warnings['eps_static'] = 'Static permittivity (ε_s) seems unusually high'
        
        return warnings