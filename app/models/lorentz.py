"""
Lorentz Oscillator model for dielectric response.

Implements the Lorentz oscillator model for resonant polarization:
ε*(ω) = ε_∞ + Σ(f_i / (ω₀ᵢ² - ω² - jγᵢω))

This model describes resonant effects and bound electron transitions.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel


class LorentzOscillatorModel(BaseModel):
    """
    Lorentz oscillator model for resonant dielectric response.
    
    Formula: ε*(ω) = ε_∞ + Σ(f_i / (ω₀ᵢ² - ω² - jγᵢω))
    
    Parameters:
    - eps_inf: Permittivity at infinite frequency
    - f_i: Oscillator strength for each resonance
    - omega0_i: Resonance angular frequency for each oscillator
    - gamma_i: Damping factor for each oscillator
    - n_oscillators: Number of oscillators (configurable)
    
    Use cases: THz/optical frequencies, resonant polarization, bound electrons
    """
    
    def __init__(self, n_oscillators: int = 1):
        """
        Initialize Lorentz oscillator model.
        
        Args:
            n_oscillators: Number of oscillators (default: 1)
        """
        if n_oscillators < 1 or n_oscillators > 5:
            raise ValueError("Number of oscillators must be between 1 and 5")
            
        self.n_oscillators = n_oscillators
        super().__init__(
            name=f"{n_oscillators}-Oscillator Lorentz Model",
            model_type=f"lorentz_{n_oscillators}"
        )
    
    def model_func(self, freq: np.ndarray, eps_inf: float, **osc_params) -> np.ndarray:
        """
        Lorentz oscillator model function.
        
        Args:
            freq: Frequency array in GHz
            eps_inf: Permittivity at infinite frequency
            **osc_params: Parameters for each oscillator (f_1, omega0_1, gamma_1, ...)
        
        Returns:
            Complex permittivity array
        """
        # Convert frequency to angular frequency in rad/s
        omega = BaseModel.angular_freq_from_ghz(freq)
        omega_squared = omega ** 2
        
        # Start with infinite frequency permittivity
        eps_complex = np.full_like(omega, eps_inf, dtype=complex)
        
        # Add contribution from each oscillator
        for i in range(1, self.n_oscillators + 1):
            f_key = f'f_{i}'
            omega0_key = f'omega0_{i}'
            gamma_key = f'gamma_{i}'
            
            if all(key in osc_params for key in [f_key, omega0_key, gamma_key]):
                f_i = osc_params[f_key]
                omega0_i = osc_params[omega0_key]  # Already in rad/s
                gamma_i = osc_params[gamma_key]
                
                # Add this oscillator's contribution
                denominator = omega0_i**2 - omega_squared - 1j * gamma_i * omega
                eps_complex += f_i / denominator
        
        return eps_complex
    
    def create_parameters(self, freq: np.ndarray, dk_exp: np.ndarray, 
                         df_exp: np.ndarray) -> lmfit.Parameters:
        """Create parameters for Lorentz oscillator model."""
        params = lmfit.Parameters()
        
        # Basic parameter estimates
        eps_inf_guess = np.min(dk_exp)
        params.add('eps_inf', value=eps_inf_guess, min=1.0, max=1000.0)
        
        # For each oscillator, make intelligent guesses
        omega_ghz = BaseModel.angular_freq_from_ghz(freq)
        
        for i in range(1, self.n_oscillators + 1):
            # Oscillator strength
            params.add(f'f_{i}', value=1.0, min=0.01, max=1000.0)
            
            # Resonance frequency (estimate from data)
            if i == 1:
                max_idx = np.argmax(df_exp)
                omega0_guess = omega_ghz[max_idx]
            else:
                # Distribute other oscillators
                omega0_guess = omega_ghz[len(omega_ghz) * i // (self.n_oscillators + 1)]
            
            params.add(f'omega0_{i}', value=omega0_guess, min=omega_ghz.min()/10, max=omega_ghz.max()*10)
            
            # Damping factor
            params.add(f'gamma_{i}', value=omega0_guess/10, min=omega0_guess/1000, max=omega0_guess)
        
        return params
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about Lorentz oscillator parameters."""
        param_info = {
            'eps_inf': {
                'name': 'ε_∞',
                'description': 'Permittivity at infinite frequency',
                'units': 'dimensionless',
                'typical_range': (1.0, 100.0),
                'physical_meaning': 'High-frequency limit beyond all resonances'
            }
        }
        
        # Add info for each oscillator
        for i in range(1, self.n_oscillators + 1):
            param_info[f'f_{i}'] = {
                'name': f'f_{i}',
                'description': f'Oscillator strength {i}',
                'units': 'rad²/s²',
                'physical_meaning': f'Strength of resonance {i}'
            }
            param_info[f'omega0_{i}'] = {
                'name': f'ω₀_{i}',
                'description': f'Resonance frequency {i}',
                'units': 'rad/s',
                'physical_meaning': f'Natural frequency of oscillator {i}'
            }
            param_info[f'gamma_{i}'] = {
                'name': f'γ_{i}',
                'description': f'Damping factor {i}',
                'units': 'rad/s',
                'physical_meaning': f'Damping rate of oscillator {i}'
            }
        
        return param_info
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {'n_oscillators': self.n_oscillators}