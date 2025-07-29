"""
Hybrid Debye-Lorentz model combining relaxation and resonance effects.

Implements a model with both Debye relaxation and Lorentz oscillators:
ε*(ω) = ε_∞ + Σ(Δεᵢ/(1+jωτᵢ)) + Σ(fₖ/(ω₀ₖ²-ω²-jγₖω))

This model describes materials with both relaxation and resonance processes.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel


class HybridDebyeLorentzModel(BaseModel):
    """
    Hybrid model combining Debye relaxation and Lorentz oscillators.
    
    Formula: ε*(ω) = ε_∞ + Σ(Δεᵢ/(1+jωτᵢ)) + Σ(fₖ/(ω₀ₖ²-ω²-jγₖω))
    
    Parameters:
    - eps_inf: Permittivity at infinite frequency
    - Debye terms: delta_eps_i, tau_i for each relaxation
    - Lorentz terms: f_k, omega0_k, gamma_k for each oscillator
    
    Use cases: Complex materials with both low-frequency relaxation and 
              high-frequency resonance effects
    """
    
    def __init__(self, n_debye: int = 1, n_lorentz: int = 1):
        """
        Initialize hybrid model.
        
        Args:
            n_debye: Number of Debye terms
            n_lorentz: Number of Lorentz terms
        """
        if n_debye < 0 or n_debye > 5:
            raise ValueError("Number of Debye terms must be between 0 and 5")
        if n_lorentz < 0 or n_lorentz > 5:
            raise ValueError("Number of Lorentz terms must be between 0 and 5")
        if n_debye == 0 and n_lorentz == 0:
            raise ValueError("Must have at least one Debye or Lorentz term")
            
        self.n_debye = n_debye
        self.n_lorentz = n_lorentz
        
        super().__init__(
            name=f"Hybrid {n_debye}D-{n_lorentz}L Model",
            model_type=f"hybrid_debye_{n_debye}_lorentz_{n_lorentz}"
        )
    
    def model_func(self, freq: np.ndarray, eps_inf: float, **params) -> np.ndarray:
        """
        Hybrid Debye-Lorentz model function.
        
        Args:
            freq: Frequency array in GHz
            eps_inf: Permittivity at infinite frequency
            **params: All model parameters
        
        Returns:
            Complex permittivity array
        """
        # Convert frequency to angular frequency in rad/s
        omega = BaseModel.angular_freq_from_ghz(freq)
        
        # Start with infinite frequency permittivity
        eps_complex = np.full_like(omega, eps_inf, dtype=complex)
        
        # Add Debye contributions
        for i in range(1, self.n_debye + 1):
            delta_eps_key = f'delta_eps_d{i}'
            tau_key = f'tau_d{i}'
            
            if delta_eps_key in params and tau_key in params:
                delta_eps = params[delta_eps_key]
                tau = params[tau_key]
                
                # Add Debye term
                denominator = 1 + 1j * omega * tau
                eps_complex += delta_eps / denominator
        
        # Add Lorentz contributions
        omega_squared = omega ** 2
        for k in range(1, self.n_lorentz + 1):
            f_key = f'f_l{k}'
            omega0_key = f'omega0_l{k}'
            gamma_key = f'gamma_l{k}'
            
            if all(key in params for key in [f_key, omega0_key, gamma_key]):
                f_k = params[f_key]
                omega0_k = params[omega0_key]
                gamma_k = params[gamma_key]
                
                # Add Lorentz term
                denominator = omega0_k**2 - omega_squared - 1j * gamma_k * omega
                eps_complex += f_k / denominator
        
        return eps_complex
    
    def create_parameters(self, freq: np.ndarray, dk_exp: np.ndarray, 
                         df_exp: np.ndarray) -> lmfit.Parameters:
        """Create parameters for hybrid model."""
        params = lmfit.Parameters()
        
        # Basic estimates
        eps_inf_guess = np.min(dk_exp)
        total_strength = np.max(dk_exp) - eps_inf_guess
        
        params.add('eps_inf', value=eps_inf_guess, min=1.0, max=1000.0)
        
        # Distribute strength between Debye and Lorentz terms
        if self.n_debye > 0:
            debye_strength = total_strength * 0.7  # Assume 70% from relaxation
            strength_per_debye = debye_strength / self.n_debye
        
        if self.n_lorentz > 0:
            lorentz_strength = total_strength * 0.3  # Assume 30% from resonance
            strength_per_lorentz = lorentz_strength / self.n_lorentz
        
        # Add Debye parameters
        for i in range(1, self.n_debye + 1):
            params.add(f'delta_eps_d{i}', 
                      value=strength_per_debye, min=0.01, max=total_strength)
            
            # Distribute relaxation times across lower frequencies
            freq_idx = min(len(freq) * i // (self.n_debye + 1), len(freq) - 1)
            tau_guess = 1 / (2 * np.pi * freq[freq_idx] * 1e9)
            params.add(f'tau_d{i}', value=tau_guess, min=1e-15, max=1.0)
        
        # Add Lorentz parameters
        omega_range = BaseModel.angular_freq_from_ghz(freq)
        for k in range(1, self.n_lorentz + 1):
            params.add(f'f_l{k}', 
                      value=strength_per_lorentz, min=0.01, max=total_strength)
            
            # Distribute resonances across higher frequencies
            omega0_guess = omega_range[len(omega_range) * k // (self.n_lorentz + 1)]
            params.add(f'omega0_l{k}', 
                      value=omega0_guess, min=omega_range.min(), max=omega_range.max()*2)
            params.add(f'gamma_l{k}', 
                      value=omega0_guess/10, min=omega0_guess/100, max=omega0_guess)
        
        return params
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about hybrid model parameters."""
        param_info = {
            'eps_inf': {
                'name': 'ε_∞',
                'description': 'Permittivity at infinite frequency',
                'units': 'dimensionless',
                'physical_meaning': 'High-frequency limit beyond all processes'
            }
        }
        
        # Debye parameters
        for i in range(1, self.n_debye + 1):
            param_info[f'delta_eps_d{i}'] = {
                'name': f'Δε_D{i}',
                'description': f'Debye relaxation strength {i}',
                'units': 'dimensionless'
            }
            param_info[f'tau_d{i}'] = {
                'name': f'τ_D{i}',
                'description': f'Debye relaxation time {i}',
                'units': 'seconds'
            }
        
        # Lorentz parameters
        for k in range(1, self.n_lorentz + 1):
            param_info[f'f_l{k}'] = {
                'name': f'f_L{k}',
                'description': f'Lorentz oscillator strength {k}',
                'units': 'rad²/s²'
            }
            param_info[f'omega0_l{k}'] = {
                'name': f'ω₀_L{k}',
                'description': f'Lorentz resonance frequency {k}',
                'units': 'rad/s'
            }
            param_info[f'gamma_l{k}'] = {
                'name': f'γ_L{k}',
                'description': f'Lorentz damping factor {k}',
                'units': 'rad/s'
            }
        
        return param_info
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'n_debye': self.n_debye,
            'n_lorentz': self.n_lorentz
        }