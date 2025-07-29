"""
Cole-Cole model for dielectric relaxation.

Implements the Cole-Cole relaxation model with symmetric broadening:
ε*(ω) = ε_∞ + Δε / (1 + (jωτ)^(1-α))

This model describes materials with broad symmetric distribution of relaxation times.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel


class ColeColeModel(BaseModel):
    """
    Cole-Cole relaxation model with symmetric broadening.
    
    Formula: ε*(ω) = ε_∞ + Δε / (1 + (jωτ)^(1-α))
    
    Parameters:
    - eps_inf: Permittivity at infinite frequency
    - delta_eps: Relaxation strength (ε_s - ε_∞)
    - tau: Relaxation time in seconds
    - alpha: Symmetric broadening factor (0 ≤ α < 1)
    
    Use cases: Materials with broad symmetric distribution of relaxation times
    """
    
    def __init__(self):
        super().__init__(name="Cole-Cole Model", model_type="cole_cole")
    
    @staticmethod
    def model_func(freq: np.ndarray, eps_inf: float, delta_eps: float, 
                   tau: float, alpha: float) -> np.ndarray:
        """
        Cole-Cole model function.
        
        Args:
            freq: Frequency array in GHz
            eps_inf: Permittivity at infinite frequency
            delta_eps: Relaxation strength
            tau: Relaxation time in seconds
            alpha: Symmetric broadening factor (0 ≤ α < 1)
        
        Returns:
            Complex permittivity array
        """
        # Convert frequency to angular frequency in rad/s
        omega = BaseModel.angular_freq_from_ghz(freq)
        
        # Calculate (jωτ)^(1-α)
        jw_tau = 1j * omega * tau
        
        # Handle the complex power carefully to avoid branch cut issues
        # Use the principal branch: z^a = exp(a * log(z))
        with np.errstate(divide='ignore', invalid='ignore'):
            log_jw_tau = np.log(jw_tau)
            power_term = np.exp((1 - alpha) * log_jw_tau)
        
        # Calculate denominator
        denominator = 1 + power_term
        
        # Calculate complex permittivity
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
        
        # Initial guesses similar to Debye model
        eps_inf_guess = np.min(dk_exp)
        eps_s_guess = np.max(dk_exp)
        delta_eps_guess = eps_s_guess - eps_inf_guess
        
        # Estimate relaxation time from peak of imaginary part
        max_df_idx = np.argmax(df_exp)
        if max_df_idx > 0 and max_df_idx < len(freq) - 1:
            peak_freq_hz = freq[max_df_idx] * 1e9
            tau_guess = 1 / (2 * np.pi * peak_freq_hz)
        else:
            mid_freq_hz = np.median(freq) * 1e9
            tau_guess = 1 / (2 * np.pi * mid_freq_hz)
        
        # Estimate alpha from the width of the loss peak
        alpha_guess = self._estimate_alpha_from_loss_width(freq, df_exp)
        
        # Add parameters with bounds
        params.add('eps_inf', value=eps_inf_guess, min=1.0, max=1000.0)
        params.add('delta_eps', value=max(delta_eps_guess, 0.1), min=0.01, max=1000.0)
        params.add('tau', value=tau_guess, min=1e-15, max=1.0)
        params.add('alpha', value=alpha_guess, min=0.0, max=0.99)  # α < 1 for causality
        
        return params
    
    def _estimate_alpha_from_loss_width(self, freq: np.ndarray, 
                                      df_exp: np.ndarray) -> float:
        """
        Estimate the α parameter from the width of the loss peak.
        
        A broader peak corresponds to larger α (more distribution).
        
        Args:
            freq: Frequency array in GHz
            df_exp: Experimental imaginary permittivity
            
        Returns:
            Estimated α value
        """
        if len(df_exp) < 5:
            return 0.1  # Default for insufficient data
        
        # Find the peak
        max_idx = np.argmax(df_exp)
        max_val = df_exp[max_idx]
        
        # Find half-maximum points
        half_max = max_val / 2
        
        # Find left half-maximum
        left_idx = max_idx
        for i in range(max_idx, -1, -1):
            if df_exp[i] <= half_max:
                left_idx = i
                break
        
        # Find right half-maximum
        right_idx = max_idx
        for i in range(max_idx, len(df_exp)):
            if df_exp[i] <= half_max:
                right_idx = i
                break
        
        # Calculate width in log frequency
        if right_idx > left_idx and left_idx >= 0 and right_idx < len(freq):
            log_width = np.log10(freq[right_idx]) - np.log10(freq[left_idx])
            
            # Empirical relationship: wider peaks have larger α
            # For Debye (α=0): log_width ≈ 2 decades
            # For very broad peaks: log_width can be > 4 decades
            alpha_est = min(0.8, max(0.0, (log_width - 2) / 3))
        else:
            alpha_est = 0.1
        
        return alpha_est
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about Cole-Cole model parameters."""
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
            'alpha': {
                'name': 'α',
                'description': 'Symmetric broadening factor',
                'units': 'dimensionless',
                'typical_range': (0.0, 0.99),
                'physical_meaning': 'Degree of symmetric distribution broadening (0=Debye, higher=broader)'
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
        alpha = params['alpha']
        
        derived = {
            'eps_static': eps_inf + delta_eps,
            'relaxation_freq_ghz': 1 / (2 * np.pi * tau) / 1e9,
            'distribution_width': alpha,  # Higher α means broader distribution
        }
        
        # Cole-Cole specific quantities
        if alpha > 0:
            # Maximum loss factor occurs at different frequency for Cole-Cole
            # ω_max * τ = (sin(απ/2))^(1/(1-α)) / (cos(απ/2))^(1/(1-α))
            alpha_rad = alpha * np.pi / 2
            if alpha < 0.99:  # Avoid division by zero
                omega_max_tau = (np.sin(alpha_rad) / np.cos(alpha_rad)) ** (1 / (1 - alpha))
                derived['loss_peak_freq_ghz'] = omega_max_tau / (2 * np.pi * tau) / 1e9
            
            # Maximum loss factor value
            derived['loss_factor_max'] = delta_eps * np.sin(alpha_rad) / (2 * np.cos(alpha_rad / 2))
        else:
            # Debye limit
            derived['loss_peak_freq_ghz'] = derived['relaxation_freq_ghz']
            derived['loss_factor_max'] = delta_eps / 2
        
        return derived
    
    def validate_parameters(self, params: Dict[str, float]) -> Dict[str, str]:
        """
        Validate fitted parameters for physical reasonableness.
        
        Args:
            params: Dictionary of fitted parameters
            
        Returns:
            Dictionary of validation warnings
        """
        warnings = {}
        
        eps_inf = params.get('eps_inf', 0)
        delta_eps = params.get('delta_eps', 0)
        tau = params.get('tau', 0)
        alpha = params.get('alpha', 0)
        
        if eps_inf < 1.0:
            warnings['eps_inf'] = 'ε_∞ should be ≥ 1 for physical materials'
        
        if delta_eps <= 0:
            warnings['delta_eps'] = 'Δε should be positive for relaxation'
            
        if tau <= 0 or tau > 1.0:
            warnings['tau'] = 'τ should be positive and typically < 1 second'
        
        if alpha < 0 or alpha >= 1:
            warnings['alpha'] = 'α should be in range [0, 1) for causality'
        elif alpha > 0.8:
            warnings['alpha'] = 'α > 0.8 indicates very broad distribution, check if physical'
            
        eps_static = eps_inf + delta_eps
        if eps_static > 1000:
            warnings['eps_static'] = 'Static permittivity seems unusually high'
        
        return warnings
    
    def get_cole_cole_plot_data(self, params: Dict[str, float], 
                              freq_range: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate data for Cole-Cole plot (ε'' vs ε').
        
        Args:
            params: Dictionary of fitted parameters
            freq_range: Frequency range for calculation
            
        Returns:
            Dictionary with real and imaginary parts
        """
        eps_complex = self.model_func(freq_range, **params)
        
        return {
            'eps_real': eps_complex.real,
            'eps_imag': eps_complex.imag,
            'frequency': freq_range
        }