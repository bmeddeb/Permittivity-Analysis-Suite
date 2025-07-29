"""
Cole-Davidson model for dielectric relaxation.

Implements the Cole-Davidson relaxation model with asymmetric broadening:
ε*(ω) = ε_∞ + Δε / (1 + jωτ)^β

This model describes materials with asymmetric distribution of relaxation times.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel


class ColeDavidsonModel(BaseModel):
    """
    Cole-Davidson relaxation model with asymmetric broadening.
    
    Formula: ε*(ω) = ε_∞ + Δε / (1 + jωτ)^β
    
    Parameters:
    - eps_inf: Permittivity at infinite frequency
    - delta_eps: Relaxation strength (ε_s - ε_∞)
    - tau: Relaxation time in seconds
    - beta: Asymmetric broadening factor (0 < β ≤ 1)
    
    Use cases: Materials with asymmetric distribution of relaxation times
    """
    
    def __init__(self):
        super().__init__(name="Cole-Davidson Model", model_type="cole_davidson")
    
    @staticmethod
    def model_func(freq: np.ndarray, eps_inf: float, delta_eps: float, 
                   tau: float, beta: float) -> np.ndarray:
        """
        Cole-Davidson model function.
        
        Args:
            freq: Frequency array in GHz
            eps_inf: Permittivity at infinite frequency
            delta_eps: Relaxation strength
            tau: Relaxation time in seconds
            beta: Asymmetric broadening factor (0 < β ≤ 1)
        
        Returns:
            Complex permittivity array
        """
        # Convert frequency to angular frequency in rad/s
        omega = BaseModel.angular_freq_from_ghz(freq)
        
        # Calculate (1 + jωτ)^β
        jw_tau_plus_1 = 1 + 1j * omega * tau
        
        # Handle the complex power carefully
        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = beta * np.log(jw_tau_plus_1)
            power_term = np.exp(log_term)
        
        # Calculate complex permittivity
        eps_complex = eps_inf + delta_eps / power_term
        
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
        
        # Estimate beta from the asymmetry of the loss peak
        beta_guess = self._estimate_beta_from_asymmetry(freq, df_exp)
        
        # Add parameters with bounds
        params.add('eps_inf', value=eps_inf_guess, min=1.0, max=1000.0)
        params.add('delta_eps', value=max(delta_eps_guess, 0.1), min=0.01, max=1000.0)
        params.add('tau', value=tau_guess, min=1e-15, max=1.0)
        params.add('beta', value=beta_guess, min=0.01, max=1.0)  # 0 < β ≤ 1
        
        return params
    
    def _estimate_beta_from_asymmetry(self, freq: np.ndarray, 
                                    df_exp: np.ndarray) -> float:
        """
        Estimate the β parameter from the asymmetry of the loss peak.
        
        Cole-Davidson model produces asymmetric peaks with a sharp high-frequency cutoff.
        Lower β values give more asymmetric peaks.
        
        Args:
            freq: Frequency array in GHz
            df_exp: Experimental imaginary permittivity
            
        Returns:
            Estimated β value
        """
        if len(df_exp) < 5:
            return 0.5  # Default for insufficient data
        
        # Find the peak
        max_idx = np.argmax(df_exp)
        max_val = df_exp[max_idx]
        
        # Find half-maximum points
        half_max = max_val / 2
        
        # Find left half-maximum (low frequency side)
        left_idx = max_idx
        for i in range(max_idx, -1, -1):
            if df_exp[i] <= half_max:
                left_idx = i
                break
        
        # Find right half-maximum (high frequency side)
        right_idx = max_idx
        for i in range(max_idx, len(df_exp)):
            if df_exp[i] <= half_max:
                right_idx = i
                break
        
        # Calculate asymmetry ratio
        if (right_idx > max_idx and left_idx < max_idx and 
            left_idx >= 0 and right_idx < len(freq)):
            
            # Width on low frequency side (in log scale)
            low_width = np.log10(freq[max_idx]) - np.log10(freq[left_idx])
            # Width on high frequency side (in log scale)
            high_width = np.log10(freq[right_idx]) - np.log10(freq[max_idx])
            
            if high_width > 0:
                asymmetry_ratio = low_width / high_width
                # For Cole-Davidson: asymmetry_ratio > 1 (broader low-freq side)
                # Lower β gives higher asymmetry
                beta_est = min(1.0, max(0.1, 1.0 / (1 + asymmetry_ratio)))
            else:
                beta_est = 0.5
        else:
            beta_est = 0.5
        
        return beta_est
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about Cole-Davidson model parameters."""
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
            'beta': {
                'name': 'β',
                'description': 'Asymmetric broadening factor',
                'units': 'dimensionless',
                'typical_range': (0.01, 1.0),
                'physical_meaning': 'Degree of asymmetric broadening (1=Debye, lower=more asymmetric)'
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
        beta = params['beta']
        
        derived = {
            'eps_static': eps_inf + delta_eps,
            'relaxation_freq_ghz': 1 / (2 * np.pi * tau) / 1e9,
            'asymmetry_factor': 1 - beta,  # Higher value means more asymmetric
        }
        
        # Cole-Davidson specific quantities
        if 0 < beta < 1:
            # The loss peak frequency is shifted from 1/(2πτ)
            # For Cole-Davidson, peak occurs at ω_max*τ = (β/(1-β))^(1/β) * sin(πβ)^(1/β)
            try:
                if beta < 0.99:
                    term1 = (beta / (1 - beta)) ** (1 / beta)
                    term2 = (np.sin(np.pi * beta)) ** (1 / beta)
                    omega_max_tau = term1 * term2
                    derived['loss_peak_freq_ghz'] = omega_max_tau / (2 * np.pi * tau) / 1e9
                else:
                    derived['loss_peak_freq_ghz'] = derived['relaxation_freq_ghz']
            except (ZeroDivisionError, OverflowError, ValueError):
                derived['loss_peak_freq_ghz'] = derived['relaxation_freq_ghz']
            
            # Maximum loss factor for Cole-Davidson
            try:
                derived['loss_factor_max'] = delta_eps * (beta ** beta) * ((1 - beta) ** (1 - beta))
            except (ValueError, OverflowError):
                derived['loss_factor_max'] = delta_eps / 2
        else:
            # Debye limit (β = 1)
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
        beta = params.get('beta', 0)
        
        if eps_inf < 1.0:
            warnings['eps_inf'] = 'ε_∞ should be ≥ 1 for physical materials'
        
        if delta_eps <= 0:
            warnings['delta_eps'] = 'Δε should be positive for relaxation'
            
        if tau <= 0 or tau > 1.0:
            warnings['tau'] = 'τ should be positive and typically < 1 second'
        
        if beta <= 0 or beta > 1:
            warnings['beta'] = 'β should be in range (0, 1] for causality'
        elif beta < 0.1:
            warnings['beta'] = 'β < 0.1 indicates very asymmetric distribution, check if physical'
            
        eps_static = eps_inf + delta_eps
        if eps_static > 1000:
            warnings['eps_static'] = 'Static permittivity seems unusually high'
        
        return warnings
    
    def get_asymmetry_analysis(self, params: Dict[str, float], 
                             freq_range: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the asymmetry characteristics of the Cole-Davidson response.
        
        Args:
            params: Dictionary of fitted parameters
            freq_range: Frequency range for analysis
            
        Returns:
            Dictionary with asymmetry analysis
        """
        eps_complex = self.model_func(freq_range, **params)
        df_calc = eps_complex.imag
        
        # Find peak
        max_idx = np.argmax(df_calc)
        peak_freq = freq_range[max_idx]
        
        # Analyze width at different levels
        levels = [0.1, 0.25, 0.5, 0.75, 0.9]  # Fractions of peak height
        analysis = {
            'peak_frequency_ghz': peak_freq,
            'asymmetry_ratios': {},
            'beta_parameter': params.get('beta', 0)
        }
        
        for level in levels:
            threshold = df_calc[max_idx] * level
            
            # Find crossing points
            left_idx = right_idx = max_idx
            
            # Search left
            for i in range(max_idx, -1, -1):
                if df_calc[i] <= threshold:
                    left_idx = i
                    break
            
            # Search right  
            for i in range(max_idx, len(df_calc)):
                if df_calc[i] <= threshold:
                    right_idx = i
                    break
            
            # Calculate asymmetry ratio
            if left_idx < max_idx < right_idx:
                left_width = np.log10(peak_freq) - np.log10(freq_range[left_idx])
                right_width = np.log10(freq_range[right_idx]) - np.log10(peak_freq)
                
                if right_width > 0:
                    asymmetry_ratios = left_width / right_width
                    analysis['asymmetry_ratios'][f'level_{level}'] = asymmetry_ratios
        
        return analysis