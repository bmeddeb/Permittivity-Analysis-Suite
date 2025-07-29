"""
Havriliak-Negami model for dielectric relaxation.

Implements the most general empirical dielectric relaxation model:
ε*(ω) = ε_∞ + Δε / (1 + (jωτ)^α)^β

This model combines both symmetric (α) and asymmetric (β) broadening factors.
"""

import numpy as np
import lmfit
from typing import Dict, Any
from .base_model import BaseModel


class HavriliakNegamiModel(BaseModel):
    """
    Havriliak-Negami relaxation model with both symmetric and asymmetric broadening.
    
    Formula: ε*(ω) = ε_∞ + Δε / (1 + (jωτ)^α)^β
    
    Parameters:
    - eps_inf: Permittivity at infinite frequency
    - delta_eps: Relaxation strength (ε_s - ε_∞)
    - tau: Relaxation time in seconds
    - alpha: Symmetric broadening factor (0 < α ≤ 1)
    - beta: Asymmetric broadening factor (0 < β ≤ 1)
    
    Special cases:
    - α = 1, β = 1: Debye model
    - α < 1, β = 1: Cole-Cole model
    - α = 1, β < 1: Cole-Davidson model
    
    Use cases: Most general model for complex relaxation distributions
    """
    
    def __init__(self):
        super().__init__(name="Havriliak-Negami Model", model_type="havriliak_negami")
    
    @staticmethod
    def model_func(freq: np.ndarray, eps_inf: float, delta_eps: float, 
                   tau: float, alpha: float, beta: float) -> np.ndarray:
        """
        Havriliak-Negami model function.
        
        Args:
            freq: Frequency array in GHz
            eps_inf: Permittivity at infinite frequency
            delta_eps: Relaxation strength
            tau: Relaxation time in seconds
            alpha: Symmetric broadening factor (0 < α ≤ 1)
            beta: Asymmetric broadening factor (0 < β ≤ 1)
        
        Returns:
            Complex permittivity array
        """
        # Convert frequency to angular frequency in rad/s
        omega = BaseModel.angular_freq_from_ghz(freq)
        
        # Calculate (jωτ)^α
        jw_tau = 1j * omega * tau
        
        # Handle the complex powers carefully to avoid branch cut issues
        with np.errstate(divide='ignore', invalid='ignore'):
            # First power: (jωτ)^α
            log_jw_tau = np.log(jw_tau)
            first_power = np.exp(alpha * log_jw_tau)
            
            # Denominator: (1 + (jωτ)^α)
            denominator_base = 1 + first_power
            
            # Second power: (1 + (jωτ)^α)^β
            log_denominator = np.log(denominator_base)
            final_denominator = np.exp(beta * log_denominator)
        
        # Calculate complex permittivity
        eps_complex = eps_inf + delta_eps / final_denominator
        
        return eps_complex
    
    def create_parameters(self, freq: np.ndarray, dk_exp: np.ndarray, 
                         df_exp: np.ndarray) -> lmfit.Parameters:
        """
        Create parameters with intelligent initial guesses.
        
        This is challenging for HN model due to parameter correlations.
        
        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental real permittivity
            df_exp: Experimental imaginary permittivity
        
        Returns:
            lmfit Parameters object
        """
        params = lmfit.Parameters()
        
        # Initial guesses for basic parameters
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
        
        # Estimate α and β from peak characteristics
        alpha_guess, beta_guess = self._estimate_alpha_beta(freq, df_exp)
        
        # Add parameters with bounds
        params.add('eps_inf', value=eps_inf_guess, min=1.0, max=1000.0)
        params.add('delta_eps', value=max(delta_eps_guess, 0.1), min=0.01, max=1000.0)
        params.add('tau', value=tau_guess, min=1e-15, max=1.0)
        params.add('alpha', value=alpha_guess, min=0.01, max=1.0)
        params.add('beta', value=beta_guess, min=0.01, max=1.0)
        
        # Add constraints to ensure causality: αβ ≤ 1
        # This is a necessary condition for the HN model to be causal
        params.add('causality_constraint', expr='alpha * beta', max=1.0)
        
        return params
    
    def _estimate_alpha_beta(self, freq: np.ndarray, 
                           df_exp: np.ndarray) -> tuple[float, float]:
        """
        Estimate α and β parameters from the loss peak characteristics.
        
        This is a simplified heuristic approach. More sophisticated methods
        could use Cole-Cole plot analysis or peak fitting.
        
        Args:
            freq: Frequency array in GHz
            df_exp: Experimental imaginary permittivity
            
        Returns:
            Tuple of (alpha_estimate, beta_estimate)
        """
        if len(df_exp) < 5:
            return 0.7, 0.8  # Default values
        
        # Find the peak
        max_idx = np.argmax(df_exp)
        max_val = df_exp[max_idx]
        
        # Analyze peak width at half maximum
        half_max = max_val / 2
        
        # Find half-maximum points
        left_idx = max_idx
        for i in range(max_idx, -1, -1):
            if df_exp[i] <= half_max:
                left_idx = i
                break
        
        right_idx = max_idx
        for i in range(max_idx, len(df_exp)):
            if df_exp[i] <= half_max:
                right_idx = i
                break
        
        if right_idx > left_idx and left_idx >= 0 and right_idx < len(freq):
            # Total width in log frequency
            total_width = np.log10(freq[right_idx]) - np.log10(freq[left_idx])
            
            # Asymmetry ratio
            left_width = np.log10(freq[max_idx]) - np.log10(freq[left_idx])
            right_width = np.log10(freq[right_idx]) - np.log10(freq[max_idx])
            
            if right_width > 0:
                asymmetry_ratio = left_width / right_width
            else:
                asymmetry_ratio = 1.0
            
            # Heuristic estimates
            # α affects overall width (smaller α → wider peak)
            alpha_est = max(0.1, min(1.0, 2.5 / total_width))
            
            # β affects asymmetry (smaller β → more asymmetric)
            beta_est = max(0.1, min(1.0, 1.0 / (0.5 + asymmetry_ratio)))
            
            # Ensure causality constraint
            if alpha_est * beta_est > 1.0:
                # Scale both parameters to satisfy constraint
                scale = 0.99 / (alpha_est * beta_est)
                alpha_est *= scale
                beta_est *= scale
                
        else:
            alpha_est, beta_est = 0.7, 0.8
        
        return alpha_est, beta_est
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about Havriliak-Negami model parameters."""
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
                'typical_range': (0.01, 1.0),
                'physical_meaning': 'Degree of symmetric distribution broadening'
            },
            'beta': {
                'name': 'β',
                'description': 'Asymmetric broadening factor',
                'units': 'dimensionless',
                'typical_range': (0.01, 1.0),
                'physical_meaning': 'Degree of asymmetric broadening'
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
        beta = params['beta']
        
        derived = {
            'eps_static': eps_inf + delta_eps,
            'relaxation_freq_ghz': 1 / (2 * np.pi * tau) / 1e9,
            'alpha_parameter': alpha,
            'beta_parameter': beta,
            'causality_product': alpha * beta,  # Must be ≤ 1
        }
        
        # Special case identification
        if abs(alpha - 1.0) < 0.01 and abs(beta - 1.0) < 0.01:
            derived['model_type'] = 'Debye (α≈1, β≈1)'
        elif abs(alpha - 1.0) < 0.01:
            derived['model_type'] = 'Cole-Davidson (α≈1)'
        elif abs(beta - 1.0) < 0.01:
            derived['model_type'] = 'Cole-Cole (β≈1)'
        else:
            derived['model_type'] = 'General Havriliak-Negami'
        
        # Loss peak frequency calculation (complex for HN model)
        try:
            if alpha > 0 and beta > 0 and alpha * beta <= 1:
                # Approximate formula for loss peak frequency
                # This is complex for general HN model
                sin_term = np.sin(np.pi * alpha / (2 * (1 + beta)))
                cos_term = np.cos(np.pi * alpha * beta / (2 * (1 + beta)))
                
                if sin_term > 0 and cos_term > 0:
                    omega_max_tau = (sin_term / cos_term) ** (1 / alpha)
                    derived['loss_peak_freq_ghz'] = omega_max_tau / (2 * np.pi * tau) / 1e9
                else:
                    derived['loss_peak_freq_ghz'] = derived['relaxation_freq_ghz']
            else:
                derived['loss_peak_freq_ghz'] = derived['relaxation_freq_ghz']
        except (ValueError, ZeroDivisionError, OverflowError):
            derived['loss_peak_freq_ghz'] = derived['relaxation_freq_ghz']
        
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
        beta = params.get('beta', 0)
        
        if eps_inf < 1.0:
            warnings['eps_inf'] = 'ε_∞ should be ≥ 1 for physical materials'
        
        if delta_eps <= 0:
            warnings['delta_eps'] = 'Δε should be positive for relaxation'
            
        if tau <= 0 or tau > 1.0:
            warnings['tau'] = 'τ should be positive and typically < 1 second'
        
        if alpha <= 0 or alpha > 1:
            warnings['alpha'] = 'α should be in range (0, 1] for physical meaning'
        
        if beta <= 0 or beta > 1:
            warnings['beta'] = 'β should be in range (0, 1] for physical meaning'
        
        # Causality constraint
        if alpha * beta > 1.0:
            warnings['causality'] = f'αβ = {alpha * beta:.3f} > 1 violates causality constraint'
        
        # Check for parameter correlation issues
        if abs(alpha - 1.0) < 0.05 and abs(beta - 1.0) < 0.05:
            warnings['over_parameterization'] = 'Parameters close to Debye model (α≈1, β≈1), consider simpler model'
        
        eps_static = eps_inf + delta_eps
        if eps_static > 1000:
            warnings['eps_static'] = 'Static permittivity seems unusually high'
        
        return warnings
    
    def get_model_complexity_analysis(self, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze the complexity and necessity of the HN model parameters.
        
        Args:
            params: Dictionary of fitted parameters
            
        Returns:
            Dictionary with complexity analysis
        """
        alpha = params.get('alpha', 1)
        beta = params.get('beta', 1)
        
        analysis = {
            'model_complexity': 'full_hn',
            'parameter_count': 5,
            'suggested_simpler_models': []
        }
        
        # Check if simpler models might be sufficient
        alpha_threshold = 0.05
        beta_threshold = 0.05
        
        if abs(alpha - 1.0) < alpha_threshold:
            if abs(beta - 1.0) < beta_threshold:
                analysis['suggested_simpler_models'].append('Debye (3 parameters)')
                analysis['model_complexity'] = 'debye_equivalent'
            else:
                analysis['suggested_simpler_models'].append('Cole-Davidson (4 parameters)')
                analysis['model_complexity'] = 'cole_davidson_equivalent'
        elif abs(beta - 1.0) < beta_threshold:
            analysis['suggested_simpler_models'].append('Cole-Cole (4 parameters)')
            analysis['model_complexity'] = 'cole_cole_equivalent'
        
        # Calculate parameter significance
        analysis['parameter_deviations'] = {
            'alpha_from_unity': abs(alpha - 1.0),
            'beta_from_unity': abs(beta - 1.0),
            'both_significant': (abs(alpha - 1.0) > alpha_threshold and 
                               abs(beta - 1.0) > beta_threshold)
        }
        
        return analysis