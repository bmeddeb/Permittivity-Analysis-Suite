"""
Multi-Pole Debye model for dielectric relaxation.

Implements a sum of multiple Debye relaxation processes:
ε*(ω) = ε_∞ + Σ(Δε_i / (1 + jωτ_i))

This model describes materials with multiple distinct relaxation times.
"""

import numpy as np
import lmfit
from typing import Dict, Any, List
from .base_model import BaseModel


class MultiPoleDebyeModel(BaseModel):
    """
    Multi-pole Debye relaxation model.
    
    Formula: ε*(ω) = ε_∞ + Σ(Δε_i / (1 + jωτ_i))
    
    Parameters:
    - eps_inf: Permittivity at infinite frequency
    - delta_eps_i: Relaxation strength for each pole
    - tau_i: Relaxation time for each pole
    - n_poles: Number of Debye poles (configurable)
    
    Use cases: Materials with multiple distinct relaxation processes
    """
    
    def __init__(self, n_poles: int = 2):
        """
        Initialize multi-pole Debye model.
        
        Args:
            n_poles: Number of Debye poles (default: 2)
        """
        if n_poles < 1 or n_poles > 10:
            raise ValueError("Number of poles must be between 1 and 10")
            
        self.n_poles = n_poles
        super().__init__(
            name=f"{n_poles}-Pole Debye Model",
            model_type=f"multi_pole_debye_{n_poles}"
        )
    
    def model_func(self, freq: np.ndarray, eps_inf: float, **pole_params) -> np.ndarray:
        """
        Multi-pole Debye model function.
        
        Args:
            freq: Frequency array in GHz
            eps_inf: Permittivity at infinite frequency
            **pole_params: Parameters for each pole (delta_eps_1, tau_1, delta_eps_2, tau_2, ...)
        
        Returns:
            Complex permittivity array
        """
        # Convert frequency to angular frequency in rad/s
        omega = BaseModel.angular_freq_from_ghz(freq)
        
        # Start with infinite frequency permittivity
        eps_complex = np.full_like(omega, eps_inf, dtype=complex)
        
        # Add contribution from each pole
        for i in range(1, self.n_poles + 1):
            delta_eps_key = f'delta_eps_{i}'
            tau_key = f'tau_{i}'
            
            if delta_eps_key in pole_params and tau_key in pole_params:
                delta_eps = pole_params[delta_eps_key]
                tau = pole_params[tau_key]
                
                # Add this pole's contribution
                denominator = 1 + 1j * omega * tau
                eps_complex += delta_eps / denominator
        
        return eps_complex
    
    def create_parameters(self, freq: np.ndarray, dk_exp: np.ndarray, 
                         df_exp: np.ndarray) -> lmfit.Parameters:
        """
        Create parameters with intelligent initial guesses for multiple poles.
        
        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental real permittivity
            df_exp: Experimental imaginary permittivity
        
        Returns:
            lmfit Parameters object
        """
        params = lmfit.Parameters()
        
        # Initial guess for eps_inf
        eps_inf_guess = np.min(dk_exp)
        eps_s_guess = np.max(dk_exp)
        total_delta_eps = eps_s_guess - eps_inf_guess
        
        # Add eps_inf parameter
        params.add('eps_inf', value=eps_inf_guess, min=1.0, max=1000.0)
        
        # Distribute relaxation strength among poles
        delta_eps_per_pole = total_delta_eps / self.n_poles
        
        # Find characteristic frequencies by analyzing the loss spectrum
        characteristic_freqs = self._find_characteristic_frequencies(freq, df_exp)
        
        # Add parameters for each pole
        for i in range(1, self.n_poles + 1):
            # Relaxation strength
            params.add(f'delta_eps_{i}', 
                      value=max(delta_eps_per_pole, 0.01),
                      min=0.001, max=total_delta_eps * 2)
            
            # Relaxation time
            if i <= len(characteristic_freqs):
                # Use identified characteristic frequency
                freq_hz = characteristic_freqs[i-1] * 1e9
                tau_guess = 1 / (2 * np.pi * freq_hz)
            else:
                # Distribute remaining poles across frequency range
                freq_range = np.logspace(np.log10(freq.min()), np.log10(freq.max()), self.n_poles + 2)
                freq_hz = freq_range[i] * 1e9
                tau_guess = 1 / (2 * np.pi * freq_hz)
            
            params.add(f'tau_{i}', 
                      value=tau_guess,
                      min=1e-15, max=1.0)
        
        # Add constraints to prevent pole overlap
        self._add_pole_ordering_constraints(params)
        
        return params
    
    def _find_characteristic_frequencies(self, freq: np.ndarray, 
                                       df_exp: np.ndarray) -> List[float]:
        """
        Find characteristic frequencies from the loss spectrum.
        
        Args:
            freq: Frequency array in GHz
            df_exp: Experimental imaginary permittivity
            
        Returns:
            List of characteristic frequencies in GHz
        """
        characteristic_freqs = []
        
        try:
            # Try to use scipy for peak finding
            from scipy.signal import find_peaks
            
            # Smooth the data to reduce noise
            if len(df_exp) > 5:
                try:
                    from scipy.ndimage import gaussian_filter1d
                    df_smooth = gaussian_filter1d(df_exp, sigma=1.0)
                except ImportError:
                    df_smooth = df_exp
            else:
                df_smooth = df_exp
            
            # Find peaks
            peaks, _ = find_peaks(df_smooth, height=np.max(df_smooth) * 0.1)
            
            for peak_idx in peaks:
                if 0 <= peak_idx < len(freq):
                    characteristic_freqs.append(freq[peak_idx])
                    
        except ImportError:
            # Fallback: simple local maxima detection
            for i in range(1, len(df_exp) - 1):
                if df_exp[i] > df_exp[i-1] and df_exp[i] > df_exp[i+1]:
                    if df_exp[i] > np.max(df_exp) * 0.1:
                        characteristic_freqs.append(freq[i])
        
        # If no peaks found, use global maximum
        if not characteristic_freqs:
            max_idx = np.argmax(df_exp)
            characteristic_freqs.append(freq[max_idx])
        
        # Sort frequencies
        characteristic_freqs.sort()
        
        return characteristic_freqs
    
    def _add_pole_ordering_constraints(self, params: lmfit.Parameters):
        """
        Add constraints to ensure poles are ordered by relaxation time.
        This helps with fitting stability and parameter identifiability.
        """
        # Order tau parameters: tau_1 > tau_2 > ... > tau_n
        for i in range(1, self.n_poles):
            constraint_expr = f'tau_{i} - tau_{i+1}'
            params.add(f'tau_diff_{i}', expr=constraint_expr, min=0)
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about multi-pole Debye model parameters."""
        param_info = {
            'eps_inf': {
                'name': 'ε_∞',
                'description': 'Permittivity at infinite frequency',
                'units': 'dimensionless',
                'typical_range': (1.0, 100.0),
                'physical_meaning': 'High-frequency limit of permittivity'
            }
        }
        
        # Add info for each pole
        for i in range(1, self.n_poles + 1):
            param_info[f'delta_eps_{i}'] = {
                'name': f'Δε_{i}',
                'description': f'Relaxation strength for pole {i}',
                'units': 'dimensionless',
                'typical_range': (0.01, 1000.0),
                'physical_meaning': f'Strength of relaxation process {i}'
            }
            
            param_info[f'tau_{i}'] = {
                'name': f'τ_{i}',
                'description': f'Relaxation time for pole {i}',
                'units': 'seconds',
                'typical_range': (1e-15, 1.0),
                'physical_meaning': f'Characteristic time for process {i}'
            }
        
        return param_info
    
    def calculate_derived_quantities(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate derived quantities from fitted parameters.
        
        Args:
            params: Dictionary of fitted parameters
            
        Returns:
            Dictionary of derived quantities
        """
        eps_inf = params['eps_inf']
        
        # Calculate total relaxation strength
        total_delta_eps = sum(params.get(f'delta_eps_{i}', 0) 
                             for i in range(1, self.n_poles + 1))
        
        derived = {
            'eps_static': eps_inf + total_delta_eps,
            'total_relaxation_strength': total_delta_eps,
            'n_poles': self.n_poles
        }
        
        # Add relaxation frequencies for each pole
        for i in range(1, self.n_poles + 1):
            tau_key = f'tau_{i}'
            if tau_key in params:
                tau = params[tau_key]
                derived[f'relaxation_freq_{i}_ghz'] = 1 / (2 * np.pi * tau) / 1e9
        
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
        if eps_inf < 1.0:
            warnings['eps_inf'] = 'ε_∞ should be ≥ 1 for physical materials'
        
        # Check each pole
        tau_values = []
        for i in range(1, self.n_poles + 1):
            delta_eps = params.get(f'delta_eps_{i}', 0)
            tau = params.get(f'tau_{i}', 0)
            
            if delta_eps <= 0:
                warnings[f'delta_eps_{i}'] = f'Δε_{i} should be positive'
            
            if tau <= 0 or tau > 1.0:
                warnings[f'tau_{i}'] = f'τ_{i} should be positive and typically < 1 second'
            else:
                tau_values.append((i, tau))
        
        # Check if relaxation times are well separated
        tau_values.sort(key=lambda x: x[1], reverse=True)  # Sort by tau (descending)
        for j in range(len(tau_values) - 1):
            tau_ratio = tau_values[j][1] / tau_values[j+1][1]
            if tau_ratio < 3:  # Times should differ by at least factor of 3
                i1, i2 = tau_values[j][0], tau_values[j+1][0]
                warnings[f'tau_separation_{i1}_{i2}'] = f'τ_{i1} and τ_{i2} are too close (ratio: {tau_ratio:.1f})'
        
        return warnings
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for storage."""
        return {'n_poles': self.n_poles}