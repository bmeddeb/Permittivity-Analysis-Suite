# app/models/lorentz.py
"""
Lorentz Oscillator model for dielectric response.

Implements the Lorentz oscillator model for resonant polarization:
ε*(ω) = ε_∞ + Σ(f_i / (ω0i² - ω² - jγiω))

This model describes resonant effects and bound electron transitions.
"""

import numpy as np
import lmfit
from typing import Optional, Dict, Any, List, Tuple, Union
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import differential_evolution
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LorentzOscillatorModel(BaseModel):
    """
    Lorentz oscillator model for dielectric spectroscopy.
    
    The Lorentz model describes resonant polarization mechanisms such as:
    - Electronic transitions (UV-Vis-IR)
    - Vibrational modes (IR)
    - Phonon resonances
    - Plasma oscillations
    
    This model is ideal for:
    - Optical materials
    - Semiconductors and dielectrics
    - Metamaterials
    - Phononic crystals
    - Materials with sharp absorption features
    
    The model supports multiple oscillators to capture complex spectra.
    
    Parameters (per oscillator):
        f_i: Oscillator strength (dimensionless)
        ω0_i: Resonance frequency (rad/s, converted from GHz)
        γ_i: Damping coefficient (rad/s, converted from GHz)
        
    Global parameters:
        ε_∞: High-frequency permittivity
    
    Attributes:
        n_oscillators (int): Number of oscillators
        constrain_parameters (bool): Apply physical constraints
        auto_detect (bool): Automatically detect number of oscillators
    """
    
    def __init__(self,
                 n_oscillators: int = 1,
                 auto_detect: bool = False,
                 constrain_parameters: bool = True,
                 name: Optional[str] = None):
        """
        Initialize the Lorentz oscillator model.
        
        Args:
            n_oscillators: Number of oscillators (if auto_detect=False)
            auto_detect: Automatically determine number from data
            constrain_parameters: Apply physical constraints
            name: Optional custom name
        """
        super().__init__(
            name=name or f"Lorentz-{n_oscillators}osc",
            frequency_range=(1e-3, 1e6)  # 1 MHz to 1 PHz
        )
        
        self.n_oscillators = n_oscillators
        self.auto_detect = auto_detect
        self.constrain_parameters = constrain_parameters
        
        # Will be updated after detection if auto_detect=True
        self._detected_oscillators = None
        
        # Update parameter names
        self._update_param_names()
        
        logger.info(f"Initialized Lorentz model with {n_oscillators} oscillators, "
                   f"auto_detect={auto_detect}")
    
    def _update_param_names(self) -> None:
        """Update parameter names based on number of oscillators."""
        self.param_names = ['eps_inf']
        
        n = self.n_oscillators if not self._detected_oscillators else self._detected_oscillators
        
        for i in range(n):
            self.param_names.extend([
                f'f_{i+1}',     # Oscillator strength
                f'omega0_{i+1}', # Resonance frequency
                f'gamma_{i+1}'   # Damping
            ])
        
        self.n_params = len(self.param_names)
    
    @staticmethod
    def model_func(freq: np.ndarray, **params) -> np.ndarray:
        """
        Calculate complex permittivity using Lorentz oscillator model.
        
        Args:
            freq: Frequency array in GHz
            **params: Model parameters including:
                - eps_inf: High-frequency permittivity
                - f_i, omega0_i, gamma_i for each oscillator
        
        Returns:
            Complex permittivity array
        """
        # Convert frequency to angular frequency (rad/s)
        omega = 2 * np.pi * freq * 1e9
        
        # Start with high-frequency permittivity
        eps_complex = params['eps_inf'] * np.ones_like(omega, dtype=complex)
        
        # Add contributions from each oscillator
        i = 1
        while f'f_{i}' in params:
            f_i = params[f'f_{i}']
            omega0_i = params[f'omega0_{i}'] * 2 * np.pi * 1e9  # Convert GHz to rad/s
            gamma_i = params[f'gamma_{i}'] * 2 * np.pi * 1e9    # Convert GHz to rad/s
            
            # Lorentz oscillator contribution
            denominator = omega0_i**2 - omega**2 - 1j * gamma_i * omega
            eps_complex += f_i * omega0_i**2 / denominator
            
            i += 1
        
        return eps_complex
    
    def detect_oscillators(self, freq: np.ndarray, dk_exp: np.ndarray, 
                          df_exp: np.ndarray) -> int:
        """
        Automatically detect number of oscillators from data.
        
        Uses peak detection in ε'' and analysis of spectral features.
        
        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental Dk
            df_exp: Experimental Df
            
        Returns:
            Detected number of oscillators
        """
        # Calculate imaginary permittivity
        eps_imag = df_exp * dk_exp
        
        # Look for peaks in ε'' (absorption peaks)
        # Use prominence to find significant peaks
        peaks, properties = find_peaks(eps_imag, 
                                     prominence=0.1 * np.max(eps_imag),
                                     width=2)  # At least 2 points wide
        
        if len(peaks) == 0:
            # No clear peaks, check for broad features
            # Look at second derivative
            d2_eps = np.gradient(np.gradient(eps_imag))
            peaks_d2, _ = find_peaks(-d2_eps, prominence=0.01 * np.max(np.abs(d2_eps)))
            n_oscillators = max(1, len(peaks_d2))
        else:
            n_oscillators = len(peaks)
        
        # Also check Dk for anomalous dispersion features
        # Lorentz oscillators cause characteristic S-shaped Dk
        dk_grad = np.gradient(dk_exp, np.log10(freq))
        dk_inflections, _ = find_peaks(np.abs(np.gradient(dk_grad)))
        
        # Each oscillator typically causes 2 inflection points
        n_from_dk = len(dk_inflections) // 2
        
        # Take maximum of estimates
        n_oscillators = max(n_oscillators, n_from_dk, 1)
        
        # Practical limit
        n_oscillators = min(n_oscillators, 5)
        
        logger.info(f"Detected {n_oscillators} oscillators from data features")
        
        return n_oscillators
    
    def create_parameters(self,
                         freq: np.ndarray,
                         dk_exp: np.ndarray,
                         df_exp: np.ndarray) -> lmfit.Parameters:
        """
        Create initial parameters for Lorentz model fitting.
        
        Uses peak detection and spectral analysis for intelligent initialization.
        
        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental Dk values
            df_exp: Experimental Df values
            
        Returns:
            lmfit.Parameters object with initial values
        """
        # Auto-detect oscillators if requested
        if self.auto_detect:
            self._detected_oscillators = self.detect_oscillators(freq, dk_exp, df_exp)
            self.n_oscillators = self._detected_oscillators
            self._update_param_names()
        
        params = lmfit.Parameters()
        
        # Estimate high-frequency permittivity
        # For Lorentz, use highest frequency value (beyond resonances)
        eps_inf_guess = dk_exp[-1]
        
        params.add('eps_inf',
                   value=eps_inf_guess,
                   min=1.0,
                   max=max(10, 2 * eps_inf_guess),
                   vary=True)
        
        # Calculate imaginary permittivity
        eps_imag = df_exp * dk_exp
        
        # Find peaks for oscillator positions
        peaks, properties = find_peaks(eps_imag, 
                                     prominence=0.05 * np.max(eps_imag))
        
        # Sort peaks by prominence
        if len(peaks) > 0:
            prominences = peak_prominences(eps_imag, peaks)[0]
            sorted_indices = np.argsort(prominences)[::-1]
            peaks = peaks[sorted_indices]
        
        # Initialize each oscillator
        for i in range(self.n_oscillators):
            # Resonance frequency
            if i < len(peaks):
                omega0_guess = freq[peaks[i]]
            else:
                # Distribute additional oscillators
                omega0_guess = freq[0] * (freq[-1] / freq[0]) ** ((i + 1) / (self.n_oscillators + 1))
            
            # Oscillator strength from peak height
            if i < len(peaks):
                peak_height = eps_imag[peaks[i]]
                # Approximate relation for Lorentz
                f_guess = peak_height * omega0_guess / (2 * np.pi)
            else:
                f_guess = 0.1
            
            # Damping from peak width
            if i < len(peaks):
                # Find FWHM
                peak_idx = peaks[i]
                half_max = eps_imag[peak_idx] / 2
                
                # Search for half-maximum points
                left_idx = np.where(eps_imag[:peak_idx] <= half_max)[0]
                right_idx = np.where(eps_imag[peak_idx:] <= half_max)[0] + peak_idx
                
                if len(left_idx) > 0 and len(right_idx) > 0:
                    f_left = freq[left_idx[-1]]
                    f_right = freq[right_idx[0]]
                    fwhm = f_right - f_left
                    gamma_guess = fwhm
                else:
                    gamma_guess = 0.1 * omega0_guess
            else:
                gamma_guess = 0.1 * omega0_guess
            
            # Add parameters with constraints
            params.add(f'f_{i+1}',
                       value=f_guess,
                       min=0.001,
                       max=1000,
                       vary=True)
            
            params.add(f'omega0_{i+1}',
                       value=omega0_guess,
                       min=freq[0] * 0.1,
                       max=freq[-1] * 10,
                       vary=True)
            
            params.add(f'gamma_{i+1}',
                       value=gamma_guess,
                       min=0.001 * omega0_guess,
                       max=omega0_guess,  # Overdamped limit
                       vary=True)
            
            # Apply constraints if requested
            if self.constrain_parameters:
                # Ensure gamma < omega0 (underdamped)
                params.add(f'quality_{i+1}', 
                          expr=f'omega0_{i+1} / gamma_{i+1}',
                          min=1.0)  # Q > 1
        
        return params
    
    def validate_parameters(self, params: lmfit.Parameters) -> bool:
        """
        Validate parameters for physical consistency.
        
        Args:
            params: Parameters to validate
            
        Returns:
            True if parameters are physically valid
        """
        # Check high-frequency permittivity
        if params['eps_inf'].value < 1:
            logger.warning("Invalid: eps_inf must be >= 1")
            return False
        
        # Check each oscillator
        for i in range(self.n_oscillators):
            f_i = params[f'f_{i+1}'].value
            omega0_i = params[f'omega0_{i+1}'].value
            gamma_i = params[f'gamma_{i+1}'].value
            
            if f_i < 0:
                logger.warning(f"Invalid: f_{i+1} must be positive")
                return False
            
            if omega0_i <= 0:
                logger.warning(f"Invalid: omega0_{i+1} must be positive")
                return False
            
            if gamma_i <= 0:
                logger.warning(f"Invalid: gamma_{i+1} must be positive")
                return False
            
            # Check for overdamping
            if gamma_i > omega0_i:
                logger.warning(f"Warning: Oscillator {i+1} is overdamped (γ > ω0)")
        
        return True
    
    @staticmethod
    def get_oscillator_properties(params: lmfit.Parameters, 
                                 osc_index: int) -> Dict[str, float]:
        """
        Get properties of a specific oscillator.
        
        Args:
            params: Model parameters
            osc_index: Oscillator index (1-based)
            
        Returns:
            Dictionary with oscillator properties
        """
        f = params[f'f_{osc_index}'].value
        omega0 = params[f'omega0_{osc_index}'].value
        gamma = params[f'gamma_{osc_index}'].value
        
        # Quality factor
        Q = omega0 / gamma
        
        # Peak frequency (slightly shifted from omega0 for damped oscillator)
        if gamma < omega0:
            omega_peak = omega0 * np.sqrt(1 - (gamma / (2 * omega0))**2)
        else:
            omega_peak = 0  # Overdamped
        
        # Maximum ε'' at resonance
        eps_imag_max = f * omega0 / gamma
        
        # Contribution to static permittivity
        delta_eps_static = f
        
        return {
            'oscillator_strength': f,
            'resonance_freq_GHz': omega0,
            'damping_GHz': gamma,
            'quality_factor': Q,
            'peak_freq_GHz': omega_peak,
            'max_absorption': eps_imag_max,
            'static_contribution': delta_eps_static,
            'overdamped': gamma >= omega0
        }
    
    @staticmethod
    def get_total_oscillator_strength(params: lmfit.Parameters) -> float:
        """Calculate sum of all oscillator strengths."""
        total_f = 0
        i = 1
        while f'f_{i}' in params:
            total_f += params[f'f_{i}'].value
            i += 1
        return total_f
    
    def get_spectral_properties(self, params: lmfit.Parameters) -> Dict[str, Any]:
        """
        Get comprehensive spectral properties.
        
        Args:
            params: Model parameters
            
        Returns:
            Dictionary with spectral analysis
        """
        # Collect all oscillator properties
        oscillators = []
        for i in range(self.n_oscillators):
            osc_props = self.get_oscillator_properties(params, i + 1)
            oscillators.append(osc_props)
        
        # Sort by frequency
        oscillators.sort(key=lambda x: x['resonance_freq_GHz'])
        
        # Overall properties
        total_f = self.get_total_oscillator_strength(params)
        eps_inf = params['eps_inf'].value
        eps_static = eps_inf + total_f
        
        # Find dominant oscillator
        dominant_idx = np.argmax([osc['oscillator_strength'] for osc in oscillators])
        
        return {
            'n_oscillators': self.n_oscillators,
            'oscillators': oscillators,
            'total_oscillator_strength': total_f,
            'eps_inf': eps_inf,
            'eps_static': eps_static,
            'dominant_oscillator': dominant_idx + 1,
            'frequency_range': [
                min(osc['resonance_freq_GHz'] for osc in oscillators),
                max(osc['resonance_freq_GHz'] for osc in oscillators)
            ]
        }
    
    def get_plot_data(self,
                      result: lmfit.model.ModelResult,
                      n_points: int = 1000,
                      show_components: bool = True,
                      freq_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Generate plot data including individual oscillator contributions.
        
        Args:
            result: Fit result
            n_points: Number of points for smooth curves
            show_components: Show individual oscillator contributions
            freq_range: Optional custom frequency range (min, max) in GHz
            
        Returns:
            Dictionary with plot data
        """
        # Get base plot data
        base_data = super().get_plot_data(result, n_points)
        
        # Add spectral properties
        spectral_props = self.get_spectral_properties(result.params)
        base_data['spectral_properties'] = spectral_props
        
        # Generate smooth frequency array
        if freq_range:
            freq_smooth = np.logspace(np.log10(freq_range[0]), 
                                    np.log10(freq_range[1]), n_points)
        else:
            freq_exp = getattr(result, 'freq', None)
            if freq_exp is not None:
                freq_smooth = np.logspace(np.log10(freq_exp.min()), 
                                        np.log10(freq_exp.max()), n_points)
            else:
                freq_smooth = np.logspace(-3, 3, n_points)
        
        # Add individual oscillator contributions if requested
        if show_components:
            components = {
                'freq': freq_smooth,
                'eps_inf': result.params['eps_inf'].value * np.ones(n_points),
                'oscillators': []
            }
            
            # Calculate each oscillator's contribution
            for i in range(self.n_oscillators):
                # Create params with only this oscillator
                single_osc_params = {
                    'eps_inf': 0,  # Don't include background
                    f'f_{i+1}': result.params[f'f_{i+1}'].value,
                    f'omega0_{i+1}': result.params[f'omega0_{i+1}'].value,
                    f'gamma_{i+1}': result.params[f'gamma_{i+1}'].value
                }
                
                # Calculate contribution
                eps_osc = self.model_func(freq_smooth, **single_osc_params)
                
                components['oscillators'].append({
                    'index': i + 1,
                    'eps_real': eps_osc.real,
                    'eps_imag': eps_osc.imag,
                    'properties': self.get_oscillator_properties(result.params, i + 1)
                })
            
            base_data['components'] = components
        
        return base_data
    
    def to_dict(self, result: lmfit.model.ModelResult) -> Dict[str, Any]:
        """
        Convert fit results to dictionary with Lorentz-specific information.
        
        Args:
            result: ModelResult from fitting
            
        Returns:
            Dictionary with fit results
        """
        # Get base dictionary
        output = super().to_dict(result)
        
        # Add Lorentz-specific information
        output['n_oscillators'] = self.n_oscillators
        output['spectral_properties'] = self.get_spectral_properties(result.params)
        
        # Add individual oscillator parameters
        output['oscillator_params'] = {}
        for i in range(self.n_oscillators):
            output['oscillator_params'][f'oscillator_{i+1}'] = {
                'f': result.params[f'f_{i+1}'].value,
                'omega0_GHz': result.params[f'omega0_{i+1}'].value,
                'gamma_GHz': result.params[f'gamma_{i+1}'].value,
                'properties': self.get_oscillator_properties(result.params, i + 1)
            }
        
        return output
    
    def estimate_optical_properties(self, params: lmfit.Parameters, 
                                  wavelength_nm: Union[float, np.ndarray]) -> Dict[str, Any]:
        """
        Calculate optical properties at specific wavelengths.
        
        Args:
            params: Model parameters
            wavelength_nm: Wavelength(s) in nanometers
            
        Returns:
            Dictionary with n, k, absorption coefficient, etc.
        """
        # Convert wavelength to frequency
        c = 2.99792458e8  # Speed of light in m/s
        wavelength_m = np.asarray(wavelength_nm) * 1e-9
        freq_hz = c / wavelength_m
        freq_ghz = freq_hz / 1e9
        
        # Calculate complex permittivity
        eps_complex = self.predict(freq_ghz, params)
        
        # Refractive index: n + ik = sqrt(ε)
        n_complex = np.sqrt(eps_complex)
        n = n_complex.real
        k = n_complex.imag
        
        # Absorption coefficient (1/m)
        alpha = 4 * np.pi * k / wavelength_m
        
        # Reflectivity at normal incidence
        R = ((n - 1)**2 + k**2) / ((n + 1)**2 + k**2)
        
        return {
            'wavelength_nm': wavelength_nm,
            'frequency_GHz': freq_ghz,
            'n': n,
            'k': k,
            'absorption_coeff_per_m': alpha,
            'absorption_coeff_per_cm': alpha / 100,
            'reflectivity': R,
            'complex_permittivity': eps_complex
        }
    
    def fit_with_constraints(self,
                           freq: np.ndarray,
                           dk_exp: np.ndarray,
                           df_exp: np.ndarray,
                           known_resonances: Optional[List[float]] = None,
                           fix_oscillator_number: bool = True,
                           **kwargs) -> lmfit.model.ModelResult:
        """
        Fit with additional constraints or known information.
        
        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental Dk
            df_exp: Experimental Df
            known_resonances: List of known resonance frequencies
            fix_oscillator_number: Don't allow changing n_oscillators
            **kwargs: Additional fitting arguments
            
        Returns:
            Fit result
        """
        # Create parameters
        params = self.create_parameters(freq, dk_exp, df_exp)
        
        # Apply known resonances if provided
        if known_resonances:
            for i, res_freq in enumerate(known_resonances[:self.n_oscillators]):
                param_name = f'omega0_{i+1}'
                if param_name in params:
                    params[param_name].value = res_freq
                    params[param_name].vary = False  # Fix known resonances
        
        # Perform fit
        result = self.fit(freq, dk_exp, df_exp, params=params, **kwargs)
        
        # Check if we should try different n_oscillators
        if not fix_oscillator_number and result.fit_metrics.r_squared < 0.95:
            logger.info("Poor fit, trying different number of oscillators")
            
            best_result = result
            best_aic = result.aic
            
            for n_osc in range(1, min(6, len(freq) // 10)):
                if n_osc == self.n_oscillators:
                    continue
                
                # Try with different number
                test_model = LorentzOscillatorModel(n_oscillators=n_osc,
                                                   constrain_parameters=self.constrain_parameters)
                
                try:
                    test_result = test_model.fit(freq, dk_exp, df_exp)
                    
                    if test_result.aic < best_aic:
                        best_aic = test_result.aic
                        best_result = test_result
                        logger.info(f"Better fit with {n_osc} oscillators (AIC: {best_aic:.1f})")
                except Exception as e:
                    logger.warning(f"Fit failed for {n_osc} oscillators: {e}")
            
            return best_result
        
        return result


class MultiLorentzModel(LorentzOscillatorModel):
    """
    Convenience class for multi-oscillator Lorentz model with auto-detection.
    """
    
    def __init__(self, max_oscillators: int = 5, name: Optional[str] = None):
        """
        Initialize multi-Lorentz model with automatic oscillator detection.
        
        Args:
            max_oscillators: Maximum number of oscillators to consider
            name: Optional custom name
        """
        super().__init__(
            n_oscillators=1,  # Will be updated by auto_detect
            auto_detect=True,
            constrain_parameters=True,
            name=name or "Multi-Lorentz"
        )
        self.max_oscillators = max_oscillators
    
    def detect_oscillators(self, freq: np.ndarray, dk_exp: np.ndarray, 
                          df_exp: np.ndarray) -> int:
        """Override to limit maximum oscillators."""
        n_detected = super().detect_oscillators(freq, dk_exp, df_exp)
        return min(n_detected, self.max_oscillators)