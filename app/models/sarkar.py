# app/models/sarkar.py
"""
Djordjevic-Sarkar logarithmic dispersion model.

This module implements the core D-S model equation and parameter estimation.
Analysis, validation, and export utilities are in separate modules.
"""

import numpy as np
import lmfit
from typing import Optional, List, Tuple
import logging
from scipy.signal import savgol_filter

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class SarkarModel(BaseModel):
    """
    Djordjevic-Sarkar logarithmic dispersion model.

    Model equation:
    ε'_r(ω) = ε_r,∞ + (ε_r,s - ε_r,∞)/(2ln(ω₂/ω₁)) * ln((ω² + ω₂²)/(ω² + ω₁²))
    ε''_r(ω) = σ_DC/(ωε₀) + (ε_r,s - ε_r,∞)/ln(ω₂/ω₁) * [arctan(ω/ω₁) - arctan(ω/ω₂)]

    Attributes:
        use_ghz (bool): If True, frequencies are in GHz; else Hz
        eps0 (float): Permittivity of free space
    """

    def __init__(self, name: Optional[str] = None, use_ghz: bool = True):
        """
        Initialize D-S model.

        Args:
            name: Optional custom name
            use_ghz: If True, frequencies in GHz; else Hz
        """
        super().__init__(
            name=name or "Djordjevic-Sarkar",
            frequency_range=(1e-3, 100)  # 1 MHz to 100 GHz typical range
        )
        self.use_ghz = use_ghz
        self.param_names = ['eps_r_inf', 'eps_r_s', 'f1', 'f2', 'sigma_dc']
        self.n_params = 5
        self.eps0 = 8.854187817e-12

    @staticmethod
    def model_func(freq: np.ndarray, **params) -> np.ndarray:
        """
        Calculate complex permittivity using D-S logarithmic model.

        Args:
            freq: Frequency array (unit determined by use_ghz parameter)
            **params: Model parameters including:
                - eps_r_inf: High-frequency permittivity
                - eps_r_s: Static permittivity
                - f1: Lower transition frequency
                - f2: Upper transition frequency
                - sigma_dc: DC conductivity
                - use_ghz: 1 if frequencies in GHz, 0 if Hz (fixed parameter)

        Returns:
            Complex permittivity array
        """
        eps_r_inf = params['eps_r_inf']
        eps_r_s = params['eps_r_s']
        f1 = params['f1']
        f2 = params['f2']
        sigma_dc = params['sigma_dc']
        use_ghz = params.get('use_ghz', 1)  # Default to GHz for backward compatibility

        # Physical constants
        eps0 = 8.854187817e-12

        # Convert to angular frequency
        if use_ghz:
            omega = 2 * np.pi * freq * 1e9
            omega1 = 2 * np.pi * f1 * 1e9
            omega2 = 2 * np.pi * f2 * 1e9
        else:
            omega = 2 * np.pi * freq
            omega1 = 2 * np.pi * f1
            omega2 = 2 * np.pi * f2

        # Calculate real part
        ln_ratio = np.log(omega2 / omega1)
        if abs(ln_ratio) < 1e-12:
            eps_real = np.full_like(omega, eps_r_s, dtype=float)
        else:
            numerator = (eps_r_s - eps_r_inf) * np.log((omega ** 2 + omega2 ** 2) / (omega ** 2 + omega1 ** 2))
            eps_real = eps_r_inf + numerator / (2 * ln_ratio)

        # Calculate imaginary part
        # DC conductivity contribution
        loss_dc = np.zeros_like(omega)
        non_zero_omega = omega > 1e-18
        loss_dc[non_zero_omega] = sigma_dc / (omega[non_zero_omega] * eps0)

        # Dispersion contribution
        if abs(ln_ratio) < 1e-12:
            loss_disp = np.zeros_like(omega)
        else:
            prefactor = (eps_r_s - eps_r_inf) / ln_ratio
            loss_disp = prefactor * (np.arctan(omega / omega1) - np.arctan(omega / omega2))

        eps_imag = loss_dc + loss_disp

        return eps_real + 1j * eps_imag

    def create_parameters(self, freq: np.ndarray, dk_exp: np.ndarray,
                          df_exp: np.ndarray) -> lmfit.Parameters:
        """
        Create initial parameters with robust estimation.

        Args:
            freq: Frequency array (GHz or Hz based on use_ghz)
            dk_exp: Experimental real permittivity (Dk)
            df_exp: Experimental loss factor (Df)

        Returns:
            lmfit.Parameters with initial guesses and bounds
        """
        params = lmfit.Parameters()

        # Add fixed parameter for frequency unit
        params.add('use_ghz', value=1 if self.use_ghz else 0, vary=False)

        # Estimate high-frequency permittivity (use median for robustness)
        n_high = max(3, len(freq) // 10)
        eps_r_inf_guess = np.median(dk_exp[-n_high:])

        # Estimate static permittivity
        n_low = max(3, len(freq) // 10)
        eps_r_s_guess = np.median(dk_exp[:n_low])

        # Ensure reasonable separation
        min_delta_eps = 0.1
        if eps_r_s_guess <= eps_r_inf_guess + min_delta_eps:
            eps_r_s_guess = eps_r_inf_guess + max(min_delta_eps, 0.1 * eps_r_inf_guess)

        # Find transition frequencies using smoothed derivative
        f1_guess, f2_guess = self._estimate_transition_frequencies(freq, dk_exp)

        # Estimate DC conductivity
        sigma_dc_guess = self._estimate_dc_conductivity(freq, dk_exp, df_exp)

        # Add parameters with bounds
        params.add('eps_r_inf',
                   value=eps_r_inf_guess,
                   min=1.0,
                   max=20.0,
                   vary=True)

        params.add('eps_r_s',
                   value=eps_r_s_guess,
                   min=eps_r_inf_guess + min_delta_eps,
                   max=50.0,
                   vary=True)

        params.add('f1',
                   value=f1_guess,
                   min=freq[0] * 0.01,
                   max=freq[-1],
                   vary=True)

        params.add('f2',
                   value=f2_guess,
                   min=f1_guess * 2,
                   max=freq[-1] * 100,
                   vary=True)

        # Add constraint on frequency ratio
        params.add('f_ratio',
                   expr='f2 / f1',
                   min=2.0,
                   max=1e6)

        params.add('sigma_dc',
                   value=sigma_dc_guess,
                   min=0,
                   max=1e-4,  # Reduced upper bound for typical dielectrics
                   vary=True)

        return params

    def _estimate_transition_frequencies(self, freq: np.ndarray,
                                         dk_exp: np.ndarray) -> Tuple[float, float]:
        """
        Estimate transition frequencies from data.

        Args:
            freq: Frequency array
            dk_exp: Experimental Dk values

        Returns:
            Tuple of (f1_guess, f2_guess)
        """
        if len(dk_exp) > 5:
            # Use Savitzky-Golay filter for smoother derivative
            window_length = min(9, len(dk_exp) if len(dk_exp) % 2 == 1 else len(dk_exp) - 1)
            if window_length >= 5:
                dk_smooth = savgol_filter(dk_exp, window_length=window_length, polyorder=2)
                d_dk = -np.gradient(dk_smooth)

                # Find peak in derivative
                peak_idx = np.argmax(d_dk)
                f_center = freq[peak_idx]

                # Set f1 and f2 around the center
                f1_guess = f_center / 10
                f2_guess = f_center * 10
            else:
                # Fallback for very noisy data
                f1_guess = freq[len(freq) // 4]
                f2_guess = freq[3 * len(freq) // 4]
        else:
            # Fallback for very few points
            f1_guess = freq[len(freq) // 4]
            f2_guess = freq[3 * len(freq) // 4]

        # Ensure bounds
        f1_guess = np.clip(f1_guess, freq[0] * 0.5, freq[-1] * 0.1)
        f2_guess = np.clip(f2_guess, freq[0] * 10, freq[-1] * 2)

        # Ensure f2 > f1
        if f2_guess <= f1_guess:
            f2_guess = f1_guess * 10

        return f1_guess, f2_guess

    def _estimate_dc_conductivity(self, freq: np.ndarray, dk_exp: np.ndarray,
                                  df_exp: np.ndarray) -> float:
        """
        Estimate DC conductivity from low-frequency data.

        Args:
            freq: Frequency array (GHz or Hz)
            dk_exp: Experimental Dk values
            df_exp: Experimental Df values

        Returns:
            Estimated DC conductivity
        """
        # Convert to Hz if needed
        freq_hz = freq * 1e9 if self.use_ghz else freq

        # Look for low-frequency data
        if len(freq) > 10 and freq_hz[0] < 1e8:  # If we have data below 100 MHz
            n_dc = min(5, len(freq) // 4)
            omega_low = 2 * np.pi * freq_hz[:n_dc]

            # Df = sigma/(omega*eps0*Dk) at low frequencies
            sigma_dc_estimates = df_exp[:n_dc] * dk_exp[:n_dc] * omega_low * self.eps0
            sigma_dc_guess = np.median(sigma_dc_estimates)
            sigma_dc_guess = np.clip(sigma_dc_guess, 1e-12, 1e-4)
        else:
            sigma_dc_guess = 1e-9  # Default for no low-freq data

        return sigma_dc_guess

    def get_transition_characteristics(self, params: lmfit.Parameters) -> dict:
        """
        Calculate basic transition characteristics.

        Args:
            params: Model parameters

        Returns:
            Dictionary with transition frequencies and dispersion strength
        """
        p = params.valuesdict()

        return {
            'f1_ghz': p['f1'] if self.use_ghz else p['f1'] / 1e9,
            'f2_ghz': p['f2'] if self.use_ghz else p['f2'] / 1e9,
            'f_center_ghz': np.sqrt(p['f1'] * p['f2']) if self.use_ghz else np.sqrt(p['f1'] * p['f2']) / 1e9,
            'f_ratio': p['f2'] / p['f1'],
            'delta_eps': p['eps_r_s'] - p['eps_r_inf'],
            'relative_dispersion': (p['eps_r_s'] - p['eps_r_inf']) / p['eps_r_s'],
            'sigma_dc': p['sigma_dc']
        }