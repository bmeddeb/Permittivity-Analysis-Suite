# app/models/cole_cole.py
"""
Cole-Cole model for dielectric relaxation.

Implements the Cole-Cole relaxation model with symmetric broadening:
ε*(ω) = ε_∞ + Δε / (1 + (jωτ)^(1-α))

This model describes materials with broad symmetric distribution of relaxation times.
"""

import numpy as np
import lmfit
from typing import Optional, Dict, Any, Tuple
from scipy.signal import find_peaks
from scipy.special import gamma
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class ColeColeModel(BaseModel):
    """
    Cole-Cole relaxation model for dielectric spectroscopy.

    The Cole-Cole model extends the Debye model by introducing a distribution
    parameter α (0 ≤ α < 1) that describes the symmetric broadening of the
    relaxation time distribution. When α = 0, it reduces to the Debye model.

    This model is ideal for:
    - Polymers with broad molecular weight distributions
    - Amorphous materials near glass transition
    - Biological tissues with cellular heterogeneity
    - Composite materials with interfacial polarization
    - Supercooled liquids

    Parameters:
        ε_inf: High-frequency permittivity
        ε_s: Static (low-frequency) permittivity
        τ: Central relaxation time
        α: Distribution parameter (0 = Debye, larger = broader)
        σ_dc: DC conductivity (optional)

    Attributes:
        conductivity_correction (bool): Whether to include DC conductivity
        constrain_alpha (bool): Whether to limit α to physically reasonable range
    """

    def __init__(self,
                 conductivity_correction: bool = False,
                 constrain_alpha: bool = True,
                 name: Optional[str] = None):
        """
        Initialize the Cole-Cole model.

        Args:
            conductivity_correction: Include DC conductivity correction
            constrain_alpha: Limit α to [0, 0.5] range (recommended)
            name: Optional custom name for the model
        """
        super().__init__(
            name=name or "Cole-Cole",
            frequency_range=(1e-6, 1e3)  # 1 Hz to 1 THz
        )

        self.conductivity_correction = conductivity_correction
        self.use_conductivity = conductivity_correction
        self.constrain_alpha = constrain_alpha

        # Update parameter names
        self._update_param_names()

        logger.info(f"Initialized Cole-Cole model with conductivity={conductivity_correction}, "
                    f"constrain_alpha={constrain_alpha}")

    def _update_param_names(self) -> None:
        """Update parameter names based on model configuration."""
        self.param_names = ['eps_inf', 'eps_s', 'tau', 'alpha']
        if self.use_conductivity:
            self.param_names.append('sigma_dc')
        self.n_params = len(self.param_names)

    @staticmethod
    def model_func(freq: np.ndarray, **params) -> np.ndarray:
        """
        Calculate complex permittivity using Cole-Cole model.

        Args:
            freq: Frequency array in GHz
            **params: Model parameters
                - eps_inf: High-frequency permittivity
                - eps_s: Static permittivity
                - tau: Central relaxation time (seconds)
                - alpha: Distribution parameter (0-1)
                - sigma_dc: DC conductivity (S/m) [optional]

        Returns:
            Complex permittivity array
        """
        # Extract parameters
        eps_inf = params['eps_inf']
        eps_s = params['eps_s']
        tau = params['tau']
        alpha = params['alpha']
        sigma_dc = params.get('sigma_dc', 0.0)

        # Convert frequency to angular frequency (rad/s)
        omega = 2 * np.pi * freq * 1e9

        # Cole-Cole equation
        # ε* = ε∞ + Δε / (1 + (jωτ)^(1-α))
        delta_eps = eps_s - eps_inf
        denominator = 1 + (1j * omega * tau) ** (1 - alpha)
        eps_complex = eps_inf + delta_eps / denominator

        # Add conductivity contribution if specified
        if sigma_dc > 0:
            eps_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
            eps_complex = eps_complex - 1j * sigma_dc / (omega * eps_0)

        return eps_complex

    def create_parameters(self,
                          freq: np.ndarray,
                          dk_exp: np.ndarray,
                          df_exp: np.ndarray) -> lmfit.Parameters:
        """
        Create initial parameters for Cole-Cole model fitting.

        Uses intelligent guessing based on data characteristics:
        - eps_s from low-frequency Dk
        - eps_inf from high-frequency Dk
        - tau from loss peak frequency (corrected for distribution)
        - alpha from loss peak width
        - sigma_dc from low-frequency slope

        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental Dk values
            df_exp: Experimental Df values

        Returns:
            lmfit.Parameters object with initial values and bounds
        """
        params = lmfit.Parameters()

        # Estimate eps_inf from high-frequency data
        n_tail = max(3, int(0.1 * len(dk_exp)))
        eps_inf_guess = np.mean(dk_exp[-n_tail:])

        # Estimate eps_s from low-frequency data
        n_head = max(3, int(0.1 * len(dk_exp)))
        eps_s_guess = np.mean(dk_exp[:n_head])

        # Ensure eps_s > eps_inf
        if eps_s_guess <= eps_inf_guess:
            eps_s_guess = eps_inf_guess + 0.1 * abs(eps_inf_guess)

        # Estimate alpha from loss peak width
        alpha_guess = self._estimate_alpha_from_width(freq, df_exp)

        # Estimate tau from loss peak, corrected for Cole-Cole broadening
        tau_guess = self._estimate_tau_from_peak(freq, df_exp, alpha_guess)

        # Add parameters with bounds
        params.add('eps_inf',
                   value=eps_inf_guess,
                   min=1.0,
                   max=eps_s_guess,
                   vary=True)

        params.add('eps_s',
                   value=eps_s_guess,
                   min=eps_inf_guess,
                   max=max(1000, 2 * eps_s_guess),
                   vary=True)

        params.add('tau',
                   value=tau_guess,
                   min=1e-15,
                   max=1e-3,
                   vary=True)

        # Alpha bounds depend on constrain_alpha setting
        alpha_max = 0.5 if self.constrain_alpha else 0.99
        params.add('alpha',
                   value=alpha_guess,
                   min=0.0,
                   max=alpha_max,
                   vary=True)

        # Add conductivity if enabled
        if self.use_conductivity:
            sigma_guess = ColeColeModel._estimate_conductivity(freq, dk_exp, df_exp)
            params.add('sigma_dc',
                       value=sigma_guess,
                       min=0.0,
                       max=100.0,
                       vary=True)

        logger.debug(f"Initial parameter guesses:")
        for name, param in params.items():
            logger.debug(f"  {name}: {param.value:.4g} [{param.min:.4g}, {param.max:.4g}]")

        return params

    @staticmethod
    def _estimate_alpha_from_width(freq: np.ndarray, df_exp: np.ndarray) -> float:
        """
        Estimate distribution parameter from loss peak width.

        For Cole-Cole, the FWHM of the loss peak increases with α.
        Uses empirical relationship between peak width and α.
        """
        # Find peak
        peaks, properties = find_peaks(df_exp, height=0.1 * np.max(df_exp))

        if len(peaks) == 0:
            return 0.1  # Default for no clear peak

        # Use most prominent peak
        if len(peaks) > 1:
            peak_heights = properties['peak_heights']
            peak_idx = int(peaks[int(np.argmax(peak_heights))])
        else:
            peak_idx = int(peaks[0])

        # Find FWHM (Full Width at Half Maximum)
        half_max = df_exp[peak_idx] / 2

        # Search for half-maximum points
        left_indices = np.where(df_exp[:peak_idx] <= half_max)[0]
        right_indices = np.where(df_exp[peak_idx:] <= half_max)[0] + peak_idx

        if len(left_indices) > 0 and len(right_indices) > 0:
            # Interpolate to find precise half-maximum frequencies
            left_idx = left_indices[-1]
            right_idx = right_indices[0]

            # Log frequency for better interpolation
            log_freq = np.log10(freq)

            # Linear interpolation around half-maximum points
            if left_idx > 0:
                f_left = 10 ** np.interp(half_max,
                                         [df_exp[left_idx - 1], df_exp[left_idx]],
                                         [log_freq[left_idx - 1], log_freq[left_idx]])
            else:
                f_left = freq[0]

            if right_idx < len(freq) - 1:
                f_right = 10 ** np.interp(half_max,
                                          [df_exp[right_idx], df_exp[right_idx + 1]],
                                          [log_freq[right_idx], log_freq[right_idx + 1]])
            else:
                f_right = freq[-1]

            # Width in decades
            width_decades = np.log10(f_right / f_left)

            # Empirical relation: wider peak → larger α
            # For Debye (α=0): width ≈ 1.14 decades
            # For α=0.5: width ≈ 2.3 decades
            alpha_est = (width_decades - 1.14) / 2.32
            alpha_est = np.clip(alpha_est, 0.0, 0.5)
        else:
            alpha_est = 0.1  # Default

        return alpha_est

    @staticmethod
    def _estimate_tau_from_peak(freq: np.ndarray, df_exp: np.ndarray, alpha: float) -> float:
        """
        Estimate central relaxation time from loss peak.

        For Cole-Cole, the peak frequency is shifted compared to Debye:
        f_max = (1/(2πτ)) * sin(απ/(2(1-α))) / sin(π/(2(1-α)))
        """
        # Find peak frequency
        peaks, properties = find_peaks(df_exp, height=0.1 * np.max(df_exp))

        if len(peaks) > 0:
            if len(peaks) > 1:
                peak_heights = properties['peak_heights']
                peak_idx = int(peaks[int(np.argmax(peak_heights))])
            else:
                peak_idx = int(peaks[0])

            f_max = freq[peak_idx]

            # Correction factor for Cole-Cole
            if alpha > 0:
                correction = np.sin(alpha * np.pi / (2 * (1 + alpha))) / np.sin(np.pi / (2 * (1 + alpha)))
            else:
                correction = 1.0  # Debye limit

            tau_guess = correction / (2 * np.pi * f_max * 1e9)
        else:
            # Fallback
            f_center = np.sqrt(freq[0] * freq[-1])
            tau_guess = 1 / (2 * np.pi * f_center * 1e9)

        return tau_guess

    @staticmethod
    def _estimate_conductivity(freq: np.ndarray, dk_exp: np.ndarray, df_exp: np.ndarray) -> float:
        """Estimate DC conductivity from low-frequency data."""
        if len(freq) < 10:
            return 0.0

        # Use lowest 10% of frequencies
        n_fit = max(5, int(0.1 * len(freq)))

        # Calculate imaginary permittivity
        omega = 2 * np.pi * freq[:n_fit] * 1e9
        eps_imag = df_exp[:n_fit] * dk_exp[:n_fit]

        # Check for 1/f behavior
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, _, _ = linregress(np.log10(omega), np.log10(eps_imag))

            if abs(slope + 1) < 0.3 and r_value ** 2 > 0.9:
                eps_0 = 8.854187817e-12
                sigma_guess = 10 ** intercept * eps_0
                return min(sigma_guess, 10.0)  # Cap at reasonable value
        except (ValueError, RuntimeError):
            pass

        return 0.0

    def validate_parameters(self, params: lmfit.Parameters) -> bool:
        """
        Validate parameter values for physical consistency.

        Args:
            params: Parameters to validate

        Returns:
            True if parameters are physically valid
        """
        # Basic constraints
        if params['eps_s'].value <= params['eps_inf'].value:
            logger.warning("Invalid: eps_s must be greater than eps_inf")
            return False

        if params['tau'].value <= 0:
            logger.warning("Invalid: tau must be positive")
            return False

        if not (0 <= params['alpha'].value < 1):
            logger.warning("Invalid: alpha must be in [0, 1)")
            return False

        if self.use_conductivity and params['sigma_dc'].value < 0:
            logger.warning("Invalid: sigma_dc must be non-negative")
            return False

        # Check for reasonable parameter ranges
        if params['alpha'].value > 0.7:
            logger.warning("Warning: alpha > 0.7 indicates extreme broadening")

        return True

    @staticmethod
    def get_relaxation_frequency(params: lmfit.Parameters) -> float:
        """
        Calculate the peak loss frequency for Cole-Cole model.

        This differs from the characteristic frequency 1/(2πτ).

        Args:
            params: Model parameters

        Returns:
            Peak loss frequency in GHz
        """
        tau = params['tau'].value
        alpha = params['alpha'].value

        # For Cole-Cole, peak occurs at modified frequency
        if alpha > 0:
            # Numerical factor for peak position
            factor = (np.sin(np.pi / (2 * (1 + alpha))) /
                      np.sin(alpha * np.pi / (2 * (1 + alpha))))
        else:
            factor = 1.0

        f_peak_hz = factor / (2 * np.pi * tau)
        return f_peak_hz / 1e9

    @staticmethod
    def get_dielectric_strength(params: lmfit.Parameters) -> float:
        """Calculate dielectric strength."""
        return params['eps_s'].value - params['eps_inf'].value

    @staticmethod
    def get_loss_peak_height(params: lmfit.Parameters) -> float:
        """
        Calculate maximum loss factor for Cole-Cole model.

        The peak height is reduced compared to Debye by factor
        depending on α.
        """
        delta_eps = ColeColeModel.get_dielectric_strength(params)
        alpha = params['alpha'].value

        # Reduction factor (empirical approximation)
        reduction = 1 - 0.377 * alpha  # Approximate for small α

        return delta_eps * reduction / 2

    @staticmethod
    def get_distribution_parameters(params: lmfit.Parameters) -> Dict[str, float]:
        """
        Calculate relaxation time distribution characteristics.

        Returns:
            Dictionary with distribution width, asymmetry, etc.
        """
        alpha = params['alpha'].value
        tau_c = params['tau'].value

        # Calculate distribution width (log scale)
        # For Cole-Cole, symmetric on log scale
        if alpha > 0:
            # Approximate FWHM of g(ln τ)
            fwhm_log = 2 * np.pi * alpha / (1 - alpha ** 2)
        else:
            fwhm_log = 0  # Delta function for Debye

        # Most probable relaxation time
        tau_max = tau_c  # Symmetric, so peak at center

        # Mean relaxation time (first moment)
        if alpha > 0:
            tau_mean = tau_c * np.pi * alpha / (2 * np.sin(np.pi * alpha / 2))
        else:
            tau_mean = tau_c

        return {
            'tau_central': tau_c,
            'tau_peak': tau_max,
            'tau_mean': tau_mean,
            'fwhm_log_decades': fwhm_log / np.log(10),
            'broadness_parameter': alpha,
            'symmetric': True  # Cole-Cole is symmetric on log scale
        }

    def get_plot_data(self,
                      result: lmfit.model.ModelResult,
                      n_points: int = 1000,
                      show_distribution: bool = True) -> Dict[str, Any]:
        """
        Generate plot data including relaxation time distribution.

        Args:
            result: Fit result
            n_points: Number of points for smooth curves
            show_distribution: Include relaxation time distribution

        Returns:
            Dictionary with plot data including distribution if requested
        """
        # Get base plot data
        plot_data = super().get_plot_data(result, n_points)

        # Add Cole-Cole specific information
        plot_data['parameters']['relaxation_freq_ghz'] = ColeColeModel.get_relaxation_frequency(result.params)
        plot_data['parameters']['distribution_width'] = ColeColeModel.get_distribution_parameters(result.params)[
            'fwhm_log_decades']

        # Add relaxation time distribution if requested
        if show_distribution:
            alpha = result.params['alpha'].value
            tau_c = result.params['tau'].value

            # Generate tau array (log scale)
            tau_array = np.logspace(np.log10(tau_c) - 3, np.log10(tau_c) + 3, 200)

            # Cole-Cole distribution function
            g_tau = ColeColeModel._calculate_distribution(tau_array, tau_c, alpha)

            plot_data['distribution'] = {
                'tau': tau_array,
                'g_tau': g_tau,
                'tau_central': tau_c,
                'alpha': alpha
            }

        return plot_data

    @staticmethod
    def _calculate_distribution(tau: np.ndarray, tau_c: float, alpha: float) -> np.ndarray:
        """
        Calculate the relaxation time distribution g(τ).

        For Cole-Cole model, this is symmetric on a log scale.
        """
        if alpha == 0:
            # Debye limit - delta function
            g = np.zeros_like(tau)
            idx = np.argmin(np.abs(tau - tau_c))
            g[idx] = 1.0
            return g

        # Cole-Cole distribution (approximate form)
        x = np.log(tau / tau_c)

        # Symmetric distribution
        sigma = np.pi * alpha / (2 * (1 - alpha))
        g = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

        # Normalize using trapezoid rule (trapz is deprecated)
        g = g / np.trapezoid(g, x)

        return g

    def to_dict(self, result: lmfit.model.ModelResult) -> Dict[str, Any]:
        """
        Convert fit results to dictionary with Cole-Cole specific info.

        Args:
            result: ModelResult from fitting

        Returns:
            Dictionary with fit results and derived quantities
        """
        # Get base dictionary
        output = super().to_dict(result)

        # Add Cole-Cole specific information
        params = result.params

        output['derived_quantities'] = {
            'relaxation_frequency_ghz': ColeColeModel.get_relaxation_frequency(params),
            'dielectric_strength': ColeColeModel.get_dielectric_strength(params),
            'loss_peak_height': ColeColeModel.get_loss_peak_height(params),
            'eps_inf': params['eps_inf'].value,
            'eps_s': params['eps_s'].value,
            'tau_seconds': params['tau'].value,
            'tau_picoseconds': params['tau'].value * 1e12,
            'alpha': params['alpha'].value,
            'distribution_parameters': ColeColeModel.get_distribution_parameters(params)
        }

        if self.use_conductivity:
            output['derived_quantities']['sigma_dc'] = params['sigma_dc'].value

        return output

    @staticmethod
    def estimate_debye_equivalent(params: lmfit.Parameters) -> Dict[str, float]:
        """
        Estimate equivalent Debye parameters for comparison.

        Useful for understanding the effective behavior.

        Args:
            params: Cole-Cole parameters

        Returns:
            Dictionary with equivalent Debye parameters
        """
        # Same static and infinite permittivity
        eps_inf = params['eps_inf'].value
        eps_s = params['eps_s'].value
        tau = params['tau'].value

        # Effective relaxation time (peak position)
        f_peak = ColeColeModel.get_relaxation_frequency(params)
        tau_eff = 1 / (2 * np.pi * f_peak * 1e9)

        return {
            'eps_inf': eps_inf,
            'eps_s': eps_s,
            'tau_effective': tau_eff,
            'note': 'Approximate Debye equivalent for peak position matching'
        }


class ColeColeWithConductivityModel(ColeColeModel):
    """
    Convenience class for Cole-Cole model with conductivity enabled by default.
    """

    def __init__(self, constrain_alpha: bool = True, name: Optional[str] = None):
        """Initialize Cole-Cole model with conductivity correction."""
        super().__init__(
            conductivity_correction=True,
            constrain_alpha=constrain_alpha,
            name=name or "Cole-Cole+Conductivity"
        )