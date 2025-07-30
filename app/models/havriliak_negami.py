# app/models/havriliak_negami.py
"""
Havriliak-Negami model for dielectric relaxation.

Implements the most general empirical dielectric relaxation model:
ε*(ω) = ε_∞ + Δε / (1 + (jωτ)^α)^β

This model combines both symmetric (α) and asymmetric (β) broadening factors.
"""

import numpy as np
import lmfit
from typing import Optional, Dict, Any, Tuple
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class HavriliakNegamiModel(BaseModel):
    """
    Havriliak-Negami relaxation model for dielectric spectroscopy.

    The H-N model is the most general empirical relaxation function,
    combining symmetric (Cole-Cole) and asymmetric (Davidson-Cole)
    broadening. It reduces to simpler models as special cases:
    - α = 1, β = 1: Debye
    - α < 1, β = 1: Cole-Cole
    - α = 1, β < 1: Davidson-Cole
    - α < 1, β < 1: Full Havriliak-Negami

    This model is ideal for:
    - Polymers with complex dynamics
    - Supercooled liquids and glasses
    - Hydrogen-bonded systems
    - Materials with asymmetric loss peaks
    - Systems with both inter- and intra-molecular interactions

    Parameters:
        ε_inf: High-frequency permittivity
        ε_s: Static (low-frequency) permittivity
        τ: Characteristic relaxation time
        α: Symmetric broadening (0 < α ≤ 1)
        β: Asymmetric broadening (0 < β ≤ 1)
        σ_dc: DC conductivity (optional)

    Attributes:
        conductivity_correction (bool): Include DC conductivity term
        constrain_parameters (bool): Apply physical constraints to α, β
        model_type (str): Special case identifier
    """

    def __init__(self,
                 conductivity_correction: bool = False,
                 constrain_parameters: bool = True,
                 name: Optional[str] = None):
        """
        Initialize the Havriliak-Negami model.

        Args:
            conductivity_correction: Include DC conductivity
            constrain_parameters: Apply constraints (αβ ≤ 1 recommended)
            name: Optional custom name for the model
        """
        super().__init__(
            name=name or "Havriliak-Negami",
            frequency_range=(1e-6, 1e3)  # 1 Hz to 1 THz
        )

        self.conductivity_correction = conductivity_correction
        self.use_conductivity = conductivity_correction
        self.constrain_parameters = constrain_parameters

        # Model type will be determined from parameters
        self.model_type = "Full H-N"

        # Update parameter names
        self._update_param_names()

        logger.info(f"Initialized H-N model with conductivity={conductivity_correction}, "
                    f"constraints={constrain_parameters}")

    def _update_param_names(self) -> None:
        """Update parameter names based on model configuration."""
        self.param_names = ['eps_inf', 'eps_s', 'tau', 'alpha', 'beta']
        if self.use_conductivity:
            self.param_names.append('sigma_dc')
        self.n_params = len(self.param_names)

    @staticmethod
    def model_func(freq: np.ndarray, **params) -> np.ndarray:
        """
        Calculate complex permittivity using Havriliak-Negami model.

        Args:
            freq: Frequency array in GHz
            **params: Model parameters
                - eps_inf: High-frequency permittivity
                - eps_s: Static permittivity
                - tau: Characteristic relaxation time (seconds)
                - alpha: Symmetric broadening (0-1)
                - beta: Asymmetric broadening (0-1)
                - sigma_dc: DC conductivity (S/m) [optional]

        Returns:
            Complex permittivity array
        """
        # Extract parameters
        eps_inf = params['eps_inf']
        eps_s = params['eps_s']
        tau = params['tau']
        alpha = params['alpha']
        beta = params['beta']
        sigma_dc = params.get('sigma_dc', 0.0)

        # Convert frequency to angular frequency (rad/s)
        omega = 2 * np.pi * freq * 1e9

        # Havriliak-Negami equation
        # ε* = ε∞ + Δε / (1 + (jωτ)^α)^β
        delta_eps = eps_s - eps_inf
        denominator = (1 + (1j * omega * tau) ** alpha) ** beta
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
        Create initial parameters for H-N model fitting.

        Uses sophisticated analysis to estimate both symmetric and
        asymmetric broadening from the experimental data shape.

        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental Dk values
            df_exp: Experimental Df values

        Returns:
            lmfit.Parameters object with initial values and bounds
        """
        params = lmfit.Parameters()

        # Estimate eps_inf and eps_s as in simpler models
        n_tail = max(3, int(0.1 * len(dk_exp)))
        eps_inf_guess = np.mean(dk_exp[-n_tail:])

        n_head = max(3, int(0.1 * len(dk_exp)))
        eps_s_guess = np.mean(dk_exp[:n_head])

        if eps_s_guess <= eps_inf_guess:
            eps_s_guess = eps_inf_guess + 0.1 * abs(eps_inf_guess)

        # Analyze peak shape for α and β estimation
        alpha_guess, beta_guess = self._estimate_shape_parameters(freq, df_exp)

        # Estimate tau considering H-N peak shift
        tau_guess = self._estimate_tau_hn(freq, df_exp, alpha_guess, beta_guess)

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

        # Shape parameters with optional constraints
        params.add('alpha',
                   value=alpha_guess,
                   min=0.01,  # Avoid numerical issues at α = 0
                   max=1.0,
                   vary=True)

        params.add('beta',
                   value=beta_guess,
                   min=0.01,
                   max=1.0,
                   vary=True)

        # Apply constraint αβ ≤ 1 if requested
        if self.constrain_parameters:
            params.add('alpha_beta_product', expr='alpha * beta')
            # This creates a derived parameter for monitoring

        # Add conductivity if enabled
        if self.use_conductivity:
            sigma_guess = HavriliakNegamiModel._estimate_conductivity(freq, dk_exp, df_exp)
            params.add('sigma_dc',
                       value=sigma_guess,
                       min=0.0,
                       max=100.0,
                       vary=True)

        logger.debug(f"Initial parameter guesses:")
        for name, param in params.items():
            if not param.expr:  # Skip derived parameters
                logger.debug(f"  {name}: {param.value:.4g} [{param.min:.4g}, {param.max:.4g}]")

        return params

    @staticmethod
    def _estimate_shape_parameters(freq: np.ndarray, df_exp: np.ndarray) -> Tuple[float, float]:
        """
        Estimate α and β from loss peak shape analysis.

        Uses asymmetry and width measurements to separate
        symmetric and asymmetric broadening contributions.
        """
        # Find main peak
        peaks, properties = find_peaks(df_exp, height=0.1 * np.max(df_exp))

        if len(peaks) == 0:
            return 0.8, 0.8  # Default for no clear peak

        # Get main peak
        if len(peaks) > 1:
            peak_heights = properties['peak_heights']
            peak_idx = int(peaks[int(np.argmax(peak_heights))])
        else:
            peak_idx = int(peaks[0])

        # Analyze peak shape
        peak_val = df_exp[peak_idx]
        half_max = peak_val / 2

        # Find points at half maximum
        left_mask = df_exp[:peak_idx] >= half_max
        right_mask = df_exp[peak_idx:] >= half_max

        if np.any(left_mask) and np.any(right_mask):
            left_idx = np.where(left_mask)[0][0]
            right_idx = np.where(right_mask)[0][-1] + peak_idx

            # Calculate asymmetry
            log_freq = np.log10(freq)
            left_width = log_freq[peak_idx] - log_freq[left_idx]
            right_width = log_freq[right_idx] - log_freq[peak_idx]

            # Asymmetry parameter
            asymmetry = (right_width - left_width) / (right_width + left_width)

            # Total width
            total_width = log_freq[right_idx] - log_freq[left_idx]

            # Empirical relations (based on H-N theory)
            # Asymmetry mainly from β, width from both α and β
            beta_guess = 1.0 - 1.5 * abs(asymmetry)  # Negative asymmetry → β < 1
            beta_guess = np.clip(beta_guess, 0.1, 1.0)

            # Width increases with decreasing α
            # For H-N: width ≈ 1.14 + f(α,β) decades
            if beta_guess < 0.9:
                # Account for β contribution to width
                alpha_guess = 1.0 - 0.4 * (total_width - 1.14) / beta_guess
            else:
                # Similar to Cole-Cole
                alpha_guess = 1.0 - 0.5 * (total_width - 1.14)

            alpha_guess = np.clip(alpha_guess, 0.1, 1.0)
        else:
            alpha_guess, beta_guess = 0.8, 0.8

        return alpha_guess, beta_guess

    @staticmethod
    def _estimate_tau_hn(freq: np.ndarray, df_exp: np.ndarray,
                         alpha: float, beta: float) -> float:
        """
        Estimate relaxation time for H-N model.

        The peak frequency in H-N is shifted from 1/(2πτ) by
        a factor depending on both α and β.
        """
        # Find peak frequency
        peak_idx = np.argmax(df_exp)
        f_peak = freq[peak_idx]

        # H-N peak position correction
        # Approximate formula for peak position
        if alpha < 1 or beta < 1:
            # Numerical approximation for correction factor
            correction = (np.sin(np.pi * alpha * beta / (2 * (1 + beta))) /
                          np.sin(np.pi * alpha / (2 * (1 + beta)))) ** (1 / alpha)
        else:
            correction = 1.0  # Debye case

        tau_guess = correction / (2 * np.pi * f_peak * 1e9)

        return tau_guess

    @staticmethod
    def _estimate_conductivity(freq: np.ndarray, dk_exp: np.ndarray,
                               df_exp: np.ndarray) -> float:
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
                return min(sigma_guess, 10.0)
        except (ValueError, RuntimeError):
            pass

        return 0.0

    def validate_parameters(self, params: lmfit.Parameters) -> bool:
        """
        Validate parameters for physical consistency.

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

        alpha = params['alpha'].value
        beta = params['beta'].value

        if not (0 < alpha <= 1):
            logger.warning(f"Invalid: alpha must be in (0, 1], got {alpha}")
            return False

        if not (0 < beta <= 1):
            logger.warning(f"Invalid: beta must be in (0, 1], got {beta}")
            return False

        # H-N specific constraint
        if self.constrain_parameters and alpha * beta > 1:
            logger.warning(f"Invalid: α×β = {alpha * beta:.3f} > 1 violates H-N constraint")
            return False

        if self.use_conductivity and params['sigma_dc'].value < 0:
            logger.warning("Invalid: sigma_dc must be non-negative")
            return False

        return True

    @staticmethod
    def identify_model_type(params: lmfit.Parameters) -> str:
        """
        Identify which special case of H-N model the parameters represent.

        Args:
            params: Model parameters

        Returns:
            String identifying the model type
        """
        alpha = params['alpha'].value
        beta = params['beta'].value

        # Tolerances for identifying special cases
        tol = 0.05

        if abs(alpha - 1.0) < tol and abs(beta - 1.0) < tol:
            return "Debye"
        elif abs(beta - 1.0) < tol:
            return "Cole-Cole"
        elif abs(alpha - 1.0) < tol:
            return "Davidson-Cole"
        else:
            return "Havriliak-Negami"

    @staticmethod
    def get_relaxation_frequency(params: lmfit.Parameters) -> float:
        """
        Calculate the peak loss frequency for H-N model.

        This is complex for H-N and requires numerical calculation.

        Args:
            params: Model parameters

        Returns:
            Peak loss frequency in GHz
        """
        tau = params['tau'].value
        alpha = params['alpha'].value
        beta = params['beta'].value

        # For special cases, use analytical formulas
        if abs(alpha - 1.0) < 0.01 and abs(beta - 1.0) < 0.01:
            # Debye
            return 1 / (2 * np.pi * tau * 1e9)

        # General H-N case - numerical solution
        def loss_at_omega(log_omega):
            omega = 10 ** log_omega
            # Calculate imaginary part
            denominator = (1 + (1j * omega * tau) ** alpha) ** beta
            eps_imag = np.imag(1 / denominator)
            return -eps_imag  # Negative for minimization

        # Search for maximum
        result = minimize_scalar(loss_at_omega,
                                 bounds=(np.log10(0.1 / tau), np.log10(10 / tau)),
                                 method='bounded')

        omega_max = 10 ** result.x
        f_max = omega_max / (2 * np.pi * 1e9)

        return f_max

    @staticmethod
    def get_dielectric_strength(params: lmfit.Parameters) -> float:
        """Calculate dielectric strength."""
        return params['eps_s'].value - params['eps_inf'].value

    @staticmethod
    def get_loss_peak_height(params: lmfit.Parameters) -> float:
        """
        Calculate maximum loss value for H-N model.

        This requires evaluation at the peak frequency.
        """
        eps_inf = params['eps_inf'].value
        eps_s = params['eps_s'].value
        tau = params['tau'].value
        alpha = params['alpha'].value
        beta = params['beta'].value

        # Get peak frequency
        f_max = HavriliakNegamiModel.get_relaxation_frequency(params)
        omega_max = 2 * np.pi * f_max * 1e9

        # Evaluate at peak
        delta_eps = eps_s - eps_inf
        denominator = (1 + (1j * omega_max * tau) ** alpha) ** beta
        eps_imag_max = delta_eps * np.imag(1 / denominator)

        return eps_imag_max

    @staticmethod
    def get_distribution_parameters(params: lmfit.Parameters) -> Dict[str, float]:
        """
        Calculate relaxation time distribution characteristics.

        H-N distribution is asymmetric and complex.
        """
        alpha = params['alpha'].value
        beta = params['beta'].value
        tau_hn = params['tau'].value

        # Most probable relaxation time
        # Different from characteristic time for H-N
        if alpha < 1 or beta < 1:
            tau_max = tau_hn * (np.sin(np.pi / (2 * (1 + beta))) /
                                np.sin(np.pi * alpha / (2 * (1 + beta)))) ** (1 / alpha)
        else:
            tau_max = tau_hn

        # Average relaxation time (first moment)
        # Complex calculation for H-N
        from scipy.special import gamma as gamma_func

        tau_mean = tau_hn * gamma_func(beta) * gamma_func(1 - beta / alpha) / gamma_func(1 - beta + beta / alpha)

        # Distribution width - approximate
        # H-N is asymmetric, so we report low and high frequency widths
        width_parameter = (1 - alpha) + (1 - beta)

        return {
            'tau_hn': tau_hn,
            'tau_peak': tau_max,
            'tau_mean': tau_mean,
            'alpha': alpha,
            'beta': beta,
            'width_parameter': width_parameter,
            'asymmetry': 1 - beta,  # Measure of asymmetry
            'model_type': HavriliakNegamiModel.identify_model_type(params)
        }

    def get_plot_data(self,
                      result: lmfit.model.ModelResult,
                      n_points: int = 1000,
                      show_distribution: bool = True,
                      show_components: bool = False) -> Dict[str, Any]:
        """
        Generate plot data including distribution and special components.

        Args:
            result: Fit result
            n_points: Number of points for smooth curves
            show_distribution: Include relaxation time distribution
            show_components: Show ε' and ε'' separately

        Returns:
            Dictionary with plot data
        """
        # Get base plot data - only pass n_points which parent accepts
        base_data = super().get_plot_data(result, n_points)

        # Add H-N specific information
        params = result.params
        dist_params = HavriliakNegamiModel.get_distribution_parameters(params)

        base_data['parameters']['model_type'] = HavriliakNegamiModel.identify_model_type(params)
        base_data['parameters']['relaxation_freq_ghz'] = HavriliakNegamiModel.get_relaxation_frequency(params)
        base_data['parameters']['distribution_info'] = dist_params

        # Add separate ε' and ε'' if requested
        if show_components:
            freq_smooth = base_data['smooth']['freq']
            eps_complex = self.predict(freq_smooth, params)

            base_data['components'] = {
                'freq': freq_smooth,
                'eps_real': eps_complex.real,
                'eps_imag': eps_complex.imag,
                'loss_tangent': eps_complex.imag / eps_complex.real
            }

        # Add distribution if requested
        if show_distribution:
            tau_array, g_tau = HavriliakNegamiModel._calculate_hn_distribution(
                params['tau'].value, params['alpha'].value, params['beta'].value
            )
            base_data['distribution'] = {
                'tau': tau_array,
                'g_tau': g_tau,
                'tau_hn': params['tau'].value,
                'alpha': params['alpha'].value,
                'beta': params['beta'].value
            }

        return base_data

    @staticmethod
    def _calculate_hn_distribution(tau_hn: float, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the H-N relaxation time distribution.

        The H-N distribution is complex and asymmetric.
        """
        # Generate tau array
        tau_min = tau_hn * 10 ** (-4)
        tau_max = tau_hn * 10 ** (4)
        tau_array = np.logspace(np.log10(tau_min), np.log10(tau_max), 500)

        # H-N distribution function (approximate)
        # This is the imaginary part of the H-N function
        omega = 1 / tau_array
        omega_hn = 1 / tau_hn

        # Calculate distribution
        x = omega / omega_hn

        # H-N distribution in frequency space
        numerator = x ** alpha * np.sin(beta * np.arctan(x ** alpha * np.sin(np.pi * alpha) /
                                                         (1 + x ** alpha * np.cos(np.pi * alpha))))
        denominator = (1 + 2 * x ** alpha * np.cos(np.pi * alpha) + x ** (2 * alpha)) ** (beta / 2)

        g_omega = numerator / denominator

        # Convert to time distribution
        g_tau = g_omega / tau_array  # Jacobian for log scale

        # Normalize
        g_tau = g_tau / np.trapezoid(g_tau, np.log(tau_array))

        return tau_array, g_tau

    def to_dict(self, result: lmfit.model.ModelResult) -> Dict[str, Any]:
        """
        Convert fit results to dictionary with H-N specific information.

        Args:
            result: ModelResult from fitting

        Returns:
            Dictionary with fit results and derived quantities
        """
        # Get base dictionary
        output = super().to_dict(result)

        # Add H-N specific information
        params = result.params

        output['model_subtype'] = HavriliakNegamiModel.identify_model_type(params)
        output['derived_quantities'] = {
            'relaxation_frequency_ghz': HavriliakNegamiModel.get_relaxation_frequency(params),
            'dielectric_strength': HavriliakNegamiModel.get_dielectric_strength(params),
            'loss_peak_height': HavriliakNegamiModel.get_loss_peak_height(params),
            'eps_inf': params['eps_inf'].value,
            'eps_s': params['eps_s'].value,
            'tau_seconds': params['tau'].value,
            'tau_picoseconds': params['tau'].value * 1e12,
            'alpha': params['alpha'].value,
            'beta': params['beta'].value,
            'alpha_beta_product': params['alpha'].value * params['beta'].value,
            'distribution_parameters': HavriliakNegamiModel.get_distribution_parameters(params)
        }

        if self.use_conductivity:
            output['derived_quantities']['sigma_dc'] = params['sigma_dc'].value

        return output

    @staticmethod
    def estimate_special_cases(params: lmfit.Parameters) -> Dict[str, Dict[str, float]]:
        """
        Estimate parameters for special cases (Debye, Cole-Cole, Davidson-Cole).

        Useful for model comparison and understanding.

        Args:
            params: H-N parameters

        Returns:
            Dictionary with equivalent parameters for each special case
        """
        eps_inf = params['eps_inf'].value
        eps_s = params['eps_s'].value
        tau_hn = params['tau'].value
        alpha = params['alpha'].value
        beta = params['beta'].value

        # Peak frequency for matching
        f_peak = HavriliakNegamiModel.get_relaxation_frequency(params)

        results = {
            'Debye': {
                'eps_inf': eps_inf,
                'eps_s': eps_s,
                'tau': 1 / (2 * np.pi * f_peak * 1e9)
            },
            'Cole-Cole': {
                'eps_inf': eps_inf,
                'eps_s': eps_s,
                'tau': tau_hn * alpha ** (1 / alpha),  # Approximate
                'alpha': (alpha + beta) / 2  # Average broadening
            },
            'Davidson-Cole': {
                'eps_inf': eps_inf,
                'eps_s': eps_s,
                'tau': tau_hn / beta,  # Approximate
                'beta': beta
            }
        }

        return results

    def fit_constrained(self,
                        freq: np.ndarray,
                        dk_exp: np.ndarray,
                        df_exp: np.ndarray,
                        constraint: str = 'alpha_beta_product',
                        **kwargs) -> lmfit.model.ModelResult:
        """
        Fit with specific constraints on α and β.

        Args:
            freq: Frequency array
            dk_exp: Experimental Dk
            df_exp: Experimental Df
            constraint: Type of constraint
                - 'alpha_beta_product': α×β ≤ 1
                - 'alpha_equals_beta': α = β (symmetric H-N)
                - 'sum_constraint': α + β ≤ constant
            **kwargs: Additional fit arguments

        Returns:
            Fit result
        """
        params = self.create_parameters(freq, dk_exp, df_exp)

        if constraint == 'alpha_beta_product':
            # This is already handled in parameter creation if constrain_parameters=True
            pass

        elif constraint == 'alpha_equals_beta':
            # Force α = β
            params['beta'].expr = 'alpha'

        elif constraint == 'sum_constraint':
            max_sum = kwargs.get('max_sum', 1.5)
            # Add constraint α + β ≤ max_sum
            # This would require custom minimization with scipy.optimize
            logger.warning(f"Sum constraint (α + β ≤ {max_sum}) not fully implemented")
            # For now, just warn but proceed with standard fitting

        return self.fit(freq, dk_exp, df_exp, params=params, **kwargs)


class HavriliakNegamiWithConductivityModel(HavriliakNegamiModel):
    """
    Convenience class for H-N model with conductivity enabled by default.
    """

    def __init__(self, constrain_parameters: bool = True, name: Optional[str] = None):
        """Initialize H-N model with conductivity correction."""
        super().__init__(
            conductivity_correction=True,
            constrain_parameters=constrain_parameters,
            name=name or "H-N+Conductivity"
        )