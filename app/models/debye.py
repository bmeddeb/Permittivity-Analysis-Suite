# app/models/debye.py
"""
Debye model for dielectric relaxation.

Implements the classic single-pole Debye relaxation model:
ε*(ω) = ε_∞ + Δε / (1 + jωτ)

This model describes materials with a single relaxation time.
"""

import numpy as np
import lmfit
from typing import Optional, Dict, Any, Tuple
from scipy.signal import find_peaks
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class DebyeModel(BaseModel):
    """
    Single Debye relaxation model for dielectric spectroscopy.

    The Debye model represents the simplest case of dielectric relaxation,
    assuming a single exponential decay of polarization. It's characterized by:
    - ε_inf: High-frequency permittivity
    - ε_s: Static (low-frequency) permittivity
    - τ: Relaxation time

    The model is ideal for:
    - Pure polar liquids (e.g., water, alcohols)
    - Dilute solutions
    - Materials with narrow relaxation time distributions

    Attributes:
        conductivity_correction (bool): Whether to include DC conductivity term
        use_conductivity (bool): Whether to fit conductivity as a parameter
    """

    def __init__(self,
                 conductivity_correction: bool = False,
                 name: Optional[str] = None):
        """
        Initialize the Debye model.

        Args:
            conductivity_correction: Include DC conductivity correction term
            name: Optional custom name for the model
        """
        super().__init__(
            name=name or "Debye",
            frequency_range=(1e-6, 1e3)  # 1 Hz to 1 THz
        )

        self.conductivity_correction = conductivity_correction
        self.use_conductivity = conductivity_correction

        # Update parameter names based on configuration
        self._update_param_names()

        logger.info(f"Initialized Debye model with conductivity_correction={conductivity_correction}")

    def _update_param_names(self) -> None:
        """Update parameter names based on model configuration."""
        self.param_names = ['eps_inf', 'eps_s', 'tau']
        if self.use_conductivity:
            self.param_names.append('sigma_dc')
        self.n_params = len(self.param_names)

    @staticmethod
    def model_func(freq: np.ndarray, **params) -> np.ndarray:
        """
        Calculate complex permittivity using Debye model.

        Args:
            freq: Frequency array in GHz
            **params: Model parameters
                - eps_inf: High-frequency permittivity
                - eps_s: Static permittivity
                - tau: Relaxation time (seconds)
                - sigma_dc: DC conductivity (S/m) [optional]

        Returns:
            Complex permittivity array
        """
        # Extract parameters
        eps_inf = params['eps_inf']
        eps_s = params['eps_s']
        tau = params['tau']
        sigma_dc = params.get('sigma_dc', 0.0)

        # Convert frequency to angular frequency (rad/s)
        omega = 2 * np.pi * freq * 1e9

        # Debye equation
        eps_complex = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)

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
        Create initial parameters for Debye model fitting.

        Uses intelligent guessing based on data characteristics:
        - eps_s from low-frequency Dk
        - eps_inf from high-frequency Dk
        - tau from Df peak frequency
        - sigma_dc from low-frequency Df slope (if enabled)

        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental Dk values
            df_exp: Experimental Df values

        Returns:
            lmfit.Parameters object with initial values and bounds
        """
        params = lmfit.Parameters()

        # Estimate eps_inf from high-frequency data
        # Use average of last 10% of data points
        n_tail = max(3, int(0.1 * len(dk_exp)))
        eps_inf_guess = np.mean(dk_exp[-n_tail:])

        # Estimate eps_s from low-frequency data
        # Use average of first 10% of data points
        n_head = max(3, int(0.1 * len(dk_exp)))
        eps_s_guess = np.mean(dk_exp[:n_head])

        # Ensure eps_s > eps_inf
        if eps_s_guess <= eps_inf_guess:
            eps_s_guess = eps_inf_guess + 0.1 * abs(eps_inf_guess)

        # Estimate relaxation time from loss peak
        # Find the frequency of maximum loss
        peaks, properties = find_peaks(df_exp, height=0.1 * np.max(df_exp))

        if len(peaks) > 0:
            # Use the most prominent peak
            if len(peaks) > 1:
                peak_heights = properties['peak_heights']
                peak_idx = peaks[int(np.argmax(peak_heights))]
            else:
                peak_idx = int(peaks[0])

            f_max = freq[peak_idx]
            # For Debye: f_max = 1 / (2π τ)
            tau_guess = 1 / (2 * np.pi * f_max * 1e9)
        else:
            # Fallback: estimate from frequency range
            f_center = np.sqrt(freq[0] * freq[-1])
            tau_guess = 1 / (2 * np.pi * f_center * 1e9)

        # Add parameters with bounds
        params.add('eps_inf',
                   value=eps_inf_guess,
                   min=1.0,  # Physical minimum
                   max=eps_s_guess,
                   vary=True)

        params.add('eps_s',
                   value=eps_s_guess,
                   min=eps_inf_guess,
                   max=max(1000, 2 * eps_s_guess),  # Allow for high permittivity materials
                   vary=True)

        params.add('tau',
                   value=tau_guess,
                   min=1e-15,  # Femtosecond scale
                   max=1e-3,  # Millisecond scale
                   vary=True)

        # Add conductivity parameter if enabled
        if self.use_conductivity:
            # Estimate from low-frequency Df slope
            if len(freq) > 10:
                # Use first 10% of data
                n_fit = max(5, int(0.1 * len(freq)))

                # Calculate imaginary permittivity
                omega = 2 * np.pi * freq[:n_fit] * 1e9
                eps_imag = df_exp[:n_fit] * dk_exp[:n_fit]

                # Fit log-log slope
                try:
                    from scipy.stats import linregress
                    slope, intercept, _, _, _ = linregress(np.log10(omega), np.log10(eps_imag))

                    if abs(slope + 1) < 0.3:  # Close to -1 slope indicates conductivity
                        eps_0 = 8.854187817e-12
                        sigma_guess = 10 ** intercept * eps_0
                    else:
                        sigma_guess = 0.0
                except (ValueError, RuntimeError):
                    sigma_guess = 0.0
            else:
                sigma_guess = 0.0

            params.add('sigma_dc',
                       value=sigma_guess,
                       min=0.0,
                       max=100.0,  # S/m
                       vary=True)

        # Log initial guesses
        logger.debug(f"Initial parameter guesses:")
        for name, param in params.items():
            logger.debug(f"  {name}: {param.value:.4g} [{param.min:.4g}, {param.max:.4g}]")

        return params

    def validate_parameters(self, params: lmfit.Parameters) -> bool:
        """
        Validate parameter values for physical consistency.

        Args:
            params: Parameters to validate

        Returns:
            True if parameters are physically valid
        """
        # Check basic Debye constraints
        if params['eps_s'].value <= params['eps_inf'].value:
            logger.warning("Invalid parameters: eps_s must be greater than eps_inf")
            return False

        if params['tau'].value <= 0:
            logger.warning("Invalid parameters: tau must be positive")
            return False

        if self.use_conductivity and params['sigma_dc'].value < 0:
            logger.warning("Invalid parameters: sigma_dc must be non-negative")
            return False

        return True

    @staticmethod
    def get_relaxation_frequency(params: lmfit.Parameters) -> float:
        """
        Calculate the relaxation frequency from parameters.

        For Debye model: f_rel = 1 / (2π τ)

        Args:
            params: Model parameters

        Returns:
            Relaxation frequency in GHz
        """
        tau = params['tau'].value
        f_rel_hz = 1 / (2 * np.pi * tau)
        return f_rel_hz / 1e9  # Convert to GHz

    @staticmethod
    def get_dielectric_strength(params: lmfit.Parameters) -> float:
        """
        Calculate the dielectric strength (Δε).

        Args:
            params: Model parameters

        Returns:
            Dielectric strength
        """
        return params['eps_s'].value - params['eps_inf'].value

    @staticmethod
    def get_loss_peak_height(params: lmfit.Parameters) -> float:
        """
        Calculate the maximum loss factor value.

        For Debye: ε''_max = Δε / 2

        Args:
            params: Model parameters

        Returns:
            Maximum loss factor
        """
        delta_eps = DebyeModel.get_dielectric_strength(params)
        return delta_eps / 2

    def to_dict(self, result: lmfit.model.ModelResult) -> Dict[str, Any]:
        """
        Convert fit results to dictionary with Debye-specific information.

        Args:
            result: ModelResult from fitting

        Returns:
            Dictionary with fit results and derived quantities
        """
        # Get base dictionary
        output = super().to_dict(result)

        # Add Debye-specific information
        params = result.params

        output['derived_quantities'] = {
            'relaxation_frequency_ghz': DebyeModel.get_relaxation_frequency(params),
            'dielectric_strength': DebyeModel.get_dielectric_strength(params),
            'loss_peak_height': DebyeModel.get_loss_peak_height(params),
            'eps_inf': params['eps_inf'].value,
            'eps_s': params['eps_s'].value,
            'tau_seconds': params['tau'].value,
            'tau_picoseconds': params['tau'].value * 1e12
        }

        if self.use_conductivity:
            output['derived_quantities']['sigma_dc'] = params['sigma_dc'].value

        return output

    def get_plot_data(self,
                      result: lmfit.model.ModelResult,
                      n_points: int = 1000) -> Dict[str, Any]:
        """
        Generate plot data in a library-agnostic format.

        Args:
            result: Fit result from model.fit()
            n_points: Number of points for smooth model curves

        Returns:
            Dictionary containing plot data:
                - 'experimental': Dict with freq, dk, df arrays
                - 'fitted': Dict with freq, dk, df arrays (at experimental points)
                - 'smooth': Dict with freq, dk, df arrays (high-resolution)
                - 'residuals': Dict with freq, dk_residuals, df_residuals
                - 'parameters': Dict with fit parameters and derived quantities
                - 'metrics': Dict with fit quality metrics
        """
        # Access custom attributes added by BaseModel.fit()
        freq_exp = getattr(result, 'freq', None)
        dk_exp = getattr(result, 'dk_exp', None)
        df_exp = getattr(result, 'df_exp', None)
        dk_fit = getattr(result, 'dk_fit', None)
        df_fit = getattr(result, 'df_fit', None)
        fit_metrics = getattr(result, 'fit_metrics', None)

        if freq_exp is None or dk_exp is None or df_exp is None:
            raise ValueError("Result object is missing required data. Ensure it was created by BaseModel.fit()")

        # Generate smooth curves for plotting
        freq_min, freq_max = freq_exp.min(), freq_exp.max()
        freq_smooth = np.logspace(np.log10(freq_min), np.log10(freq_max), n_points)
        smooth_complex = self.predict(freq_smooth, result.params)
        dk_smooth = smooth_complex.real
        df_smooth = smooth_complex.imag / smooth_complex.real  # Convert to loss factor

        # Calculate residuals
        dk_residuals = dk_exp - dk_fit
        df_residuals = df_exp - df_fit

        # Prepare parameters info
        params_info = {
            'eps_inf': result.params['eps_inf'].value,
            'eps_s': result.params['eps_s'].value,
            'tau': result.params['tau'].value,
            'tau_ps': result.params['tau'].value * 1e12,  # picoseconds
            'relaxation_freq_ghz': DebyeModel.get_relaxation_frequency(result.params),
            'dielectric_strength': DebyeModel.get_dielectric_strength(result.params),
            'loss_peak_height': DebyeModel.get_loss_peak_height(result.params)
        }

        # Add uncertainties if available
        for param_name in ['eps_inf', 'eps_s', 'tau']:
            if result.params[param_name].stderr is not None:
                params_info[f'{param_name}_stderr'] = result.params[param_name].stderr

        if self.use_conductivity and 'sigma_dc' in result.params:
            params_info['sigma_dc'] = result.params['sigma_dc'].value
            if result.params['sigma_dc'].stderr is not None:
                params_info['sigma_dc_stderr'] = result.params['sigma_dc'].stderr

        # Prepare metrics
        if fit_metrics:
            metrics = {
                'r_squared': fit_metrics.r_squared,
                'rmse': fit_metrics.rmse,
                'mape': fit_metrics.mape,
                'chi_squared': fit_metrics.chi_squared,
                'reduced_chi_squared': fit_metrics.reduced_chi_squared,
                'success': result.success
            }
        else:
            metrics = {'success': result.success}

        return {
            'experimental': {
                'freq': freq_exp,
                'dk': dk_exp,
                'df': df_exp
            },
            'fitted': {
                'freq': freq_exp,
                'dk': dk_fit,
                'df': df_fit
            },
            'smooth': {
                'freq': freq_smooth,
                'dk': dk_smooth,
                'df': df_smooth
            },
            'residuals': {
                'freq': freq_exp,
                'dk': dk_residuals,
                'df': df_residuals
            },
            'parameters': params_info,
            'metrics': metrics,
            'model_name': self.name
        }

    def export_for_circuit_simulation(self,
                                      params: lmfit.Parameters) -> Dict[str, float]:
        """
        Export parameters in format suitable for circuit simulation.

        Converts Debye parameters to equivalent RC circuit values.

        Args:
            params: Model parameters

        Returns:
            Dictionary with circuit element values
        """
        eps_inf = params['eps_inf'].value
        eps_s = params['eps_s'].value
        tau = params['tau'].value

        # For parallel RC circuit representation
        # C∞ represents high-frequency capacitance
        # R and C represent the relaxation

        eps_0 = 8.854187817e-12  # F/m

        # Assuming unit area and thickness for normalized values
        C_inf = eps_inf * eps_0  # High-frequency capacitance
        C_relax = (eps_s - eps_inf) * eps_0  # Relaxation capacitance
        R_relax = tau / C_relax  # Relaxation resistance

        circuit_params = {
            'model': 'Debye',
            'C_inf_F': C_inf,
            'C_relax_F': C_relax,
            'R_relax_Ohm': R_relax,
            'tau_s': tau,
            'eps_inf': eps_inf,
            'eps_s': eps_s,
            'circuit_type': 'parallel_RC',
            'description': 'Parallel RC circuit with series C_inf'
        }

        if self.use_conductivity and 'sigma_dc' in params:
            sigma_dc = params['sigma_dc'].value
            if sigma_dc > 0:
                R_dc = 1 / sigma_dc  # DC resistance per unit area/length
                circuit_params['R_dc_Ohm'] = R_dc
                circuit_params['sigma_dc_S/m'] = sigma_dc

        return circuit_params

    @staticmethod
    def from_circuit_parameters(C_inf: float,
                                C_relax: float,
                                R_relax: float,
                                sigma_dc: Optional[float] = None) -> Dict[str, float]:
        """
        Convert circuit parameters back to Debye parameters.

        Args:
            C_inf: High-frequency capacitance (F)
            C_relax: Relaxation capacitance (F)
            R_relax: Relaxation resistance (Ohm)
            sigma_dc: DC conductivity (S/m) [optional]

        Returns:
            Dictionary with Debye model parameters
        """
        eps_0 = 8.854187817e-12  # F/m

        eps_inf = C_inf / eps_0
        delta_eps = C_relax / eps_0
        eps_s = eps_inf + delta_eps
        tau = R_relax * C_relax

        params = {
            'eps_inf': eps_inf,
            'eps_s': eps_s,
            'tau': tau
        }

        if sigma_dc is not None:
            params['sigma_dc'] = sigma_dc

        return params


class DebyeWithConductivityModel(DebyeModel):
    """
    Convenience class for Debye model with conductivity correction enabled by default.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize Debye model with conductivity correction."""
        super().__init__(
            conductivity_correction=True,
            name=name or "Debye+Conductivity"
        )