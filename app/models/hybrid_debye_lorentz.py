# app/models/hybrid_debye_lorentz.py
"""
Hybrid Debye-Lorentz model combining relaxation and resonance effects.

Implements a model with both Debye relaxation and Lorentz oscillators:
ε*(ω) = ε_∞ + Σ(Δεᵢ/(1+jωτᵢ)) + Σ(fₖ/(ω₀ₖ²-ω²-jγₖω))

This model describes materials with both relaxation and resonance processes.
"""

import numpy as np
import lmfit
from typing import Optional, Dict, Any, List, Tuple, Union
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class HybridDebyeLorentzModel(BaseModel):
    """
    Hybrid model combining Debye relaxation and Lorentz oscillator terms.

    This model is ideal for materials exhibiting both:
    - Low-frequency relaxation processes (ionic, interfacial, dipolar)
    - High-frequency resonances (phonons, electronic transitions)

    Common applications:
    - Semiconductors with ionic conductivity
    - Biological tissues (cellular relaxation + water resonance)
    - Composite materials with multiple mechanisms
    - Polymers with ionic impurities
    - Minerals and geological materials

    The model equation:
    ε*(ω) = ε_∞ + Σᵢ[Δεᵢ/(1+jωτᵢ)] + Σₖ[fₖω₀ₖ²/(ω₀ₖ²-ω²-jγₖω)]

    Parameters:
        n_debye: Number of Debye relaxation terms
        n_lorentz: Number of Lorentz oscillator terms
        auto_detect: Automatically determine number of each type
        constrain_parameters: Apply physical constraints
        shared_eps_inf: Use single ε_∞ for both mechanisms
    """

    def __init__(self,
                 n_debye: int = 1,
                 n_lorentz: int = 1,
                 auto_detect: bool = False,
                 constrain_parameters: bool = True,
                 shared_eps_inf: bool = True,
                 name: Optional[str] = None):
        """
        Initialize the hybrid model.

        Args:
            n_debye: Number of Debye relaxation terms
            n_lorentz: Number of Lorentz oscillator terms
            auto_detect: Automatically detect number of processes
            constrain_parameters: Apply physical constraints
            shared_eps_inf: Use single high-frequency permittivity
            name: Optional custom name
        """
        super().__init__(
            name=name or f"Hybrid-{n_debye}D-{n_lorentz}L",
            frequency_range=(1e-6, 1e6)  # 1 Hz to 1 PHz
        )

        self.n_debye = n_debye
        self.n_lorentz = n_lorentz
        self.auto_detect = auto_detect
        self.constrain_parameters = constrain_parameters
        self.shared_eps_inf = shared_eps_inf

        # Will be updated after detection if auto_detect=True
        self._detected_debye = None
        self._detected_lorentz = None

        # Frequency thresholds for process separation
        self.transition_freq = 1.0  # GHz, typical boundary

        # Update parameter names
        self._update_param_names()

        logger.info(f"Initialized hybrid model with {n_debye} Debye and "
                    f"{n_lorentz} Lorentz processes")

    def _update_param_names(self) -> None:
        """Update parameter names based on current configuration."""
        self.param_names = ['eps_inf']

        # Use detected values if available
        n_d = self._detected_debye if self._detected_debye else self.n_debye
        n_l = self._detected_lorentz if self._detected_lorentz else self.n_lorentz

        # Debye parameters
        for i in range(n_d):
            self.param_names.extend([
                f'delta_eps_{i + 1}',  # Relaxation strength
                f'tau_{i + 1}'  # Relaxation time
            ])

        # Lorentz parameters
        for i in range(n_l):
            self.param_names.extend([
                f'f_{i + 1}',  # Oscillator strength
                f'omega0_{i + 1}',  # Resonance frequency
                f'gamma_{i + 1}'  # Damping
            ])

        self.n_params = len(self.param_names)

    @staticmethod
    def model_func(freq: np.ndarray, **params) -> np.ndarray:
        """
        Calculate complex permittivity using hybrid model.

        Args:
            freq: Frequency array in GHz
            **params: Model parameters

        Returns:
            Complex permittivity array
        """
        omega = 2 * np.pi * freq * 1e9  # Convert to rad/s

        # Start with high-frequency permittivity
        eps_complex = params['eps_inf'] * np.ones_like(omega, dtype=complex)

        # Add Debye contributions
        i = 1
        while f'delta_eps_{i}' in params:
            delta_eps = params[f'delta_eps_{i}']
            tau = params[f'tau_{i}']

            eps_complex += delta_eps / (1 + 1j * omega * tau)
            i += 1

        # Add Lorentz contributions
        i = 1
        while f'f_{i}' in params:
            f_i = params[f'f_{i}']
            omega0_i = params[f'omega0_{i}'] * 2 * np.pi * 1e9
            gamma_i = params[f'gamma_{i}'] * 2 * np.pi * 1e9

            denominator = omega0_i ** 2 - omega ** 2 - 1j * gamma_i * omega
            eps_complex += f_i * omega0_i ** 2 / denominator

            i += 1

        return eps_complex

    def detect_processes(self, freq: np.ndarray, dk_exp: np.ndarray,
                         df_exp: np.ndarray) -> Tuple[int, int]:
        """
        Automatically detect number of Debye and Lorentz processes.

        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental Dk
            df_exp: Experimental Df

        Returns:
            Tuple of (n_debye, n_lorentz)
        """
        # Calculate loss spectrum
        eps_imag = df_exp * dk_exp

        # Separate frequency regions
        low_freq_mask = freq < self.transition_freq
        high_freq_mask = freq >= self.transition_freq

        # Detect Debye processes (low frequency)
        n_debye = 0
        if np.any(low_freq_mask):
            # Look for monotonic decrease characteristic of relaxation
            df_low = df_exp[low_freq_mask]
            freq_low = freq[low_freq_mask]

            # Check for peaks in Df (indicates relaxation)
            peaks_df, _ = find_peaks(df_low)
            n_debye = len(peaks_df)

            # Also check for slope changes
            if n_debye == 0 and len(df_low) > 3:
                d_log_df = np.gradient(np.log10(df_low), np.log10(freq_low))
                slope_changes = find_peaks(np.abs(np.gradient(d_log_df)))[0]
                n_debye = max(1, len(slope_changes))

        # Detect Lorentz processes (high frequency)
        n_lorentz = 0
        if np.any(high_freq_mask):
            eps_imag_high = eps_imag[high_freq_mask]

            # Look for resonance peaks
            peaks_lorentz, properties = find_peaks(
                eps_imag_high,
                prominence=0.1 * np.max(eps_imag_high)
            )
            n_lorentz = len(peaks_lorentz)

        # Practical limits
        n_debye = min(max(n_debye, 1), 3)
        n_lorentz = min(max(n_lorentz, 0), 5)

        logger.info(f"Detected {n_debye} Debye and {n_lorentz} Lorentz processes")

        return n_debye, n_lorentz

    def create_parameters(self,
                          freq: np.ndarray,
                          dk_exp: np.ndarray,
                          df_exp: np.ndarray) -> lmfit.Parameters:
        """
        Create initial parameters for hybrid model fitting.

        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental Dk values
            df_exp: Experimental Df values

        Returns:
            lmfit.Parameters object with initial values
        """
        # Auto-detect if requested
        if self.auto_detect:
            self._detected_debye, self._detected_lorentz = self.detect_processes(
                freq, dk_exp, df_exp
            )
            self.n_debye = self._detected_debye
            self.n_lorentz = self._detected_lorentz
            self._update_param_names()

        params = lmfit.Parameters()

        # Estimate high-frequency permittivity
        # Use highest frequency value as initial guess
        eps_inf_guess = dk_exp[-1]

        params.add('eps_inf',
                   value=eps_inf_guess,
                   min=1.0,
                   max=max(10, 2 * eps_inf_guess),
                   vary=True)

        # Initialize Debye processes
        self._init_debye_params(params, freq, dk_exp, df_exp)

        # Initialize Lorentz processes
        self._init_lorentz_params(params, freq, dk_exp, df_exp)

        # Apply constraints if requested
        if self.constrain_parameters:
            self._apply_constraints(params)

        return params

    def _init_debye_params(self, params: lmfit.Parameters,
                           freq: np.ndarray, dk_exp: np.ndarray,
                           df_exp: np.ndarray) -> None:
        """Initialize Debye relaxation parameters."""
        # Focus on low-frequency region
        low_freq_mask = freq < self.transition_freq

        if not np.any(low_freq_mask):
            # No low-frequency data, use defaults
            for i in range(self.n_debye):
                params.add(f'delta_eps_{i + 1}', value=1.0, min=0, max=100)
                params.add(f'tau_{i + 1}', value=10 ** (-9 - i), min=1e-15, max=1e-3)
            return

        freq_low = freq[low_freq_mask]
        dk_low = dk_exp[low_freq_mask]
        df_low = df_exp[low_freq_mask]

        # Estimate static permittivity
        eps_s_guess = dk_low[0] if len(dk_low) > 0 else dk_exp[0]
        eps_inf_current = params['eps_inf'].value
        total_delta_eps = max(0, eps_s_guess - eps_inf_current)

        # Find characteristic frequencies for each Debye process
        if self.n_debye == 1:
            # Single Debye
            # Find frequency of maximum Df
            if len(df_low) > 0:
                max_idx = np.argmax(df_low)
                f_max = freq_low[max_idx]
                tau_guess = 1 / (2 * np.pi * f_max * 1e9)
            else:
                tau_guess = 1e-9

            params.add('delta_eps_1',
                       value=total_delta_eps,
                       min=0,
                       max=100,
                       vary=True)

            params.add('tau_1',
                       value=tau_guess,
                       min=1e-15,
                       max=1e-3,
                       vary=True)

        else:
            # Multiple Debye processes
            # Distribute logarithmically
            tau_values = np.logspace(-12, -6, self.n_debye)

            for i in range(self.n_debye):
                # Equal distribution of relaxation strength
                delta_eps_i = total_delta_eps / self.n_debye

                params.add(f'delta_eps_{i + 1}',
                           value=delta_eps_i,
                           min=0,
                           max=100,
                           vary=True)

                params.add(f'tau_{i + 1}',
                           value=tau_values[i],
                           min=1e-15,
                           max=1e-3,
                           vary=True)

    def _init_lorentz_params(self, params: lmfit.Parameters,
                             freq: np.ndarray, dk_exp: np.ndarray,
                             df_exp: np.ndarray) -> None:
        """Initialize Lorentz oscillator parameters."""
        if self.n_lorentz == 0:
            return

        # Focus on high-frequency region
        high_freq_mask = freq >= self.transition_freq

        if not np.any(high_freq_mask):
            # No high-frequency data, use defaults
            for i in range(self.n_lorentz):
                params.add(f'f_{i + 1}', value=0.1, min=0.001, max=100)
                params.add(f'omega0_{i + 1}', value=10 ** (i + 1), min=0.1, max=1e6)
                params.add(f'gamma_{i + 1}', value=1.0, min=0.001, max=100)
            return

        freq_high = freq[high_freq_mask]
        eps_imag_high = df_exp[high_freq_mask] * dk_exp[high_freq_mask]

        # Find peaks for oscillator positions
        peaks, properties = find_peaks(eps_imag_high,
                                       prominence=0.05 * np.max(eps_imag_high))

        # Initialize each oscillator
        for i in range(self.n_lorentz):
            if i < len(peaks):
                # Use detected peak
                peak_idx = peaks[i]
                omega0_guess = freq_high[peak_idx]
                f_guess = eps_imag_high[peak_idx] * omega0_guess / (2 * np.pi)

                # Estimate damping from peak width
                if len(properties['widths']) > i:
                    width_idx = properties['widths'][i]
                    gamma_guess = freq_high[int(peak_idx + width_idx / 2)] - \
                                  freq_high[int(peak_idx - width_idx / 2)]
                else:
                    gamma_guess = 0.1 * omega0_guess
            else:
                # Default initialization
                omega0_guess = 10 ** (i + 1)
                f_guess = 0.1
                gamma_guess = 1.0

            params.add(f'f_{i + 1}',
                       value=f_guess,
                       min=0.001,
                       max=1000,
                       vary=True)

            params.add(f'omega0_{i + 1}',
                       value=omega0_guess,
                       min=freq[0] * 0.1,
                       max=freq[-1] * 10,
                       vary=True)

            params.add(f'gamma_{i + 1}',
                       value=gamma_guess,
                       min=0.001 * omega0_guess,
                       max=omega0_guess,
                       vary=True)

    def _apply_constraints(self, params: lmfit.Parameters) -> None:
        """Apply physical constraints to parameters."""
        # Ensure Debye relaxation times are ordered
        if self.n_debye > 1:
            for i in range(1, self.n_debye):
                params.add(f'tau_ratio_{i}',
                           expr=f'tau_{i + 1} / tau_{i}',
                           min=2.0)  # Each tau at least 2x previous

        # Ensure Lorentz frequencies are ordered
        if self.n_lorentz > 1:
            for i in range(1, self.n_lorentz):
                params.add(f'freq_ratio_{i}',
                           expr=f'omega0_{i + 1} / omega0_{i}',
                           min=1.2)  # Each frequency at least 20% higher

        # Ensure underdamped oscillators
        for i in range(self.n_lorentz):
            params.add(f'Q_{i + 1}',
                       expr=f'omega0_{i + 1} / gamma_{i + 1}',
                       min=0.5)  # Q > 0.5

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

        # Check Debye parameters
        for i in range(self.n_debye):
            delta_eps = params[f'delta_eps_{i + 1}'].value
            tau = params[f'tau_{i + 1}'].value

            if delta_eps < 0:
                logger.warning(f"Invalid: delta_eps_{i + 1} must be positive")
                return False

            if tau <= 0:
                logger.warning(f"Invalid: tau_{i + 1} must be positive")
                return False

        # Check Lorentz parameters
        for i in range(self.n_lorentz):
            f = params[f'f_{i + 1}'].value
            omega0 = params[f'omega0_{i + 1}'].value
            gamma = params[f'gamma_{i + 1}'].value

            if f < 0:
                logger.warning(f"Invalid: f_{i + 1} must be positive")
                return False

            if omega0 <= 0:
                logger.warning(f"Invalid: omega0_{i + 1} must be positive")
                return False

            if gamma <= 0:
                logger.warning(f"Invalid: gamma_{i + 1} must be positive")
                return False

        return True

    def get_process_contributions(self, params: lmfit.Parameters,
                                  freq: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate individual contributions from each process.

        Args:
            params: Model parameters
            freq: Frequency array in GHz

        Returns:
            Dictionary with contribution arrays
        """
        omega = 2 * np.pi * freq * 1e9

        contributions = {
            'eps_inf': params['eps_inf'].value * np.ones_like(omega, dtype=complex),
            'debye_total': np.zeros_like(omega, dtype=complex),
            'lorentz_total': np.zeros_like(omega, dtype=complex),
            'debye_individual': [],
            'lorentz_individual': []
        }

        # Calculate Debye contributions
        for i in range(self.n_debye):
            delta_eps = params[f'delta_eps_{i + 1}'].value
            tau = params[f'tau_{i + 1}'].value

            contribution = delta_eps / (1 + 1j * omega * tau)
            contributions['debye_individual'].append(contribution)
            contributions['debye_total'] += contribution

        # Calculate Lorentz contributions
        for i in range(self.n_lorentz):
            f = params[f'f_{i + 1}'].value
            omega0 = params[f'omega0_{i + 1}'].value * 2 * np.pi * 1e9
            gamma = params[f'gamma_{i + 1}'].value * 2 * np.pi * 1e9

            contribution = f * omega0 ** 2 / (omega0 ** 2 - omega ** 2 - 1j * gamma * omega)
            contributions['lorentz_individual'].append(contribution)
            contributions['lorentz_total'] += contribution

        return contributions

    def get_characteristic_frequencies(self, params: lmfit.Parameters) -> Dict[str, List[float]]:
        """
        Get characteristic frequencies for all processes.

        Args:
            params: Model parameters

        Returns:
            Dictionary with frequency lists
        """
        frequencies = {
            'debye_relaxation': [],
            'lorentz_resonance': [],
            'lorentz_peak': []
        }

        # Debye relaxation frequencies (1/2πτ)
        for i in range(self.n_debye):
            tau = params[f'tau_{i + 1}'].value
            f_relax = 1 / (2 * np.pi * tau) / 1e9  # Convert to GHz
            frequencies['debye_relaxation'].append(f_relax)

        # Lorentz resonance and peak frequencies
        for i in range(self.n_lorentz):
            omega0 = params[f'omega0_{i + 1}'].value
            gamma = params[f'gamma_{i + 1}'].value

            frequencies['lorentz_resonance'].append(omega0)

            # Peak frequency (slightly shifted for damped oscillator)
            if gamma < omega0:
                omega_peak = omega0 * np.sqrt(1 - (gamma / (2 * omega0)) ** 2)
                frequencies['lorentz_peak'].append(omega_peak)
            else:
                frequencies['lorentz_peak'].append(0)  # Overdamped

        return frequencies

    def get_plot_data(self,
                      result: lmfit.model.ModelResult,
                      n_points: int = 1000,
                      show_components: bool = True,
                      freq_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Generate plot data including individual process contributions.

        Args:
            result: Fit result
            n_points: Number of points for smooth curves
            show_components: Show individual contributions
            freq_range: Optional custom frequency range

        Returns:
            Dictionary with plot data
        """
        # Get base plot data
        base_data = super().get_plot_data(result, n_points)

        # Generate smooth frequency array
        if freq_range:
            freq_smooth = np.logspace(np.log10(freq_range[0]),
                                      np.log10(freq_range[1]), n_points)
        else:
            freq_exp = result.freq
            freq_smooth = np.logspace(np.log10(freq_exp.min()),
                                      np.log10(freq_exp.max()), n_points)

        # Calculate model values
        eps_model = self.predict(freq_smooth, result.params)

        base_data.update({
            'freq': freq_smooth,
            'dk_fit': eps_model.real,
            'df_fit': eps_model.imag / eps_model.real,
            'eps_real': eps_model.real,
            'eps_imag': eps_model.imag
        })

        # Add component contributions if requested
        if show_components:
            contributions = self.get_process_contributions(result.params, freq_smooth)

            components_data = {
                'eps_inf': contributions['eps_inf'].real,
                'debye': {
                    'total_real': contributions['debye_total'].real,
                    'total_imag': contributions['debye_total'].imag,
                    'individual': []
                },
                'lorentz': {
                    'total_real': contributions['lorentz_total'].real,
                    'total_imag': contributions['lorentz_total'].imag,
                    'individual': []
                }
            }

            # Individual Debye processes
            for i, debye_contrib in enumerate(contributions['debye_individual']):
                components_data['debye']['individual'].append({
                    'index': i + 1,
                    'real': debye_contrib.real,
                    'imag': debye_contrib.imag,
                    'tau': result.params[f'tau_{i + 1}'].value,
                    'delta_eps': result.params[f'delta_eps_{i + 1}'].value
                })

            # Individual Lorentz oscillators
            for i, lorentz_contrib in enumerate(contributions['lorentz_individual']):
                components_data['lorentz']['individual'].append({
                    'index': i + 1,
                    'real': lorentz_contrib.real,
                    'imag': lorentz_contrib.imag,
                    'omega0': result.params[f'omega0_{i + 1}'].value,
                    'gamma': result.params[f'gamma_{i + 1}'].value,
                    'f': result.params[f'f_{i + 1}'].value
                })

            base_data['components'] = components_data

        # Add characteristic frequencies
        base_data['characteristic_frequencies'] = self.get_characteristic_frequencies(result.params)

        return base_data

    def to_dict(self, result: lmfit.model.ModelResult) -> Dict[str, Any]:
        """
        Convert fit results to dictionary with hybrid-specific information.

        Args:
            result: ModelResult from fitting

        Returns:
            Dictionary with fit results
        """
        # Get base dictionary
        output = super().to_dict(result)

        # Add hybrid-specific information
        output['n_debye'] = self.n_debye
        output['n_lorentz'] = self.n_lorentz
        output['transition_frequency_ghz'] = self.transition_freq

        # Separate Debye and Lorentz parameters
        output['debye_params'] = {}
        for i in range(self.n_debye):
            output['debye_params'][f'process_{i + 1}'] = {
                'delta_eps': result.params[f'delta_eps_{i + 1}'].value,
                'tau': result.params[f'tau_{i + 1}'].value,
                'relaxation_freq_ghz': 1 / (2 * np.pi * result.params[f'tau_{i + 1}'].value) / 1e9
            }

        output['lorentz_params'] = {}
        for i in range(self.n_lorentz):
            output['lorentz_params'][f'oscillator_{i + 1}'] = {
                'f': result.params[f'f_{i + 1}'].value,
                'omega0_ghz': result.params[f'omega0_{i + 1}'].value,
                'gamma_ghz': result.params[f'gamma_{i + 1}'].value,
                'q_factor': result.params[f'omega0_{i + 1}'].value / result.params[f'gamma_{i + 1}'].value
            }

        return output

    def analyze_frequency_regions(self, result: lmfit.model.ModelResult) -> Dict[str, Any]:
        """
        Analyze model behavior in different frequency regions.

        Args:
            result: Fit result

        Returns:
            Dictionary with regional analysis
        """
        # Define frequency regions
        freq_regions = {
            'low': (result.freq.min(), self.transition_freq),
            'high': (self.transition_freq, result.freq.max())
        }

        analysis = {}

        for region_name, (f_min, f_max) in freq_regions.items():
            # Create frequency array for region
            freq_region = np.logspace(np.log10(f_min), np.log10(f_max), 100)

            # Get contributions
            contributions = self.get_process_contributions(result.params, freq_region)

            # Calculate dominance
            debye_power = np.mean(np.abs(contributions['debye_total']) ** 2)
            lorentz_power = np.mean(np.abs(contributions['lorentz_total']) ** 2)
            total_power = debye_power + lorentz_power

            analysis[region_name] = {
                'frequency_range_ghz': (f_min, f_max),
                'debye_dominance': debye_power / (total_power + 1e-10),
                'lorentz_dominance': lorentz_power / (total_power + 1e-10),
                'dominant_mechanism': 'debye' if debye_power > lorentz_power else 'lorentz'
            }

        return analysis

    def suggest_model_simplification(self, result: lmfit.model.ModelResult) -> str:
        """
        Suggest if a simpler model would suffice.

        Args:
            result: Fit result

        Returns:
            Suggestion string
        """
        suggestions = []

        # Check if Debye terms are negligible
        debye_negligible = []
        for i in range(self.n_debye):
            delta_eps = result.params[f'delta_eps_{i + 1}'].value
            if delta_eps < 0.01:  # Very small contribution
                debye_negligible.append(i + 1)

        if len(debye_negligible) == self.n_debye:
            suggestions.append("All Debye terms negligible - consider pure Lorentz model")
        elif debye_negligible:
            suggestions.append(f"Debye terms {debye_negligible} negligible - consider reducing n_debye")

        # Check if Lorentz terms are negligible
        lorentz_negligible = []
        for i in range(self.n_lorentz):
            f = result.params[f'f_{i + 1}'].value
            if f < 0.001:  # Very small contribution
                lorentz_negligible.append(i + 1)

        if len(lorentz_negligible) == self.n_lorentz:
            suggestions.append("All Lorentz terms negligible - consider pure Debye model")
        elif lorentz_negligible:
            suggestions.append(f"Lorentz oscillators {lorentz_negligible} negligible - consider reducing n_lorentz")

        # Check parameter uncertainties
        high_uncertainty = []
        for param_name in self.param_names:
            param = result.params[param_name]
            if param.stderr and param.stderr / abs(param.value) > 0.5:
                high_uncertainty.append(param_name)

        if high_uncertainty:
            suggestions.append(f"High uncertainty in: {', '.join(high_uncertainty)}")
            suggestions.append("Consider reducing model complexity")

        if not suggestions:
            suggestions.append("Model complexity appears appropriate for the data")

        return "\n".join(suggestions)


class AdaptiveHybridModel(HybridDebyeLorentzModel):
    """
    Adaptive hybrid model that automatically determines optimal configuration.
    """

    def __init__(self, max_debye: int = 3, max_lorentz: int = 5,
                 name: Optional[str] = None):
        """
        Initialize adaptive hybrid model.

        Args:
            max_debye: Maximum number of Debye processes
            max_lorentz: Maximum number of Lorentz oscillators
            name: Optional custom name
        """
        super().__init__(
            n_debye=1,
            n_lorentz=1,
            auto_detect=True,
            name=name or "Adaptive-Hybrid"
        )
        self.max_debye = max_debye
        self.max_lorentz = max_lorentz

    def find_optimal_configuration(self, freq: np.ndarray, dk_exp: np.ndarray,
                                   df_exp: np.ndarray) -> Tuple[int, int]:
        """
        Find optimal number of Debye and Lorentz processes using AIC.

        Args:
            freq: Frequency array
            dk_exp: Experimental Dk
            df_exp: Experimental Df

        Returns:
            Tuple of (optimal_n_debye, optimal_n_lorentz)
        """
        best_aic = np.inf
        best_config = (1, 0)

        # Try different configurations
        for n_d in range(0, self.max_debye + 1):
            for n_l in range(0, self.max_lorentz + 1):
                if n_d == 0 and n_l == 0:
                    continue  # Skip null model

                try:
                    # Create model with this configuration
                    test_model = HybridDebyeLorentzModel(
                        n_debye=n_d,
                        n_lorentz=n_l,
                        auto_detect=False
                    )

                    # Fit model
                    result = test_model.fit(freq, dk_exp, df_exp)

                    # Calculate AIC
                    aic = result.aic if hasattr(result, 'aic') else \
                        2 * test_model.n_params + result.chisqr

                    logger.info(f"Config ({n_d}D, {n_l}L): AIC = {aic:.1f}")

                    if aic < best_aic:
                        best_aic = aic
                        best_config = (n_d, n_l)

                except Exception as e:
                    logger.warning(f"Fit failed for ({n_d}D, {n_l}L): {e}")

        logger.info(f"Optimal configuration: {best_config[0]} Debye, {best_config[1]} Lorentz")
        return best_config

    def fit(self, freq: np.ndarray, dk_exp: np.ndarray,
            df_exp: np.ndarray, **kwargs) -> lmfit.model.ModelResult:
        """
        Fit with automatic configuration selection.

        Args:
            freq: Frequency array
            dk_exp: Experimental Dk
            df_exp: Experimental Df
            **kwargs: Additional fitting arguments

        Returns:
            Fit result with optimal configuration
        """
        # Find optimal configuration
        n_d_opt, n_l_opt = self.find_optimal_configuration(freq, dk_exp, df_exp)

        # Update model configuration
        self.n_debye = n_d_opt
        self.n_lorentz = n_l_opt
        self._update_param_names()

        # Perform final fit
        return super().fit(freq, dk_exp, df_exp, **kwargs)