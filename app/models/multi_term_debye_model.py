# app/models/multi_term_debye_model.py
"""
Multi-term Debye model for dielectric spectroscopy.
"""

from typing import Optional
import lmfit
import numpy as np

from .base_model import BaseModel


class MultiTermDebyeModel(BaseModel):
    """
    Multi-term Debye model (sum of Debye relaxations).

    Model equation:
    ε_r(ω) = ε_∞ + Σₖ [Δε_k/(1 + jωτ_k)]

    This model represents the dielectric response as a sum of multiple
    Debye relaxation processes, each with its own relaxation time.
    """

    def __init__(self,
                 n_terms: int = 4,
                 tau_min: float = 1e-12,
                 tau_max: float = 1e-6,
                 distribution: str = 'logarithmic',
                 name: Optional[str] = None):
        """
        Initialize multi-term Debye model.

        Args:
            n_terms: Number of Debye terms
            tau_min: Minimum relaxation time (s)
            tau_max: Maximum relaxation time (s)
            distribution: 'logarithmic' or 'optimized'
            name: Optional custom name
        """
        super().__init__(
            name=name or f"MultiDebye-{n_terms}",
            frequency_range=(1e-3, 100)
        )

        self.n_terms = n_terms
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.distribution = distribution

    @staticmethod
    def model_func(freq: np.ndarray, **params) -> np.ndarray:
        """
        Calculate complex permittivity using multi-term Debye.

        Args:
            freq: Frequency array in GHz
            **params: Model parameters including:
                - eps_inf: High-frequency permittivity
                - delta_eps_i: Permittivity increment for term i
                - tau_i: Relaxation time for term i (if optimized)
                - n_terms: Number of terms (fixed parameter)
                - tau_min: Minimum tau (fixed parameter)
                - tau_max: Maximum tau (fixed parameter)
                - distribution: Distribution type (fixed parameter)

        Returns:
            Complex permittivity
        """
        omega = 2 * np.pi * freq * 1e9

        # High-frequency permittivity
        eps_r = params['eps_inf'] * np.ones_like(omega, dtype=complex)

        # Get fixed parameters
        n_terms = int(params.get('n_terms', 4))
        distribution = params.get('distribution', 'logarithmic')
        tau_min = params.get('tau_min', 1e-12)
        tau_max = params.get('tau_max', 1e-6)

        # Get tau values
        if distribution == 'logarithmic':
            # Fixed distribution
            tau_values = np.logspace(np.log10(tau_min),
                                     np.log10(tau_max),
                                     n_terms)
        else:
            # From parameters
            tau_values = [params[f'tau_{i + 1}'] for i in range(n_terms)]

        # Sum Debye terms
        for i in range(n_terms):
            tau = tau_values[i]
            delta_eps = params[f'delta_eps_{i + 1}']
            eps_r += delta_eps / (1 + 1j * omega * tau)

        return eps_r

    def create_parameters(self, freq: np.ndarray, dk_exp: np.ndarray,
                          df_exp: np.ndarray) -> lmfit.Parameters:
        """
        Create initial parameters for multi-Debye model.

        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental real permittivity
            df_exp: Experimental loss factor

        Returns:
            lmfit.Parameters with initial guesses
        """
        params = lmfit.Parameters()

        # Add fixed parameters for model function
        params.add('n_terms', value=self.n_terms, vary=False)
        params.add('distribution', value=self.distribution, vary=False)
        params.add('tau_min', value=self.tau_min, vary=False)
        params.add('tau_max', value=self.tau_max, vary=False)

        # High-frequency permittivity
        eps_inf_guess = np.median(dk_exp[-max(3, len(dk_exp) // 10):])
        params.add('eps_inf', value=eps_inf_guess, min=1.0, max=10.0)

        # Total relaxation strength
        delta_eps_total = dk_exp[0] - eps_inf_guess

        # Initialize each term
        for i in range(self.n_terms):
            # Equal distribution initially
            params.add(f'delta_eps_{i + 1}',
                       value=max(0.1, delta_eps_total / self.n_terms),
                       min=0, max=10)

            if self.distribution == 'optimized':
                # Log-spaced initial tau values
                if self.n_terms > 1:
                    tau_init = self.tau_min * (self.tau_max / self.tau_min) ** (i / (self.n_terms - 1))
                else:
                    tau_init = np.sqrt(self.tau_min * self.tau_max)

                params.add(f'tau_{i + 1}',
                           value=tau_init,
                           min=self.tau_min / 10,
                           max=self.tau_max * 10)

        return params

    def get_relaxation_times(self, params: lmfit.Parameters) -> np.ndarray:
        """
        Get the relaxation times from fitted parameters.

        Args:
            params: Fitted parameters

        Returns:
            Array of relaxation times in seconds
        """
        if self.distribution == 'logarithmic':
            return np.logspace(np.log10(self.tau_min),
                               np.log10(self.tau_max),
                               self.n_terms)
        else:
            return np.array([params[f'tau_{i + 1}'].value
                             for i in range(self.n_terms)])

    def get_relaxation_strengths(self, params: lmfit.Parameters) -> np.ndarray:
        """
        Get the relaxation strengths from fitted parameters.

        Args:
            params: Fitted parameters

        Returns:
            Array of relaxation strengths (delta_eps values)
        """
        return np.array([params[f'delta_eps_{i + 1}'].value
                         for i in range(self.n_terms)])

    def get_model_summary(self, params: lmfit.Parameters) -> str:
        """
        Generate a summary of the model parameters.

        Args:
            params: Fitted parameters

        Returns:
            Summary string
        """
        p = params.valuesdict()
        summary = f"Multi-Term Debye Model ({self.n_terms} terms)\n"
        summary += f"Distribution: {self.distribution}\n"
        summary += f"ε∞ = {p['eps_inf']:.3f}\n"

        tau_values = self.get_relaxation_times(params)

        for i in range(self.n_terms):
            summary += f"Term {i + 1}: Δε = {p[f'delta_eps_{i + 1}']:.3f}, "
            summary += f"τ = {tau_values[i]:.3e} s, "
            summary += f"f = {1 / (2 * np.pi * tau_values[i]):.3e} Hz\n"

        return summary