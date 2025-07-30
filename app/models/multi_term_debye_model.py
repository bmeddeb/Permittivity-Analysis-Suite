from typing import Optional
import lmfit
import numpy as np
from app.models import BaseModel


class MultiTermDebyeModel(BaseModel):
    """
    Multi-term Debye model (sum of Debye relaxations).

    Model equation:
    ε_r(ω) = ε_∞ + Σₖ [Δε_k/(1 + jωτ_k)]
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

        # Set parameter names
        self.param_names = ['eps_inf']
        for i in range(n_terms):
            self.param_names.append(f'delta_eps_{i + 1}')
            if distribution == 'optimized':
                self.param_names.append(f'tau_{i + 1}')

        self.n_params = len(self.param_names)

    def model_func(self, freq: np.ndarray, **params) -> np.ndarray:
        """
        Calculate complex permittivity using multi-term Debye.

        Args:
            freq: Frequency array in GHz
            **params: Model parameters

        Returns:
            Complex permittivity
        """
        omega = 2 * np.pi * freq * 1e9

        # High-frequency permittivity
        eps_r = params['eps_inf'] * np.ones_like(omega, dtype=complex)

        # Get tau values
        if self.distribution == 'logarithmic':
            # Fixed distribution
            tau_values = np.logspace(np.log10(self.tau_min),
                                     np.log10(self.tau_max),
                                     self.n_terms)
        else:
            # From parameters
            tau_values = [params[f'tau_{i + 1}'] for i in range(self.n_terms)]

        # Sum Debye terms
        for i, tau in enumerate(tau_values):
            delta_eps = params[f'delta_eps_{i + 1}']
            eps_r += delta_eps / (1 + 1j * omega * tau)

        return eps_r

    def create_parameters(self, freq: np.ndarray, dk_exp: np.ndarray,
                          df_exp: np.ndarray) -> lmfit.Parameters:
        """Create initial parameters for multi-Debye model."""
        params = lmfit.Parameters()

        # High-frequency permittivity
        eps_inf_guess = np.mean(dk_exp[-len(dk_exp) // 10:])
        params.add('eps_inf', value=eps_inf_guess, min=1.0, max=10.0)

        # Total relaxation
        delta_eps_total = dk_exp[0] - eps_inf_guess

        # Initialize each term
        for i in range(self.n_terms):
            # Equal distribution initially
            params.add(f'delta_eps_{i + 1}',
                       value=delta_eps_total / self.n_terms,
                       min=0, max=10)

            if self.distribution == 'optimized':
                tau_init = self.tau_min * (self.tau_max / self.tau_min) ** (i / (self.n_terms - 1))
                params.add(f'tau_{i + 1}',
                           value=tau_init,
                           min=self.tau_min / 10,
                           max=self.tau_max * 10)

        return params