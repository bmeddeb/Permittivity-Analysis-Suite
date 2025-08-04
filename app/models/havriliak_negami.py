from __future__ import annotations

import numpy as np
import numpy.typing as npt
from lmfit.model import Model

from ._eval_funcs import hn_eval
from ._mixins import ScaledResidualMixin

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


class HavriliakNegamiModel(ScaledResidualMixin, Model):
    """lmfit Model wrapper for the Havriliak–Negami dispersion."""

    def __init__(self, prefix: str = "", **kws):
        super().__init__(hn_eval, independent_vars=["f_ghz"], prefix=prefix, **kws)
        self.set_param_hint("eps_inf", value=1.0, min=1.0)
        self.set_param_hint("delta_eps", value=1.0, min=0.0)
        self.set_param_hint("tau", value=1e-9, min=1e-15)
        self.set_param_hint("alpha", value=0.8, min=0.0, max=1.0)
        self.set_param_hint("beta", value=0.8, min=0.0, max=1.0)

    def guess(self, data: ComplexArray, f_ghz: FloatArray, **overrides: float):
        # Handle edge case where imaginary part is flat
        imag_data = np.imag(data)
        if np.allclose(imag_data, imag_data[0]):
            peak_idx = len(f_ghz) // 2
        else:
            peak_idx = np.argmax(imag_data)
        tau_guess = 1 / (2 * np.pi * f_ghz[peak_idx] * 1e9)
        params = self.make_params(
            eps_inf=np.real(data[-1]),
            delta_eps=np.real(data[0]) - np.real(data[-1]),
            tau=tau_guess,
            alpha=0.8,
            beta=0.8,
        )
        # Apply overrides to parameter values
        for key, value in overrides.items():
            if key in params:
                params[key].set(value=value)
        return params
