from __future__ import annotations

import numpy.typing as npt
from lmfit.model import Model

from ._eval_funcs import dsarkar_eval
from ._mixins import ScaledResidualMixin

FloatArray = npt.NDArray
ComplexArray = npt.NDArray


class DSarkarModel(ScaledResidualMixin, Model):
    """Model implementation of the Djordjevic–Sarkar dispersion."""

    def __init__(self, prefix: str = "", **kws):
        super().__init__(dsarkar_eval, independent_vars=["f_ghz"], prefix=prefix, **kws)
        self.set_param_hint("eps_inf", min=1.0)
        self.set_param_hint("delta_eps", min=0.0)
        self.set_param_hint("omega1", min=1e7)
        self.set_param_hint("omega2", expr=f"{prefix}omega1*10")

    def guess(self, data: ComplexArray, f_ghz: FloatArray, **overrides):
        eps_inf = float((data[-1]).real)
        delta = float((data[0] - data[-1]).real)
        omega_peak = _guess_peak(f_ghz, data)
        params = self.make_params(
            eps_inf=eps_inf,
            delta_eps=delta,
            omega1=omega_peak,
            omega2=omega_peak * 10,
        )
        # Apply overrides to parameter values
        for key, value in overrides.items():
            if key in params:
                params[key].set(value=value)
        return params


def _guess_peak(f_ghz: FloatArray, data: ComplexArray) -> float:
    """Estimate omega1 based on maximum loss point."""
    import numpy as np

    # Handle edge case where imaginary part is flat
    imag_data = np.imag(data)
    if np.allclose(imag_data, imag_data[0]):
        idx = len(f_ghz) // 2
    else:
        idx = int(np.argmax(imag_data))
    f_hz = f_ghz[idx] * 1e9
    return 2 * np.pi * f_hz
