from __future__ import annotations

import numpy as np


class ScaledResidualMixin:
    """Normalise real and imag residuals by their dynamic range."""

    def _residual(self, params, data, *args, **kws):  # lmfit hook
        # Filter out None values from args
        filtered_args = [arg for arg in args if arg is not None]

        # Avoid double-passing of f_ghz if it's in both args and kwargs
        if "f_ghz" in kws and filtered_args:
            filtered_args = ()

        model = self.eval(params, *filtered_args, **kws)
        scale_r = np.ptp(np.real(data)) or 1.0
        scale_i = np.ptp(np.imag(data)) or 1.0
        res = (np.real(model) - np.real(data)) / scale_r
        resi = (np.imag(model) - np.imag(data)) / scale_i
        return np.concatenate((res, resi)).ravel()
