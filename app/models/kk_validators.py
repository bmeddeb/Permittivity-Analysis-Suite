"""Kramers-Kronig validation utilities with advanced methods."""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd
from lmfit import Parameters
from lmfit.model import Model, ModelResult
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, get_window, hilbert
from scipy.stats import linregress

# Optional Numba acceleration
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def _epsilon_to_df_columns(eps_complex: np.ndarray, loss_repr: str):
    """Convert complex epsilon array into (Dk, Df) columns.

    Parameters
    ----------
    eps_complex:
        Complex permittivity values.
    loss_repr:
        Representation of the loss component. ``"eps_imag"`` uses the
        imaginary part directly while ``"tan_delta"`` interprets values as
        tangent-delta.
    """
    dk = np.real(eps_complex)
    eps_imag = np.imag(eps_complex)
    if loss_repr == "eps_imag":
        df_col = eps_imag
    elif loss_repr == "tan_delta":
        df_col = eps_imag / (dk + 1e-18)
    else:
        raise ValueError("loss_repr must be 'eps_imag' or 'tan_delta'")
    return dk, df_col


if NUMBA_AVAILABLE:

    @njit(parallel=True, fastmath=True)
    def _kk_trapz_numba(
        omega: np.ndarray, eps_imag: np.ndarray, eps_inf: float
    ) -> np.ndarray:
        """Numba-accelerated trapezoidal Kramers-Kronig on non-uniform grids."""
        n = omega.size
        dk_kk = np.empty(n, dtype=np.float64)
        for i in prange(n):
            w_i = omega[i]
            integral = 0.0
            for j in range(n - 1):
                # Skip the two intervals adjacent to the singularity
                if j == i or j + 1 == i:
                    continue
                wj, wj1 = omega[j], omega[j + 1]
                # Integrand for Kramers-Kronig
                fj = (wj * eps_imag[j]) / (wj**2 - w_i**2)
                fj1 = (wj1 * eps_imag[j + 1]) / (wj1**2 - w_i**2)
                # Standard trapezoidal rule
                integral += 0.5 * (fj + fj1) * (wj1 - wj)
            dk_kk[i] = eps_inf + (2.0 / np.pi) * integral
        return dk_kk

else:

    def _kk_trapz_numba(
        omega: np.ndarray, eps_imag: np.ndarray, eps_inf: float
    ) -> np.ndarray:
        """Pure-Python trapezoidal integration fallback."""
        n = omega.size
        dk_kk = np.empty(n, dtype=float)
        for i in range(n):
            w_i = omega[i]
            integral = 0.0
            for j in range(n - 1):
                if j == i or j + 1 == i:
                    continue
                wj, wj1 = omega[j], omega[j + 1]
                fj = (wj * eps_imag[j]) / (wj**2 - w_i**2)
                fj1 = (wj1 * eps_imag[j + 1]) / (wj1**2 - w_i**2)
                integral += 0.5 * (fj + fj1) * (wj1 - wj)
            dk_kk[i] = eps_inf + (2.0 / np.pi) * integral
        return dk_kk


class KramersKronigValidator:
    """Validate dielectric data against the Kramers-Kronig relations.

    This enhanced validator supports multiple KK transform methods including
    fast Hilbert transform for uniform grids and robust trapezoidal integration
    for non-uniform grids, with optional Numba acceleration.

    Parameters
    ----------
    df:
        DataFrame containing ``Frequency (GHz)``, ``Dk`` and ``Df`` columns.
    eps_inf:
        Optional high-frequency permittivity value. If ``None`` an estimate is
        made from the data.
    loss_repr:
        Indicates whether ``Df`` represents the imaginary part directly or a
        tangent-delta value.
    method:
        KK transform method: 'auto' selects based on grid uniformity,
        'hilbert' uses FFT-based transform, 'trapz' uses trapezoidal integration.
    eps_inf_method:
        Method for estimating eps_inf: 'mean' uses tail average, 'fit' uses
        linear regression on 1/f^2.
    tail_fraction:
        Fraction of high-frequency data to use for eps_inf estimation.
    window:
        Optional window function for Hilbert transform to reduce edge effects.
    resample_points:
        Number of points for resampling when using Hilbert on non-uniform grids.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        eps_inf: float | None = None,
        *,
        loss_repr: Literal["eps_imag", "tan_delta"] = "eps_imag",
        method: Literal["auto", "hilbert", "trapz"] = "auto",
        eps_inf_method: Literal["mean", "fit"] = "fit",
        tail_fraction: float = 0.1,
        window: str | None = None,
        resample_points: int | None = None,
        **_: Any,
    ) -> None:
        self.df = df.copy()
        self.loss_repr = loss_repr
        self.method = method
        self.eps_inf_method = eps_inf_method
        self.tail_fraction = tail_fraction
        self.window = window
        self.resample_points = resample_points

        # Convert loss representation if needed
        if loss_repr == "tan_delta":
            self.df["Df"] = self.df["Df"] * (self.df["Dk"] + 1e-18)
        elif loss_repr != "eps_imag":
            raise ValueError("loss_repr must be 'eps_imag' or 'tan_delta'")

        self.explicit_eps_inf = eps_inf
        self._eps_inf_cache: float | None = None
        self._report: str = ""
        self.validated: bool = False
        self.results: Dict[str, Any] = {}

        # Validate data
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate input data format and values."""
        required_cols = ["Frequency (GHz)", "Dk", "Df"]
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(self.df) < 2:
            raise ValueError("DataFrame must contain at least 2 data points")

        # Check for numeric data
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise ValueError(f"Column '{col}' must contain numeric data")

        # Check for NaN values
        if self.df[required_cols].isnull().any().any():
            raise ValueError("DataFrame contains NaN values")

        # Check frequency ordering
        freq_ghz = self.df["Frequency (GHz)"].values
        if not np.all(np.diff(freq_ghz) > 0):
            raise ValueError("Frequencies must be strictly increasing")

    def _is_grid_uniform(self, rtol: float = 1e-5) -> bool:
        """Check if frequency grid is uniformly spaced."""
        freq_hz = self.df["Frequency (GHz)"].values * 1e9
        diffs = np.diff(freq_hz)
        return bool(np.allclose(diffs, diffs[0], rtol=rtol))

    def _estimate_eps_inf(self) -> float:
        """Return or estimate the high-frequency permittivity."""
        if self.explicit_eps_inf is not None:
            self._eps_inf_cache = float(self.explicit_eps_inf)
            return self._eps_inf_cache

        if self._eps_inf_cache is None:
            dk = self.df["Dk"].values
            n_tail = max(3, int(self.tail_fraction * len(dk)))
            tail_dk = dk[-n_tail:]

            if self.eps_inf_method == "fit" and n_tail >= 3:
                # Fit Dk vs 1/f^2
                freq_hz = self.df["Frequency (GHz)"].values[-n_tail:] * 1e9
                slope, intercept, *_ = linregress(1 / freq_hz**2, tail_dk)
                self._eps_inf_cache = float(intercept)
            else:
                # Use mean of tail
                self._eps_inf_cache = float(np.mean(tail_dk))

        return self._eps_inf_cache

    def _detect_peaks(self) -> int:
        """Detect number of peaks in loss data."""
        df_data = self.df["Df"].values
        if np.allclose(df_data, df_data[0]):
            return 0
        peaks, _ = find_peaks(df_data, height=0.1 * np.max(df_data))
        return len(peaks)

    def _kk_hilbert(self, eps_inf: float) -> np.ndarray:
        """Apply Hilbert transform for KK on uniform grids."""
        df_data = self.df["Df"].values
        if self.window:
            w = get_window(self.window, len(df_data))
            df_windowed = df_data * w
            dk_kk = eps_inf - np.imag(hilbert(df_windowed)) / (w + 1e-12)
        else:
            dk_kk = eps_inf - np.imag(hilbert(df_data))
        return dk_kk

    def _kk_resample_hilbert(self, eps_inf: float) -> np.ndarray:
        """Resample to uniform grid and apply Hilbert transform."""
        freq_ghz = self.df["Frequency (GHz)"].values
        df_data = self.df["Df"].values

        # Determine resampling points
        num = self.resample_points or min(4096, 4 * len(freq_ghz))
        freq_uniform = np.linspace(freq_ghz.min(), freq_ghz.max(), num)

        # Interpolate to uniform grid
        interp_df = interp1d(
            freq_ghz,
            df_data,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        df_uniform = interp_df(freq_uniform)

        # Apply Hilbert transform
        if self.window:
            w = get_window(self.window, num)
            df_uniform = df_uniform * w
            dk_uniform = eps_inf - np.imag(hilbert(df_uniform)) / (w + 1e-12)
        else:
            dk_uniform = eps_inf - np.imag(hilbert(df_uniform))

        # Interpolate back to original frequencies
        interp_dk = interp1d(
            freq_uniform,
            dk_uniform,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return interp_dk(freq_ghz)

    def _kk_trapz(self, eps_inf: float) -> np.ndarray:
        """Apply trapezoidal integration for KK transform."""
        freq_hz = self.df["Frequency (GHz)"].values * 1e9
        omega = 2 * np.pi * freq_hz
        df_data = self.df["Df"].values
        return _kk_trapz_numba(omega, df_data, eps_inf)

    def validate(self, causality_threshold: float = 0.05) -> float:
        """Run the validation returning the mean relative error.

        Parameters
        ----------
        causality_threshold:
            Threshold for mean relative error to determine pass/fail status.

        Returns
        -------
        float
            Mean relative error between experimental and KK-transformed Dk.
        """
        eps_inf = self._estimate_eps_inf()

        # Determine method
        uniform = self._is_grid_uniform()
        method = self.method
        if method == "auto":
            method = "hilbert" if uniform else "trapz"
            logger.info(f"Auto-selected '{method}' method based on grid uniformity")

        # Apply KK transform
        if method == "hilbert":
            if uniform:
                dk_kk = self._kk_hilbert(eps_inf)
            else:
                logger.info(
                    "Non-uniform grid detected; resampling for Hilbert transform"
                )
                dk_kk = self._kk_resample_hilbert(eps_inf)
        else:
            dk_kk = self._kk_trapz(eps_inf)

        # Store KK results
        self.df["Dk_KK"] = dk_kk

        # Calculate metrics
        dk_exp = self.df["Dk"].values
        diff = dk_exp - dk_kk
        rel_err = np.mean(np.abs(diff) / (np.abs(dk_exp) + 1e-18))
        rmse = float(np.sqrt(np.mean(diff**2)))

        # Store results
        self.results = {
            "dk_kk": dk_kk,
            "mean_relative_error": rel_err,
            "rmse": rmse,
            "causality_status": "PASS" if rel_err < causality_threshold else "FAIL",
            "method_used": method,
            "eps_inf": eps_inf,
            "num_peaks": self._detect_peaks(),
            "grid_uniform": uniform,
        }

        # Generate report
        self._report = (
            f"\n{' Kramers-Kronig Causality Report ':=^50}\n"
            f" ▸ Causality Status:      {self.results['causality_status']}\n"
            f" ▸ Mean Relative Error:   {rel_err * 100:.2f}%\n"
            f" ▸ RMSE (Dk vs. Dk_KK):   {rmse:.4f}\n"
            f" ▸ Method Used:           {method}\n"
            f" ▸ Grid Type:             {'Uniform' if uniform else 'Non-uniform'}\n"
            f" ▸ Number of Peaks:       {self.results['num_peaks']}\n"
            f" ▸ ε_∞ (estimated):       {eps_inf:.4f}\n"
            f"{'=' * 50}\n"
        )

        self.validated = True
        return rel_err

    def get_report(self) -> str:
        """Return the textual validation report."""
        if not self._report:
            raise RuntimeError("validate() must be called before get_report()")
        return self._report

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostic information."""
        if not self.validated:
            self.validate()

        freq_ghz = self.df["Frequency (GHz)"].values
        df_data = self.df["Df"].values

        # Find frequency of maximum loss
        max_df_idx = np.argmax(df_data)

        return {
            "grid_uniform": self.results["grid_uniform"],
            "num_points": len(self.df),
            "freq_range_ghz": (float(freq_ghz.min()), float(freq_ghz.max())),
            "eps_inf": self.results["eps_inf"],
            "eps_inf_method": self.eps_inf_method,
            "method_used": self.results["method_used"],
            "max_df_freq_ghz": float(freq_ghz[max_df_idx]),
            "num_peaks": self.results["num_peaks"],
            "numba_available": NUMBA_AVAILABLE,
            **self.results,
        }

    @property
    def is_causal(self) -> bool:
        """Check if the data passes causality test."""
        if not self.results:
            raise RuntimeError("Must call validate() first")
        return self.results["causality_status"] == "PASS"

    @property
    def relative_error(self) -> float:
        """Get mean relative error."""
        if not self.results:
            raise RuntimeError("Must call validate() first")
        return self.results["mean_relative_error"]

    # Context manager support
    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.results.clear()
        self._eps_inf_cache = None
        self._report = ""
        return False

    # ------------------------------------------------------------------
    @classmethod
    def from_model(
        cls,
        model: Model,
        params: Parameters | ModelResult | None,
        f_ghz: np.ndarray,
        *,
        loss_repr: Literal["eps_imag", "tan_delta"] = "eps_imag",
        **validator_kwargs: Any,
    ) -> "KramersKronigValidator":
        """Build a validator directly from an :class:`lmfit` model."""
        if isinstance(params, ModelResult):
            params = params.params
        elif params is None:
            params = model.make_params()

        eps_complex = model.eval(params, f_ghz=f_ghz)
        dk, df_col = _epsilon_to_df_columns(eps_complex, loss_repr)

        df = pd.DataFrame({"Frequency (GHz)": f_ghz, "Dk": dk, "Df": df_col})

        eps_inf_val = params.get("eps_inf", None)
        eps_inf = eps_inf_val.value if eps_inf_val is not None else None

        return cls(df, eps_inf=eps_inf, loss_repr=loss_repr, **validator_kwargs)
