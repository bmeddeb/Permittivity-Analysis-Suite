import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Literal
from scipy.signal import hilbert, get_window, find_peaks
from scipy.interpolate import interp1d
from scipy.stats import linregress
from functools import lru_cache

# Optional Numba acceleration
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def _kk_trapz_numba(omega: np.ndarray, eps_imag: np.ndarray, eps_inf: float) -> np.ndarray:
        """
        Numba-accelerated trapezoidal Kramers-Kronig on non-uniform grids.
        Expects NumPy arrays as input.
        """
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
                fj = (wj * eps_imag[j]) / (wj ** 2 - w_i ** 2)
                fj1 = (wj1 * eps_imag[j + 1]) / (wj1 ** 2 - w_i ** 2)
                # Standard trapezoidal rule
                integral += 0.5 * (fj + fj1) * (wj1 - wj)
            dk_kk[i] = eps_inf + (2.0 / np.pi) * integral
        return dk_kk
else:
    def _kk_trapz_numba(omega: np.ndarray, eps_imag: np.ndarray, eps_inf: float) -> np.ndarray:
        """
        Pure-Python trapezoidal integration fallback.
        Expects NumPy arrays as input.
        """
        n = omega.size
        dk_kk = np.empty(n, dtype=float)
        for i in range(n):
            w_i = omega[i]
            integral = 0.0
            for j in range(n - 1):
                if j == i or j + 1 == i:
                    continue
                wj, wj1 = omega[j], omega[j + 1]
                fj = (wj * eps_imag[j]) / (wj ** 2 - w_i ** 2)
                fj1 = (wj1 * eps_imag[j + 1]) / (wj1 ** 2 - w_i ** 2)
                integral += 0.5 * (fj + fj1) * (wj1 - wj)
            dk_kk[i] = eps_inf + (2.0 / np.pi) * integral
        return dk_kk


class KramersKronigValidator:
    """
    Validates experimental dielectric data via Kramers-Kronig.

    Automatically dispatches to a fast Hilbert transform (uniform grids)
    or a robust trapezoidal integration (non-uniform grids).
    """

    def __init__(
            self,
            df: pd.DataFrame,
            method: Literal['auto', 'hilbert', 'trapz'] = 'auto',
            eps_inf_method: Literal['mean', 'fit'] = 'fit',
            eps_inf: Optional[float] = None,
            tail_fraction: float = 0.1,
            min_tail_points: int = 3,
            window: Optional[str] = None,
            resample_points: Optional[int] = None
    ):
        if method not in ('auto', 'hilbert', 'trapz'):
            raise ValueError("method must be 'auto', 'hilbert', or 'trapz'.")
        if eps_inf_method not in ('mean', 'fit'):
            raise ValueError("eps_inf_method must be 'mean' or 'fit'.")

        # Validate window parameter
        if window is not None:
            valid_windows = ['hamming', 'hann', 'blackman', 'bartlett', 'kaiser', 'tukey']
            if window not in valid_windows:
                raise ValueError(f"window must be one of {valid_windows} or None")

        self.df = df
        self.method = method
        self.eps_inf_method = eps_inf_method
        self.explicit_eps_inf = eps_inf
        self.tail_fraction = tail_fraction
        self.min_tail_points = min_tail_points
        self.window = window
        self.resample_points = resample_points
        self.results: Dict[str, Any] = {}
        self._eps_inf_cache: Optional[float] = None
        self._setup()

    def _setup(self):
        """Parse and validate input DataFrame."""
        # Check for required columns
        required_columns = ['Frequency (GHz)', 'Dk', 'Df']
        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check minimum number of data points
        if len(self.df) < 2:
            raise ValueError("DataFrame must contain at least 2 data points")

        try:
            # Attempt to convert to numeric, coercing errors to NaN
            freq_ghz = pd.to_numeric(self.df['Frequency (GHz)'], errors='coerce')
            dk_exp = pd.to_numeric(self.df['Dk'], errors='coerce')
            df_exp = pd.to_numeric(self.df['Df'], errors='coerce')
        except KeyError as e:
            # This handles cases where columns are missing entirely
            logger.error(f"Missing columns: {e}")
            raise ValueError("Could not find required columns: 'Frequency (GHz)', 'Dk', 'Df'.") from e

        # Check if coercion created any NaNs, which implies non-numeric data or original NaNs
        if freq_ghz.isnull().any() or dk_exp.isnull().any() or df_exp.isnull().any():
            raise ValueError("DataFrame contains non-numeric values or NaNs.")

        # Check for negative frequencies
        if (freq_ghz < 0).any():
            raise ValueError("Negative frequencies detected")

        # Assign to self if all checks pass
        self.freq_ghz = freq_ghz
        self.dk_exp = dk_exp
        self.df_exp = df_exp
        self.freq_hz = self.freq_ghz * 1e9

        if not np.all(np.diff(self.freq_hz.to_numpy()) > 0):
            raise ValueError("Frequencies must be strictly increasing.")

    def _is_grid_uniform(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Check if frequency grid is uniformly spaced."""
        diffs = np.diff(self.freq_hz.to_numpy())
        return np.allclose(diffs, diffs[0], rtol=rtol, atol=atol)

    def _estimate_eps_inf(self) -> float:
        """Estimate ε_inf from the high-frequency tail of Dk, or use explicit eps_inf if provided."""
        # Use cached value if available
        if self._eps_inf_cache is not None:
            return self._eps_inf_cache

        if self.explicit_eps_inf is not None:
            logger.info(f"Using explicit eps_inf: {self.explicit_eps_inf}")
            self._eps_inf_cache = self.explicit_eps_inf
            return self._eps_inf_cache

        N = self.freq_hz.size
        n_tail = max(self.min_tail_points, int(self.tail_fraction * N))
        tail_freq = self.freq_hz.iloc[-n_tail:].to_numpy()
        tail_dk = self.dk_exp.iloc[-n_tail:].to_numpy()

        if self.eps_inf_method == 'fit':
            # linregress requires at least 3 points.
            if len(tail_dk) < 3:
                logger.warning(
                    f"Tail size for 'fit' method is {len(tail_dk)}, which is less than 3. "
                    "Falling back to 'mean' method for estimating eps_inf."
                )
                result = float(np.mean(tail_dk))
            else:
                # Fit Dk vs 1/f^2, physically motivated for many models
                slope, intercept, *_ = linregress(1 / tail_freq ** 2, tail_dk)
                result = float(intercept)
        else:
            # Default to mean
            result = float(np.mean(tail_dk))

        self._eps_inf_cache = result
        return result

    def _detect_peaks(self, df_exp_np: np.ndarray) -> int:
        """Detect number of peaks in Df using scipy."""
        # Use 10% of maximum as minimum height threshold
        peaks, _ = find_peaks(df_exp_np, height=0.1 * np.max(df_exp_np))
        return len(peaks)

    def _kk_resample_hilbert(self, eps_inf: float) -> np.ndarray:
        """Resample onto uniform grid and apply Hilbert transform."""
        num = self.resample_points or min(4096, 4 * self.freq_hz.size)
        ωu = np.linspace(self.freq_hz.min(), self.freq_hz.max(), num)

        freq_hz_np = self.freq_hz.to_numpy()
        df_exp_np = self.df_exp.to_numpy()

        interp_df = interp1d(freq_hz_np, df_exp_np, kind='cubic', fill_value='extrapolate')
        dfu = interp_df(ωu)
        if self.window:
            w = get_window(self.window, num)
            dfu = dfu * w
            # Add a small epsilon to avoid division by zero
            dk_u = eps_inf - np.imag(hilbert(dfu)) / (w + 1e-12)
        else:
            dk_u = eps_inf - np.imag(hilbert(dfu))
        interp_back = interp1d(ωu, dk_u, kind='cubic', fill_value='extrapolate')
        return interp_back(freq_hz_np)

    def _kk_transform(self) -> np.ndarray:
        """Select and run the KK transform method.

        Always performs the KK transform to validate causality, regardless of
        the number of peaks in Df. Even single-peak data can have causality
        violations due to noise, measurement errors, or non-Debye behavior.
        """
        eps_inf = self._estimate_eps_inf()
        uniform = self._is_grid_uniform()
        omega = self.freq_hz.to_numpy()
        df_exp_np = self.df_exp.to_numpy()

        # Log peak information for diagnostics
        num_peaks = self._detect_peaks(df_exp_np)
        logger.info(f"Detected {num_peaks} peak(s) in Df data")

        # Always perform KK transform
        method = self.method
        if method == 'auto':
            method = 'hilbert' if uniform else 'trapz'
            logger.info(f"Auto-selected '{method}' method based on grid uniformity")

        if method == 'hilbert':
            if not uniform:
                logger.warning(
                    "Non-uniform grid detected for 'hilbert' method; resampling to uniform grid"
                )
                return self._kk_resample_hilbert(eps_inf)

            data = df_exp_np
            if self.window:
                w = get_window(self.window, data.size)
                data = data * w
                # Add small epsilon to avoid division by zero
                return eps_inf - np.imag(hilbert(data)) / (w + 1e-12)
            return eps_inf - np.imag(hilbert(data))
        else:
            # trapz method
            logger.info("Using trapezoidal integration for KK transform")
            return _kk_trapz_numba(omega, df_exp_np, eps_inf)

    def validate(self, causality_threshold: float = 0.05) -> Dict[str, Any]:
        """Compute and return causality metrics."""
        if self.freq_hz.size < self.min_tail_points:
            raise ValueError(
                f"Insufficient data points for validation. Need at least {self.min_tail_points}, but have {self.freq_hz.size}.")
        logger.info("🔬 Running Kramers-Kronig causality check...")
        dk_kk = self._kk_transform()
        eps = 1e-9

        dk_exp_np = self.dk_exp.to_numpy()

        rel_err = np.abs(dk_kk - dk_exp_np) / (np.abs(dk_exp_np) + eps)
        mean_rel = float(np.mean(rel_err))
        rmse = float(np.sqrt(np.mean((dk_kk - dk_exp_np) ** 2)))
        status = "PASS" if mean_rel <= causality_threshold else "FAIL"
        logger.info(f"Causality: {status} (Mean Rel Err: {mean_rel:.2%})")
        self.results = {
            'dk_kk': dk_kk,
            'mean_relative_error': mean_rel,
            'rmse': rmse,
            'causality_status': status,
        }
        return self.results

    @property
    def is_causal(self) -> bool:
        """Check if the data passes causality test."""
        if not self.results:
            raise RuntimeError("Must call validate() first")
        return self.results['causality_status'] == 'PASS'

    @property
    def relative_error(self) -> float:
        """Get mean relative error."""
        if not self.results:
            raise RuntimeError("Must call validate() first")
        return self.results['mean_relative_error']

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostic information."""
        if not self.results:
            self.validate()

        freq_np = self.freq_hz.to_numpy()
        df_np = self.df_exp.to_numpy()

        # Find frequency of maximum Df
        max_df_idx = np.argmax(df_np)

        return {
            'grid_uniform': self._is_grid_uniform(),
            'num_points': len(self.freq_hz),
            'freq_range_ghz': (float(self.freq_ghz.min()), float(self.freq_ghz.max())),
            'eps_inf': self._estimate_eps_inf(),
            'method_used': self.method,
            'max_df_freq_ghz': float(self.freq_ghz.iloc[max_df_idx]),
            'num_peaks': self._detect_peaks(df_np),
            **self.results
        }

    def get_report(self) -> str:
        """Formatted summary of the last validation."""
        if not self.results:
            return "Validation has not been run. Call .validate() first."
        report = (
            f"\n{' Kramers-Kronig Causality Report ':=^50}\n"
            f" ▸ Causality Status:      {self.results['causality_status']}\n"
            f" ▸ Mean Relative Error:   {self.results['mean_relative_error']:.2%}\n"
            f" ▸ RMSE (Dk vs. Dk_KK):   {self.results['rmse']:.4f}\n"
            f"{'=' * 50}"
        )
        return report

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns results in a standardized dictionary for model comparison.
        """
        if not self.results:
            self.validate()
        return {
            'model_name': 'Kramers-Kronig',
            'freq_ghz': self.freq_ghz.to_numpy(),
            'eps_fit': self.results['dk_kk'] + 1j * self.df_exp.to_numpy(),
            'success': self.results['causality_status'] == 'PASS',
            'rmse': self.results['rmse'],
            'aic': np.inf,
            'bic': np.inf,
            'params': {'eps_inf': self._estimate_eps_inf()},
            'causality_report': self.results,
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up resources."""
        self.results.clear()
        self._eps_inf_cache = None
        return False