# app/models/base_model.py
"""
Base class for dielectric models providing a standardized interface
for fitting complex permittivity data using lmfit.
"""

import numpy as np
import lmfit
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union, Any, List
import warnings
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FitQualityMetrics:
    """Container for fit quality metrics."""
    rmse: float
    mape: float
    r_squared: float
    chi_squared: float
    reduced_chi_squared: float
    max_error: float
    dk_rmse: float
    df_rmse: float


class BaseModel(lmfit.Model, ABC):
    """
    Enhanced base model for dielectric spectroscopy data fitting.

    This class provides a standardized interface for creating and fitting
    dielectric models to experimental data. It handles complex permittivity
    data and provides comprehensive fit quality metrics.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 frequency_range: Optional[Tuple[float, float]] = None,
                 **kwargs):
        """
        Initialize the base model.

        Args:
            name: Optional model name (defaults to class name)
            frequency_range: Optional (min, max) frequency range in GHz
            **kwargs: Additional arguments passed to lmfit.Model
        """
        # Set model metadata
        self.name = name or self.__class__.__name__
        self.frequency_range = frequency_range or (1e-3, 1e3)

        # Initialize lmfit Model with our model function
        # We pass the static model_func directly
        super().__init__(self.model_func, independent_vars=['freq'],
                         name=name or self.__class__.__name__, **kwargs)

        # Storage for fit quality metrics
        self._last_fit_metrics: Optional[FitQualityMetrics] = None

        # Flag for preprocessing behavior
        self.remove_duplicates: bool = True

        logger.info(f"Initialized {self.name} model")

    @staticmethod
    @abstractmethod
    def model_func(freq: np.ndarray, **params) -> np.ndarray:
        """
        Mathematical function describing the dielectric model.

        This must be implemented by all subclasses as a static method.

        Args:
            freq: Frequency array in GHz
            **params: Model parameters as keyword arguments

        Returns:
            Complex permittivity values
        """
        pass

    @abstractmethod
    def create_parameters(self,
                          freq: np.ndarray,
                          dk_exp: np.ndarray,
                          df_exp: np.ndarray) -> lmfit.Parameters:
        """
        Create lmfit.Parameters with initial guesses and bounds.

        This must be implemented by all subclasses.

        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental real permittivity (Dk)
            df_exp: Experimental loss factor (Df)

        Returns:
            lmfit.Parameters object configured for this model
        """
        pass

    def validate_data(self,
                      freq: np.ndarray,
                      dk_exp: np.ndarray,
                      df_exp: np.ndarray) -> None:
        """
        Validate input data before fitting.

        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental real permittivity
            df_exp: Experimental loss factor

        Raises:
            ValueError: If data is invalid
        """
        # Check array shapes
        if not (freq.shape == dk_exp.shape == df_exp.shape):
            raise ValueError("Frequency, Dk, and Df arrays must have the same shape")

        # Check for NaN or infinite values
        if np.any(~np.isfinite(freq)) or np.any(~np.isfinite(dk_exp)) or np.any(~np.isfinite(df_exp)):
            raise ValueError("Data contains NaN or infinite values")

        # Check frequency range
        if np.any(freq <= 0):
            raise ValueError("Frequencies must be positive")

        # Check Df is non-negative
        if np.any(df_exp < 0):
            raise ValueError("Df (loss factor) must be non-negative")

        # Check frequency ordering
        if not np.all(np.diff(freq) > 0):
            raise ValueError("Frequencies must be monotonically increasing")

        # Warn if outside model's frequency range
        if np.min(freq) < self.frequency_range[0] or np.max(freq) > self.frequency_range[1]:
            warnings.warn(
                f"Data frequency range ({np.min(freq):.3f}-{np.max(freq):.3f} GHz) "
                f"extends outside model's recommended range "
                f"({self.frequency_range[0]:.3f}-{self.frequency_range[1]:.3f} GHz)",
                UserWarning
            )

    def preprocess_data(self,
                        freq: np.ndarray,
                        dk_exp: np.ndarray,
                        df_exp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data before fitting.

        Default implementation removes duplicate frequencies by averaging
        the corresponding dk and df values.

        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental real permittivity
            df_exp: Experimental loss factor

        Returns:
            Tuple of preprocessed (freq, dk_exp, df_exp)
        """
        if self.remove_duplicates:
            # Find unique frequencies and their indices
            unique_freq, inverse_indices = np.unique(freq, return_inverse=True)

            if len(unique_freq) < len(freq):
                logger.warning(f"Found {len(freq) - len(unique_freq)} duplicate frequencies. "
                               "Averaging corresponding values.")

                # Average dk and df values for duplicate frequencies
                unique_dk = np.zeros(len(unique_freq))
                unique_df = np.zeros(len(unique_freq))
                counts = np.zeros(len(unique_freq))

                for i, idx in enumerate(inverse_indices):
                    unique_dk[idx] += dk_exp[i]
                    unique_df[idx] += df_exp[i]
                    counts[idx] += 1

                unique_dk /= counts
                unique_df /= counts

                return unique_freq, unique_dk, unique_df

        return freq, dk_exp, df_exp

    def fit(self,
            freq: np.ndarray,
            dk_exp: np.ndarray,
            df_exp: np.ndarray,
            params: Optional[lmfit.Parameters] = None,
            weights: Optional[Union[str, np.ndarray]] = None,
            method: str = 'leastsq',
            **kwargs) -> lmfit.model.ModelResult:
        """
        Fit the model to experimental complex permittivity data.

        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental real permittivity
            df_exp: Experimental loss factor
            params: Initial parameters (if None, will be created automatically)
            weights: Weighting method or array
            method: Optimization method (default: 'leastsq')
            **kwargs: Additional arguments for the minimizer

        Returns:
            lmfit ModelResult object with fit results
        """
        # Validate input data
        self.validate_data(freq, dk_exp, df_exp)

        # Preprocess data if needed
        freq, dk_exp, df_exp = self.preprocess_data(freq, dk_exp, df_exp)

        # Create parameters if not provided
        if params is None:
            params = self.create_parameters(freq, dk_exp, df_exp)

        # Convert to complex data
        eps_exp = dk_exp + 1j * (dk_exp * df_exp)

        # Calculate weights if string method provided
        if isinstance(weights, str):
            weights = self._calculate_weights(freq, dk_exp, df_exp, method=weights)

        # Use lmfit's built-in complex fitting
        result = super().fit(eps_exp, params, freq=freq, weights=weights,
                             method=method, **kwargs)

        # Store additional data for analysis
        result.freq = freq
        result.dk_exp = dk_exp
        result.df_exp = df_exp

        # Calculate fitted dk and df
        eps_fit = result.best_fit
        result.dk_fit = eps_fit.real
        result.df_fit = np.divide(eps_fit.imag, eps_fit.real,
                                  out=np.zeros_like(eps_fit.real),
                                  where=eps_fit.real != 0)

        # Calculate fit quality metrics
        self._calculate_fit_metrics(result)

        logger.info(f"Fit completed. Success: {result.success}, χ²_red: {result.redchi:.4f}")

        return result

    def _calculate_weights(self,
                           freq: np.ndarray,
                           dk_exp: np.ndarray,
                           df_exp: np.ndarray,
                           method: str = 'uniform') -> np.ndarray:
        """
        Calculate weights for weighted fitting.

        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental real permittivity
            df_exp: Experimental loss factor
            method: Weighting method

        Returns:
            Weight array for complex data
        """
        n_points = len(freq)

        if method == 'uniform':
            weights = np.ones(n_points)
        elif method == 'relative':
            # Weight by inverse of magnitude
            eps_mag = np.sqrt(dk_exp ** 2 + (dk_exp * df_exp) ** 2)
            weights = 1.0 / np.maximum(eps_mag, 0.1)
        elif method == 'frequency':
            # Weight by frequency
            weights = freq / np.max(freq)
        else:
            raise ValueError(f"Unknown weighting method: {method}")

        # Normalize weights
        weights = weights / np.mean(weights)

        return weights

    def _calculate_fit_metrics(self, result: lmfit.model.ModelResult) -> None:
        """
        Calculate comprehensive fit quality metrics.

        Args:
            result: ModelResult from fitting
        """
        # Calculate RMSE for dk and df separately
        dk_rmse = np.sqrt(np.mean((result.dk_fit - result.dk_exp) ** 2))
        df_rmse = np.sqrt(np.mean((result.df_fit - result.df_exp) ** 2))

        # Combined RMSE
        rmse = np.sqrt((dk_rmse ** 2 + df_rmse ** 2) / 2)

        # Mean Absolute Percentage Error
        dk_mape = np.mean(np.abs(result.dk_fit - result.dk_exp) /
                          (np.abs(result.dk_exp) + 1e-10)) * 100
        df_mape = np.mean(np.abs(result.df_fit - result.df_exp) /
                          (np.abs(result.df_exp) + 1e-10)) * 100
        mape = (dk_mape + df_mape) / 2

        # R-squared
        dk_ss_res = np.sum((result.dk_exp - result.dk_fit) ** 2)
        dk_ss_tot = np.sum((result.dk_exp - np.mean(result.dk_exp)) ** 2)
        dk_r2 = 1 - dk_ss_res / (dk_ss_tot + 1e-10) if dk_ss_tot > 1e-10 else 0.0

        df_ss_res = np.sum((result.df_exp - result.df_fit) ** 2)
        df_ss_tot = np.sum((result.df_exp - np.mean(result.df_exp)) ** 2)
        df_r2 = 1 - df_ss_res / (df_ss_tot + 1e-10) if df_ss_tot > 1e-10 else 0.0

        r_squared = (dk_r2 + df_r2) / 2

        # Maximum error
        dk_max_error = np.max(np.abs(result.dk_fit - result.dk_exp))
        df_max_error = np.max(np.abs(result.df_fit - result.df_exp))
        max_error = max(dk_max_error, df_max_error)

        # Store metrics
        self._last_fit_metrics = FitQualityMetrics(
            rmse=rmse,
            mape=mape,
            r_squared=r_squared,
            chi_squared=result.chisqr,
            reduced_chi_squared=result.redchi,
            max_error=max_error,
            dk_rmse=dk_rmse,
            df_rmse=df_rmse
        )

        # Add to result
        result.fit_metrics = self._last_fit_metrics

    def get_fit_report(self, result: lmfit.model.ModelResult) -> str:
        """
        Generate a comprehensive fit report.

        Args:
            result: ModelResult from fitting

        Returns:
            Formatted report string
        """
        if not hasattr(result, 'fit_metrics'):
            return "No fit metrics available. Run fit() first."

        metrics = result.fit_metrics

        report = f"\n{'=' * 60}\n"
        report += f"{self.name} Fit Report\n"
        report += f"{'=' * 60}\n\n"

        # Fit quality
        report += "Fit Quality:\n"
        report += f"  Success: {result.success}\n"
        report += f"  R²: {metrics.r_squared:.4f}\n"
        report += f"  RMSE: {metrics.rmse:.4g}\n"
        report += f"  MAPE: {metrics.mape:.2f}%\n"
        report += f"  χ²: {metrics.chi_squared:.4g}\n"
        report += f"  χ²_reduced: {metrics.reduced_chi_squared:.4g}\n"
        report += f"  Max Error: {metrics.max_error:.4g}\n"
        report += f"  Dk RMSE: {metrics.dk_rmse:.4g}\n"
        report += f"  Df RMSE: {metrics.df_rmse:.4g}\n\n"

        # Parameters
        report += "Fitted Parameters:\n"
        for name, param in result.params.items():
            if param.vary:
                report += f"  {name}: {param.value:.6g}"
                if param.stderr is not None:
                    report += f" ± {param.stderr:.6g}"
                report += "\n"

        # Data info
        report += f"\nData Info:\n"
        report += f"  Frequency range: {result.freq.min():.3f} - {result.freq.max():.3f} GHz\n"
        report += f"  Number of points: {len(result.freq)}\n"
        report += f"  Degrees of freedom: {result.nfree}\n"

        report += f"{'=' * 60}\n"

        return report

    def predict(self,
                freq: np.ndarray,
                params: Union[lmfit.Parameters, Dict[str, float]]) -> np.ndarray:
        """
        Predict complex permittivity at given frequencies.

        Args:
            freq: Frequency array in GHz
            params: Parameters as lmfit.Parameters or dict

        Returns:
            Complex permittivity predictions
        """
        # Convert dict to Parameters if needed
        if isinstance(params, dict):
            param_obj = lmfit.Parameters()
            for name, value in params.items():
                param_obj.add(name, value=value)
            params = param_obj

        # Use lmfit's eval method
        return self.eval(params, freq=freq)

    def to_dict(self, result: lmfit.model.ModelResult) -> Dict[str, Any]:
        """
        Convert fit results to dictionary for storage/comparison.

        Args:
            result: ModelResult from fitting

        Returns:
            Dictionary containing all relevant fit information
        """
        # Extract parameter values and uncertainties
        params_dict = {}
        params_stderr = {}
        for name, param in result.params.items():
            if param.vary:
                params_dict[name] = param.value
                if param.stderr is not None:
                    params_stderr[f"{name}_stderr"] = param.stderr

        # Prepare output dictionary
        output = {
            'model_name': self.name,
            'success': result.success,
            'parameters': params_dict,
            'uncertainties': params_stderr,
            'fit_metrics': {
                'rmse': result.fit_metrics.rmse,
                'mape': result.fit_metrics.mape,
                'r_squared': result.fit_metrics.r_squared,
                'chi_squared': result.fit_metrics.chi_squared,
                'reduced_chi_squared': result.fit_metrics.reduced_chi_squared,
                'dk_rmse': result.fit_metrics.dk_rmse,
                'df_rmse': result.fit_metrics.df_rmse,
            },
            'freq_range': [float(result.freq.min()), float(result.freq.max())],
            'n_points': len(result.freq),
            'n_params': result.nvarys,
            'message': result.message if hasattr(result, 'message') else ''
        }

        return output