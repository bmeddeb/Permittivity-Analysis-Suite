"""
Base model class for dielectric fitting models.

This module provides the abstract base class that all dielectric models
inherit from, providing standardized interfaces for fitting and evaluation.
"""

import numpy as np
import lmfit
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
from datetime import datetime


class BaseModel(ABC):
    """
    An enhanced base model that provides a standardized interface for 
    creating and fitting dielectric models using lmfit.
    
    This class handles the conversion between experimental data (Dk, Df)
    and complex permittivity, as well as providing common fitting utilities.
    """
    
    def __init__(self, name: str, model_type: str):
        """
        Initialize the base model.
        
        Args:
            name: Human-readable name of the model
            model_type: Machine-readable model type identifier
        """
        self.name = name
        self.model_type = model_type
        self._lmfit_model = None
        self._last_fit_result = None
        
    @staticmethod
    @abstractmethod
    def model_func(freq: np.ndarray, **params) -> np.ndarray:
        """
        The mathematical function describing the dielectric model.
        This must be implemented by all subclasses.

        Args:
            freq: Frequency array in GHz.
            **params: Model parameters as keyword arguments.

        Returns:
            Complex permittivity array.
        """
        pass

    @abstractmethod
    def create_parameters(self, freq: np.ndarray, dk_exp: np.ndarray, 
                         df_exp: np.ndarray) -> lmfit.Parameters:
        """
        Create an lmfit.Parameters object with initial guesses and bounds.
        This must be implemented by all subclasses.

        Args:
            freq: Frequency array (GHz).
            dk_exp: Experimental real permittivity.
            df_exp: Experimental imaginary permittivity.

        Returns:
            The parameters for the model.
        """
        pass
    
    def _create_lmfit_model(self):
        """Create the lmfit Model object if not already created."""
        if self._lmfit_model is None:
            self._lmfit_model = lmfit.Model(self._residual_func, 
                                          independent_vars=['freq', 'dk_exp', 'df_exp'])
    
    def _residual_func(self, freq: np.ndarray, dk_exp: np.ndarray, 
                      df_exp: np.ndarray, **params) -> np.ndarray:
        """
        Residual function for fitting complex permittivity data.
        
        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental Dk values
            df_exp: Experimental Df values  
            **params: Model parameters
            
        Returns:
            Concatenated residuals for real and imaginary parts
        """
        # Get model prediction
        eps_complex = self.model_func(freq, **params)
        
        # Calculate residuals for both real and imaginary parts
        dk_residual = eps_complex.real - dk_exp
        df_residual = eps_complex.imag - df_exp
        
        # Return concatenated residuals
        return np.concatenate([dk_residual, df_residual])
    
    def fit(self, freq: np.ndarray, dk_exp: np.ndarray, df_exp: np.ndarray,
            weights: Optional[np.ndarray] = None, **fit_kwargs) -> Dict[str, Any]:
        """
        Fit the model to experimental data.

        Args:
            freq: Frequency array in GHz
            dk_exp: Experimental Dk values
            df_exp: Experimental Df values
            weights: Optional weights for residuals
            **fit_kwargs: Additional arguments passed to lmfit minimize

        Returns:
            Dictionary containing fit results and metadata
        """
        # Create parameters
        params = self.create_parameters(freq, dk_exp, df_exp)
        
        # Create lmfit model
        self._create_lmfit_model()
        
        # Set up experimental data
        exp_data = np.concatenate([dk_exp, df_exp])
        
        # Handle weights
        if weights is not None:
            # Duplicate weights for both real and imaginary parts
            fit_weights = np.concatenate([weights, weights])
        else:
            fit_weights = None
            
        # Perform fit
        try:
            result = self._lmfit_model.fit(exp_data, params, 
                                         freq=freq, dk_exp=dk_exp, df_exp=df_exp,
                                         weights=fit_weights, **fit_kwargs)
            self._last_fit_result = result
            
            # Extract fitted parameters
            fitted_params = {}
            param_uncertainties = {}
            
            for name, param in result.params.items():
                fitted_params[name] = param.value
                if param.stderr is not None:
                    param_uncertainties[name] = param.stderr
                else:
                    param_uncertainties[name] = 0.0
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(result, freq, dk_exp, df_exp)
            
            # Generate fitted curve
            fitted_curve = self._generate_fitted_curve(freq, fitted_params)
            
            # Determine convergence status
            if result.success:
                status = 'success'
            elif result.errorbars:
                status = 'warning'  # Fit completed but with warnings
            else:
                status = 'failed'
            
            return {
                'success': result.success,
                'status': status,
                'message': result.message,
                'iterations': result.nfev,
                'fitted_parameters': fitted_params,
                'parameter_uncertainties': param_uncertainties,
                'quality_metrics': quality_metrics,
                'fitted_curve': fitted_curve,
                'raw_result': result  # Keep for advanced users
            }
            
        except Exception as e:
            return {
                'success': False,
                'status': 'failed',
                'message': f'Fit failed with error: {str(e)}',
                'iterations': 0,
                'fitted_parameters': {},
                'parameter_uncertainties': {},
                'quality_metrics': {},
                'fitted_curve': {'freq': [], 'dk': [], 'df': []},
                'raw_result': None
            }
    
    def _calculate_quality_metrics(self, result: lmfit.model.ModelResult, 
                                 freq: np.ndarray, dk_exp: np.ndarray, 
                                 df_exp: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for the fit."""
        # Basic lmfit metrics
        metrics = {
            'chi_squared': result.chisqr,
            'reduced_chi_squared': result.redchi,
            'aic': result.aic,
            'bic': result.bic
        }
        
        # Calculate R-squared for Dk and Df separately
        fitted_complex = self.model_func(freq, **result.best_values)
        
        # R-squared for Dk
        ss_res_dk = np.sum((dk_exp - fitted_complex.real) ** 2)
        ss_tot_dk = np.sum((dk_exp - np.mean(dk_exp)) ** 2)
        metrics['r_squared_dk'] = 1 - (ss_res_dk / ss_tot_dk) if ss_tot_dk > 0 else 0
        
        # R-squared for Df  
        ss_res_df = np.sum((df_exp - fitted_complex.imag) ** 2)
        ss_tot_df = np.sum((df_exp - np.mean(df_exp)) ** 2)
        metrics['r_squared_df'] = 1 - (ss_res_df / ss_tot_df) if ss_tot_df > 0 else 0
        
        # Overall R-squared
        metrics['r_squared_overall'] = (metrics['r_squared_dk'] + metrics['r_squared_df']) / 2
        
        # RMSE
        n_points = len(freq)
        metrics['rmse_dk'] = np.sqrt(ss_res_dk / n_points)
        metrics['rmse_df'] = np.sqrt(ss_res_df / n_points)
        
        return metrics
    
    def _generate_fitted_curve(self, freq: np.ndarray, 
                             fitted_params: Dict[str, float]) -> Dict[str, list]:
        """Generate fitted curve data for plotting."""
        try:
            fitted_complex = self.model_func(freq, **fitted_params)
            return {
                'freq': freq.tolist(),
                'dk': fitted_complex.real.tolist(),
                'df': fitted_complex.imag.tolist()
            }
        except Exception:
            return {'freq': [], 'dk': [], 'df': []}
    
    def evaluate(self, freq: np.ndarray, **params) -> np.ndarray:
        """
        Evaluate the model at given frequencies with specified parameters.
        
        Args:
            freq: Frequency array in GHz
            **params: Model parameters
            
        Returns:
            Complex permittivity array
        """
        return self.model_func(freq, **params)
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about model parameters.
        Should be overridden by subclasses to provide parameter descriptions.
        
        Returns:
            Dictionary with parameter information
        """
        return {}
    
    @staticmethod
    def dk_df_to_complex(dk: np.ndarray, df: np.ndarray) -> np.ndarray:
        """Convert Dk and Df to complex permittivity."""
        return dk - 1j * df
    
    @staticmethod  
    def complex_to_dk_df(eps_complex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert complex permittivity to Dk and Df."""
        return eps_complex.real, eps_complex.imag
    
    @staticmethod
    def angular_freq_from_ghz(freq_ghz: np.ndarray) -> np.ndarray:
        """Convert frequency in GHz to angular frequency in rad/s."""
        return 2 * np.pi * freq_ghz * 1e9