# models/hybrid_model.py
import numpy as np
import lmfit
from utils.helpers import get_numeric_data
from .base_model import BaseModel


class HybridModel(BaseModel):
    def __init__(self, N=2):
        """
        Initialize Hybrid Debye-Lorentz model with N terms
        """
        self.N = N
    
    def create_parameters(self, freq, dk_exp, df_exp):
        """
        Create lmfit.Parameters object for Hybrid model
        """
        params = lmfit.Parameters()
        
        dk_range = np.max(dk_exp) - np.min(dk_exp)
        freq_min = np.min(freq)
        freq_max = np.max(freq)
        freq_mean = np.mean(freq)
        
        # Debye strengths - split the dielectric range
        delta_eps_initial = dk_range / self.N
        
        # Characteristic frequencies - spread across measurement range
        if self.N == 1:
            f_k_initial = [freq_mean]
        else:
            f_k_initial = np.logspace(
                np.log10(freq_min / 2),
                np.log10(freq_max * 2),
                self.N
            )
        
        # Conductivity terms - start small
        sigma_k_initial = 0.01
        
        # High-frequency permittivity
        eps_inf_initial = np.min(dk_exp) * 0.8
        
        # Parameter bounds
        max_eps_inf = np.min(dk_exp) * 0.95  # Leave margin
        
        # Add parameters
        for i in range(self.N):
            params.add(f'delta_eps_{i}', value=delta_eps_initial, min=0.0, max=dk_range * 3)
            params.add(f'f_k_{i}', value=f_k_initial[i], min=freq_min / 100, max=freq_max * 100)
            params.add(f'sigma_k_{i}', value=sigma_k_initial, min=0.0, max=10.0)
        
        params.add('eps_inf', value=eps_inf_initial, min=1.0, max=max_eps_inf)
        
        return params
    
    def model_function(self, params, freq):
        """
        Hybrid Debye-Lorentz model function for lmfit
        """
        # Extract parameter values
        delta_eps = np.array([params[f'delta_eps_{i}'].value for i in range(self.N)])
        f_k = np.array([params[f'f_k_{i}'].value for i in range(self.N)])
        sigma_k = np.array([params[f'sigma_k_{i}'].value for i in range(self.N)])
        eps_inf = params['eps_inf'].value

        # Use frequency in GHz directly
        F = freq.reshape(-1, 1)  # shape (M,1)
        FK = f_k.reshape(1, -1)  # shape (1,N)

        # Debye relaxation terms
        debye_terms = delta_eps / (1 + 1j * (F / FK))

        # Modified Lorentz conductivity terms
        # Note: This formulation may need adjustment based on your specific hybrid model
        lorentz_terms = 1j * F * (sigma_k / (F ** 2 + FK ** 2))

        return eps_inf + np.sum(debye_terms, axis=1) + np.sum(lorentz_terms, axis=1)


    def analyze(self, df):
        freq_ghz, dk_exp, df_exp = get_numeric_data(df)

        print(f"Hybrid Model (N={self.N} terms, lmfit)")
        print(f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
        print(f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

        # Create parameters and print initial guesses
        params = self.create_parameters(freq_ghz, dk_exp, df_exp)
        
        delta_eps_initial = [params[f'delta_eps_{i}'].value for i in range(self.N)]
        f_k_initial = [params[f'f_k_{i}'].value for i in range(self.N)]
        sigma_k_initial = [params[f'sigma_k_{i}'].value for i in range(self.N)]
        
        print(f"Initial delta_eps: {delta_eps_initial}")
        print(f"Initial f_k (GHz): {f_k_initial}")
        print(f"Initial sigma_k: {sigma_k_initial}")
        print(f"Initial eps_inf: {params['eps_inf'].value:.3f}")

        # Display bounds for transparency
        print(f"Bounds - delta_eps: [0.0, {params['delta_eps_0'].max:.1f}]")
        print(f"Bounds - f_k: [{params['f_k_0'].min:.2e}, {params['f_k_0'].max:.2e}] GHz")
        print(f"Bounds - sigma_k: [0.0, {params['sigma_k_0'].max:.1f}]")
        print(f"Bounds - eps_inf: [1.0, {params['eps_inf'].max:.3f}]")

        # Fit the model using lmfit
        result = self.fit_model(freq_ghz, dk_exp, df_exp)

        # Calculate final fitted values
        eps_fit = self.model_function(result.params, freq_ghz)

        min_imag = np.min(eps_fit.imag)
        max_imag = np.max(eps_fit.imag)

        print(f"Fitted - Imaginary part range: {min_imag:.6f} to {max_imag:.6f}")
        
        # Extract fitted parameters
        delta_eps_fitted = [result.params[f'delta_eps_{i}'].value for i in range(self.N)]
        f_k_fitted = [result.params[f'f_k_{i}'].value for i in range(self.N)]
        sigma_k_fitted = [result.params[f'sigma_k_{i}'].value for i in range(self.N)]
        eps_inf_fitted = result.params['eps_inf'].value
        
        print(f"Fitted delta_eps: {delta_eps_fitted}")
        print(f"Fitted f_k (GHz): {f_k_fitted}")
        print(f"Fitted sigma_k: {sigma_k_fitted}")
        print(f"Fitted eps_inf: {eps_inf_fitted:.3f} ± {result.params['eps_inf'].stderr or 0:.3f}")
        print(f"Optimization success: {result.success}")
        print(f"AIC: {result.aic:.2f}")
        print(f"BIC: {result.bic:.2f}")

        # Physical interpretation
        dominant_debye = np.argmax(delta_eps_fitted)
        dominant_lorentz = np.argmax(sigma_k_fitted)
        print(f"Dominant Debye term: {dominant_debye} at {f_k_fitted[dominant_debye]:.2f} GHz")
        print(f"Dominant Lorentz term: {dominant_lorentz} at {f_k_fitted[dominant_lorentz]:.2f} GHz")

        # Check if terms are well-separated
        f_k_sorted = np.sort(f_k_fitted)
        if self.N > 1:
            min_separation = np.min([f_k_sorted[i+1] / f_k_sorted[i] for i in range(self.N-1)])
            print(f"Minimum frequency separation factor: {min_separation:.2f}")
            if min_separation < 2:
                print("Warning: Some relaxation frequencies are very close - consider reducing N")

        # Handle negative imaginary parts using BaseModel method
        eps_fit_corrected = self.handle_negative_imaginary(eps_fit, "Hybrid model")

        # Create params_fit array for compatibility
        params_fit = delta_eps_fitted + f_k_fitted + sigma_k_fitted + [eps_inf_fitted]

        return {
            "freq_ghz": freq_ghz,
            "dk_fit": eps_fit_corrected.real,
            "df_fit": eps_fit_corrected.imag,
            "eps_fit": eps_fit_corrected,
            "params_fit": params_fit,
            "dk_exp": dk_exp,
            "success": result.success,
            "cost": result.chisqr,
            "aic": result.aic,
            "bic": result.bic,
            "lmfit_result": result
        }