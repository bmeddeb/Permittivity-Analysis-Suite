# models/sarkar_model.py
import numpy as np
import lmfit
from utils.helpers import get_numeric_data
from .base_model import BaseModel

class SarkarModel(BaseModel):
    
    def create_parameters(self, freq, dk_exp, df_exp):
        """
        Create lmfit.Parameters object for Sarkar model
        """
        params = lmfit.Parameters()
        
        # Better initial guess based on data
        eps_inf_guess = np.min(dk_exp) * 0.9  # Slightly below minimum Dk
        eps_s_guess = np.max(dk_exp) * 1.1  # Slightly above maximum Dk
        
        # Estimate characteristic frequency from data
        f_p_guess = np.mean(freq)
        
        # Physical bounds - ensure eps_s > eps_inf and add margins
        min_eps_s = np.max(dk_exp) * 0.8
        max_eps_s = np.max(dk_exp) * 2.0
        max_eps_inf = np.min(dk_exp) * 0.95  # Leave margin and ensure eps_inf < eps_s
        
        # Ensure initial guess satisfies eps_s > eps_inf
        if eps_s_guess <= max_eps_inf:
            eps_s_guess = max_eps_inf * 1.1
        
        # Add parameters with bounds
        params.add('eps_s', value=eps_s_guess, min=min_eps_s, max=max_eps_s)
        params.add('eps_inf', value=eps_inf_guess, min=1.0, max=max_eps_inf)
        params.add('f_p', value=f_p_guess, min=freq[0] * 0.1, max=freq[-1] * 10.0)
        
        return params
    
    def model_function(self, params, freq):
        """
        Sarkar (modified Debye) model function for lmfit
        """
        # Extract parameter values
        eps_s = params['eps_s'].value
        eps_inf = params['eps_inf'].value
        f_p = params['f_p'].value

        # Sarkar equation: eps_inf + (eps_s - eps_inf) / (1 + j*(freq/f_p))
        complex_perm = eps_inf + (eps_s - eps_inf) / (1 + 1j * (freq / f_p))
        return complex_perm

    def analyze(self, df):
        freq_ghz, dk_exp, df_exp = get_numeric_data(df)

        print(f"Sarkar Model (lmfit)")
        print(f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
        print(f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

        # Create parameters and print initial guesses
        params = self.create_parameters(freq_ghz, dk_exp, df_exp)
        
        print(f"Initial guess - eps_s: {params['eps_s'].value:.3f}")
        print(f"Initial guess - eps_inf: {params['eps_inf'].value:.3f}")
        print(f"Initial guess - f_p (GHz): {params['f_p'].value:.3f}")
        
        print(f"Bounds - eps_s: [{params['eps_s'].min:.3f}, {params['eps_s'].max:.3f}]")
        print(f"Bounds - eps_inf: [{params['eps_inf'].min:.3f}, {params['eps_inf'].max:.3f}]")
        print(f"Bounds - f_p: [{params['f_p'].min:.2f}, {params['f_p'].max:.2f}] GHz")

        # Fit the model using lmfit
        result = self.fit_model(freq_ghz, dk_exp, df_exp)

        # Calculate final fitted values
        eps_fit = self.model_function(result.params, freq_ghz)

        min_imag = np.min(eps_fit.imag)
        max_imag = np.max(eps_fit.imag)

        print(f"Fitted - Imaginary part range: {min_imag:.6f} to {max_imag:.6f}")
        print(f"Fitted - eps_s: {result.params['eps_s'].value:.3f} ± {result.params['eps_s'].stderr or 0:.3f}")
        print(f"Fitted - eps_inf: {result.params['eps_inf'].value:.3f} ± {result.params['eps_inf'].stderr or 0:.3f}")
        print(f"Fitted - f_p (GHz): {result.params['f_p'].value:.3f} ± {result.params['f_p'].stderr or 0:.3f}")
        print(f"Optimization success: {result.success}")
        print(f"AIC: {result.aic:.2f}")
        print(f"BIC: {result.bic:.2f}")

        # Physical interpretation
        eps_s_fit = result.params['eps_s'].value
        eps_inf_fit = result.params['eps_inf'].value
        f_p_fit = result.params['f_p'].value
        
        dielectric_strength = eps_s_fit - eps_inf_fit
        print(f"Dielectric strength (eps_s - eps_inf): {dielectric_strength:.3f}")

        # Calculate characteristic frequency for comparison
        f_char_ghz = f_p_fit  # In Sarkar model, f_p is the characteristic frequency
        print(f"Characteristic frequency: {f_char_ghz:.3f} GHz")

        # Check if relaxation is within measurement range
        freq_min, freq_max = np.min(freq_ghz), np.max(freq_ghz)
        if f_char_ghz < freq_min:
            print(f"Note: Characteristic frequency ({f_char_ghz:.3f} GHz) is below measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")
        elif f_char_ghz > freq_max:
            print(f"Note: Characteristic frequency ({f_char_ghz:.3f} GHz) is above measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")
        else:
            print(f"Note: Characteristic frequency ({f_char_ghz:.3f} GHz) is within measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")

        # Check parameter validity
        if eps_s_fit <= eps_inf_fit:
            print(f"Warning: eps_s ({eps_s_fit:.3f}) <= eps_inf ({eps_inf_fit:.3f}) - unphysical!")
        else:
            print(f"Physical check: eps_s ({eps_s_fit:.3f}) > eps_inf ({eps_inf_fit:.3f}) ✓")

        # Compare to Debye model
        tau_equivalent = 1/(2*np.pi*f_p_fit*1e9)
        print(f"Note: Sarkar model is equivalent to Debye with tau = 1/(2π*f_p) = {tau_equivalent:.3e} seconds")

        # Handle negative imaginary parts using BaseModel method
        eps_fit_corrected = self.handle_negative_imaginary(eps_fit, "Sarkar model")

        return {
            "freq_ghz": freq_ghz,
            "eps_fit": eps_fit_corrected,
            "dk_fit": eps_fit_corrected.real,
            "df_fit": eps_fit_corrected.imag,
            "params_fit": [eps_s_fit, eps_inf_fit, f_p_fit],
            "dk_exp": dk_exp,
            "success": result.success,
            "cost": result.chisqr,
            "aic": result.aic,
            "bic": result.bic,
            "lmfit_result": result
        }