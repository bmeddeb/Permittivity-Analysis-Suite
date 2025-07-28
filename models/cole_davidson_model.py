# models/cole_davidson_model.py
import numpy as np
import lmfit
from utils.helpers import get_numeric_data
from .base_model import BaseModel

class ColeDavidsonModel(BaseModel):
    
    def create_parameters(self, freq, dk_exp, df_exp):
        """
        Create lmfit.Parameters object for Cole-Davidson model
        """
        params = lmfit.Parameters()
        
        # Better parameter initialization
        dk_range = np.max(dk_exp) - np.min(dk_exp)
        freq_mean_hz = np.mean(freq) * 1e9
        
        # Initial parameter estimates
        delta_eps_guess = dk_range  # Dielectric strength
        tau_guess = 1 / (2 * np.pi * freq_mean_hz)  # Relaxation time (seconds)
        beta_guess = 0.7  # Asymmetry parameter (0 < beta <= 1)
        eps_inf_guess = np.min(dk_exp) * 0.9  # High-frequency permittivity
        
        # Physical bounds for Cole-Davidson parameters
        max_delta_eps = dk_range * 3
        max_eps_inf = np.min(dk_exp) * 0.95  # Leave some margin
        
        # Add parameters with bounds
        params.add('delta_eps', value=delta_eps_guess, min=0.0, max=max_delta_eps)
        params.add('tau', value=tau_guess, min=1e-15, max=1e-6)
        params.add('beta', value=beta_guess, min=0.01, max=1.0)  # Slightly above 0 for stability
        params.add('eps_inf', value=eps_inf_guess, min=1.0, max=max_eps_inf)
        
        return params
    
    def model_function(self, params, freq):
        """
        Cole-Davidson relaxation model function for lmfit
        """
        # Extract parameter values
        delta_eps = params['delta_eps'].value
        tau = params['tau'].value
        beta = params['beta'].value
        eps_inf = params['eps_inf'].value

        # Convert frequency to angular frequency (rad/s)
        omega = 2 * np.pi * freq * 1e9  # GHz -> rad/s

        # Cole-Davidson equation: eps_inf + delta_eps / (1 + j*omega*tau)^beta
        return eps_inf + delta_eps / (1 + 1j * omega * tau) ** beta

    def analyze(self, df):
        freq_ghz, dk_exp, df_exp = get_numeric_data(df)

        print(f"Cole-Davidson Model (lmfit)")
        print(f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
        print(f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

        # Create parameters and print initial guesses
        params = self.create_parameters(freq_ghz, dk_exp, df_exp)
        
        print(f"Initial guess - delta_eps: {params['delta_eps'].value:.3f}")
        print(f"Initial guess - tau (s): {params['tau'].value:.3e}")
        print(f"Initial guess - beta: {params['beta'].value:.3f}")
        print(f"Initial guess - eps_inf: {params['eps_inf'].value:.3f}")
        
        print(f"Bounds - delta_eps: [{params['delta_eps'].min:.3f}, {params['delta_eps'].max:.3f}]")
        print(f"Bounds - tau: [{params['tau'].min:.3e}, {params['tau'].max:.3e}]")
        print(f"Bounds - beta: [{params['beta'].min:.3f}, {params['beta'].max:.3f}]")
        print(f"Bounds - eps_inf: [{params['eps_inf'].min:.3f}, {params['eps_inf'].max:.3f}]")

        # Fit the model using lmfit
        result = self.fit_model(freq_ghz, dk_exp, df_exp)

        # Calculate final fitted values
        eps_fit = self.model_function(result.params, freq_ghz)

        min_imag = np.min(eps_fit.imag)
        max_imag = np.max(eps_fit.imag)

        print(f"Fitted - Imaginary part range: {min_imag:.6f} to {max_imag:.6f}")
        print(f"Fitted - delta_eps: {result.params['delta_eps'].value:.3f} ± {result.params['delta_eps'].stderr or 0:.3f}")
        print(f"Fitted - tau (s): {result.params['tau'].value:.3e} ± {result.params['tau'].stderr or 0:.3e}")
        print(f"Fitted - beta: {result.params['beta'].value:.3f} ± {result.params['beta'].stderr or 0:.3f}")
        print(f"Fitted - eps_inf: {result.params['eps_inf'].value:.3f} ± {result.params['eps_inf'].stderr or 0:.3f}")
        print(f"Optimization success: {result.success}")
        print(f"AIC: {result.aic:.2f}")
        print(f"BIC: {result.bic:.2f}")

        # Calculate characteristic frequency for interpretation
        tau_fit = result.params['tau'].value
        beta_fit = result.params['beta'].value
        f_char_ghz = 1 / (2 * np.pi * tau_fit) / 1e9  # Convert to GHz
        print(f"Characteristic frequency: {f_char_ghz:.3f} GHz")

        # Check parameter validity
        if beta_fit <= 0 or beta_fit > 1:
            print(f"Warning: beta parameter ({beta_fit:.3f}) outside valid range (0, 1]")

        # Interpret beta parameter
        if beta_fit > 0.9:
            print("Note: beta ≈ 1, behavior close to Debye model")
        elif beta_fit < 0.3:
            print("Note: beta < 0.3, highly asymmetric relaxation")

        # Check if relaxation is within measurement range
        freq_min, freq_max = np.min(freq_ghz), np.max(freq_ghz)
        if f_char_ghz < freq_min:
            print(f"Note: Relaxation frequency ({f_char_ghz:.3f} GHz) is below measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")
        elif f_char_ghz > freq_max:
            print(f"Note: Relaxation frequency ({f_char_ghz:.3f} GHz) is above measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")
        else:
            print(f"Note: Relaxation frequency ({f_char_ghz:.3f} GHz) is within measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")

        # Handle negative imaginary parts using BaseModel method
        eps_fit_corrected = self.handle_negative_imaginary(eps_fit, "Cole-Davidson model")

        return {
            "freq_ghz": freq_ghz,
            "eps_fit": eps_fit_corrected,
            "dk_fit": eps_fit_corrected.real,
            "df_fit": eps_fit_corrected.imag,
            "params_fit": [result.params['delta_eps'].value, result.params['tau'].value, result.params['beta'].value, result.params['eps_inf'].value],
            "dk_exp": dk_exp,
            "success": result.success,
            "cost": result.chisqr,
            "aic": result.aic,
            "bic": result.bic,
            "lmfit_result": result
        }