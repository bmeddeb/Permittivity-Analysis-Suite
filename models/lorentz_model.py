# models/lorentz_model.py
import numpy as np
import lmfit
from utils.helpers import get_numeric_data
from .base_model import BaseModel


class LorentzModel(BaseModel):
    def __init__(self, N=2):
        """
        Initialize Lorentz model with N oscillators
        """
        self.N = N

    def create_parameters(self, freq, dk_exp, df_exp):
        """
        Create lmfit.Parameters object for Lorentz model
        """
        params = lmfit.Parameters()

        # Convert frequency range to rad/s for parameter estimation
        freq_rad_s = freq * 1e9 * 2 * np.pi
        freq_min = np.min(freq_rad_s)
        freq_max = np.max(freq_rad_s)
        freq_mean = np.mean(freq_rad_s)

        # Better initial parameter estimates
        dk_range = np.max(dk_exp) - np.min(dk_exp)

        # Oscillator strengths - reasonable starting values
        f_initial = dk_range / self.N

        # Resonance frequencies - spread across and beyond measured frequency range
        if self.N == 1:
            w0_initial = [freq_mean]
        else:
            # Spread resonances logarithmically around the measurement range
            w0_initial = np.logspace(
                np.log10(freq_min / 2),  # Below measurement range
                np.log10(freq_max * 2),  # Above measurement range
                self.N
            )

        # Damping coefficients - start moderate
        # 10% of resonance frequency
        gamma_initial = [w0 * 0.1 for w0 in w0_initial]

        # High-frequency permittivity
        eps_inf_initial = np.min(dk_exp) * 0.8

        # Add oscillator parameters
        for i in range(self.N):
            params.add(f'f_{i}', value=f_initial, min=0.0, max=dk_range * 5)
            params.add(
                f'w0_{i}', value=w0_initial[i], min=freq_min / 100, max=freq_max * 100)
            params.add(
                f'gamma_{i}', value=gamma_initial[i], min=freq_min / 1000, max=freq_max * 10)

        # High-frequency permittivity
        max_eps_inf = np.min(dk_exp) * 0.95
        params.add('eps_inf', value=eps_inf_initial, min=1.0, max=max_eps_inf)

        return params

    def model_function(self, params, freq):
        """
        Lorentz oscillator model function for lmfit
        """
        # Extract parameter values
        f = np.array([params[f'f_{i}'].value for i in range(self.N)])
        w0 = np.array([params[f'w0_{i}'].value for i in range(self.N)])
        gamma = np.array([params[f'gamma_{i}'].value for i in range(self.N)])
        eps_inf = params['eps_inf'].value

        # Convert frequency to angular frequency (rad/s)
        w = 2 * np.pi * freq * 1e9  # GHz -> rad/s
        w = w.reshape(-1, 1)

        # Lorentz oscillator terms: f / (w0^2 - w^2 - j*gamma*w)
        terms = f / (w0 ** 2 - w ** 2 - 1j * gamma * w)
        return eps_inf + np.sum(terms, axis=1)

    def analyze(self, df):
        freq_ghz, dk_exp, df_exp = get_numeric_data(df)

        print(f"Lorentz Model (N={self.N} oscillators, lmfit)")
        print(
            f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
        print(
            f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

        # Create parameters and print initial guesses
        params = self.create_parameters(freq_ghz, dk_exp, df_exp)

        print(
            f"Initial guess - f: {[params[f'f_{i}'].value for i in range(self.N)]}")
        print(
            f"Initial guess - w0 (rad/s): {[params[f'w0_{i}'].value for i in range(self.N)]}")
        print(
            f"Initial guess - gamma (rad/s): {[params[f'gamma_{i}'].value for i in range(self.N)]}")
        print(f"Initial guess - eps_inf: {params['eps_inf'].value:.3f}")

        # Display bounds for transparency
        print(f"Bounds - f: [0.0, {params['f_0'].max:.1f}]")
        print(
            f"Bounds - w0: [{params['w0_0'].min:.2e}, {params['w0_0'].max:.2e}] rad/s")
        print(
            f"Bounds - gamma: [{params['gamma_0'].min:.2e}, {params['gamma_0'].max:.2e}] rad/s")
        print(
            f"Bounds - eps_inf: [{params['eps_inf'].min:.1f}, {params['eps_inf'].max:.3f}]")

        # Fit the model using lmfit
        result = self.fit_model(freq_ghz, dk_exp, df_exp)

        # Calculate final fitted values
        eps_fit = self.model_function(result.params, freq_ghz)

        min_imag = np.min(eps_fit.imag)
        max_imag = np.max(eps_fit.imag)

        print(
            f"Fitted - Imaginary part range: {min_imag:.6f} to {max_imag:.6f}")

        # Print fitted parameters with uncertainties
        f_fitted = [result.params[f'f_{i}'].value for i in range(self.N)]
        w0_fitted = [result.params[f'w0_{i}'].value for i in range(self.N)]
        gamma_fitted = [
            result.params[f'gamma_{i}'].value for i in range(self.N)]
        eps_inf_fitted = result.params['eps_inf'].value

        print(f"Fitted - f: {f_fitted}")
        print(f"Fitted - w0 (rad/s): {w0_fitted}")
        print(
            f"Fitted - w0 (GHz): {[w0 / (2 * np.pi * 1e9) for w0 in w0_fitted]}")
        print(f"Fitted - gamma (rad/s): {gamma_fitted}")
        print(
            f"Fitted - eps_inf: {eps_inf_fitted:.3f} ± {result.params['eps_inf'].stderr or 0:.3f}")
        print(f"Optimization success: {result.success}")
        print(f"AIC: {result.aic:.2f}")
        print(f"BIC: {result.bic:.2f}")

        # Physical interpretation
        resonance_freqs_ghz = [w0 / (2 * np.pi * 1e9) for w0 in w0_fitted]
        quality_factors = [w0 / gamma for w0,
                           gamma in zip(w0_fitted, gamma_fitted)]

        for i in range(self.N):
            print(
                f"Oscillator {i}: f_res = {resonance_freqs_ghz[i]:.2f} GHz, Q = {quality_factors[i]:.1f}")

        # Check if resonances are within measurement range
        freq_min_ghz, freq_max_ghz = np.min(freq_ghz), np.max(freq_ghz)
        for i, f_res in enumerate(resonance_freqs_ghz):
            if f_res < freq_min_ghz:
                print(
                    f"Note: Oscillator {i} resonance ({f_res:.2f} GHz) is below measurement range")
            elif f_res > freq_max_ghz:
                print(
                    f"Note: Oscillator {i} resonance ({f_res:.2f} GHz) is above measurement range")
            else:
                print(
                    f"Note: Oscillator {i} resonance ({f_res:.2f} GHz) is within measurement range")

        # Check for overdamped oscillators
        for i in range(self.N):
            if quality_factors[i] < 0.5:
                print(
                    f"Warning: Oscillator {i} is heavily overdamped (Q = {quality_factors[i]:.2f})")

        # Handle negative imaginary parts using BaseModel method
        eps_fit_corrected = self.handle_negative_imaginary(
            eps_fit, "Lorentz model")

        # Create params_fit array for compatibility
        params_fit = f_fitted + w0_fitted + gamma_fitted + [eps_inf_fitted]

        # Extract fitted parameters in expected format
        fitted_params = self.extract_fitted_params(result, "lorentz")

        return {
            "freq_ghz": freq_ghz,
            "eps_fit": eps_fit_corrected,
            "dk_fit": eps_fit_corrected.real,
            "df_fit": eps_fit_corrected.imag,
            "params_fit": params_fit,
            "fitted_params": fitted_params,  # Add expected fitted_params dictionary
            "dk_exp": dk_exp,
            "success": result.success,
            "cost": result.chisqr,
            "aic": result.aic,
            "bic": result.bic,
            "lmfit_result": result
        }
