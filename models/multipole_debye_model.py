# models/multipole_debye_model.py
import numpy as np
import lmfit
from utils.helpers import get_numeric_data
from .base_model import BaseModel


class MultiPoleDebyeModel(BaseModel):
    def __init__(self, N=3):
        """
        Initialize Multi-pole Debye model with N terms
        """
        self.N = N

    def create_parameters(self, freq, dk_exp, df_exp):
        """
        Create lmfit.Parameters object for Multi-pole Debye model
        """
        params = lmfit.Parameters()

        # Better initial parameter estimates
        dk_range = np.max(dk_exp) - np.min(dk_exp)
        delta_eps_initial = dk_range / self.N  # Split the dielectric strength

        # Estimate relaxation times based on frequency range
        freq_min_hz = np.min(freq) * 1e9
        freq_max_hz = np.max(freq) * 1e9

        # Spread relaxation times logarithmically across frequency range
        # 10x faster than max freq
        tau_min = 1.0 / (2 * np.pi * freq_max_hz * 10)
        # 10x slower than min freq
        tau_max = 1.0 / (2 * np.pi * freq_min_hz / 10)
        tau_initial = np.logspace(np.log10(tau_min), np.log10(tau_max), self.N)

        eps_inf_initial = np.min(dk_exp) * 0.9  # Slightly below minimum

        # Physical bounds
        max_delta_eps = dk_range * 2
        max_eps_inf = np.min(dk_exp) * 0.95  # Leave margin

        # Add parameters
        for i in range(self.N):
            params.add(
                f'delta_eps_{i}', value=delta_eps_initial, min=0.0, max=max_delta_eps)
            params.add(f'tau_{i}', value=tau_initial[i], min=1e-15, max=1e-6)

        params.add('eps_inf', value=eps_inf_initial, min=1.0, max=max_eps_inf)

        return params

    def model_function(self, params, freq):
        """
        Multi-pole Debye model function for lmfit
        """
        # Extract parameter values
        delta_eps = np.array(
            [params[f'delta_eps_{i}'].value for i in range(self.N)])
        tau = np.array([params[f'tau_{i}'].value for i in range(self.N)])
        eps_inf = params['eps_inf'].value

        # Convert frequency to angular frequency (rad/s)
        omega = 2 * np.pi * freq * 1e9  # freq in GHz -> rad/s

        # Reshape for broadcasting
        omega_reshaped = omega.reshape(-1, 1)

        # Calculate Debye terms: delta_eps / (1 + j*omega*tau)
        terms = delta_eps / (1 + 1j * omega_reshaped * tau)

        return eps_inf + np.sum(terms, axis=1)

    def objective(self, params, freq, dk_exp, df_exp, N):
        """
        Objective function fitting both real and imaginary parts separately
        """
        eps_fit = self.model(params, freq, N)

        # Calculate residuals for both components
        dk_residual = np.real(eps_fit) - dk_exp
        df_residual = np.imag(eps_fit) - df_exp

        # Weight the smaller Df values more heavily
        weight_dk = 1.0
        weight_df = 10.0  # Higher weight for loss factor

        residual = np.concatenate([
            weight_dk * dk_residual,
            weight_df * df_residual
        ])

        # Strong penalty for negative imaginary parts
        negative_penalty = np.sum(np.minimum(
            np.imag(eps_fit), 0) ** 2) * 1000.0
        if negative_penalty > 0:
            residual = np.append(residual, np.sqrt(negative_penalty))

        return residual

    def analyze(self, df):
        freq_ghz, dk_exp, df_exp = get_numeric_data(df)

        print(f"Multipole Debye Model (N={self.N}, lmfit)")
        print(
            f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
        print(
            f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

        # Create parameters and print initial guesses
        params = self.create_parameters(freq_ghz, dk_exp, df_exp)

        delta_eps_initial = [
            params[f'delta_eps_{i}'].value for i in range(self.N)]
        tau_initial = [params[f'tau_{i}'].value for i in range(self.N)]

        print(f"Initial guess - delta_eps: {delta_eps_initial}")
        print(f"Initial guess - tau (s): {tau_initial}")
        print(f"Initial guess - eps_inf: {params['eps_inf'].value:.3f}")

        print(f"Bounds - delta_eps: [0.0, {params['delta_eps_0'].max:.3f}]")
        print(f"Bounds - tau: [1e-15, 1e-6]")
        print(f"Bounds - eps_inf: [1.0, {params['eps_inf'].max:.3f}]")

        # Fit the model using lmfit
        result = self.fit_model(freq_ghz, dk_exp, df_exp)

        # Calculate final fitted values
        eps_fit = self.model_function(result.params, freq_ghz)

        min_imag = np.min(eps_fit.imag)
        max_imag = np.max(eps_fit.imag)

        print(
            f"Fitted - Imaginary part range: {min_imag:.6f} to {max_imag:.6f}")

        # Extract fitted parameters
        delta_eps_fitted = [
            result.params[f'delta_eps_{i}'].value for i in range(self.N)]
        tau_fitted = [result.params[f'tau_{i}'].value for i in range(self.N)]
        eps_inf_fitted = result.params['eps_inf'].value

        print(f"Fitted - delta_eps: {delta_eps_fitted}")
        print(f"Fitted - tau (s): {tau_fitted}")
        print(
            f"Fitted - eps_inf: {eps_inf_fitted:.3f} ± {result.params['eps_inf'].stderr or 0:.3f}")
        print(f"Optimization success: {result.success}")
        print(f"AIC: {result.aic:.2f}")
        print(f"BIC: {result.bic:.2f}")

        # Physical interpretation
        characteristic_freqs_ghz = [
            1 / (2 * np.pi * tau) / 1e9 for tau in tau_fitted]

        print("Relaxation processes:")
        for i in range(self.N):
            print(
                f"  Process {i}: Δε = {delta_eps_fitted[i]:.3f}, f_char = {characteristic_freqs_ghz[i]:.2f} GHz")

        # Check if relaxations are within measurement range
        freq_min_ghz, freq_max_ghz = np.min(freq_ghz), np.max(freq_ghz)
        processes_in_range = 0

        for i, f_char in enumerate(characteristic_freqs_ghz):
            if f_char < freq_min_ghz:
                print(
                    f"  Process {i} relaxation ({f_char:.2f} GHz) is below measurement range")
            elif f_char > freq_max_ghz:
                print(
                    f"  Process {i} relaxation ({f_char:.2f} GHz) is above measurement range")
            else:
                print(
                    f"  Process {i} relaxation ({f_char:.2f} GHz) is within measurement range")
                processes_in_range += 1

        print(
            f"Processes within measurement range: {processes_in_range}/{self.N}")

        # Check for dominant processes
        dominant_process = np.argmax(delta_eps_fitted)
        print(
            f"Dominant process: {dominant_process} (Δε = {delta_eps_fitted[dominant_process]:.3f})")

        # Check time constant separation
        tau_sorted = np.sort(tau_fitted)
        if self.N > 1:
            min_separation = np.min(
                [tau_sorted[i+1] / tau_sorted[i] for i in range(self.N-1)])
            print(
                f"Minimum time constant separation factor: {min_separation:.2f}")
            if min_separation < 3:
                print(
                    "Warning: Some relaxation times are very close - consider reducing N")

        # Handle negative imaginary parts using BaseModel method
        eps_fit_corrected = self.handle_negative_imaginary(
            eps_fit, "Multipole Debye model")

        # Create params_fit array for compatibility
        params_fit = delta_eps_fitted + tau_fitted + [eps_inf_fitted]

        # Extract fitted parameters in expected format
        fitted_params = self.extract_fitted_params(result, "multipole_debye")

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
