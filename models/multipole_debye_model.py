# models/multipole_debye_model.py
import numpy as np
from utils.helpers import get_numeric_data
from .base_model import BaseModel


class MultiPoleDebyeModel(BaseModel):
    def model(self, params, freq, N):
        """
        Multi-pole Debye model
        freq: frequency in GHz
        """
        delta_eps = params[:N]
        tau = params[N:2 * N]  # relaxation times in seconds
        eps_inf = params[-1]

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
        negative_penalty = np.sum(np.minimum(np.imag(eps_fit), 0) ** 2) * 1000.0
        if negative_penalty > 0:
            residual = np.append(residual, np.sqrt(negative_penalty))

        return residual

    def analyze(self, df, N=3):
        freq_ghz, dk_exp, df_exp = get_numeric_data(df)

        print(f"Multipole Debye (N={N})")
        print(f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
        print(f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

        # Better initial parameter estimates
        dk_range = np.max(dk_exp) - np.min(dk_exp)
        delta_eps0 = np.ones(N) * dk_range / N  # Split the dielectric strength

        # Estimate relaxation times based on frequency range
        freq_min_hz = np.min(freq_ghz) * 1e9
        freq_max_hz = np.max(freq_ghz) * 1e9

        # Spread relaxation times logarithmically across frequency range
        tau_min = 1.0 / (2 * np.pi * freq_max_hz * 10)  # 10x faster than max freq
        tau_max = 1.0 / (2 * np.pi * freq_min_hz / 10)  # 10x slower than min freq
        tau0 = np.logspace(np.log10(tau_min), np.log10(tau_max), N)

        eps_inf0 = np.min(dk_exp) * 0.9  # Slightly below minimum

        p0 = np.concatenate([delta_eps0, tau0, [eps_inf0]])

        print(f"Initial guess - delta_eps: {delta_eps0}")
        print(f"Initial guess - tau (s): {tau0}")
        print(f"Initial guess - eps_inf: {eps_inf0:.3f}")

        # Physical bounds - add margins to prevent bounds errors
        max_delta_eps = dk_range * 2
        max_eps_inf = np.min(dk_exp) * 0.95  # Leave margin

        lower_bounds = np.concatenate([
            np.zeros(N),  # delta_eps >= 0
            np.ones(N) * 1e-15,  # tau >= 1 fs (very fast)
            [1.0]  # eps_inf >= 1
        ])
        upper_bounds = np.concatenate([
            np.ones(N) * max_delta_eps,  # delta_eps reasonable upper bound
            np.ones(N) * 1e-6,  # tau <= 1 µs (very slow)
            [max_eps_inf]  # eps_inf < min(dk) with margin
        ])

        # Display bounds for transparency
        print(f"Bounds - delta_eps: [0.0, {max_delta_eps:.3f}]")
        print(f"Bounds - tau: [1e-15, 1e-6]")
        print(f"Bounds - eps_inf: [1.0, {max_eps_inf:.3f}]")

        # Create parameter names for debugging
        param_names = []
        for i in range(N):
            param_names.append(f'delta_eps_{i}')
        for i in range(N):
            param_names.append(f'tau_{i}')
        param_names.append('eps_inf')

        # Use safe least_squares from BaseModel
        res = self.safe_least_squares(
            self.objective,
            p0,
            bounds=(lower_bounds, upper_bounds),
            args=(freq_ghz, dk_exp, df_exp, N),
            param_names=param_names
        )

        # Calculate final fitted values
        eps_fit = self.model(res.x, freq_ghz, N)

        min_imag = np.min(eps_fit.imag)
        max_imag = np.max(eps_fit.imag)

        print(f"Fitted - Imaginary part range: {min_imag:.6f} to {max_imag:.6f}")
        print(f"Fitted - delta_eps: {res.x[:N]}")
        print(f"Fitted - tau (s): {res.x[N:2 * N]}")
        print(f"Fitted - eps_inf: {res.x[-1]:.3f}")
        print(f"Optimization success: {res.success}")

        # Physical interpretation
        characteristic_freqs_ghz = 1 / (2 * np.pi * res.x[N:2 * N]) / 1e9

        print("Relaxation processes:")
        for i in range(N):
            print(f"  Process {i}: Δε = {res.x[i]:.3f}, f_char = {characteristic_freqs_ghz[i]:.2f} GHz")

        # Check if relaxations are within measurement range
        freq_min_ghz, freq_max_ghz = np.min(freq_ghz), np.max(freq_ghz)
        processes_in_range = 0

        for i, f_char in enumerate(characteristic_freqs_ghz):
            if f_char < freq_min_ghz:
                print(f"  Process {i} relaxation ({f_char:.2f} GHz) is below measurement range")
            elif f_char > freq_max_ghz:
                print(f"  Process {i} relaxation ({f_char:.2f} GHz) is above measurement range")
            else:
                print(f"  Process {i} relaxation ({f_char:.2f} GHz) is within measurement range")
                processes_in_range += 1

        print(f"Processes within measurement range: {processes_in_range}/{N}")

        # Check for dominant processes
        dominant_process = np.argmax(res.x[:N])
        print(f"Dominant process: {dominant_process} (Δε = {res.x[dominant_process]:.3f})")

        # Check time constant separation
        tau_sorted = np.sort(res.x[N:2 * N])
        if N > 1:
            min_separation = np.min(tau_sorted[1:] / tau_sorted[:-1])
            print(f"Minimum time constant separation factor: {min_separation:.2f}")
            if min_separation < 3:
                print("Warning: Some relaxation times are very close - consider reducing N")

        # Handle negative imaginary parts using BaseModel method
        eps_fit_corrected = self.handle_negative_imaginary(eps_fit, "Multipole Debye model")

        return {
            "freq": freq_ghz,
            "eps_fit": eps_fit_corrected,
            "dk_fit": eps_fit_corrected.real,
            "df_fit": eps_fit_corrected.imag,
            "params_fit": res.x,
            "dk_exp": dk_exp,
            "success": res.success,
            "cost": res.cost
        }