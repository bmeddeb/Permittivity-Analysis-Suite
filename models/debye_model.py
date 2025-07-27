# models/debye_model.py
import numpy as np
from .base_model import BaseModel
from utils.helpers import get_numeric_data

class DebyeModel(BaseModel):
    def model(self, params, freq):
        """
        Debye relaxation model
        freq: frequency in GHz
        """
        delta_eps, tau, eps_inf = params

        # Convert frequency to angular frequency (rad/s)
        omega = 2 * np.pi * freq * 1e9  # GHz -> rad/s

        # Debye equation: eps_inf + delta_eps / (1 + j*omega*tau)
        return eps_inf + delta_eps / (1 + 1j * omega * tau)

    def objective(self, params, freq, dk_exp, df_exp):
        """
        Objective function fitting both real and imaginary parts separately
        """
        eps_fit = self.model(params, freq)

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

    def analyze(self, df):
        freq_ghz, dk_exp, df_exp = get_numeric_data(df)

        print(f"Debye Model")
        print(f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
        print(f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

        # Better parameter initialization based on experimental data
        eps_inf_guess = np.min(dk_exp) * 0.9  # High-frequency limit
        delta_eps_guess = np.max(dk_exp) - eps_inf_guess  # Dielectric strength

        # Estimate relaxation time from frequency range
        freq_mean_hz = np.mean(freq_ghz) * 1e9
        tau_guess = 1 / (2 * np.pi * freq_mean_hz)  # Relaxation time in seconds

        p0 = [delta_eps_guess, tau_guess, eps_inf_guess]

        print(f"Initial guess - delta_eps: {delta_eps_guess:.3f}")
        print(f"Initial guess - tau (s): {tau_guess:.3e}")
        print(f"Initial guess - eps_inf: {eps_inf_guess:.3f}")

        # Physical bounds
        max_delta_eps = (np.max(dk_exp) - np.min(dk_exp)) * 2
        max_eps_inf = np.min(dk_exp) * 0.95  # Leave some margin

        lower_bounds = [
            0.0,                    # delta_eps >= 0 (dielectric strength positive)
            1e-15,                  # tau >= 1 fs (very fast relaxation)
            1.0                     # eps_inf >= 1 (physical permittivity)
        ]
        upper_bounds = [
            max_delta_eps,          # delta_eps reasonable upper bound
            1e-6,                   # tau <= 1 µs (very slow relaxation)
            max_eps_inf             # eps_inf < min(dk_exp) for physical ordering
        ]

        print(f"Bounds - delta_eps: [0.0, {max_delta_eps:.3f}]")
        print(f"Bounds - tau: [1e-15, 1e-6]")
        print(f"Bounds - eps_inf: [1.0, {max_eps_inf:.3f}]")

        # Use safe least_squares from BaseModel
        param_names = ['delta_eps', 'tau', 'eps_inf']
        res = self.safe_least_squares(
            self.objective,
            p0,
            bounds=(lower_bounds, upper_bounds),
            args=(freq_ghz, dk_exp, df_exp),
            param_names=param_names
        )

        # Calculate final fitted values
        eps_fit = self.model(res.x, freq_ghz)

        min_imag = np.min(eps_fit.imag)
        max_imag = np.max(eps_fit.imag)

        print(f"Fitted - Imaginary part range: {min_imag:.6f} to {max_imag:.6f}")
        print(f"Fitted - delta_eps: {res.x[0]:.3f}")
        print(f"Fitted - tau (s): {res.x[1]:.3e}")
        print(f"Fitted - eps_inf: {res.x[2]:.3f}")
        print(f"Optimization success: {res.success}")

        # Calculate characteristic frequency for interpretation
        f_char_ghz = 1 / (2 * np.pi * res.x[1]) / 1e9  # Convert to GHz
        print(f"Characteristic frequency: {f_char_ghz:.3f} GHz")

        # Check if relaxation is within measurement range
        freq_min, freq_max = np.min(freq_ghz), np.max(freq_ghz)
        if f_char_ghz < freq_min:
            print(f"Note: Relaxation frequency ({f_char_ghz:.3f} GHz) is below measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")
        elif f_char_ghz > freq_max:
            print(f"Note: Relaxation frequency ({f_char_ghz:.3f} GHz) is above measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")
        else:
            print(f"Note: Relaxation frequency ({f_char_ghz:.3f} GHz) is within measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")

        # Handle negative imaginary parts using BaseModel method
        eps_fit_corrected = self.handle_negative_imaginary(eps_fit, "Debye model")

        return {
            "freq_ghz": freq_ghz,
            "eps_fit": eps_fit_corrected,
            "dk_fit": eps_fit_corrected.real,
            "df_fit": eps_fit_corrected.imag,
            "params_fit": res.x,
            "dk_exp": dk_exp,
            "success": res.success,
            "cost": res.cost
        }