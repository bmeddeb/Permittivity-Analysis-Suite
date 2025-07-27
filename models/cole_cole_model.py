# models/cole_cole_model.py
import numpy as np
from utils.helpers import get_numeric_data
from .base_model import BaseModel

class ColeColeModel(BaseModel):
    def model(self, params, freq):
        """
        Cole-Cole relaxation model
        freq: frequency in GHz
        """
        delta_eps, tau, alpha, eps_inf = params

        # Convert frequency to angular frequency (rad/s)
        omega = 2 * np.pi * freq * 1e9  # GHz -> rad/s

        # Cole-Cole equation: eps_inf + delta_eps / (1 + (j*omega*tau)^(1-alpha))
        return eps_inf + delta_eps / (1 + (1j * omega * tau) ** (1 - alpha))

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
        weight_df = 12.0  # Higher weight for loss factor

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

        print(f"Cole-Cole Model")
        print(f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
        print(f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

        # Better parameter initialization
        dk_range = np.max(dk_exp) - np.min(dk_exp)
        freq_mean_hz = np.mean(freq_ghz) * 1e9

        # Initial parameter estimates
        delta_eps0 = dk_range  # Dielectric strength
        tau0 = 1 / (2 * np.pi * freq_mean_hz)  # Relaxation time (seconds)
        alpha0 = 0.1  # Distribution parameter (0 <= alpha < 1)
        eps_inf0 = np.min(dk_exp) * 0.9  # High-frequency permittivity

        p0 = [delta_eps0, tau0, alpha0, eps_inf0]

        print(f"Initial guess - delta_eps: {delta_eps0:.3f}")
        print(f"Initial guess - tau (s): {tau0:.3e}")
        print(f"Initial guess - alpha: {alpha0:.3f}")
        print(f"Initial guess - eps_inf: {eps_inf0:.3f}")

        # Physical bounds for Cole-Cole parameters
        max_delta_eps = dk_range * 3
        max_eps_inf = np.min(dk_exp) * 0.95  # Leave some margin

        lower_bounds = [
            0.0,                    # delta_eps >= 0
            1e-15,                  # tau >= 1 fs (very fast)
            0.0,                    # alpha >= 0 (Debye limit)
            1.0                     # eps_inf >= 1
        ]
        upper_bounds = [
            max_delta_eps,          # delta_eps reasonable upper bound
            1e-6,                   # tau <= 1 µs (very slow)
            0.99,                   # alpha < 1 (Cole-Cole constraint)
            max_eps_inf             # eps_inf < min(dk_exp)
        ]

        print(f"Bounds - delta_eps: [0.0, {max_delta_eps:.3f}]")
        print(f"Bounds - tau: [1e-15, 1e-6]")
        print(f"Bounds - alpha: [0.0, 0.99]")
        print(f"Bounds - eps_inf: [1.0, {max_eps_inf:.3f}]")

        # Use safe least_squares from BaseModel
        param_names = ['delta_eps', 'tau', 'alpha', 'eps_inf']
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
        print(f"Fitted - alpha: {res.x[2]:.3f}")
        print(f"Fitted - eps_inf: {res.x[3]:.3f}")
        print(f"Optimization success: {res.success}")

        # Calculate characteristic frequency for interpretation
        f_char_ghz = 1 / (2 * np.pi * res.x[1]) / 1e9  # Convert to GHz
        print(f"Characteristic frequency: {f_char_ghz:.3f} GHz")

        # Check parameter validity
        if res.x[2] < 0 or res.x[2] >= 1:
            print(f"Warning: alpha parameter ({res.x[2]:.3f}) outside valid range [0, 1)")

        # Interpret alpha parameter
        if res.x[2] < 0.05:
            print("Note: alpha ≈ 0, behavior close to Debye model")
        elif res.x[2] > 0.5:
            print("Note: alpha > 0.5, highly distributed relaxation")
        else:
            print("Note: moderate distribution of relaxation times")

        # Physical interpretation of distribution
        if res.x[2] > 0.1:
            distribution_factor = 1 / (1 - res.x[2])
            print(f"Relaxation time distribution factor: {distribution_factor:.1f}x broader than Debye")

        # Check if relaxation is within measurement range
        freq_min, freq_max = np.min(freq_ghz), np.max(freq_ghz)
        if f_char_ghz < freq_min:
            print(f"Note: Relaxation frequency ({f_char_ghz:.3f} GHz) is below measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")
        elif f_char_ghz > freq_max:
            print(f"Note: Relaxation frequency ({f_char_ghz:.3f} GHz) is above measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")
        else:
            print(f"Note: Relaxation frequency ({f_char_ghz:.3f} GHz) is within measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")

        # Handle negative imaginary parts using BaseModel method
        eps_fit_corrected = self.handle_negative_imaginary(eps_fit, "Cole-Cole model")

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