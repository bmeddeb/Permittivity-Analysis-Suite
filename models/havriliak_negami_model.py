# models/havriliak_negami_model.py
import numpy as np
from utils.helpers import get_numeric_data
from .base_model import BaseModel

class HavriliakNegamiModel(BaseModel):
    def model(self, params, freq):
        """
        Havriliak-Negami model
        freq: frequency in GHz
        """
        delta_eps, tau, alpha, beta, eps_inf = params

        # Convert frequency to angular frequency (rad/s)
        omega = 2 * np.pi * freq * 1e9  # GHz -> rad/s

        # Havriliak-Negami equation: eps_inf + delta_eps / (1 + (j*omega*tau)^alpha)^beta
        return eps_inf + delta_eps / (1 + (1j * omega * tau) ** alpha) ** beta

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
        weight_df = 15.0  # Higher weight for loss factor

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

        print(f"Havriliak-Negami Model")
        print(f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
        print(f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

        # Better parameter initialization
        dk_range = np.max(dk_exp) - np.min(dk_exp)
        freq_mean_hz = np.mean(freq_ghz) * 1e9

        # Initial parameter estimates
        delta_eps0 = dk_range  # Dielectric strength
        tau0 = 1 / (2 * np.pi * freq_mean_hz)  # Relaxation time (seconds)
        alpha0 = 0.5  # Shape parameter (0 < alpha <= 1)
        beta0 = 0.8  # Asymmetry parameter (0 < beta <= 1)
        eps_inf0 = np.min(dk_exp) * 0.9  # High-frequency permittivity

        p0 = [delta_eps0, tau0, alpha0, beta0, eps_inf0]

        print(f"Initial guess - delta_eps: {delta_eps0:.3f}")
        print(f"Initial guess - tau (s): {tau0:.3e}")
        print(f"Initial guess - alpha: {alpha0:.3f}")
        print(f"Initial guess - beta: {beta0:.3f}")
        print(f"Initial guess - eps_inf: {eps_inf0:.3f}")

        # Physical bounds for Havriliak-Negami parameters
        max_delta_eps = dk_range * 3
        max_eps_inf = np.min(dk_exp) * 0.95  # Leave some margin

        lower_bounds = [
            0.0,                    # delta_eps >= 0
            1e-15,                  # tau >= 1 fs (very fast)
            0.01,                   # alpha > 0 (slightly above 0 for numerical stability)
            0.01,                   # beta > 0 (slightly above 0 for numerical stability)
            1.0                     # eps_inf >= 1
        ]
        upper_bounds = [
            max_delta_eps,          # delta_eps reasonable upper bound
            1e-6,                   # tau <= 1 µs (very slow)
            1.0,                    # alpha <= 1
            1.0,                    # beta <= 1
            max_eps_inf             # eps_inf < min(dk_exp)
        ]

        print(f"Bounds - delta_eps: [0.0, {max_delta_eps:.3f}]")
        print(f"Bounds - tau: [1e-15, 1e-6]")
        print(f"Bounds - alpha: [0.01, 1.0]")
        print(f"Bounds - beta: [0.01, 1.0]")
        print(f"Bounds - eps_inf: [1.0, {max_eps_inf:.3f}]")

        # Use safe least_squares from BaseModel
        param_names = ['delta_eps', 'tau', 'alpha', 'beta', 'eps_inf']
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
        print(f"Fitted - beta: {res.x[3]:.3f}")
        print(f"Fitted - eps_inf: {res.x[4]:.3f}")
        print(f"Optimization success: {res.success}")

        # Calculate characteristic frequency for interpretation
        f_char_ghz = 1 / (2 * np.pi * res.x[1]) / 1e9  # Convert to GHz
        print(f"Characteristic frequency: {f_char_ghz:.3f} GHz")

        # Check parameter validity
        if res.x[2] <= 0 or res.x[2] > 1:
            print(f"Warning: alpha parameter ({res.x[2]:.3f}) outside valid range (0, 1]")
        if res.x[3] <= 0 or res.x[3] > 1:
            print(f"Warning: beta parameter ({res.x[3]:.3f}) outside valid range (0, 1]")

        # Interpret parameters
        if res.x[2] < 0.05 and res.x[3] > 0.95:
            print("Note: α ≈ 0, β ≈ 1, behavior close to Debye model")
        elif res.x[2] > 0.05 and res.x[3] > 0.95:
            print("Note: β ≈ 1, behavior close to Cole-Cole model")
        elif res.x[2] < 0.05 and res.x[3] < 0.95:
            print("Note: α ≈ 0, behavior close to Cole-Davidson model")
        else:
            print("Note: Full Havriliak-Negami behavior with both symmetric and asymmetric distribution")

        # Check if relaxation is within measurement range
        freq_min, freq_max = np.min(freq_ghz), np.max(freq_ghz)
        if f_char_ghz < freq_min:
            print(f"Note: Relaxation frequency ({f_char_ghz:.3f} GHz) is below measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")
        elif f_char_ghz > freq_max:
            print(f"Note: Relaxation frequency ({f_char_ghz:.3f} GHz) is above measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")
        else:
            print(f"Note: Relaxation frequency ({f_char_ghz:.3f} GHz) is within measurement range ({freq_min:.1f}-{freq_max:.1f} GHz)")

        # Handle negative imaginary parts using BaseModel method
        eps_fit_corrected = self.handle_negative_imaginary(eps_fit, "Havriliak-Negami model")

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