# models/sarkar_model.py
import numpy as np
from utils.helpers import get_numeric_data
from .base_model import BaseModel

class SarkarModel(BaseModel):
    def model(self, params, freq):
        """
        Sarkar (modified Debye) model
        freq: frequency in GHz
        """
        eps_s, eps_inf, f_p = params

        # Note: Physical constraint eps_s > eps_inf is enforced by bounds
        # No need to modify parameters during evaluation

        # Sarkar equation: eps_inf + (eps_s - eps_inf) / (1 + j*(freq/f_p))
        complex_perm = eps_inf + (eps_s - eps_inf) / (1 + 1j * (freq / f_p))
        return complex_perm

    def objective(self, params, freq, dk_exp, df_exp):
        """
        Objective function fitting both real and imaginary parts separately
        """
        eps_fit = self.model(params, freq)

        # Calculate residuals for both real and imaginary parts
        dk_residual = np.real(eps_fit) - dk_exp
        df_residual = np.imag(eps_fit) - df_exp

        # Weight the smaller Df values more heavily
        weight_dk = 1.0
        weight_df = 10.0  # Higher weight for Df fitting

        residual = np.concatenate([
            weight_dk * dk_residual,
            weight_df * df_residual
        ])

        # FIXED: Removed penalty append - it was causing shape mismatch errors
        # The bounds constraints and handle_negative_imaginary() will handle any issues
        return residual

    def analyze(self, df):
        freq_ghz, dk_exp, df_exp = get_numeric_data(df)

        print(f"Sarkar Model")
        print(f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
        print(f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

        # Better initial guess based on data
        eps_inf_guess = np.min(dk_exp) * 0.9  # Slightly below minimum Dk
        eps_s_guess = np.max(dk_exp) * 1.1  # Slightly above maximum Dk

        # Estimate characteristic frequency from data
        f_p_guess = np.mean(freq_ghz)

        p0 = [eps_s_guess, eps_inf_guess, f_p_guess]

        print(f"Initial guess - eps_s: {eps_s_guess:.3f}")
        print(f"Initial guess - eps_inf: {eps_inf_guess:.3f}")
        print(f"Initial guess - f_p (GHz): {f_p_guess:.3f}")

        # Physical bounds - ensure eps_s > eps_inf and add margins
        min_eps_s = np.max(dk_exp) * 0.8
        max_eps_s = np.max(dk_exp) * 2.0
        max_eps_inf = np.min(dk_exp) * 0.95  # Leave margin and ensure eps_inf < eps_s

        # Ensure initial guess satisfies eps_s > eps_inf
        if eps_s_guess <= max_eps_inf:
            eps_s_guess = max_eps_inf * 1.1
            p0[0] = eps_s_guess
            print(f"Adjusted eps_s guess to {eps_s_guess:.3f} to ensure eps_s > eps_inf")

        lower_bounds = [
            min_eps_s,                  # eps_s must be > max(dk_exp) * 0.8
            1.0,                        # eps_inf > 1 (physical)
            freq_ghz[0] * 0.1          # f_p lower bound
        ]
        upper_bounds = [
            max_eps_s,                  # eps_s upper bound
            max_eps_inf,                # eps_inf < min(dk_exp) with margin
            freq_ghz[-1] * 10.0        # f_p upper bound
        ]

        # Display bounds for transparency
        print(f"Bounds - eps_s: [{min_eps_s:.3f}, {max_eps_s:.3f}]")
        print(f"Bounds - eps_inf: [1.0, {max_eps_inf:.3f}]")
        print(f"Bounds - f_p: [{lower_bounds[2]:.2f}, {upper_bounds[2]:.2f}] GHz")

        # Use safe least_squares from BaseModel
        param_names = ['eps_s', 'eps_inf', 'f_p']
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
        print(f"Fitted - eps_s: {res.x[0]:.3f}")
        print(f"Fitted - eps_inf: {res.x[1]:.3f}")
        print(f"Fitted - f_p (GHz): {res.x[2]:.3f}")
        print(f"Optimization success: {res.success}")

        # Physical interpretation
        dielectric_strength = res.x[0] - res.x[1]
        print(f"Dielectric strength (eps_s - eps_inf): {dielectric_strength:.3f}")

        # Calculate characteristic frequency for comparison
        f_char_ghz = res.x[2]  # In Sarkar model, f_p is the characteristic frequency
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
        if res.x[0] <= res.x[1]:
            print(f"Warning: eps_s ({res.x[0]:.3f}) <= eps_inf ({res.x[1]:.3f}) - unphysical!")
        else:
            print(f"Physical check: eps_s ({res.x[0]:.3f}) > eps_inf ({res.x[1]:.3f}) ✓")

        # Compare to Debye model
        print(f"Note: Sarkar model is equivalent to Debye with tau = 1/(2π*f_p) = {1/(2*np.pi*res.x[2]*1e9):.3e} seconds")

        # Handle negative imaginary parts using BaseModel method
        eps_fit_corrected = self.handle_negative_imaginary(eps_fit, "Sarkar model")

        return {
            "freq": freq_ghz,
            "eps_fit": eps_fit_corrected,
            "dk_fit": eps_fit_corrected.real,
            "df_fit": eps_fit_corrected.imag,
            "params_fit": res.x,
            "success": res.success,
            "cost": res.cost
        }