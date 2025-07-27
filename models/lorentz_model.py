# models/lorentz_model.py
import numpy as np
from utils.helpers import get_numeric_data
from .base_model import BaseModel


class LorentzModel(BaseModel):
    def model(self, params, freq, N):
        """
        Lorentz oscillator model
        freq: frequency in GHz
        """
        f = params[:N]  # oscillator strengths
        w0 = params[N:2 * N]  # resonance frequencies (rad/s)
        gamma = params[2 * N:3 * N]  # damping coefficients (rad/s)
        eps_inf = params[-1]  # high-frequency permittivity

        # Convert frequency to angular frequency (rad/s)
        w = 2 * np.pi * freq * 1e9  # GHz -> rad/s
        w = w.reshape(-1, 1)

        # Lorentz oscillator terms: f / (w0^2 - w^2 - j*gamma*w)
        terms = f / (w0 ** 2 - w ** 2 - 1j * gamma * w)
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
        weight_df = 20.0  # Even higher weight for Lorentz since it can be tricky

        residual = np.concatenate([
            weight_dk * dk_residual,
            weight_df * df_residual
        ])

        # Strong penalty for negative imaginary parts
        negative_penalty = np.sum(np.minimum(np.imag(eps_fit), 0) ** 2) * 1000.0
        if negative_penalty > 0:
            residual = np.append(residual, np.sqrt(negative_penalty))

        return residual

    def analyze(self, df, N=2):
        freq_ghz, dk_exp, df_exp = get_numeric_data(df)

        print(f"Lorentz Model (N={N} oscillators)")
        print(f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
        print(f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

        # Convert frequency range to rad/s for parameter estimation
        freq_rad_s = freq_ghz * 1e9 * 2 * np.pi
        freq_min = np.min(freq_rad_s)
        freq_max = np.max(freq_rad_s)
        freq_mean = np.mean(freq_rad_s)

        # Better initial parameter estimates
        dk_range = np.max(dk_exp) - np.min(dk_exp)

        # Oscillator strengths - reasonable starting values
        f0 = np.ones(N) * dk_range / N

        # Resonance frequencies - spread across and beyond measured frequency range
        if N == 1:
            w0_0 = np.array([freq_mean])
        else:
            # Spread resonances logarithmically around the measurement range
            w0_0 = np.logspace(
                np.log10(freq_min / 2),  # Below measurement range
                np.log10(freq_max * 2),  # Above measurement range
                N
            )

        # Damping coefficients - start moderate
        gamma0 = w0_0 * 0.1  # 10% of resonance frequency

        # High-frequency permittivity
        eps_inf0 = np.min(dk_exp) * 0.8

        p0 = np.concatenate([f0, w0_0, gamma0, [eps_inf0]])

        print(f"Initial guess - f: {f0}")
        print(f"Initial guess - w0 (rad/s): {w0_0}")
        print(f"Initial guess - gamma (rad/s): {gamma0}")
        print(f"Initial guess - eps_inf: {eps_inf0:.3f}")

        # Physical bounds - add margins to prevent bounds errors
        max_eps_inf = np.min(dk_exp) * 0.95  # Leave margin

        lower_bounds = np.concatenate([
            np.zeros(N),  # f >= 0
            np.ones(N) * freq_min / 100,  # w0 >= freq_min/100
            np.ones(N) * freq_min / 1000,  # gamma >= freq_min/1000
            [1.0]  # eps_inf >= 1
        ])
        upper_bounds = np.concatenate([
            np.ones(N) * dk_range * 5,  # f <= reasonable upper bound
            np.ones(N) * freq_max * 100,  # w0 <= freq_max*100
            np.ones(N) * freq_max * 10,  # gamma <= freq_max*10
            [max_eps_inf]  # eps_inf < min(dk) with margin
        ])

        # Display bounds for transparency
        print(f"Bounds - f: [0.0, {upper_bounds[:N]}]")
        print(f"Bounds - w0: [{lower_bounds[N]:.2e}, {upper_bounds[N]:.2e}]")
        print(f"Bounds - gamma: [{lower_bounds[2 * N]:.2e}, {upper_bounds[2 * N]:.2e}]")
        print(f"Bounds - eps_inf: [1.0, {max_eps_inf:.3f}]")

        # Create parameter names for debugging
        param_names = []
        for i in range(N):
            param_names.append(f'f_{i}')
        for i in range(N):
            param_names.append(f'w0_{i}')
        for i in range(N):
            param_names.append(f'gamma_{i}')
        param_names.append('eps_inf')

        # Use safe least_squares from BaseModel
        res = self.safe_least_squares(
            self.objective,
            p0,
            bounds=(lower_bounds, upper_bounds),
            args=(freq_ghz, dk_exp, df_exp, N),
            param_names=param_names,
            max_nfev=15000  # More iterations for Lorentz
        )

        # Calculate final fitted values
        eps_fit = self.model(res.x, freq_ghz, N)

        min_imag = np.min(eps_fit.imag)
        max_imag = np.max(eps_fit.imag)

        print(f"Fitted - Imaginary part range: {min_imag:.6f} to {max_imag:.6f}")
        print(f"Fitted - f: {res.x[:N]}")
        print(f"Fitted - w0 (rad/s): {res.x[N:2 * N]}")
        print(f"Fitted - w0 (GHz): {res.x[N:2 * N] / (2 * np.pi * 1e9)}")
        print(f"Fitted - gamma (rad/s): {res.x[2 * N:3 * N]}")
        print(f"Fitted - eps_inf: {res.x[-1]:.3f}")
        print(f"Optimization success: {res.success}")

        # Physical interpretation
        resonance_freqs_ghz = res.x[N:2 * N] / (2 * np.pi * 1e9)
        quality_factors = res.x[N:2 * N] / res.x[2 * N:3 * N]

        for i in range(N):
            print(f"Oscillator {i}: f_res = {resonance_freqs_ghz[i]:.2f} GHz, Q = {quality_factors[i]:.1f}")

        # Check if resonances are within measurement range
        freq_min_ghz, freq_max_ghz = np.min(freq_ghz), np.max(freq_ghz)
        for i, f_res in enumerate(resonance_freqs_ghz):
            if f_res < freq_min_ghz:
                print(f"Note: Oscillator {i} resonance ({f_res:.2f} GHz) is below measurement range")
            elif f_res > freq_max_ghz:
                print(f"Note: Oscillator {i} resonance ({f_res:.2f} GHz) is above measurement range")
            else:
                print(f"Note: Oscillator {i} resonance ({f_res:.2f} GHz) is within measurement range")

        # Check for overdamped oscillators
        for i in range(N):
            if quality_factors[i] < 0.5:
                print(f"Warning: Oscillator {i} is heavily overdamped (Q = {quality_factors[i]:.2f})")

        # Handle negative imaginary parts using BaseModel method
        eps_fit_corrected = self.handle_negative_imaginary(eps_fit, "Lorentz model")

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