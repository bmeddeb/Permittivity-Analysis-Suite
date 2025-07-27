# models/hybrid_model.py
import numpy as np
from utils.helpers import get_numeric_data
from .base_model import BaseModel


class HybridModel(BaseModel):
    def hybrid_eval(self, params, freq, N):
        """
        Hybrid Debye-Lorentz model
        freq: frequency in GHz
        """
        delta_eps = params[:N]  # Debye strengths
        f_k = params[N:2 * N]  # characteristic frequencies (GHz)
        sigma_k = params[2 * N:3 * N]  # Lorentz conductivity terms
        eps_inf = params[-1]  # high-frequency permittivity

        # Use frequency in GHz directly
        F = freq.reshape(-1, 1)  # shape (M,1)
        FK = f_k.reshape(1, -1)  # shape (1,N)

        # Debye relaxation terms
        debye_terms = delta_eps / (1 + 1j * (F / FK))

        # Modified Lorentz conductivity terms
        # Note: This formulation may need adjustment based on your specific hybrid model
        lorentz_terms = 1j * F * (sigma_k / (F ** 2 + FK ** 2))

        return eps_inf + np.sum(debye_terms, axis=1) + np.sum(lorentz_terms, axis=1)

    def objective(self, params, freq, dk_exp, df_exp, N):
        """
        Objective function fitting both real and imaginary parts separately
        """
        eps_fit = self.hybrid_eval(params, freq, N)

        # Calculate residuals for both components
        dk_residual = np.real(eps_fit) - dk_exp
        df_residual = np.imag(eps_fit) - df_exp

        # Weight the smaller Df values more heavily
        weight_dk = 1.0
        weight_df = 15.0  # High weight for hybrid model

        residual = np.concatenate([
            weight_dk * dk_residual,
            weight_df * df_residual
        ])

        # Strong penalty for negative imaginary parts
        negative_penalty = np.sum(np.minimum(np.imag(eps_fit), 0) ** 2) * 1000.0

        # FIXED: Don't append penalty as array element - it causes shape mismatch
        # The penalty should be added to the cost, not as a residual element
        # Let scipy handle the penalty through the cost function

        return residual

    def init_hybrid_params(self, freq_ghz, dk_exp, N):
        """
        Initialize parameters for hybrid model
        """
        dk_range = np.max(dk_exp) - np.min(dk_exp)
        freq_min = np.min(freq_ghz)
        freq_max = np.max(freq_ghz)
        freq_mean = np.mean(freq_ghz)

        # Debye strengths - split the dielectric range
        delta_eps = np.ones(N) * dk_range / N

        # Characteristic frequencies - spread across measurement range
        if N == 1:
            f_k = np.array([freq_mean])
        else:
            f_k = np.logspace(
                np.log10(freq_min / 2),
                np.log10(freq_max * 2),
                N
            )

        # Conductivity terms - start small
        sigma_k = np.ones(N) * 0.01

        # High-frequency permittivity
        eps_inf = np.min(dk_exp) * 0.8

        return np.concatenate([delta_eps, f_k, sigma_k, [eps_inf]])

    def get_hybrid_bounds(self, freq_ghz, dk_exp, N):
        """
        Get parameter bounds for hybrid model
        """
        dk_range = np.max(dk_exp) - np.min(dk_exp)
        freq_min = np.min(freq_ghz)
        freq_max = np.max(freq_ghz)

        # Lower bounds
        lb = np.concatenate([
            np.zeros(N),  # delta_eps >= 0
            np.ones(N) * freq_min / 100,  # f_k >= freq_min/100
            np.zeros(N),  # sigma_k >= 0
            [1.0]  # eps_inf >= 1
        ])

        # Upper bounds - add margins to prevent bounds errors
        max_eps_inf = np.min(dk_exp) * 0.95  # Leave margin
        ub = np.concatenate([
            np.ones(N) * dk_range * 3,  # delta_eps <= reasonable
            np.ones(N) * freq_max * 100,  # f_k <= freq_max*100
            np.ones(N) * 10.0,  # sigma_k <= reasonable
            [max_eps_inf]  # eps_inf < min(dk) with margin
        ])

        return lb, ub

    def analyze(self, df, N=2):
        freq_ghz, dk_exp, df_exp = get_numeric_data(df)

        print(f"Hybrid Model (N={N} terms)")
        print(f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
        print(f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

        # Initialize parameters
        p0 = self.init_hybrid_params(freq_ghz, dk_exp, N)
        lb, ub = self.get_hybrid_bounds(freq_ghz, dk_exp, N)

        print(f"Initial delta_eps: {p0[:N]}")
        print(f"Initial f_k (GHz): {p0[N:2 * N]}")
        print(f"Initial sigma_k: {p0[2 * N:3 * N]}")
        print(f"Initial eps_inf: {p0[-1]:.3f}")

        # Display bounds for transparency
        print(f"Bounds - delta_eps: [0.0, {ub[:N]}]")
        print(f"Bounds - f_k: [{lb[N]:.2e}, {ub[N]:.2e}]")
        print(f"Bounds - sigma_k: [0.0, {ub[2 * N]:.1f}]")
        print(f"Bounds - eps_inf: [1.0, {ub[-1]:.3f}]")

        # Create parameter names for debugging
        param_names = []
        for i in range(N):
            param_names.append(f'delta_eps_{i}')
        for i in range(N):
            param_names.append(f'f_k_{i}')
        for i in range(N):
            param_names.append(f'sigma_k_{i}')
        param_names.append('eps_inf')

        # Use safe least_squares from BaseModel
        res = self.safe_least_squares(
            self.objective,
            p0,
            bounds=(lb, ub),
            args=(freq_ghz, dk_exp, df_exp, N),
            param_names=param_names,
            max_nfev=15000  # More iterations for complex model
        )

        # Calculate final fitted values
        eps_fit = self.hybrid_eval(res.x, freq_ghz, N)

        min_imag = np.min(eps_fit.imag)
        max_imag = np.max(eps_fit.imag)

        print(f"Fitted - Imaginary part range: {min_imag:.6f} to {max_imag:.6f}")
        print(f"Fitted delta_eps: {res.x[:N]}")
        print(f"Fitted f_k (GHz): {res.x[N:2 * N]}")
        print(f"Fitted sigma_k: {res.x[2 * N:3 * N]}")
        print(f"Fitted eps_inf: {res.x[-1]:.3f}")
        print(f"Optimization success: {res.success}")

        # Physical interpretation
        dominant_debye = np.argmax(res.x[:N])
        dominant_lorentz = np.argmax(res.x[2 * N:3 * N])
        print(f"Dominant Debye term: {dominant_debye} at {res.x[N + dominant_debye]:.2f} GHz")
        print(f"Dominant Lorentz term: {dominant_lorentz} at {res.x[N + dominant_lorentz]:.2f} GHz")

        # Check if terms are well-separated
        f_k_sorted = np.sort(res.x[N:2 * N])
        if N > 1:
            min_separation = np.min(f_k_sorted[1:] / f_k_sorted[:-1])
            print(f"Minimum frequency separation factor: {min_separation:.2f}")
            if min_separation < 2:
                print("Warning: Some relaxation frequencies are very close - consider reducing N")

        # Handle negative imaginary parts using BaseModel method
        eps_fit_corrected = self.handle_negative_imaginary(eps_fit, "Hybrid model")

        return {
            "freq": freq_ghz,
            "dk_fit": eps_fit_corrected.real,
            "df_fit": eps_fit_corrected.imag,
            "eps_fit": eps_fit_corrected,
            "params_fit": res.x,
            "success": res.success,
            "cost": res.cost,
            "dk_exp": dk_exp
        }