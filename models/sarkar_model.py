# models/sarkar_model.py
import numpy as np
import lmfit
from scipy.optimize import differential_evolution
from utils.helpers import get_numeric_data
from .base_model import BaseModel


class SarkarModel(BaseModel):

    def create_parameters(self, freq, dk_exp, df_exp):
        """
        Refactored lmfit.Parameters for Sarkar model with bound-adjustment and expression constraints
        """
        params = lmfit.Parameters()

        # ----------------------------------------------------------
        # 1. Initial guesses
        # ----------------------------------------------------------
        # Real permittivity bounds
        min_eps = np.min(dk_exp)
        max_eps = np.max(dk_exp)

        # Guess eps_s just above max, eps_inf just below min
        eps_s_guess = max_eps * 1.05
        eps_inf_guess = min_eps * 0.95

        # Guess characteristic frequency at Df peak
        f_p_guess = freq[np.argmax(df_exp)]

        # ----------------------------------------------------------
        # 2. Define numeric bounds
        # ----------------------------------------------------------
        lb = [max_eps * 0.5,  # eps_s >= 0.5*max_eps
              1.0,  # eps_inf >= 1
              freq[0] * 0.1]  # f_p >= 0.1*min_freq
        ub = [max_eps * 3.0,  # eps_s <= 3*max_eps
              min_eps * 1.0,  # eps_inf <= min_eps
              freq[-1] * 10.0]  # f_p <= 10*max_freq
        p0 = [eps_s_guess, eps_inf_guess, f_p_guess]

        # ----------------------------------------------------------
        # 3. Adjust guesses to fit within bounds
        # ----------------------------------------------------------
        # Uses BaseModel.check_and_adjust_bounds
        p0_adj = self.check_and_adjust_bounds(
            p0, lb, ub, param_names=['eps_s', 'eps_inf', 'f_p']
        )
        eps_s0, eps_inf0, f_p0 = p0_adj

        # ----------------------------------------------------------
        # 4. Add parameters with expression constraint for eps_inf
        # ----------------------------------------------------------
        # delta_eps ensures eps_s > eps_inf by at least 1e-3
        params.add('eps_s', value=eps_s0, min=lb[0], max=ub[0])
        params.add('delta_eps', value=eps_s0 - eps_inf0,
                   min=1e-3, max=(ub[0] - lb[1]))
        # eps_inf expressed as eps_s - delta_eps
        params.add('eps_inf', expr='eps_s - delta_eps')

        # Characteristic frequency
        params.add('f_p', value=f_p0, min=lb[2], max=ub[2])

        return params

    def model_function(self, params, freq):
        """
        Sarkar (modified Debye) model function for lmfit
        """
        # Extract parameter values
        eps_s = params['eps_s'].value
        eps_inf = params['eps_inf'].value
        f_p = params['f_p'].value

        # Sarkar equation: eps_inf + (eps_s - eps_inf) / (1 + j*(freq/f_p))
        complex_perm = eps_inf + (eps_s - eps_inf) / (1 + 1j * (freq / f_p))
        return complex_perm

    def analyze(self, df, use_explorer=True):
        """
        Analyze data with the Sarkar model, using robust fitting strategies.

        Args:
            df (pd.DataFrame): Input data with frequency, Dk, and Df.
            use_explorer (bool): If True, runs multi-start and global optimization.
                                 If False, runs a single standard fit.
        """
        freq_ghz, dk_exp, df_exp = get_numeric_data(df)

        print(f"Sarkar Model (lmfit)")
        print(
            f"Experimental Dk range: {np.min(dk_exp):.3f} to {np.max(dk_exp):.3f}")
        print(
            f"Experimental Df range: {np.min(df_exp):.6f} to {np.max(df_exp):.6f}")

        # --- Integration of Sarkar Explorer Logic ---

        # 1. Standard fit (initial guess)
        print("\nRunning initial fit...")
        # Use the base fit_model method to get the initial result
        initial_params = self.create_parameters(freq_ghz, dk_exp, df_exp)
        initial_result = lmfit.minimize(self.residual_function, initial_params, args=(freq_ghz, dk_exp, df_exp),
                                        method='leastsq')
        best_result = initial_result
        best_method = 'initial'
        print(f"  Initial fit Chi-squared: {initial_result.chisqr:.4f}")

        if use_explorer:
            # 2. Multi-start local optimization
            print("\nRunning multi-start local fits...")
            n_starts = 20
            best_multistart_result = None
            best_multistart_chisqr = np.inf

            for i in range(n_starts):
                p_rand = initial_params.copy()
                # Create random starting values within bounds
                if 'eps_s' in p_rand:
                    p_rand['eps_s'].value = np.random.uniform(p_rand['eps_s'].min,
                                                              p_rand['eps_s'].max)
                if 'delta_eps' in p_rand:
                    p_rand['delta_eps'].value = np.random.uniform(p_rand['delta_eps'].min,
                                                                  p_rand['delta_eps'].max)
                if 'f_p' in p_rand:
                    p_rand['f_p'].value = np.random.uniform(
                        p_rand['f_p'].min, p_rand['f_p'].max)

                res = lmfit.minimize(self.residual_function, p_rand, args=(
                    freq_ghz, dk_exp, df_exp), method='leastsq')

                if res.chisqr < best_multistart_chisqr:
                    best_multistart_chisqr = res.chisqr
                    best_multistart_result = res

            print(
                f"  Best multi-start Chi-squared: {best_multistart_chisqr:.4f}")
            if best_multistart_chisqr < best_result.chisqr:
                best_result = best_multistart_result
                best_method = 'multi-start'

            # 3. Global optimization (Differential Evolution) followed by local refinement
            print("\nRunning global optimization (Differential Evolution)...")

            def objective_func(x, varnames, params_template):
                params_de = params_template.copy()
                for name, val in zip(varnames, x):
                    params_de[name].value = val
                residuals = self.residual_function(
                    params_de, freq_ghz, dk_exp, df_exp)
                return np.sum(residuals ** 2)

            bounds = []
            varnames = []
            for name, par in initial_params.items():
                if not par.expr:
                    bounds.append((par.min, par.max))
                    varnames.append(name)

            de_result = differential_evolution(objective_func, bounds, args=(varnames, initial_params), maxiter=200,
                                               tol=0.01, seed=1)

            p_final = initial_params.copy()
            for name, val in zip(varnames, de_result.x):
                p_final[name].value = val

            global_result = lmfit.minimize(self.residual_function, p_final, args=(freq_ghz, dk_exp, df_exp),
                                           method='leastsq')

            print(f"  Global fit (DE) Chi-squared: {global_result.chisqr:.4f}")
            if global_result.chisqr < best_result.chisqr:
                best_result = global_result
                best_method = 'global'

        print(
            f"\n🏆 Best fit found using '{best_method}' method (Chi-squared: {best_result.chisqr:.4f})\n")

        # --- End of Explorer Integration ---

        # Use the 'best_result' for all subsequent calculations
        result = best_result

        # Calculate final fitted values
        eps_fit = self.model_function(result.params, freq_ghz)

        # Handle negative imaginary parts
        eps_fit_corrected = self.handle_negative_imaginary(
            eps_fit, "Sarkar model")

        # Physical interpretation
        eps_s_fit = result.params['eps_s'].value
        eps_inf_fit = result.params['eps_inf'].value
        f_p_fit = result.params['f_p'].value

        print(
            f"Fitted - eps_s: {result.params['eps_s'].value:.3f} ± {result.params['eps_s'].stderr or 0:.3f}")
        print(
            f"Fitted - eps_inf: {result.params['eps_inf'].value:.3f} ± {result.params['eps_inf'].stderr or 0:.3f}")
        print(
            f"Fitted - f_p (GHz): {result.params['f_p'].value:.3f} ± {result.params['f_p'].stderr or 0:.3f}")
        print(f"Optimization success: {result.success}")
        print(f"AIC: {result.aic:.2f}")
        print(f"BIC: {result.bic:.2f}")

        dielectric_strength = eps_s_fit - eps_inf_fit
        print(
            f"Dielectric strength (eps_s - eps_inf): {dielectric_strength:.3f}")

        if eps_s_fit <= eps_inf_fit:
            print(
                f"Warning: eps_s ({eps_s_fit:.3f}) <= eps_inf ({eps_inf_fit:.3f}) - unphysical!")
        else:
            print(
                f"Physical check: eps_s ({eps_s_fit:.3f}) > eps_inf ({eps_inf_fit:.3f}) ✓")

        # Extract fitted parameters in expected format
        fitted_params = self.extract_fitted_params(result, "sarkar")

        return {
            "freq_ghz": freq_ghz,
            "eps_fit": eps_fit_corrected,
            "dk_fit": eps_fit_corrected.real,
            "df_fit": eps_fit_corrected.imag,
            "params_fit": [eps_s_fit, eps_inf_fit, f_p_fit],
            "fitted_params": fitted_params,  # Add expected fitted_params dictionary
            "dk_exp": dk_exp,
            "success": result.success,
            "cost": result.chisqr,
            "aic": result.aic,
            "bic": result.bic,
            "lmfit_result": result,
            "best_fit_method": best_method
        }
