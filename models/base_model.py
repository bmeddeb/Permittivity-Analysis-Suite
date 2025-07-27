# base_model.py
import numpy as np


class BaseModel:

    def check_and_adjust_bounds(self, p0, lower_bounds, upper_bounds, param_names=None):
        """
        Check if initial parameters are within bounds and adjust if necessary

        Args:
            p0: Initial parameter guess (list or array)
            lower_bounds: Lower bounds (list or array)
            upper_bounds: Upper bounds (list or array)
            param_names: Optional parameter names for debugging (list)

        Returns:
            Adjusted p0 that is guaranteed to be within bounds
        """
        p0_adjusted = np.array(p0, dtype=float)
        lower_bounds = np.array(lower_bounds, dtype=float)
        upper_bounds = np.array(upper_bounds, dtype=float)

        for i, (val, lb, ub) in enumerate(zip(p0_adjusted, lower_bounds, upper_bounds)):
            param_name = param_names[i] if param_names else f"p[{i}]"

            # Check for invalid bounds
            if lb >= ub:
                raise ValueError(f"Invalid bounds for {param_name}: lower_bound ({lb}) >= upper_bound ({ub})")

            # Adjust if outside bounds
            if val <= lb:
                p0_adjusted[i] = lb + (ub - lb) * 0.05  # 5% above lower bound
                print(
                    f"Warning: {param_name} initial guess {val:.3e} <= lower bound {lb:.3e}. Adjusted to {p0_adjusted[i]:.3e}")
            elif val >= ub:
                p0_adjusted[i] = ub - (ub - lb) * 0.05  # 5% below upper bound
                print(
                    f"Warning: {param_name} initial guess {val:.3e} >= upper bound {ub:.3e}. Adjusted to {p0_adjusted[i]:.3e}")

        return p0_adjusted.tolist()

    def safe_least_squares(self, objective_func, p0, bounds, args=(), param_names=None, **kwargs):
        """
        Safe wrapper around scipy's least_squares with automatic bounds checking

        Args:
            objective_func: The objective function to minimize
            p0: Initial parameter guess
            bounds: Tuple of (lower_bounds, upper_bounds)
            args: Additional arguments to pass to objective_func
            param_names: Optional parameter names for debugging
            **kwargs: Additional keyword arguments for least_squares

        Returns:
            scipy.optimize.OptimizeResult
        """
        from scipy.optimize import least_squares

        lower_bounds, upper_bounds = bounds

        # Adjust initial guess to be within bounds
        p0_safe = self.check_and_adjust_bounds(p0, lower_bounds, upper_bounds, param_names)

        # Set default parameters if not provided
        default_kwargs = {
            'method': 'trf',
            'max_nfev': 10000,
            'ftol': 1e-12,
            'xtol': 1e-12
        }

        # Update with user-provided kwargs
        for key, value in default_kwargs.items():
            kwargs.setdefault(key, value)

        print(f"Calling least_squares with adjusted p0: {p0_safe}")

        return least_squares(
            objective_func,
            p0_safe,
            args=args,
            bounds=bounds,
            **kwargs
        )

    def calculate_rmse_components(self, eps_fit, dk_exp, df_exp):
        """
        Calculate RMSE for both real and imaginary components

        Args:
            eps_fit: Complex fitted permittivity
            dk_exp: Experimental real permittivity
            df_exp: Experimental imaginary permittivity

        Returns:
            Dictionary with rmse, rmse_dk, rmse_df
        """
        dk_fit = np.real(eps_fit)
        df_fit = np.imag(eps_fit)

        rmse_dk = np.sqrt(np.mean((dk_fit - dk_exp) ** 2))
        rmse_df = np.sqrt(np.mean((df_fit - df_exp) ** 2))
        rmse_total = np.sqrt(rmse_dk ** 2 + rmse_df ** 2)

        return {
            'rmse': rmse_total,
            'rmse_dk': rmse_dk,
            'rmse_df': rmse_df
        }

    def handle_negative_imaginary(self, eps_fit, model_name="Model"):
        """
        Handle negative imaginary parts by clamping to small positive values

        Args:
            eps_fit: Complex fitted permittivity
            model_name: Name of the model for warning messages

        Returns:
            Corrected complex permittivity with non-negative imaginary part
        """
        min_imag = np.min(eps_fit.imag)

        if min_imag < 0:
            print(
                f"Warning: {model_name} produced negative Df values (min: {min_imag:.6f}). Clamping to small positive values.")
            eps_fit_corrected = eps_fit.real + 1j * np.maximum(eps_fit.imag, 1e-6)
            return eps_fit_corrected
        else:
            return eps_fit