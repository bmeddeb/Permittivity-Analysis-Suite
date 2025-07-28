# base_model.py
import numpy as np
import lmfit
from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def create_parameters(self, freq, dk_exp, df_exp):
        """
        Create lmfit.Parameters object with parameter definitions

        Args:
            freq: Frequency array (GHz)
            dk_exp: Experimental real permittivity
            df_exp: Experimental imaginary permittivity

        Returns:
            lmfit.Parameters object
        """
        pass

    @abstractmethod
    def model_function(self, params, freq):
        """
        Model function that returns complex permittivity

        Args:
            params: lmfit.Parameters object
            freq: Frequency array (GHz)

        Returns:
            Complex permittivity array
        """
        pass

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
                raise ValueError(
                    f"Invalid bounds for {param_name}: lower_bound ({lb}) >= upper_bound ({ub})")

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

    def residual_function(self, params, freq, dk_exp, df_exp):
        """
        Residual function for lmfit optimization

        Args:
            params: lmfit.Parameters object
            freq: Frequency array (GHz)
            dk_exp: Experimental real permittivity
            df_exp: Experimental imaginary permittivity

        Returns:
            Weighted residual array
        """
        eps_fit = self.model_function(params, freq)

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

        return residual

    def fit_model(self, freq, dk_exp, df_exp):
        """
        Fit the model using lmfit

        Args:
            freq: Frequency array (GHz)
            dk_exp: Experimental real permittivity
            df_exp: Experimental imaginary permittivity

        Returns:
            lmfit.MinimizerResult object
        """
        # Create parameters
        params = self.create_parameters(freq, dk_exp, df_exp)

        # Use lmfit's minimize function directly for residual fitting
        result = lmfit.minimize(
            self.residual_function,
            params,
            args=(freq, dk_exp, df_exp),
            method='leastsq',
            max_nfev=10000
        )

        return result

    @staticmethod
    def compare_models(results_dict):
        """
        Compare multiple model results using AIC and BIC criteria

        Args:
            results_dict: Dictionary with model names as keys and result dicts as values
                         Each result dict must contain 'aic', 'bic', and 'lmfit_result' keys

        Returns:
            Dictionary with comparison summary
        """
        if not results_dict:
            return {"error": "No results provided for comparison"}

        # Extract AIC and BIC values
        comparison_data = []
        for model_name, result in results_dict.items():
            if 'aic' in result and 'bic' in result:
                comparison_data.append({
                    'model': model_name,
                    'aic': result['aic'],
                    'bic': result['bic'],
                    'success': result.get('success', False),
                    'chisqr': result.get('cost', float('inf'))
                })

        if not comparison_data:
            return {"error": "No valid AIC/BIC data found in results"}

        # Sort by AIC (lower is better)
        sorted_by_aic = sorted(comparison_data, key=lambda x: x['aic'])
        sorted_by_bic = sorted(comparison_data, key=lambda x: x['bic'])

        # Calculate relative probabilities (AIC weights)
        min_aic = sorted_by_aic[0]['aic']
        for item in comparison_data:
            delta_aic = item['aic'] - min_aic
            item['aic_weight'] = np.exp(-0.5 * delta_aic)

        # Normalize weights
        total_weight = sum(item['aic_weight'] for item in comparison_data)
        for item in comparison_data:
            item['aic_weight'] /= total_weight

        best_aic_model = sorted_by_aic[0]['model']
        best_bic_model = sorted_by_bic[0]['model']

        summary = {
            'best_aic_model': best_aic_model,
            'best_bic_model': best_bic_model,
            'aic_ranking': [item['model'] for item in sorted_by_aic],
            'bic_ranking': [item['model'] for item in sorted_by_bic],
            'model_details': comparison_data,
            'recommendation': best_aic_model if best_aic_model == best_bic_model else f"AIC favors {best_aic_model}, BIC favors {best_bic_model}"
        }

        return summary

    @staticmethod
    def print_model_comparison(comparison_result):
        """
        Print a formatted model comparison table
        """
        if 'error' in comparison_result:
            print(f"Error: {comparison_result['error']}")
            return

        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)

        # Create table header
        print(
            f"{'Model':<20} {'AIC':<12} {'BIC':<12} {'AIC Weight':<12} {'Success':<8}")
        print("-"*70)

        # Sort by AIC for display
        details = sorted(
            comparison_result['model_details'], key=lambda x: x['aic'])

        for model in details:
            success_str = "✓" if model['success'] else "✗"
            print(f"{model['model']:<20} {model['aic']:<12.2f} {model['bic']:<12.2f} "
                  f"{model['aic_weight']:<12.3f} {success_str:<8}")

        print("-"*70)
        print(f"Best AIC Model: {comparison_result['best_aic_model']}")
        print(f"Best BIC Model: {comparison_result['best_bic_model']}")
        print(f"Recommendation: {comparison_result['recommendation']}")
        print("="*70)

    @staticmethod
    def get_parameter_count(model_name, n_terms=None):
        """
        Get the number of parameters for each model type
        """
        param_counts = {
            'debye': 3,                    # delta_eps, tau, eps_inf
            'cole_cole': 4,                # delta_eps, tau, alpha, eps_inf
            'cole_davidson': 4,            # delta_eps, tau, beta, eps_inf
            'havriliak_negami': 5,         # delta_eps, tau, alpha, beta, eps_inf
            'sarkar': 3,                   # eps_s, eps_inf, f_p
            'kk': 0,                       # Model-free (no fitting parameters)
        }

        # Variable parameter models
        if model_name == 'multipole_debye':
            return 2 * (n_terms or 3) + 1  # N*delta_eps + N*tau + eps_inf
        elif model_name == 'lorentz':
            return 3 * (n_terms or 2) + 1  # N*f + N*w0 + N*gamma + eps_inf
        elif model_name == 'hybrid':
            # N*delta_eps + N*f_k + N*sigma_k + eps_inf
            return 3 * (n_terms or 2) + 1

        return param_counts.get(model_name, 0)

    @staticmethod
    def calculate_composite_score(result, method="balanced"):
        """
        Calculate composite score combining AIC and RMSE

        Args:
            result: Model result dictionary with 'aic' and 'rmse' keys
            method: Scoring method ('balanced', 'aic_focused', 'rmse_focused')

        Returns:
            Composite score (lower is better)
        """
        aic = result.get('aic', float('inf'))
        rmse = result.get('rmse', float('inf'))

        if method == "balanced":
            # Balanced: 70% AIC, 30% RMSE (scaled)
            rmse_scaled = rmse * 1000  # Scale RMSE to similar magnitude as AIC
            return 0.7 * aic + 0.3 * rmse_scaled
        elif method == "aic_focused":
            # Statistical rigor: 90% AIC, 10% RMSE
            rmse_scaled = rmse * 1000
            return 0.9 * aic + 0.1 * rmse_scaled
        elif method == "rmse_focused":
            # Accuracy focused: 30% AIC, 70% RMSE
            rmse_scaled = rmse * 1000
            return 0.3 * aic + 0.7 * rmse_scaled
        else:
            raise ValueError(f"Unknown scoring method: {method}")

    @staticmethod
    def enhanced_model_comparison(results_dict, selection_method="balanced"):
        """
        Enhanced model comparison with multi-criteria selection

        Args:
            results_dict: Dictionary with model names as keys and result dicts as values
            selection_method: Selection criteria ('balanced', 'aic_focused', 'rmse_focused')

        Returns:
            Dictionary with enhanced comparison and selection results
        """
        if not results_dict:
            return {"error": "No results provided for comparison"}

        # Extract and enhance comparison data
        comparison_data = []
        valid_results = {}

        for model_name, result in results_dict.items():
            if result is None or not isinstance(result, dict):
                continue

            # Skip if essential metrics are missing
            if 'aic' not in result or 'rmse' not in result:
                continue

            valid_results[model_name] = result

            # Get parameter count (try to infer from lmfit_result if available)
            n_params = BaseModel.get_parameter_count(model_name)
            if 'lmfit_result' in result and hasattr(result['lmfit_result'], 'nvarys'):
                n_params = result['lmfit_result'].nvarys

            comparison_data.append({
                'model': model_name,
                'aic': result['aic'],
                'bic': result.get('bic', float('inf')),
                'rmse': result['rmse'],
                'rmse_dk': result.get('rmse_dk', 0),
                'rmse_df': result.get('rmse_df', 0),
                'success': result.get('success', False),
                'n_params': n_params,
                'composite_score': BaseModel.calculate_composite_score(result, selection_method)
            })

        if not comparison_data:
            return {"error": "No valid results with required metrics (AIC, RMSE)"}

        # Sort by composite score (lower is better)
        comparison_data.sort(key=lambda x: x['composite_score'])

        # Calculate rankings
        for i, item in enumerate(comparison_data):
            item['overall_rank'] = i + 1

        # Calculate AIC weights for the valid models
        min_aic = min(item['aic'] for item in comparison_data)
        for item in comparison_data:
            delta_aic = item['aic'] - min_aic
            item['aic_weight'] = np.exp(-0.5 * delta_aic)

        # Normalize AIC weights
        total_weight = sum(item['aic_weight'] for item in comparison_data)
        for item in comparison_data:
            item['aic_weight'] /= total_weight

        # Best models by different criteria
        best_overall = comparison_data[0]['model']
        best_aic = min(comparison_data, key=lambda x: x['aic'])['model']
        best_rmse = min(comparison_data, key=lambda x: x['rmse'])['model']

        # Generate selection rationale
        best_result = comparison_data[0]
        rationale = BaseModel.generate_selection_rationale(
            best_result, comparison_data, selection_method)

        return {
            'best_model': best_overall,
            'best_aic_model': best_aic,
            'best_rmse_model': best_rmse,
            'selection_method': selection_method,
            'selection_rationale': rationale,
            'comparison_data': comparison_data,
            'summary': {
                'total_models': len(comparison_data),
                'successful_fits': sum(1 for item in comparison_data if item['success']),
                'aic_range': [min(item['aic'] for item in comparison_data),
                              max(item['aic'] for item in comparison_data)],
                'rmse_range': [min(item['rmse'] for item in comparison_data),
                               max(item['rmse'] for item in comparison_data)]
            }
        }

    @staticmethod
    def generate_selection_rationale(best_result, all_results, method):
        """Generate human-readable explanation for model selection"""
        model_name = best_result['model']
        aic = best_result['aic']
        rmse = best_result['rmse']

        # Find if there are close alternatives
        alternatives = [r for r in all_results if r['model']
                        != model_name and r['overall_rank'] <= 3]

        rationale = f"Selected {model_name.replace('_', ' ').title()} model "

        if method == "balanced":
            rationale += f"based on balanced criteria (AIC: {aic:.1f}, RMSE: {rmse:.4f})"
        elif method == "aic_focused":
            rationale += f"for statistical rigor (AIC: {aic:.1f})"
        elif method == "rmse_focused":
            rationale += f"for best accuracy (RMSE: {rmse:.4f})"

        if alternatives:
            alt_names = [r['model'].replace('_', ' ').title()
                         for r in alternatives[:2]]
            rationale += f". Close alternatives: {', '.join(alt_names)}"

        return rationale

    @staticmethod
    def print_enhanced_comparison(comparison_result):
        """
        Print enhanced model comparison with detailed metrics
        """
        if 'error' in comparison_result:
            print(f"Error: {comparison_result['error']}")
            return

        print("\n" + "="*80)
        print("ENHANCED MODEL COMPARISON")
        print("="*80)

        # Selection summary
        print(
            f"🏆 Best Model: {comparison_result['best_model'].replace('_', ' ').title()}")
        print(
            f"📊 Selection Method: {comparison_result['selection_method'].replace('_', ' ').title()}")
        print(f"💡 Rationale: {comparison_result['selection_rationale']}")
        print()

        # Detailed comparison table
        print(
            f"{'Model':<18} {'Rank':<5} {'AIC':<8} {'RMSE':<8} {'R²':<6} {'Params':<7} {'Status':<8}")
        print("-"*80)

        for item in comparison_result['comparison_data']:
            # Calculate R² from RMSE (approximate)
            rmse_normalized = item['rmse'] / max(0.1, np.mean(
                [item.get('rmse_dk', 0), item.get('rmse_df', 0)]))
            r_squared = max(0, 1 - rmse_normalized**2)

            success_str = "✓" if item['success'] else "✗"
            model_display = item['model'].replace('_', ' ').title()[:17]

            print(f"{model_display:<18} {item['overall_rank']:<5} {item['aic']:<8.1f} "
                  f"{item['rmse']:<8.4f} {r_squared:<6.3f} {item['n_params']:<7} {success_str:<8}")

        # Summary statistics
        summary = comparison_result['summary']
        print("-"*80)
        print(
            f"Summary: {summary['successful_fits']}/{summary['total_models']} successful fits")
        print(
            f"AIC range: {summary['aic_range'][0]:.1f} to {summary['aic_range'][1]:.1f}")
        print(
            f"RMSE range: {summary['rmse_range'][0]:.4f} to {summary['rmse_range'][1]:.4f}")
        print("="*80)

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
            eps_fit_corrected = eps_fit.real + \
                1j * np.maximum(eps_fit.imag, 1e-6)
            return eps_fit_corrected
        else:
            return eps_fit

    def extract_fitted_params(self, lmfit_result, model_type="unknown"):
        """
        Extract fitted parameters from lmfit result into a dictionary format
        that tests and analysis expect.

        Args:
            lmfit_result: lmfit result object with fitted parameters
            model_type: Type of model for parameter name mapping

        Returns:
            Dictionary with fitted parameter names and values
        """
        if lmfit_result is None or not hasattr(lmfit_result, 'params'):
            return {}

        fitted_params = {}

        # Extract all fitted parameters
        for param_name, param_obj in lmfit_result.params.items():
            fitted_params[param_name] = param_obj.value

        # Convert common parameter names to expected format
        if 'delta_eps' in fitted_params and 'eps_inf' in fitted_params:
            # Calculate eps_s from delta_eps and eps_inf
            fitted_params['eps_s'] = fitted_params['delta_eps'] + \
                fitted_params['eps_inf']

        return fitted_params
