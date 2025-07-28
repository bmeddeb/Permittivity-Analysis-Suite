# analysis/postprocessing.py
import numpy as np


class PostProcessor:
    """
    Handles post-processing of model results including RMSE computation
    and model selection/comparison.
    """

    def __init__(self, selection_method="balanced"):
        """
        Initialize post-processor.

        Args:
            selection_method: Selection criteria ('balanced', 'aic_focused', 'rmse_focused')
        """
        self.selection_method = selection_method

    def handle(self, raw_results, df, analysis_mode):
        """
        Process raw model results and perform final selection/comparison.

        Args:
            raw_results: Dictionary of raw model results from ModelRunner
            df: Original dataframe for RMSE computation
            analysis_mode: 'manual', 'auto', or 'auto_compare'

        Returns:
            Dictionary with processed results based on analysis mode
        """
        # First, compute RMSE for all results
        processed_results = self._compute_rmse_for_all(raw_results, df)

        # Handle different analysis modes
        if analysis_mode == "auto":
            return self._auto_select_best_model(processed_results)
        elif analysis_mode == "auto_compare":
            return self._auto_compare_models(processed_results)
        else:
            # Manual mode: return results as-is
            return {'model_results': processed_results}

    def _compute_rmse_for_all(self, results, df):
        """
        Compute RMSE for each model result that has eps_fit.

        Args:
            results: Dictionary of model results
            df: DataFrame with experimental data

        Returns:
            Dictionary of results with RMSE computed
        """
        processed_results = {}

        for key, res in results.items():
            if res is not None:
                print(f"Computing RMSE for {key}...")
                rmse_result = self._compute_rmse(res, df)
                if rmse_result is not None:
                    if isinstance(rmse_result, dict):
                        # New format with separate RMSE components
                        res.update(rmse_result)
                    else:
                        # Legacy format (single RMSE value)
                        res["rmse"] = rmse_result

                print(f"{key} RMSE: {res.get('rmse', 'N/A')}")

            processed_results[key] = res

        return processed_results

    def _compute_rmse(self, result, df):
        """
        Compute RMSE for a single model result.

        Args:
            result: Model result dictionary
            df: DataFrame with experimental data

        Returns:
            Dictionary with RMSE components or None
        """
        freq_exp = df.iloc[:, 0].values
        dk_exp = df.iloc[:, 1].values
        df_exp = df.iloc[:, 2].values  # Also include Df in RMSE calculation

        if "eps_fit" not in result:
            return None

        if "freq" in result:
            freq_fit = result["freq"]
        elif "freq_ghz" in result:
            freq_fit = result["freq_ghz"]
        else:
            return None

        freq_fit = np.array(freq_fit)
        eps_fit = np.array(result["eps_fit"])
        eps_fit_real = eps_fit.real
        eps_fit_imag = eps_fit.imag

        # Remove NaNs
        mask = (~np.isnan(freq_fit) & ~np.isnan(eps_fit_real) &
                ~np.isnan(eps_fit_imag))
        freq_fit = freq_fit[mask]
        eps_fit_real = eps_fit_real[mask]
        eps_fit_imag = eps_fit_imag[mask]

        if len(freq_fit) == 0:
            return None

        # Sort by frequency
        sort_idx = np.argsort(freq_fit)
        freq_fit = freq_fit[sort_idx]
        eps_fit_real = eps_fit_real[sort_idx]
        eps_fit_imag = eps_fit_imag[sort_idx]

        # Interpolate both real and imaginary parts
        dk_interp = np.interp(freq_exp, freq_fit, eps_fit_real)
        df_interp = np.interp(freq_exp, freq_fit, eps_fit_imag)

        # Calculate RMSE for both components
        rmse_dk = np.sqrt(np.mean((dk_interp - dk_exp) ** 2))
        rmse_df = np.sqrt(np.mean((df_interp - df_exp) ** 2))

        # Combined RMSE (you can adjust weighting if needed)
        rmse_total = np.sqrt(rmse_dk ** 2 + rmse_df ** 2)

        return {
            'rmse': rmse_total,
            'rmse_dk': rmse_dk,
            'rmse_df': rmse_df
        }

    def _auto_select_best_model(self, results):
        """
        Automatically select the best model using enhanced comparison

        Args:
            results: Dictionary of processed model results

        Returns:
            Dictionary with best model result and comparison info
        """
        from models.base_model import BaseModel

        # Filter out failed models - use same logic as optimize_model_n
        valid_results = {}
        for k, v in results.items():
            if v is not None:
                has_aic = 'aic' in v and v['aic'] != float('inf')
                has_rmse_or_fittable = (
                    'rmse' in v and v['rmse'] != float('inf')) or 'eps_fit' in v
                if has_aic and has_rmse_or_fittable:
                    valid_results[k] = v

        if not valid_results:
            return {"error": "No successful model fits found"}

        # Perform enhanced comparison
        comparison = BaseModel.enhanced_model_comparison(
            valid_results, self.selection_method)

        if "error" in comparison:
            return comparison

        best_model_name = comparison['best_model']
        best_result = valid_results[best_model_name]

        # Return the best model with comparison metadata
        return {
            "best_model_name": best_model_name,
            "best_model_result": best_result,
            "selection_rationale": comparison['selection_rationale'],
            "selection_method": self.selection_method,
            "comparison_summary": {
                "total_models_tested": comparison['summary']['total_models'],
                "successful_fits": comparison['summary']['successful_fits'],
                # Top 3 alternatives
                "alternatives": [item['model'] for item in comparison['comparison_data'][1:4]]
            },
            "all_results": valid_results  # For detailed analysis if needed
        }

    def _auto_compare_models(self, results):
        """
        Run enhanced model comparison and return detailed results

        Args:
            results: Dictionary of processed model results

        Returns:
            Dictionary with all results plus enhanced comparison data
        """
        from models.base_model import BaseModel

        # Filter out failed models - use same logic as optimize_model_n
        valid_results = {}
        for k, v in results.items():
            if v is not None:
                has_aic = 'aic' in v and v['aic'] != float('inf')
                has_rmse_or_fittable = (
                    'rmse' in v and v['rmse'] != float('inf')) or 'eps_fit' in v
                if has_aic and has_rmse_or_fittable:
                    valid_results[k] = v

        if not valid_results:
            return {"error": "No successful model fits found"}

        # Perform enhanced comparison
        comparison = BaseModel.enhanced_model_comparison(
            valid_results, self.selection_method)

        return {
            "analysis_mode": "auto_compare",
            "results": results,  # All results including failed ones
            "valid_results": valid_results,  # Only successful ones
            "enhanced_comparison": comparison,
            "selection_method": self.selection_method
        }
