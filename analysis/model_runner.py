# analysis/model_runner.py
from models.debye_model import DebyeModel
from models.hybrid_model import HybridModel
from models.multipole_debye_model import MultiPoleDebyeModel
from models.cole_cole_model import ColeColeModel
from models.cole_davidson_model import ColeDavidsonModel
from models.havriliak_negami_model import HavriliakNegamiModel
from models.lorentz_model import LorentzModel
from models.sarkar_model import SarkarModel
from models.kk_model import KKModel
import numpy as np

# Model Registry - centralized configuration for all models
MODEL_REGISTRY = {
    "debye": {
        "class": DebyeModel,
        "optimize_n": False,
        "default_n": None,
        "n_range": None
    },
    "multipole_debye": {
        "class": MultiPoleDebyeModel,
        "optimize_n": True,
        "default_n": 3,
        "n_range": range(1, 6),
        "param_key": "multipole_terms"
    },
    "cole_cole": {
        "class": ColeColeModel,
        "optimize_n": False,
        "default_n": None,
        "n_range": None
    },
    "cole_davidson": {
        "class": ColeDavidsonModel,
        "optimize_n": False,
        "default_n": None,
        "n_range": None
    },
    "havriliak_negami": {
        "class": HavriliakNegamiModel,
        "optimize_n": False,
        "default_n": None,
        "n_range": None
    },
    "lorentz": {
        "class": LorentzModel,
        "optimize_n": True,
        "default_n": 2,
        "n_range": range(1, 5),
        "param_key": "lorentz_terms"
    },
    "sarkar": {
        "class": SarkarModel,
        "optimize_n": False,
        "default_n": None,
        "n_range": None
    },
    "hybrid": {
        "class": HybridModel,
        "optimize_n": True,
        "default_n": 2,
        "n_range": range(1, 6),
        "param_key": "hybrid_terms"
    },
    "kk": {
        "class": KKModel,
        "optimize_n": False,
        "default_n": None,
        "n_range": None
    }
}


class ModelRunner:
    """
    Handles model execution using the MODEL_REGISTRY pattern.
    Supports both manual and auto modes with N-parameter optimization.
    """

    def __init__(self, model_params, analysis_mode):
        """
        Initialize model runner.

        Args:
            model_params: Dictionary with model parameters
            analysis_mode: 'manual', 'auto', or 'auto_compare'
        """
        self.model_params = model_params
        self.analysis_mode = analysis_mode
        self.selection_method = model_params.get(
            'selection_method', 'balanced')

    def run(self, models_to_run, df):
        """
        Run the specified models on the data.

        Args:
            models_to_run: List of model names to run
            df: DataFrame with (preprocessed) experimental data

        Returns:
            Dict[str, result]: Dictionary mapping model names to their results
        """
        # Determine models to run based on analysis mode
        if self.analysis_mode in ["auto", "auto_compare"]:
            # Auto mode: run all models with N-optimization
            print(f"Analysis mode: {self.analysis_mode}")
            print(
                "Auto mode: Will optimize N for each variable model (Hybrid, Multipole, Lorentz)")
            models_to_run = ["debye", "cole_cole", "cole_davidson", "havriliak_negami",
                             "sarkar", "multipole_debye", "lorentz", "hybrid"]
            print("Running all models with N-optimization for automatic selection...")
        else:
            # Manual mode: use selected models
            print(f"Analysis mode: {self.analysis_mode}")
            hybrid_terms = self.model_params.get('hybrid_terms', 2)
            multipole_terms = self.model_params.get('multipole_terms', 3)
            lorentz_terms = self.model_params.get('lorentz_terms', 2)
            print(
                f"Manual mode using USER parameters: Hybrid={hybrid_terms}, Multipole={multipole_terms}, Lorentz={lorentz_terms}")
            print(f"Running selected models: {models_to_run}")

        results = {}

        # Run the selected models using the registry
        for model_name, model_config in MODEL_REGISTRY.items():
            if model_name not in models_to_run:
                continue

            print(f"Running {model_name} model...")
            try:
                if model_config["optimize_n"] and self.analysis_mode in ("auto", "auto_compare"):
                    # Auto mode: optimize N for models that support it
                    results[model_name] = self._optimize_model_n(
                        model_config["class"], df, model_config["n_range"],
                        model_name, self.selection_method
                    )
                else:
                    # Manual mode or non-optimizing models: use specified/default N
                    model_class = model_config["class"]

                    if model_config["optimize_n"]:
                        # Models with N parameter - use manual values from UI or defaults
                        param_key = model_config.get("param_key")
                        if param_key:
                            n_value = self.model_params.get(
                                param_key) or model_config["default_n"]
                        else:
                            n_value = model_config["default_n"]

                        print(f"  Using N={n_value}")
                        results[model_name] = model_class(
                            N=n_value).analyze(df)
                    else:
                        # Models without N parameter (Debye, Cole-Cole, etc.)
                        results[model_name] = model_class().analyze(df)

            except Exception as e:
                print(f"{model_name} model failed: {e}")
                results[model_name] = None

        return results

    def _optimize_model_n(self, model_class, df, n_range, model_name, selection_method="balanced"):
        """
        Optimize N parameter for variable models by testing multiple values

        Args:
            model_class: Model class (e.g., HybridModel, MultiPoleDebyeModel)
            df: DataFrame with experimental data
            n_range: Range of N values to test (e.g., range(1, 6))
            model_name: Name for logging
            selection_method: Selection criteria ('balanced', 'aic_focused', 'rmse_focused')

        Returns:
            Dictionary with best result and optimal N value
        """
        from models.base_model import BaseModel

        print(f"  Optimizing N for {model_name} over range {list(n_range)}...")

        best_result = None
        best_n = None
        best_score = float('inf')
        all_results = {}

        for n in n_range:
            try:
                print(f"    Testing N={n}...")
                model = model_class(N=n)
                result = model.analyze(df)

                if result is not None:
                    # Check if we have essential metrics (more lenient than strict success flag)
                    has_aic = 'aic' in result and result['aic'] != float('inf')
                    has_rmse_or_fittable = (
                        'rmse' in result and result['rmse'] != float('inf')) or 'eps_fit' in result

                    if has_aic and has_rmse_or_fittable:
                        # Calculate RMSE if not present
                        if 'rmse' not in result or result['rmse'] == float('inf'):
                            rmse_result = self._compute_rmse(result, df)
                            if rmse_result is not None:
                                if isinstance(rmse_result, dict):
                                    result.update(rmse_result)
                                else:
                                    result["rmse"] = rmse_result

                        # Calculate composite score based on selection method
                        composite_score = BaseModel.calculate_composite_score(
                            result, selection_method)
                        all_results[n] = {
                            'result': result,
                            'composite_score': composite_score,
                            'aic': result.get('aic', float('inf')),
                            'rmse': result.get('rmse', float('inf'))
                        }

                        aic_val = result.get('aic', float('inf'))
                        rmse_val = result.get('rmse', float('inf'))
                        success_flag = result.get('success', 'Unknown')
                        print(
                            f"      N={n}: AIC={aic_val:.2f}, RMSE={rmse_val:.4f}, Score={composite_score:.2f}, Success={success_flag}")

                        if composite_score < best_score:
                            best_score = composite_score
                            best_result = result
                            best_n = n
                    else:
                        print(
                            f"      N={n}: Failed (missing essential metrics: AIC={has_aic}, RMSE={has_rmse_or_fittable})")
                else:
                    print(f"      N={n}: Failed (None result)")

            except Exception as e:
                print(f"      N={n}: Error - {e}")
                continue

        if best_result is not None:
            # Add optimization metadata to the result
            best_result['optimal_n'] = best_n
            best_result['optimization_summary'] = {
                'tested_n_values': list(all_results.keys()),
                'best_n': best_n,
                'selection_method': selection_method,
                'all_scores': {n: data['composite_score'] for n, data in all_results.items()}
            }

            # Ensure RMSE is calculated for the optimized result
            if 'rmse' not in best_result or best_result['rmse'] == float('inf'):
                rmse_result = self._compute_rmse(best_result, df)
                if rmse_result is not None:
                    if isinstance(rmse_result, dict):
                        best_result.update(rmse_result)
                    else:
                        best_result["rmse"] = rmse_result

            print(
                f"  🏆 Optimal N for {model_name}: N={best_n} (Score: {best_score:.2f})")
            return best_result
        else:
            print(f"  ❌ No successful fits for {model_name}")
            return None

    def _compute_rmse(self, result, df):
        """Helper method to compute RMSE for a result"""
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
