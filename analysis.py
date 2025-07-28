# analysis.py
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


def optimize_model_n(model_class, df, n_range, model_name, selection_method="balanced"):
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
                has_rmse_or_fittable = ('rmse' in result and result['rmse'] != float('inf')) or 'eps_fit' in result
                
                if has_aic and has_rmse_or_fittable:
                    # Calculate RMSE if not present
                    if 'rmse' not in result or result['rmse'] == float('inf'):
                        rmse_result = compute_rmse(result, df)
                        if rmse_result is not None:
                            if isinstance(rmse_result, dict):
                                result.update(rmse_result)
                            else:
                                result["rmse"] = rmse_result
                    
                    # Calculate composite score based on selection method
                    composite_score = BaseModel.calculate_composite_score(result, selection_method)
                    all_results[n] = {
                        'result': result,
                        'composite_score': composite_score,
                        'aic': result.get('aic', float('inf')),
                        'rmse': result.get('rmse', float('inf'))
                    }
                    
                    aic_val = result.get('aic', float('inf'))
                    rmse_val = result.get('rmse', float('inf'))
                    success_flag = result.get('success', 'Unknown')
                    print(f"      N={n}: AIC={aic_val:.2f}, RMSE={rmse_val:.4f}, Score={composite_score:.2f}, Success={success_flag}")
                    
                    if composite_score < best_score:
                        best_score = composite_score
                        best_result = result
                        best_n = n
                else:
                    print(f"      N={n}: Failed (missing essential metrics: AIC={has_aic}, RMSE={has_rmse_or_fittable})")
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
            rmse_result = compute_rmse(best_result, df)
            if rmse_result is not None:
                if isinstance(rmse_result, dict):
                    best_result.update(rmse_result)
                else:
                    best_result["rmse"] = rmse_result
        
        print(f"  🏆 Optimal N for {model_name}: N={best_n} (Score: {best_score:.2f})")
        return best_result
    else:
        print(f"  ❌ No successful fits for {model_name}")
        return None


def compute_rmse(result, df):
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


def run_analysis(df, selected_models, model_params, analysis_mode="manual"):
    """
    Enhanced analysis with integrated preprocessing support

    Args:
        df: DataFrame with experimental data
        selected_models: List of model names (ignored in auto mode)
        model_params: Dictionary with keys like:
            - 'hybrid_terms': int
            - 'multipole_terms': int
            - 'lorentz_terms': int
            - 'selection_method': str ('balanced', 'aic_focused', 'rmse_focused')
            - 'preprocessing_mode': str ('auto', 'manual', 'none')
            - 'preprocessing_method': str (for manual mode)
            - 'preprocessing_selection_method': str ('hybrid', 'rule_based', 'quality_based')
        analysis_mode: 'manual' or 'auto' or 'auto_compare'
    """
    # Import enhanced preprocessing
    from utils.enhanced_preprocessing import EnhancedDielectricPreprocessor
    
    # Extract parameters with defaults
    selection_method = model_params.get('selection_method', 'balanced')
    preprocessing_mode = model_params.get('preprocessing_mode', 'auto')
    preprocessing_method = model_params.get('preprocessing_method', 'smoothing_spline')
    preprocessing_selection_method = model_params.get('preprocessing_selection_method', 'hybrid')
    
    # Initialize preprocessing results
    preprocessing_info = None
    original_df = df.copy()
    
    # Apply preprocessing based on mode
    if preprocessing_mode != 'none':
        print(f"=== PREPROCESSING ({preprocessing_mode.upper()} MODE) ===")
        preprocessor = EnhancedDielectricPreprocessor()
        
        if preprocessing_mode == 'auto':
            # Auto preprocessing with selected method
            df, preprocessing_info = preprocessor.preprocess(
                df, apply_smoothing=True, selection_method=preprocessing_selection_method
            )
            print(f"Auto preprocessing: {preprocessing_info.get('dk_algorithm', 'None')} applied")
        
        elif preprocessing_mode == 'manual':
            # Manual preprocessing with specific algorithm
            print(f"Manual preprocessing: {preprocessing_method}")
            # Extract manual parameters for UI overrides
            manual_params = model_params.get('preprocessing_params', {})
            df, preprocessing_info = preprocessor.preprocess_manual(
                df, preprocessing_method, manual_params
            )
        
        # Add original data reference for comparison
        if preprocessing_info:
            preprocessing_info['original_data'] = original_df
            print(f"Preprocessing applied: {preprocessing_info.get('smoothing_applied', False)}")
            print(f"Noise score: {preprocessing_info.get('noise_metrics', {}).get('overall_noise_score', 'N/A'):.3f}")
    else:
        print("=== NO PREPROCESSING ===")
        preprocessing_info = {
            'preprocessing_mode': 'none',
            'smoothing_applied': False,
            'original_data': original_df,
            'noise_metrics': {},
            'recommendations': ['Preprocessing disabled by user']
        }

    # Use different parameter sets for auto vs manual mode
    if analysis_mode in ["auto", "auto_compare"]:
        # Auto mode: Optimize N for each variable model
        print(f"Analysis mode: {analysis_mode}")
        print("Auto mode: Will optimize N for each variable model (Hybrid, Multipole, Lorentz)")
        models_to_run = ["debye", "cole_cole", "cole_davidson", "havriliak_negami", 
                        "sarkar", "multipole_debye", "lorentz", "hybrid"]
        print("Running all models with N-optimization for automatic selection...")
        
        # We'll optimize these during model running
        hybrid_terms = None      # Will be optimized
        multipole_terms = None   # Will be optimized  
        lorentz_terms = None     # Will be optimized
    else:
        # Manual mode: Use UI slider values as requested by user
        hybrid_terms = model_params.get('hybrid_terms', 2)
        multipole_terms = model_params.get('multipole_terms', 3)
        lorentz_terms = model_params.get('lorentz_terms', 2)
        print(f"Analysis mode: {analysis_mode}")
        print(f"Manual mode using USER parameters: Hybrid={hybrid_terms}, Multipole={multipole_terms}, Lorentz={lorentz_terms}")
        models_to_run = selected_models
        print(f"Running selected models: {selected_models}")

    results = {}

    # Run the selected models
    if "debye" in models_to_run:
        print("Running Debye model...")
        try:
            results["debye"] = DebyeModel().analyze(df)
        except Exception as e:
            print(f"Debye model failed: {e}")
            results["debye"] = None

    if "multipole_debye" in models_to_run:
        if analysis_mode in ["auto", "auto_compare"] and multipole_terms is None:
            # Auto mode: optimize N
            print("Running Multipole Debye model with N-optimization...")
            try:
                results["multipole_debye"] = optimize_model_n(
                    MultiPoleDebyeModel, df, range(1, 6), 
                    "Multipole Debye", selection_method
                )
            except Exception as e:
                print(f"Multipole Debye optimization failed: {e}")
                results["multipole_debye"] = None
        else:
            # Manual mode: use specified N
            print(f"Running Multipole Debye model with N={multipole_terms}...")
            try:
                results["multipole_debye"] = MultiPoleDebyeModel(N=multipole_terms).analyze(df)
            except Exception as e:
                print(f"Multipole Debye model failed: {e}")
                results["multipole_debye"] = None

    if "cole_cole" in models_to_run:
        print("Running Cole-Cole model...")
        try:
            results["cole_cole"] = ColeColeModel().analyze(df)
        except Exception as e:
            print(f"Cole-Cole model failed: {e}")
            results["cole_cole"] = None

    if "cole_davidson" in models_to_run:
        print("Running Cole-Davidson model...")
        try:
            results["cole_davidson"] = ColeDavidsonModel().analyze(df)
        except Exception as e:
            print(f"Cole-Davidson model failed: {e}")
            results["cole_davidson"] = None

    if "havriliak_negami" in models_to_run:
        print("Running Havriliak-Negami model...")
        try:
            results["havriliak_negami"] = HavriliakNegamiModel().analyze(df)
        except Exception as e:
            print(f"Havriliak-Negami model failed: {e}")
            results["havriliak_negami"] = None

    if "lorentz" in models_to_run:
        if analysis_mode in ["auto", "auto_compare"] and lorentz_terms is None:
            # Auto mode: optimize N
            print("Running Lorentz model with N-optimization...")
            try:
                results["lorentz"] = optimize_model_n(
                    LorentzModel, df, range(1, 5), 
                    "Lorentz", selection_method
                )
            except Exception as e:
                print(f"Lorentz optimization failed: {e}")
                results["lorentz"] = None
        else:
            # Manual mode: use specified N
            print(f"Running Lorentz model with N={lorentz_terms}...")
            try:
                results["lorentz"] = LorentzModel(N=lorentz_terms).analyze(df)
            except Exception as e:
                print(f"Lorentz model failed: {e}")
                results["lorentz"] = None

    if "sarkar" in models_to_run:
        print("Running Sarkar model...")
        try:
            results["sarkar"] = SarkarModel().analyze(df)
        except Exception as e:
            print(f"Sarkar model failed: {e}")
            results["sarkar"] = None

    if "hybrid" in models_to_run:
        if analysis_mode in ["auto", "auto_compare"] and hybrid_terms is None:
            # Auto mode: optimize N
            print("Running Hybrid model with N-optimization...")
            try:
                results["hybrid"] = optimize_model_n(
                    HybridModel, df, range(1, 6), 
                    "Hybrid", selection_method
                )
            except Exception as e:
                print(f"Hybrid optimization failed: {e}")
                results["hybrid"] = None
        else:
            # Manual mode: use specified N
            print(f"Running Hybrid model with N={hybrid_terms}...")
            try:
                results["hybrid"] = HybridModel(N=hybrid_terms).analyze(df)
            except Exception as e:
                print(f"Hybrid model failed: {e}")
                results["hybrid"] = None

    if "kk" in models_to_run:
        print("Running KK model...")
        try:
            results["kk"] = KKModel().analyze(df)
        except Exception as e:
            print(f"KK model failed: {e}")
            results["kk"] = None

    # Compute RMSE for each model that has eps_fit
    for key, res in results.items():
        if res is not None:
            print(f"Computing RMSE for {key}...")
            rmse_result = compute_rmse(res, df)
            if rmse_result is not None:
                if isinstance(rmse_result, dict):
                    # New format with separate RMSE components
                    res.update(rmse_result)
                else:
                    # Legacy format (single RMSE value)
                    res["rmse"] = rmse_result

            print(f"{key} RMSE: {res.get('rmse', 'N/A')}")

    # Handle auto-selection modes
    if analysis_mode == "auto":
        # Auto mode: return only the best model
        auto_result = auto_select_best_model(results, selection_method)
        if isinstance(auto_result, dict) and 'error' not in auto_result:
            auto_result['preprocessing_info'] = preprocessing_info
        return auto_result
    elif analysis_mode == "auto_compare":
        # Auto-compare mode: return all results with enhanced comparison
        compare_result = auto_compare_models(results, selection_method)
        if isinstance(compare_result, dict) and 'error' not in compare_result:
            compare_result['preprocessing_info'] = preprocessing_info
        return compare_result
    else:
        # Manual mode: return results as-is with preprocessing info
        return {
            'model_results': results,
            'preprocessing_info': preprocessing_info
        }


def auto_select_best_model(results, selection_method="balanced"):
    """
    Automatically select the best model using enhanced comparison
    
    Returns:
        Dictionary with best model result and comparison info
    """
    from models.base_model import BaseModel
    
    # Filter out failed models - use same logic as optimize_model_n
    valid_results = {}
    for k, v in results.items():
        if v is not None:
            has_aic = 'aic' in v and v['aic'] != float('inf')
            has_rmse_or_fittable = ('rmse' in v and v['rmse'] != float('inf')) or 'eps_fit' in v
            if has_aic and has_rmse_or_fittable:
                valid_results[k] = v
    
    if not valid_results:
        return {"error": "No successful model fits found"}
    
    # Perform enhanced comparison
    comparison = BaseModel.enhanced_model_comparison(valid_results, selection_method)
    
    if "error" in comparison:
        return comparison
    
    best_model_name = comparison['best_model']
    best_result = valid_results[best_model_name]
    
    # Return the best model with comparison metadata
    return {
        "best_model_name": best_model_name,
        "best_model_result": best_result,
        "selection_rationale": comparison['selection_rationale'],
        "selection_method": selection_method,
        "comparison_summary": {
            "total_models_tested": comparison['summary']['total_models'],
            "successful_fits": comparison['summary']['successful_fits'],
            "alternatives": [item['model'] for item in comparison['comparison_data'][1:4]]  # Top 3 alternatives
        },
        "all_results": valid_results  # For detailed analysis if needed
    }


def auto_compare_models(results, selection_method="balanced"):
    """
    Run enhanced model comparison and return detailed results
    
    Returns:
        Dictionary with all results plus enhanced comparison data
    """
    from models.base_model import BaseModel
    
    # Filter out failed models - use same logic as optimize_model_n
    valid_results = {}
    for k, v in results.items():
        if v is not None:
            has_aic = 'aic' in v and v['aic'] != float('inf')
            has_rmse_or_fittable = ('rmse' in v and v['rmse'] != float('inf')) or 'eps_fit' in v
            if has_aic and has_rmse_or_fittable:
                valid_results[k] = v
    
    if not valid_results:
        return {"error": "No successful model fits found"}
    
    # Perform enhanced comparison
    comparison = BaseModel.enhanced_model_comparison(valid_results, selection_method)
    
    return {
        "analysis_mode": "auto_compare",
        "results": results,  # All results including failed ones
        "valid_results": valid_results,  # Only successful ones
        "enhanced_comparison": comparison,
        "selection_method": selection_method
    }