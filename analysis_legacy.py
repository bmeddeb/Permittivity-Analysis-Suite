# analysis.py
# Model imports moved to analysis.model_runner module
import numpy as np

# MODEL_REGISTRY moved to analysis.model_runner module

# Legacy functions moved to analysis/ package classes:
# - optimize_model_n -> ModelRunner._optimize_model_n
# - compute_rmse -> PostProcessor._compute_rmse
# - auto_select_best_model -> PostProcessor._auto_select_best_model
# - auto_compare_models -> PostProcessor._auto_compare_models


# run_analysis function moved to analysis package (__init__.py)
# This provides backward compatibility for any direct imports from analysis.py
def run_analysis(df, selected_models, model_params, analysis_mode="manual"):
    """
    Legacy wrapper - imports from the analysis package.
    Prefer: from analysis import run_analysis
    """
    from analysis import run_analysis as package_run_analysis
    return package_run_analysis(df, selected_models, model_params, analysis_mode)
