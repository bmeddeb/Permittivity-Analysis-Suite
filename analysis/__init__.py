# analysis package
from .preprocessing import Preprocessor
from .model_runner import ModelRunner
from .postprocessing import PostProcessor


def run_analysis(df, selected_models, model_params, analysis_mode="manual"):
    """
    Enhanced analysis with separated concerns using dedicated classes.

    Args:
        df: DataFrame with experimental data
        selected_models: List of model names (ignored in auto mode)
        model_params: Dictionary with analysis parameters
        analysis_mode: 'manual', 'auto', or 'auto_compare'

    Returns:
        Dictionary with analysis results and metadata
    """
    # Step 1: Preprocessing
    preprocessor = Preprocessor(model_params)
    df_processed, prep_info = preprocessor.apply(df)

    # Step 2: Model execution
    runner = ModelRunner(model_params, analysis_mode)
    raw_results = runner.run(selected_models, df_processed)

    # Step 3: Post-processing and selection
    selection_method = model_params.get('selection_method', 'balanced')
    post = PostProcessor(selection_method)
    final = post.handle(raw_results, df_processed, analysis_mode)

    # Add preprocessing info to final result
    final['preprocessing_info'] = prep_info

    return final


__all__ = ['Preprocessor', 'ModelRunner', 'PostProcessor', 'run_analysis']
