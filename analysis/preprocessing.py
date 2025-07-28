# analysis/preprocessing.py
from utils.enhanced_preprocessing import EnhancedDielectricPreprocessor


class Preprocessor:
    """
    Handles data preprocessing for dielectric analysis.
    Wraps the EnhancedDielectricPreprocessor with a clean interface.
    """

    def __init__(self, model_params):
        """
        Initialize preprocessor with parameters.

        Args:
            model_params: Dictionary containing preprocessing parameters:
                - 'preprocessing_mode': str ('auto', 'manual', 'none')
                - 'preprocessing_method': str (for manual mode)
                - 'preprocessing_selection_method': str ('hybrid', 'rule_based', 'quality_based')
                - 'preprocessing_params': dict (manual parameters for UI overrides)
        """
        self.preprocessing_mode = model_params.get(
            'preprocessing_mode', 'auto')
        self.preprocessing_method = model_params.get(
            'preprocessing_method', 'smoothing_spline')
        self.preprocessing_selection_method = model_params.get(
            'preprocessing_selection_method', 'hybrid')
        self.manual_params = model_params.get('preprocessing_params', {})

        self.preprocessor = EnhancedDielectricPreprocessor()

    def apply(self, df):
        """
        Apply preprocessing to the input DataFrame.

        Args:
            df: DataFrame with experimental data

        Returns:
            Tuple[DataFrame, dict]: (processed_dataframe, preprocessing_info)
        """
        original_df = df.copy()
        preprocessing_info = None

        if self.preprocessing_mode != 'none':
            print(
                f"=== PREPROCESSING ({self.preprocessing_mode.upper()} MODE) ===")

            if self.preprocessing_mode == 'auto':
                # Auto preprocessing with selected method
                df, preprocessing_info = self.preprocessor.preprocess(
                    df, apply_smoothing=True, selection_method=self.preprocessing_selection_method
                )
                print(
                    f"Auto preprocessing: {preprocessing_info.get('dk_algorithm', 'None')} applied")

            elif self.preprocessing_mode == 'manual':
                # Manual preprocessing with specific algorithm
                print(f"Manual preprocessing: {self.preprocessing_method}")
                df, preprocessing_info = self.preprocessor.preprocess_manual(
                    df, self.preprocessing_method, self.manual_params
                )

            # Add original data reference for comparison
            if preprocessing_info:
                preprocessing_info['original_data'] = original_df
                print(
                    f"Preprocessing applied: {preprocessing_info.get('smoothing_applied', False)}")
                noise_score = preprocessing_info.get(
                    'noise_metrics', {}).get('overall_noise_score', 'N/A')
                if isinstance(noise_score, (int, float)):
                    print(f"Noise score: {noise_score:.3f}")
                else:
                    print(f"Noise score: {noise_score}")
        else:
            print("=== NO PREPROCESSING ===")
            preprocessing_info = {
                'preprocessing_mode': 'none',
                'smoothing_applied': False,
                'original_data': original_df,
                'noise_metrics': {},
                'recommendations': ['Preprocessing disabled by user']
            }

        return df, preprocessing_info
