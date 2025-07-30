# app/models/sarkar_analyzer.py
"""
Analysis and validation utilities for the Djordjevic-Sarkar model.

This module provides model suitability assessment, parameter validation,
quality metrics, and model comparison functionality.
"""

import numpy as np
import pandas as pd
import lmfit
from typing import Dict, Tuple, List, Any, Optional
import logging

from .sarkar import SarkarModel
from .kramers_kronig_validator import KramersKronigValidator

logger = logging.getLogger(__name__)


class SarkarModelAnalyzer:
    """
    Analyzer for Djordjevic-Sarkar model results.

    Provides validation, quality assessment, and comparison with other models.
    """

    def __init__(self, model: SarkarModel):
        """
        Initialize analyzer with a SarkarModel instance.

        Args:
            model: SarkarModel instance
        """
        self.model = model

    def assess_data_suitability(self, freq: np.ndarray, dk_exp: np.ndarray,
                                df_exp: np.ndarray) -> Dict[str, Any]:
        """
        Assess if D-S model is appropriate for the data.

        Args:
            freq: Frequency array
            dk_exp: Experimental Dk
            df_exp: Experimental Df

        Returns:
            Assessment dictionary with suitability analysis
        """
        assessment = {
            'suitable': True,
            'confidence': 1.0,
            'reasons': [],
            'warnings': [],
            'alternatives': []
        }

        # Check frequency span
        freq_span = np.log10(freq[-1] / freq[0])
        if freq_span < 1:
            assessment['warnings'].append(f"Narrow frequency range ({freq_span:.1f} decades)")
            assessment['alternatives'].append("Consider single-term Debye model")
            assessment['confidence'] *= 0.8

        # Check for monotonic Dk
        dk_diff = np.diff(dk_exp)
        if not np.all(dk_diff <= 0):
            non_monotonic_pct = 100 * np.sum(dk_diff > 0) / len(dk_diff)
            if non_monotonic_pct > 10:
                assessment['suitable'] = False
                assessment['reasons'].append(f"Dk is non-monotonic ({non_monotonic_pct:.0f}% increasing)")
                assessment['alternatives'].append("Check data quality or consider Lorentz model")
                assessment['confidence'] *= 0.3
            elif non_monotonic_pct > 5:
                assessment['warnings'].append(f"Some non-monotonic behavior ({non_monotonic_pct:.0f}%)")
                assessment['confidence'] *= 0.7

        # Check dispersion strength
        dispersion = (dk_exp[0] - dk_exp[-1]) / dk_exp[0]
        if dispersion < 0.01:
            assessment['warnings'].append(f"Very weak dispersion ({dispersion:.1%})")
            assessment['alternatives'].append("May not need frequency-dependent model")
            assessment['confidence'] *= 0.6
        elif dispersion > 0.8:
            assessment['warnings'].append(f"Very strong dispersion ({dispersion:.1%})")
            assessment['alternatives'].append("Check for multiple relaxation mechanisms")

        # Check for resonance-like features
        if len(dk_exp) > 5:
            d2_dk = np.gradient(np.gradient(dk_exp))
            max_d2 = np.max(np.abs(d2_dk))
            if max_d2 > 0.1 * np.max(dk_exp):
                assessment['warnings'].append("Possible resonance features detected")
                assessment['alternatives'].append("Consider Hybrid Debye-Lorentz model")
                assessment['confidence'] *= 0.7

        # Check Df behavior
        if np.any(df_exp <= 0):
            assessment['suitable'] = False
            assessment['reasons'].append("Negative or zero Df values detected")
            assessment['confidence'] = 0

        # Overall confidence score
        assessment['confidence'] = max(0, min(1, assessment['confidence']))

        return assessment

    def validate_parameters(self, params: lmfit.Parameters) -> Tuple[bool, List[str]]:
        """
        Validate fitted parameters for physical consistency.

        Args:
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        p = params.valuesdict()

        # Physical bounds
        if p['eps_r_inf'] < 1:
            issues.append("eps_r_inf < 1 is non-physical")

        if p['eps_r_s'] <= p['eps_r_inf']:
            issues.append("eps_r_s must be > eps_r_inf for normal dispersion")

        # Check frequency ratio
        ratio = p['f2'] / p['f1']
        if ratio < 2:
            issues.append(f"f2/f1 = {ratio:.1f} is too small (min 2)")
        elif ratio > 1e6:
            issues.append(f"f2/f1 = {ratio:.1e} is unrealistically large")

        # Check for reasonable dispersion
        delta_eps = p['eps_r_s'] - p['eps_r_inf']
        relative_dispersion = delta_eps / p['eps_r_s']
        if relative_dispersion > 0.9:
            issues.append(f"Dispersion {relative_dispersion:.1%} > 90% is unusual")

        # Validate sigma_dc
        if p['sigma_dc'] < 0:
            issues.append("Negative conductivity is non-physical")
        elif p['sigma_dc'] > 1e-4:
            issues.append(f"sigma_dc = {p['sigma_dc']:.1e} S/m is high for typical dielectric")

        return len(issues) == 0, issues

    def validate_with_kramers_kronig(self, result: lmfit.model.ModelResult,
                                     causality_threshold: float = 0.05) -> Dict[str, Any]:
        """
        Validate fit results using Kramers-Kronig relations.

        Args:
            result: Fit result from SarkarModel
            causality_threshold: Threshold for causality check

        Returns:
            Dictionary with K-K validation results
        """
        # Get fitted values
        fitted_complex = self.model.predict(result.freq, result.params)
        dk_fit = fitted_complex.real
        df_fit = fitted_complex.imag / fitted_complex.real

        # Create DataFrame for K-K validator
        kk_df = pd.DataFrame({
            'Frequency (GHz)': result.freq if self.model.use_ghz else result.freq / 1e9,
            'Dk': dk_fit,
            'Df': df_fit
        })

        # Run K-K validation
        try:
            with KramersKronigValidator(kk_df, method='auto') as validator:
                kk_results = validator.validate(causality_threshold)
                kk_diagnostics = validator.get_diagnostics()

            validation_results = {
                'causality_status': kk_results.get('causality_status', 'UNKNOWN'),
                'mean_relative_error': kk_results.get('mean_relative_error', 0.0),
                'rmse': kk_results.get('rmse', 0.0),
                'is_causal': kk_results.get('causality_status', '') == 'PASS',
                'diagnostics': kk_diagnostics
            }

            # Add comparison with experimental K-K if possible
            exp_df = pd.DataFrame({
                'Frequency (GHz)': result.freq if self.model.use_ghz else result.freq / 1e9,
                'Dk': result.dk_exp,
                'Df': result.df_exp
            })

            with KramersKronigValidator(exp_df, method='auto') as exp_validator:
                exp_kk_results = exp_validator.validate(causality_threshold)
                validation_results['experimental_causality'] = {
                    'status': exp_kk_results['causality_status'],
                    'mean_relative_error': exp_kk_results['mean_relative_error']
                }

        except Exception as e:
            logger.error(f"K-K validation failed: {e}")
            validation_results = {
                'causality_status': 'ERROR',
                'error': str(e),
                'is_causal': False
            }

        return validation_results

    def calculate_enhanced_quality_metrics(self, result: lmfit.model.ModelResult) -> Dict[str, float]:
        """
        Calculate comprehensive quality metrics for the fit.

        Args:
            result: Fit result

        Returns:
            Dictionary of quality metrics
        """
        # Get base metrics from the model
        if hasattr(result, 'fit_metrics'):
            base_metrics = {
                'r_squared': result.fit_metrics.r_squared,
                'rmse': result.fit_metrics.rmse,
                'mape': result.fit_metrics.mape,
                'chi_squared': result.fit_metrics.chi_squared,
                'reduced_chi_squared': result.fit_metrics.reduced_chi_squared,
                'dk_rmse': result.fit_metrics.dk_rmse,
                'df_rmse': result.fit_metrics.df_rmse
            }
        else:
            base_metrics = {}

        metrics = {
            'r_squared': base_metrics.get('r_squared', 0),
            'rmse': base_metrics.get('rmse', np.inf),
            'mape': base_metrics.get('mape', 100),
            'chi_squared': base_metrics.get('chi_squared', np.inf),
            'reduced_chi_squared': base_metrics.get('reduced_chi_squared', np.inf),
            'dk_rmse': base_metrics.get('dk_rmse', np.inf),
            'df_rmse': base_metrics.get('df_rmse', np.inf)
        }

        # Get fitted values
        eps_fit = self.model.predict(result.freq, result.params)
        dk_fit = eps_fit.real
        df_fit = eps_fit.imag / eps_fit.real

        # Smoothness metric (lower is smoother)
        if len(dk_fit) > 3:
            dk_smoothness = np.std(np.diff(np.diff(dk_fit))) / (np.std(dk_fit) + 1e-10)
            df_smoothness = np.std(np.diff(np.diff(df_fit))) / (np.std(df_fit) + 1e-10)
        else:
            dk_smoothness = 0
            df_smoothness = 0

        metrics['dk_smoothness'] = dk_smoothness
        metrics['df_smoothness'] = df_smoothness

        # Parameter stability (condition number)
        if hasattr(result, 'covar') and result.covar is not None:
            try:
                condition_number = np.linalg.cond(result.covar)
                metrics['parameter_stability'] = 1 / (1 + condition_number / 100)
            except:
                metrics['parameter_stability'] = 0.5
        else:
            metrics['parameter_stability'] = 0.5

        # Physical consistency score
        is_valid, issues = self.validate_parameters(result.params)
        metrics['physical_consistency'] = 1.0 if is_valid else 1.0 / (1 + len(issues))

        # Extrapolation quality (based on frequency ratio)
        p = result.params.valuesdict()
        f_ratio = p['f2'] / p['f1']
        metrics['extrapolation_quality'] = min(1.0, f_ratio / 100)

        # K-K validation score
        kk_validation = self.validate_with_kramers_kronig(result)
        if kk_validation.get('is_causal', False):
            metrics['causality_score'] = 1.0 - kk_validation.get('mean_relative_error', 0.5)
        else:
            metrics['causality_score'] = 0.0

        # Overall quality score (0-1)
        metrics['overall_quality'] = (
                metrics['r_squared'] * 0.25 +
                (1 - min(1, metrics['mape'] / 100)) * 0.25 +
                metrics['physical_consistency'] * 0.15 +
                metrics['parameter_stability'] * 0.10 +
                metrics['extrapolation_quality'] * 0.10 +
                metrics['causality_score'] * 0.15
        )

        return metrics

    def compare_with_models(self, freq: np.ndarray, dk_exp: np.ndarray,
                            df_exp: np.ndarray,
                            models_to_compare: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare D-S model with other models.

        Args:
            freq: Frequency array
            dk_exp: Experimental Dk
            df_exp: Experimental Df
            models_to_compare: List of model names to compare against

        Returns:
            Comparison results
        """
        if models_to_compare is None:
            models_to_compare = ['MultiDebye-2', 'MultiDebye-4', 'MultiDebye-6']

        # Fit D-S model
        ds_result = self.model.fit(freq, dk_exp, df_exp)
        ds_quality = self.calculate_enhanced_quality_metrics(ds_result)

        comparison = {
            'sarkar_model': {
                'result': ds_result,
                'n_params': ds_result.nvarys,  # Get from result instead of model
                'aic': ds_result.aic if hasattr(ds_result, 'aic') else self._calculate_aic(ds_result),
                'bic': ds_result.bic if hasattr(ds_result, 'bic') else self._calculate_bic(ds_result),
                'quality': ds_quality,
                'validation': self.validate_parameters(ds_result.params)
            },
            'other_models': {}
        }

        # Compare with other models
        for model_name in models_to_compare:
            if model_name.startswith('MultiDebye'):
                n_terms = int(model_name.split('-')[1])
                comparison['other_models'][model_name] = self._fit_multi_debye(
                    freq, dk_exp, df_exp, n_terms
                )

        # Find best model by BIC
        all_models = [('sarkar', comparison['sarkar_model'])] + \
                     [(name, data) for name, data in comparison['other_models'].items()]

        best_model = min(all_models, key=lambda x: x[1].get('bic', np.inf))
        comparison['best_model'] = best_model[0]

        # Generate recommendation
        comparison['recommendation'] = self._generate_recommendation(comparison)

        return comparison

    def _fit_multi_debye(self, freq: np.ndarray, dk_exp: np.ndarray,
                         df_exp: np.ndarray, n_terms: int) -> Dict[str, Any]:
        """Fit Multi-Term Debye model for comparison."""
        try:
            from .multi_term_debye_model import MultiTermDebyeModel

            md_model = MultiTermDebyeModel(n_terms=n_terms)
            md_result = md_model.fit(freq, dk_exp, df_exp)

            return {
                'result': md_result,
                'n_params': md_result.nvarys,  # Get from result
                'aic': md_result.aic if hasattr(md_result, 'aic') else self._calculate_aic(md_result),
                'bic': md_result.bic if hasattr(md_result, 'bic') else self._calculate_bic(md_result),
                'rmse': md_result.fit_metrics.rmse if hasattr(md_result, 'fit_metrics') else np.inf,
                'r_squared': md_result.fit_metrics.r_squared if hasattr(md_result, 'fit_metrics') else 0
            }
        except Exception as e:
            logger.warning(f"Multi-Debye fit failed for n={n_terms}: {e}")
            return {'error': str(e), 'bic': np.inf}

    def _calculate_aic(self, result: lmfit.model.ModelResult) -> float:
        """Calculate AIC if not available."""
        n_data = len(result.freq) * 2  # Real and imaginary parts
        n_params = result.nvarys
        chi2 = result.chisqr if hasattr(result, 'chisqr') else np.inf
        return n_data * np.log(chi2 / n_data) + 2 * n_params

    def _calculate_bic(self, result: lmfit.model.ModelResult) -> float:
        """Calculate BIC if not available."""
        n_data = len(result.freq) * 2
        n_params = result.nvarys
        chi2 = result.chisqr if hasattr(result, 'chisqr') else np.inf
        return n_data * np.log(chi2 / n_data) + n_params * np.log(n_data)

    def _generate_recommendation(self, comparison: Dict[str, Any]) -> str:
        """Generate model recommendation based on comparison."""
        sarkar_bic = comparison['sarkar_model']['bic']
        best_other = min(comparison['other_models'].items(),
                         key=lambda x: x[1].get('bic', np.inf),
                         default=(None, {'bic': np.inf}))

        if best_other[0] is None:
            return "Only Sarkar model available"

        bic_diff = sarkar_bic - best_other[1]['bic']

        if bic_diff < -10:
            return "Sarkar model strongly preferred"
        elif bic_diff < -2:
            return "Sarkar model preferred"
        elif bic_diff < 2:
            return "Models comparable - choose based on physics"
        elif bic_diff < 10:
            return f"{best_other[0]} preferred"
        else:
            return f"{best_other[0]} strongly preferred"

    def generate_analysis_report(self, result: lmfit.model.ModelResult) -> str:
        """
        Generate comprehensive analysis report.

        Args:
            result: Fit result

        Returns:
            Formatted report string
        """
        # Get all analysis components
        suitability = self.assess_data_suitability(result.freq, result.dk_exp, result.df_exp)
        param_validation = self.validate_parameters(result.params)
        quality_metrics = self.calculate_enhanced_quality_metrics(result)
        kk_validation = self.validate_with_kramers_kronig(result)
        transition_chars = self.model.get_transition_characteristics(result.params)

        report = f"\n{'=' * 70}\n"
        report += f"Djordjevic-Sarkar Model Analysis Report\n"
        report += f"{'=' * 70}\n\n"

        # Data Suitability
        report += "Data Suitability Assessment:\n"
        report += f"  Suitable: {'Yes' if suitability['suitable'] else 'No'}\n"
        report += f"  Confidence: {suitability['confidence']:.1%}\n"
        if suitability['warnings']:
            report += "  Warnings:\n"
            for warning in suitability['warnings']:
                report += f"    - {warning}\n"
        if suitability['alternatives']:
            report += "  Alternative models to consider:\n"
            for alt in suitability['alternatives']:
                report += f"    - {alt}\n"
        report += "\n"

        # Parameter Validation
        report += "Parameter Validation:\n"
        report += f"  Valid: {'Yes' if param_validation[0] else 'No'}\n"
        if not param_validation[0]:
            report += "  Issues:\n"
            for issue in param_validation[1]:
                report += f"    - {issue}\n"
        report += "\n"

        # Fit Quality
        report += "Fit Quality Metrics:\n"
        report += f"  R²: {quality_metrics['r_squared']:.4f}\n"
        report += f"  RMSE: {quality_metrics['rmse']:.4g}\n"
        report += f"  MAPE: {quality_metrics['mape']:.2f}%\n"
        report += f"  Overall Quality: {quality_metrics['overall_quality']:.3f}\n"
        report += f"  Causality Score: {quality_metrics['causality_score']:.3f}\n"
        report += "\n"

        # K-K Validation
        report += "Kramers-Kronig Validation:\n"
        report += f"  Causality Status: {kk_validation.get('causality_status', 'N/A')}\n"
        if 'mean_relative_error' in kk_validation:
            report += f"  Mean Relative Error: {kk_validation['mean_relative_error']:.2%}\n"
        if 'experimental_causality' in kk_validation:
            exp_status = kk_validation['experimental_causality']['status']
            report += f"  Experimental Data Causality: {exp_status}\n"
        report += "\n"

        # Transition Characteristics
        report += "Transition Characteristics:\n"
        report += f"  f₁: {transition_chars['f1_ghz']:.3f} GHz\n"
        report += f"  f₂: {transition_chars['f2_ghz']:.3f} GHz\n"
        report += f"  Center: {transition_chars['f_center_ghz']:.3f} GHz\n"
        report += f"  Ratio (f₂/f₁): {transition_chars['f_ratio']:.1f}\n"
        report += f"  Dispersion: {transition_chars['relative_dispersion']:.1%}\n"
        report += f"  DC Conductivity: {transition_chars['sigma_dc']:.2e} S/m\n"

        report += f"{'=' * 70}\n"

        return report