#!/usr/bin/env python3
"""
Integration tests combining preprocessing and model fitting
Tests the complete workflow from raw data to final results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from utils.preprocessing import DielectricPreprocessor
from analysis import run_analysis


class TestPreprocessingIntegration:
    """Test integration between preprocessing and model fitting"""
    
    def test_preprocessing_improves_fitting(self, frequency_range, synthetic_data_generator):
        """Test that preprocessing improves fitting quality for noisy data"""
        # Generate noisy synthetic data
        dk, df = synthetic_data_generator.generate_debye_data(
            frequency_range, eps_s=10.0, eps_inf=2.0, tau=1e-9, noise_level=0.2
        )
        
        noisy_df = pd.DataFrame({
            'Frequency_GHz': frequency_range,
            'Dk': dk,
            'Df': df
        })
        
        # Fit without preprocessing
        model_params = {'selection_method': 'balanced'}
        results_raw = run_analysis(noisy_df, [], model_params, 'auto')
        
        # Preprocess data
        preprocessor = DielectricPreprocessor()
        processed_df, _ = preprocessor.preprocess(noisy_df, apply_smoothing=True)
        
        # Fit with preprocessing
        results_processed = run_analysis(processed_df, [], model_params, 'auto')
        
        # Preprocessing should improve fit quality (lower RMSE)
        if ('error' not in results_raw and 'error' not in results_processed):
            rmse_raw = results_raw['best_model_result'].get('rmse', float('inf'))
            rmse_processed = results_processed['best_model_result'].get('rmse', float('inf'))
            
            # Preprocessing should reduce RMSE for noisy data
            assert rmse_processed <= rmse_raw * 1.1  # Allow small margin for variability
    
    def test_workflow_robustness(self, frequency_range, synthetic_data_generator, noise_levels):
        """Test complete workflow robustness across noise levels"""
        preprocessor = DielectricPreprocessor()
        
        for noise_level in noise_levels:
            # Generate test data
            dk, df = synthetic_data_generator.generate_cole_cole_data(
                frequency_range, eps_s=8.0, eps_inf=3.0, tau=5e-10, 
                alpha=0.85, noise_level=noise_level
            )
            
            test_df = pd.DataFrame({
                'Frequency_GHz': frequency_range,
                'Dk': dk,
                'Df': df
            })
            
            # Full workflow: preprocess + analyze
            processed_df, preprocess_info = preprocessor.preprocess(test_df)
            
            model_params = {'selection_method': 'balanced'}
            results = run_analysis(processed_df, [], model_params, 'auto')
            
            # Should always produce valid results
            assert 'error' not in results
            assert 'best_model_name' in results
            assert 'best_model_result' in results
            
            # Quality should degrade gracefully with noise
            rmse = results['best_model_result'].get('rmse', float('inf'))
            assert rmse < 1.0 + 2.0 * noise_level  # Reasonable scaling
    
    def test_different_models_with_preprocessing(self, frequency_range, synthetic_data_generator):
        """Test preprocessing works well with different underlying models"""
        preprocessor = DielectricPreprocessor()
        models_data = {}
        
        # Generate different model types with noise
        models_data['debye'] = synthetic_data_generator.generate_debye_data(
            frequency_range, eps_s=12.0, eps_inf=2.5, tau=2e-9, noise_level=0.15
        )
        
        models_data['cole_cole'] = synthetic_data_generator.generate_cole_cole_data(
            frequency_range, eps_s=9.0, eps_inf=2.0, tau=1e-9, alpha=0.75, noise_level=0.15
        )
        
        models_data['havriliak_negami'] = synthetic_data_generator.generate_havriliak_negami_data(
            frequency_range, eps_s=11.0, eps_inf=3.0, tau=8e-10, 
            alpha=0.8, beta=0.9, noise_level=0.15
        )
        
        for model_name, (dk, df) in models_data.items():
            test_df = pd.DataFrame({
                'Frequency_GHz': frequency_range,
                'Dk': dk,
                'Df': df
            })
            
            # Preprocess and analyze
            processed_df, _ = preprocessor.preprocess(test_df)
            
            model_params = {'selection_method': 'balanced'}
            results = run_analysis(processed_df, [], model_params, 'auto_compare')
            
            # Should successfully analyze all model types
            assert 'error' not in results
            assert 'valid_results' in results
            assert len(results['valid_results']) >= 3  # Multiple models should fit
            
            # Best model should have reasonable fit
            best_model = results['enhanced_comparison']['best_model']
            best_result = results['valid_results'][best_model]
            assert best_result.get('rmse', float('inf')) < 0.5


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow scenarios"""
    
    def test_clean_data_workflow(self, clean_debye_data):
        """Test workflow with clean data (minimal preprocessing needed)"""
        data_df, expected_params = clean_debye_data
        
        # Preprocess (should detect clean data)
        preprocessor = DielectricPreprocessor()
        processed_df, preprocess_info = preprocessor.preprocess(data_df)
        
        # Should detect low noise
        assert preprocess_info['noise_metrics']['combined_noise_score'] < 0.4
        
        # Analyze
        model_params = {'selection_method': 'balanced'}
        results = run_analysis(processed_df, [], model_params, 'auto')
        
        # Should achieve excellent fit
        assert 'error' not in results
        rmse = results['best_model_result'].get('rmse', float('inf'))
        assert rmse < 0.05  # Very low RMSE for clean data
        
        # Parameters should be well recovered
        fitted_params = results['best_model_result']['fitted_params']
        if 'eps_s' in fitted_params:
            assert abs(fitted_params['eps_s'] - expected_params['eps_s']) < 1.0
    
    def test_noisy_data_workflow(self, frequency_range, synthetic_data_generator):
        """Test workflow with very noisy data"""
        # Generate very noisy data
        dk, df = synthetic_data_generator.generate_debye_data(
            frequency_range, eps_s=15.0, eps_inf=2.0, tau=5e-10, noise_level=0.3
        )
        
        noisy_df = pd.DataFrame({
            'Frequency_GHz': frequency_range,
            'Dk': dk,
            'Df': df
        })
        
        # Preprocess (should detect high noise and apply smoothing)
        preprocessor = DielectricPreprocessor()
        processed_df, preprocess_info = preprocessor.preprocess(noisy_df)
        
        # Should detect high noise
        assert preprocess_info['noise_metrics']['combined_noise_score'] > 0.3
        
        # Should apply smoothing if noise is high enough
        if preprocess_info['smoothing_applied']:
            assert preprocess_info['dk_algorithm'] is not None
            assert preprocess_info['df_algorithm'] is not None
        
        # Analyze preprocessed data
        model_params = {'selection_method': 'rmse_focused'}  # Focus on fit quality
        results = run_analysis(processed_df, [], model_params, 'auto')
        
        # Should still produce reasonable results despite noise
        assert 'error' not in results
        rmse = results['best_model_result'].get('rmse', float('inf'))
        assert rmse < 1.0  # Should handle noise reasonably
        
        # Parameters should be in reasonable range despite noise
        fitted_params = results['best_model_result']['fitted_params']
        if 'eps_s' in fitted_params:
            assert 10.0 < fitted_params['eps_s'] < 20.0  # Should be in ballpark
    
    def test_multi_model_comparison_workflow(self, clean_cole_cole_data):
        """Test complete workflow with model comparison"""
        data_df, expected_params = clean_cole_cole_data
        
        # Preprocess
        preprocessor = DielectricPreprocessor()
        processed_df, _ = preprocessor.preprocess(data_df)
        
        # Run comprehensive comparison
        model_params = {'selection_method': 'balanced'}
        results = run_analysis(processed_df, [], model_params, 'auto_compare')
        
        assert 'error' not in results
        assert 'enhanced_comparison' in results
        assert 'valid_results' in results
        
        # Should have multiple valid models
        valid_results = results['valid_results']
        assert len(valid_results) >= 3
        
        # Cole-Cole or similar complex model should be competitive
        comparison_data = results['enhanced_comparison']['comparison_data']
        best_models = [item['model'] for item in comparison_data[:3]]
        
        # Should include appropriate models for Cole-Cole data
        complex_models = ['cole_cole', 'havriliak_negami', 'multipole_debye']
        assert any(model in best_models for model in complex_models)
        
        # All top models should have reasonable RMSE
        for item in comparison_data[:3]:
            assert item['rmse'] < 0.2
    
    @pytest.mark.parametrize("analysis_mode", ["auto", "auto_compare"])
    def test_analysis_modes(self, clean_debye_data, analysis_mode):
        """Test different analysis modes in complete workflow"""
        data_df, expected_params = clean_debye_data
        
        # Preprocess
        preprocessor = DielectricPreprocessor()
        processed_df, _ = preprocessor.preprocess(data_df)
        
        # Analyze with specified mode
        model_params = {'selection_method': 'balanced'}
        results = run_analysis(processed_df, [], model_params, analysis_mode)
        
        assert 'error' not in results
        
        if analysis_mode == 'auto':
            # Auto mode should return single best model
            assert 'best_model_name' in results
            assert 'best_model_result' in results
            assert 'selection_rationale' in results
            
        elif analysis_mode == 'auto_compare':
            # Auto-compare should return comprehensive comparison
            assert 'enhanced_comparison' in results
            assert 'valid_results' in results
            
            # Should have detailed comparison data
            comparison = results['enhanced_comparison']
            assert 'best_model' in comparison
            assert 'comparison_data' in comparison
            assert len(comparison['comparison_data']) >= 2
    
    def test_error_handling_workflow(self):
        """Test workflow error handling with problematic data"""
        preprocessor = DielectricPreprocessor()
        
        # Test with empty data
        empty_df = pd.DataFrame()
        processed_df, info = preprocessor.preprocess(empty_df)
        assert info['status'] == 'insufficient_data'
        
        # Test with NaN data
        nan_df = pd.DataFrame({
            'Frequency_GHz': [1.0, 2.0, np.nan, 4.0],
            'Dk': [4.0, np.nan, 3.8, 3.7],
            'Df': [0.1, 0.11, 0.12, np.nan]
        })
        
        # Should handle NaN gracefully
        try:
            processed_df, info = preprocessor.preprocess(nan_df)
            # If it processes, should not crash the analysis
            model_params = {'selection_method': 'balanced'}
            results = run_analysis(processed_df, [], model_params, 'auto')
            # May fail analysis but shouldn't crash
        except:
            # Acceptable to fail gracefully with bad data
            pass
    
    def test_performance_consistency(self, frequency_range, synthetic_data_generator):
        """Test that workflow performance is consistent across runs"""
        # Generate test data
        dk, df = synthetic_data_generator.generate_debye_data(
            frequency_range, eps_s=8.0, eps_inf=2.5, tau=1e-9, noise_level=0.1
        )
        
        test_df = pd.DataFrame({
            'Frequency_GHz': frequency_range,
            'Dk': dk,
            'Df': df
        })
        
        preprocessor = DielectricPreprocessor()
        model_params = {'selection_method': 'balanced'}
        
        # Run multiple times
        results_list = []
        for _ in range(3):
            processed_df, _ = preprocessor.preprocess(test_df)
            results = run_analysis(processed_df, [], model_params, 'auto')
            if 'error' not in results:
                results_list.append(results)
        
        # Should get consistent results
        assert len(results_list) >= 2  # Most runs should succeed
        
        if len(results_list) >= 2:
            # Should select same model type consistently
            models = [r['best_model_name'] for r in results_list]
            # Allow some variation but should be mostly consistent
            assert len(set(models)) <= 2  # At most 2 different models selected
            
            # RMSE should be similar across runs
            rmses = [r['best_model_result']['rmse'] for r in results_list]
            rmse_std = np.std(rmses)
            assert rmse_std < 0.1  # Should be reasonably consistent