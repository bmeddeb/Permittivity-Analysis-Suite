#!/usr/bin/env python3
"""
Test suite for enhanced preprocessing module with spline-based methods
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from utils.enhanced_preprocessing import (
    EnhancedNoiseAnalyzer, SplineSmoothing, EnhancedAlgorithmSelector, 
    EnhancedDielectricPreprocessor
)


class TestEnhancedNoiseAnalyzer:
    """Test enhanced noise analysis with domain-specific metrics"""
    
    def test_roughness_metric(self):
        """Test domain-specific roughness metric"""
        # Smooth sinusoidal signal
        x = np.linspace(0, 4*np.pi, 100)
        smooth_signal = np.sin(x)
        smooth_roughness = EnhancedNoiseAnalyzer.roughness_metric(smooth_signal)
        
        # Noisy signal
        noisy_signal = smooth_signal + 0.2 * np.random.randn(100)
        noisy_roughness = EnhancedNoiseAnalyzer.roughness_metric(noisy_signal)
        
        # Noisy signal should have higher roughness
        assert noisy_roughness > smooth_roughness
        assert smooth_roughness >= 0
        assert noisy_roughness >= 0
    
    def test_spectral_snr(self):
        """Test spectral SNR calculation for frequency-domain data"""
        # Create signal with dominant low-frequency component
        x = np.linspace(0, 10, 200)
        low_freq_signal = np.sin(0.5 * x) + 0.1 * np.sin(10 * x)  # Low + high freq
        high_snr = EnhancedNoiseAnalyzer.spectral_snr(low_freq_signal)
        
        # Add high-frequency noise
        noisy_signal = low_freq_signal + 0.3 * np.sin(50 * x)
        low_snr = EnhancedNoiseAnalyzer.spectral_snr(noisy_signal)
        
        # Signal with less high-frequency content should have higher SNR
        assert high_snr > low_snr
        assert high_snr > 0
    
    def test_local_variance_ratio(self):
        """Test local variance ratio for non-stationary noise detection"""
        # Stationary signal
        stationary = np.random.randn(100)
        stationary_ratio = EnhancedNoiseAnalyzer.local_variance_ratio(stationary)
        
        # Non-stationary signal (changing variance)
        non_stationary = np.concatenate([
            0.1 * np.random.randn(50),  # Low variance
            1.0 * np.random.randn(50)   # High variance
        ])
        non_stationary_ratio = EnhancedNoiseAnalyzer.local_variance_ratio(non_stationary)
        
        # Non-stationary should have different local variance ratio
        assert abs(non_stationary_ratio - 1.0) > abs(stationary_ratio - 1.0)
    
    def test_comprehensive_analysis(self, clean_debye_data, noisy_debye_data):
        """Test comprehensive noise analysis on dielectric data"""
        clean_df, _ = clean_debye_data
        noisy_df, _ = noisy_debye_data
        
        # Analyze both datasets
        clean_metrics = EnhancedNoiseAnalyzer.comprehensive_analysis(
            clean_df['Dk'].values, clean_df['Df'].values, clean_df['Frequency_GHz'].values
        )
        
        noisy_metrics = EnhancedNoiseAnalyzer.comprehensive_analysis(
            noisy_df['Dk'].values, noisy_df['Df'].values, noisy_df['Frequency_GHz'].values
        )
        
        # Noisy data should have higher overall noise score
        assert noisy_metrics['overall_noise_score'] > clean_metrics['overall_noise_score']
        
        # Check that all expected metrics are present
        expected_keys = [
            'dk_roughness', 'df_roughness', 'dk_spectral_snr', 'df_spectral_snr',
            'dk_derivative_var', 'df_derivative_var', 'dk_local_var_ratio', 'df_local_var_ratio',
            'dk_noise_variance', 'df_noise_variance', 'overall_noise_score'
        ]
        
        for key in expected_keys:
            assert key in clean_metrics
            assert key in noisy_metrics
            assert isinstance(clean_metrics[key], (int, float))


class TestSplineSmoothing:
    """Test spline-based smoothing algorithms"""
    
    def test_interpolating_spline(self):
        """Test interpolating spline (should pass through all points)"""
        x = np.linspace(0, 10, 20)
        y = np.sin(x) + 0.1 * np.random.randn(20)
        
        y_spline = SplineSmoothing.interpolating_spline(x, y)
        
        # Should pass exactly through original points
        np.testing.assert_allclose(y, y_spline, rtol=1e-10)
        assert len(y_spline) == len(y)
    
    def test_smoothing_spline_with_noise_var(self):
        """Test smoothing spline with noise variance parameter"""
        x = np.linspace(0, 10, 50)
        true_signal = np.sin(x)
        noise = 0.1 * np.random.randn(50)
        noisy_signal = true_signal + noise
        
        # Apply smoothing spline with known noise variance
        noise_var = np.var(noise)
        smoothed = SplineSmoothing.smoothing_spline(x, noisy_signal, noise_var=noise_var)
        
        # Smoothed signal should be closer to true signal than noisy signal
        original_mse = np.mean((noisy_signal - true_signal)**2)
        smoothed_mse = np.mean((smoothed - true_signal)**2)
        
        assert smoothed_mse < original_mse
        assert len(smoothed) == len(x)
    
    def test_smoothing_spline_automatic_s(self):
        """Test smoothing spline with automatic s parameter selection"""
        x = np.linspace(0, 10, 30)
        y = x**2 + 0.5 * np.random.randn(30)
        
        smoothed = SplineSmoothing.smoothing_spline(x, y)
        
        # Should produce reasonable smoothing
        assert len(smoothed) == len(y)
        assert np.var(np.diff(smoothed)) < np.var(np.diff(y))  # Should be smoother
    
    def test_pchip_smoothing(self):
        """Test PCHIP (shape-preserving) smoothing"""
        x = np.linspace(0, 10, 20)
        y = np.exp(-x/3) + 0.05 * np.random.randn(20)  # Monotonic decreasing
        
        smoothed = SplineSmoothing.pchip_smoothing(x, y)
        
        assert len(smoothed) == len(y)
        # PCHIP should preserve monotonicity better than high-order polynomials
        assert np.all(np.diff(smoothed) <= 0.1)  # Mostly decreasing
    
    def test_lowess_smoothing(self):
        """Test LOWESS robust smoothing"""
        x = np.linspace(0, 10, 40)
        y = np.sin(x)
        
        # Add some outliers
        y[10] = 5.0
        y[30] = -5.0
        
        smoothed = SplineSmoothing.lowess_smoothing(x, y, frac=0.3)
        
        assert len(smoothed) == len(y)
        # LOWESS should be more robust to outliers
        assert abs(smoothed[10]) < abs(y[10])  # Outlier should be reduced
        assert abs(smoothed[30]) < abs(y[30])


class TestEnhancedAlgorithmSelector:
    """Test enhanced algorithm selection"""
    
    def test_rule_based_selection_clean_data(self):
        """Test rule-based selection for clean data"""
        selector = EnhancedAlgorithmSelector()
        
        # Simulate clean data metrics
        noise_score = 0.05  # Very low noise
        n_points = 50
        spectral_snr = 25.0  # High SNR
        
        alg_name, params = selector.rule_based_selection(noise_score, n_points, spectral_snr)
        
        # Should select interpolating spline for clean data
        assert alg_name == "interpolating_spline"
        assert params == {}
    
    def test_rule_based_selection_noisy_data(self):
        """Test rule-based selection for noisy data"""
        selector = EnhancedAlgorithmSelector()
        
        # Simulate noisy data metrics
        noise_score = 0.6  # High noise
        n_points = 50
        spectral_snr = 3.0  # Low SNR
        
        alg_name, params = selector.rule_based_selection(noise_score, n_points, spectral_snr)
        
        # Should select robust smoothing for noisy data
        assert alg_name in ["lowess", "median"]
        assert isinstance(params, dict)
    
    def test_quality_based_evaluation(self, clean_debye_data):
        """Test quality-based algorithm evaluation"""
        clean_df, _ = clean_debye_data
        selector = EnhancedAlgorithmSelector()
        
        x = clean_df['Frequency_GHz'].values
        y = clean_df['Dk'].values
        
        # Create noise metrics
        noise_metrics = EnhancedNoiseAnalyzer.comprehensive_analysis(
            y, clean_df['Df'].values, x
        )
        
        alg_name, params, score = selector.quality_based_evaluation(x, y, noise_metrics)
        
        assert alg_name in selector.all_algorithms.keys()
        assert isinstance(params, dict)
        assert 0 <= score <= 1  # Quality score should be normalized
    
    def test_hybrid_selection(self, noisy_debye_data):
        """Test hybrid selection method"""
        noisy_df, _ = noisy_debye_data
        selector = EnhancedAlgorithmSelector()
        
        x = noisy_df['Frequency_GHz'].values
        y = noisy_df['Dk'].values
        
        noise_metrics = EnhancedNoiseAnalyzer.comprehensive_analysis(
            y, noisy_df['Df'].values, x
        )
        
        alg_name, alg_func, params = selector.select_algorithm(
            x, y, noise_metrics, method='hybrid'
        )
        
        assert alg_name in selector.all_algorithms.keys()
        assert callable(alg_func)
        assert isinstance(params, dict)
    
    @pytest.mark.parametrize("method", ["rule_based", "quality_based", "hybrid"])
    def test_selection_methods(self, clean_debye_data, method):
        """Test all selection methods"""
        clean_df, _ = clean_debye_data
        selector = EnhancedAlgorithmSelector()
        
        x = clean_df['Frequency_GHz'].values
        y = clean_df['Dk'].values
        
        noise_metrics = EnhancedNoiseAnalyzer.comprehensive_analysis(
            y, clean_df['Df'].values, x
        )
        
        alg_name, alg_func, params = selector.select_algorithm(
            x, y, noise_metrics, method=method
        )
        
        assert alg_name in selector.all_algorithms.keys()
        assert callable(alg_func)
        
        # Test that algorithm actually works
        result = alg_func(x, y, **params)
        assert len(result) == len(y)
        assert not np.any(np.isnan(result))


class TestEnhancedDielectricPreprocessor:
    """Test the main enhanced preprocessing class"""
    
    def test_preprocessing_clean_data(self, clean_debye_data):
        """Test preprocessing on clean data"""
        clean_df, _ = clean_debye_data
        preprocessor = EnhancedDielectricPreprocessor()
        
        processed_df, info = preprocessor.preprocess(clean_df, apply_smoothing=True)
        
        # Should detect clean data
        assert info['noise_metrics']['overall_noise_score'] < 0.3
        
        # May or may not apply smoothing depending on threshold
        assert len(processed_df) == len(clean_df)
        assert list(processed_df.columns) == list(clean_df.columns)
        
        # Should have recommendations
        assert 'recommendations' in info
        assert isinstance(info['recommendations'], list)
    
    def test_preprocessing_noisy_data(self, noisy_debye_data):
        """Test preprocessing on noisy data"""
        noisy_df, _ = noisy_debye_data
        preprocessor = EnhancedDielectricPreprocessor()
        
        processed_df, info = preprocessor.preprocess(noisy_df, apply_smoothing=True)
        
        # Should detect noise and apply smoothing
        assert info['noise_metrics']['overall_noise_score'] > 0.1
        
        if info['smoothing_applied']:
            assert info['dk_algorithm'] is not None
            assert info['df_algorithm'] is not None
            
            # Check noise reduction
            original_roughness = EnhancedNoiseAnalyzer.roughness_metric(noisy_df['Dk'].values)
            processed_roughness = EnhancedNoiseAnalyzer.roughness_metric(processed_df['Dk'].values)
            assert processed_roughness <= original_roughness
    
    @pytest.mark.parametrize("selection_method", ["rule_based", "quality_based", "hybrid"])
    def test_selection_methods_integration(self, noisy_debye_data, selection_method):
        """Test different selection methods in full preprocessing"""
        noisy_df, _ = noisy_debye_data
        preprocessor = EnhancedDielectricPreprocessor()
        
        processed_df, info = preprocessor.preprocess(
            noisy_df, apply_smoothing=True, selection_method=selection_method
        )
        
        assert info['selection_method'] == selection_method
        assert len(processed_df) == len(noisy_df)
        
        # All methods should produce valid results
        assert not processed_df.isnull().any().any()
        assert all(processed_df['Dk'] > 0)  # Physical constraint
        assert all(processed_df['Df'] >= 0)  # Physical constraint
    
    def test_spline_parameter_tuning(self, frequency_range, synthetic_data_generator):
        """Test that spline parameters are tuned based on noise characteristics"""
        preprocessor = EnhancedDielectricPreprocessor()
        
        # Generate data with known noise level
        dk, df = synthetic_data_generator.generate_debye_data(
            frequency_range, eps_s=10.0, eps_inf=2.0, tau=1e-9, noise_level=0.15
        )
        
        test_df = pd.DataFrame({
            'Frequency_GHz': frequency_range,
            'Dk': dk,
            'Df': df
        })
        
        processed_df, info = preprocessor.preprocess(test_df, apply_smoothing=True)
        
        # Should apply smoothing due to noise
        if info['smoothing_applied']:
            # Check that spline methods are preferred for dielectric data
            assert info['dk_algorithm'] in ['smoothing_spline', 'interpolating_spline', 
                                          'pchip', 'lowess', 'savitzky_golay']
            
            # If smoothing spline was used, should have appropriate parameters
            if info['dk_algorithm'] == 'smoothing_spline':
                assert 'noise_var' in info.get('dk_params', {}) or len(info.get('dk_params', {})) == 0
    
    def test_edge_cases(self):
        """Test preprocessing edge cases"""
        preprocessor = EnhancedDielectricPreprocessor()
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        processed, info = preprocessor.preprocess(empty_df)
        assert info['status'] == 'insufficient_data'
        
        # Very small DataFrame
        small_df = pd.DataFrame({
            'Frequency_GHz': [1.0, 2.0],
            'Dk': [4.0, 3.9],
            'Df': [0.1, 0.11]
        })
        processed, info = preprocessor.preprocess(small_df)
        assert info['status'] == 'insufficient_data'
    
    def test_algorithm_robustness(self, frequency_range, synthetic_data_generator):
        """Test that preprocessing is robust across different data characteristics"""
        preprocessor = EnhancedDielectricPreprocessor()
        
        # Test different data patterns
        test_cases = [
            # Flat data (low variation)
            (np.full_like(frequency_range, 4.0), np.full_like(frequency_range, 0.01)),
            # Monotonic data
            (np.linspace(10, 2, len(frequency_range)), np.linspace(0.1, 0.01, len(frequency_range))),
            # High dynamic range
            (10 * np.exp(-frequency_range/10), 0.1 * frequency_range),
        ]
        
        for dk, df in test_cases:
            test_df = pd.DataFrame({
                'Frequency_GHz': frequency_range,
                'Dk': dk,
                'Df': df
            })
            
            # Should handle all cases without crashing
            try:
                processed_df, info = preprocessor.preprocess(test_df, apply_smoothing=True)
                
                # Basic sanity checks
                assert len(processed_df) == len(test_df)
                assert not processed_df.isnull().any().any()
                assert 'overall_noise_score' in info['noise_metrics']
                
            except Exception as e:
                pytest.fail(f"Preprocessing failed on edge case: {e}")
    
    def test_consistency_across_runs(self, clean_debye_data):
        """Test that preprocessing is consistent across multiple runs"""
        clean_df, _ = clean_debye_data
        preprocessor = EnhancedDielectricPreprocessor()
        
        # Run preprocessing multiple times
        results = []
        for _ in range(3):
            processed_df, info = preprocessor.preprocess(clean_df, apply_smoothing=True)
            results.append((processed_df, info))
        
        # Should get consistent results
        if len(results) > 1:
            # Algorithm selection should be consistent for clean data
            algorithms = [info['dk_algorithm'] for _, info in results]
            assert len(set(algorithms)) <= 2  # Allow some variation
            
            # If smoothing was applied, results should be very similar
            if results[0][1]['smoothing_applied'] and results[1][1]['smoothing_applied']:
                df1, df2 = results[0][0], results[1][0]
                np.testing.assert_allclose(df1['Dk'].values, df2['Dk'].values, rtol=0.01)
    
    def test_get_algorithm_info(self):
        """Test algorithm information retrieval"""
        preprocessor = EnhancedDielectricPreprocessor()
        info = preprocessor.get_algorithm_info()
        
        assert isinstance(info, dict)
        assert 'primary_algorithms' in info
        assert 'secondary_algorithms' in info
        assert 'selection_methods' in info
        
        for key, value in info.items():
            assert isinstance(value, str)
            assert len(value) > 10  # Should be descriptive