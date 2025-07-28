#!/usr/bin/env python3
"""
Test suite for the preprocessing module using synthetic data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from utils.preprocessing import (
    NoiseAnalyzer, SmoothingAlgorithms, SmoothinSelector, DielectricPreprocessor
)


class TestNoiseAnalyzer:
    """Test noise analysis functionality"""
    
    def test_roughness_calculation(self):
        """Test roughness calculation for different noise levels"""
        # Clean signal
        clean_data = np.sin(np.linspace(0, 4*np.pi, 100))
        clean_roughness = NoiseAnalyzer.calculate_roughness(clean_data)
        
        # Noisy signal
        noisy_data = clean_data + 0.1 * np.random.randn(100)
        noisy_roughness = NoiseAnalyzer.calculate_roughness(noisy_data)
        
        assert noisy_roughness > clean_roughness
        assert clean_roughness >= 0
        assert noisy_roughness >= 0
    
    def test_snr_calculation(self):
        """Test SNR calculation"""
        # High SNR signal
        signal = np.sin(np.linspace(0, 4*np.pi, 100))
        small_noise = signal + 0.01 * np.random.randn(100)
        high_snr = NoiseAnalyzer.calculate_snr(small_noise)
        
        # Low SNR signal
        large_noise = signal + 0.5 * np.random.randn(100)
        low_snr = NoiseAnalyzer.calculate_snr(large_noise)
        
        assert high_snr > low_snr
        assert high_snr > 0  # Should be positive for decent signals
    
    def test_derivative_variance(self):
        """Test derivative variance calculation"""
        # Smooth signal
        smooth_data = np.sin(np.linspace(0, 2*np.pi, 50))
        smooth_var = NoiseAnalyzer.calculate_derivative_variance(smooth_data)
        
        # Noisy signal
        noisy_data = smooth_data + 0.2 * np.random.randn(50)
        noisy_var = NoiseAnalyzer.calculate_derivative_variance(noisy_data)
        
        assert noisy_var > smooth_var
        assert smooth_var >= 0
    
    def test_comprehensive_noise_analysis(self, clean_debye_data, noisy_debye_data):
        """Test comprehensive noise analysis on synthetic dielectric data"""
        clean_df, _ = clean_debye_data
        noisy_df, _ = noisy_debye_data
        
        # Analyze clean data
        clean_metrics = NoiseAnalyzer.analyze_noise(
            clean_df['Dk'].values, clean_df['Df'].values, clean_df['Frequency_GHz'].values
        )
        
        # Analyze noisy data
        noisy_metrics = NoiseAnalyzer.analyze_noise(
            noisy_df['Dk'].values, noisy_df['Df'].values, noisy_df['Frequency_GHz'].values
        )
        
        # Noisy data should have higher noise scores
        assert noisy_metrics['combined_noise_score'] > clean_metrics['combined_noise_score']
        assert noisy_metrics['dk_roughness'] > clean_metrics['dk_roughness']
        assert noisy_metrics['df_roughness'] > clean_metrics['df_roughness']
        
        # Check that all metrics are present
        expected_keys = ['dk_roughness', 'df_roughness', 'dk_snr', 'df_snr', 
                        'dk_derivative_var', 'df_derivative_var', 'dk_high_freq', 
                        'df_high_freq', 'combined_noise_score']
        
        for key in expected_keys:
            assert key in clean_metrics
            assert key in noisy_metrics


class TestSmoothingAlgorithms:
    """Test smoothing algorithms"""
    
    def test_moving_average(self):
        """Test moving average smoothing"""
        # Create noisy step function
        data = np.concatenate([np.ones(25), 2*np.ones(25)]) + 0.1*np.random.randn(50)
        smoothed = SmoothingAlgorithms.moving_average(data, window_size=5)
        
        # Smoothed data should be less noisy
        assert np.var(np.diff(smoothed)) < np.var(np.diff(data))
        assert len(smoothed) == len(data)
    
    def test_savitzky_golay(self):
        """Test Savitzky-Golay filter"""
        # Create polynomial signal with noise
        x = np.linspace(0, 10, 100)
        data = x**2 + 0.5*x + 1 + 0.2*np.random.randn(100)
        smoothed = SmoothingAlgorithms.savitzky_golay(data, window_size=11, poly_order=2)
        
        # Should preserve polynomial trends better than moving average
        assert len(smoothed) == len(data)
        assert np.var(smoothed) > 0  # Should not be completely flat
    
    def test_gaussian_filter(self):
        """Test Gaussian filter"""
        data = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.1*np.random.randn(100)
        smoothed = SmoothingAlgorithms.gaussian_filter(data, sigma=1.0)
        
        assert len(smoothed) == len(data)
        assert np.var(np.diff(smoothed)) < np.var(np.diff(data))
    
    def test_median_filter(self):
        """Test median filter for outlier removal"""
        data = np.sin(np.linspace(0, 4*np.pi, 100))
        # Add some outliers
        data[20] = 10
        data[50] = -10
        data[80] = 15
        
        smoothed = SmoothingAlgorithms.median_filter(data, window_size=5)
        
        # Outliers should be reduced
        assert np.max(smoothed) < np.max(data)
        assert np.min(smoothed) > np.min(data)
        assert len(smoothed) == len(data)


class TestSmoothingSelector:
    """Test automatic smoothing algorithm selection"""
    
    def test_algorithm_selection_clean_data(self, clean_debye_data):
        """Test that clean data gets minimal or no smoothing"""
        clean_df, _ = clean_debye_data
        selector = SmoothinSelector()
        
        dk_data = clean_df['Dk'].values
        frequency = clean_df['Frequency_GHz'].values
        
        # Analyze noise (should be low)
        noise_metrics = NoiseAnalyzer.analyze_noise(
            dk_data, clean_df['Df'].values, frequency
        )
        
        # Select algorithm
        alg_name, alg_func, params = selector.select_best_algorithm(
            dk_data, frequency, noise_metrics
        )
        
        # Should select a gentle smoothing algorithm for clean data
        assert alg_name in selector.algorithms.keys()
        assert callable(alg_func)
        assert isinstance(params, dict)
    
    def test_algorithm_selection_noisy_data(self, noisy_debye_data):
        """Test that noisy data gets appropriate smoothing"""
        noisy_df, _ = noisy_debye_data
        selector = SmoothinSelector()
        
        dk_data = noisy_df['Dk'].values
        frequency = noisy_df['Frequency_GHz'].values
        
        # Analyze noise (should be higher)
        noise_metrics = NoiseAnalyzer.analyze_noise(
            dk_data, noisy_df['Df'].values, frequency
        )
        
        # Select algorithm
        alg_name, alg_func, params = selector.select_best_algorithm(
            dk_data, frequency, noise_metrics
        )
        
        # Should select appropriate smoothing
        assert alg_name in selector.algorithms.keys()
        assert callable(alg_func)
        
        # Apply smoothing and check it reduces noise
        smoothed = alg_func(dk_data, **params)
        assert NoiseAnalyzer.calculate_roughness(smoothed) < NoiseAnalyzer.calculate_roughness(dk_data)
    
    def test_smoothing_quality_evaluation(self):
        """Test smoothing quality evaluation"""
        selector = SmoothinSelector()
        
        # Create test data
        original = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.1*np.random.randn(100)
        good_smooth = SmoothingAlgorithms.gaussian_filter(original, sigma=0.5)
        over_smooth = SmoothingAlgorithms.gaussian_filter(original, sigma=5.0)
        
        # Evaluate quality
        good_score = selector.evaluate_smoothing_quality(original, good_smooth, 0.3)
        over_score = selector.evaluate_smoothing_quality(original, over_smooth, 0.3)
        
        # Good smoothing should score higher than over-smoothing
        assert good_score > over_score
        assert good_score >= 0
        assert over_score >= 0


class TestDielectricPreprocessor:
    """Test main preprocessing functionality"""
    
    def test_preprocessing_clean_data(self, clean_debye_data):
        """Test preprocessing on clean data"""
        clean_df, expected_params = clean_debye_data
        preprocessor = DielectricPreprocessor()
        
        # Preprocess data
        processed_df, info = preprocessor.preprocess(clean_df, apply_smoothing=True)
        
        # Should detect low noise and apply minimal smoothing
        assert info['noise_metrics']['combined_noise_score'] < 0.5
        assert 'good' in ' '.join(info.get('recommendations', [])).lower()
        
        # Data should be preserved well
        assert len(processed_df) == len(clean_df)
        assert list(processed_df.columns) == list(clean_df.columns)
        
        # Values should be close to original for clean data
        np.testing.assert_allclose(
            processed_df['Dk'].values, clean_df['Dk'].values, rtol=0.1
        )
    
    def test_preprocessing_noisy_data(self, noisy_debye_data):
        """Test preprocessing on noisy data"""
        noisy_df, expected_params = noisy_debye_data
        preprocessor = DielectricPreprocessor()
        
        # Preprocess data
        processed_df, info = preprocessor.preprocess(noisy_df, apply_smoothing=True)
        
        # Should detect noise and apply smoothing
        assert info['noise_metrics']['combined_noise_score'] > 0.1
        
        if info['smoothing_applied']:
            assert info['dk_algorithm'] is not None
            assert info['df_algorithm'] is not None
            
            # Check that smoothing reduced noise
            original_roughness = NoiseAnalyzer.calculate_roughness(noisy_df['Dk'].values)
            processed_roughness = NoiseAnalyzer.calculate_roughness(processed_df['Dk'].values)
            assert processed_roughness <= original_roughness
    
    def test_preprocessing_different_noise_levels(self, frequency_range, synthetic_data_generator, noise_levels):
        """Test preprocessing across different noise levels"""
        preprocessor = DielectricPreprocessor()
        
        for noise_level in noise_levels:
            # Generate data with specific noise level
            dk, df = synthetic_data_generator.generate_debye_data(
                frequency_range, noise_level=noise_level
            )
            
            test_df = pd.DataFrame({
                'Frequency_GHz': frequency_range,
                'Dk': dk,
                'Df': df
            })
            
            # Preprocess
            processed_df, info = preprocessor.preprocess(test_df, apply_smoothing=True)
            
            # Higher noise should result in higher noise scores
            noise_score = info['noise_metrics']['combined_noise_score']
            
            if noise_level == 0.0:
                assert noise_score < 0.3  # Clean data should have low noise score
            elif noise_level >= 0.2:
                assert noise_score > 0.3  # High noise should be detected
            
            # Verify output format
            assert len(processed_df) == len(test_df)
            assert list(processed_df.columns) == ['Frequency_GHz', 'Dk', 'Df']
    
    def test_preprocessing_edge_cases(self):
        """Test preprocessing edge cases"""
        preprocessor = DielectricPreprocessor()
        
        # Empty dataframe
        empty_df = pd.DataFrame()
        processed, info = preprocessor.preprocess(empty_df)
        assert info['status'] == 'insufficient_data'
        
        # Very small dataframe
        small_df = pd.DataFrame({
            'Frequency_GHz': [1.0, 2.0],
            'Dk': [4.0, 3.9],
            'Df': [0.1, 0.11]
        })
        processed, info = preprocessor.preprocess(small_df)
        assert info['status'] == 'insufficient_data'
        
        # Single point
        single_df = pd.DataFrame({
            'Frequency_GHz': [1.0],
            'Dk': [4.0],
            'Df': [0.1]
        })
        processed, info = preprocessor.preprocess(single_df)
        assert info['status'] == 'insufficient_data'
    
    def test_preprocessing_without_smoothing(self, noisy_debye_data):
        """Test preprocessing with smoothing disabled"""
        noisy_df, _ = noisy_debye_data
        preprocessor = DielectricPreprocessor()
        
        # Preprocess without smoothing
        processed_df, info = preprocessor.preprocess(noisy_df, apply_smoothing=False)
        
        # Should not apply smoothing even if noise is detected
        assert not info['smoothing_applied']
        assert info['dk_algorithm'] is None
        assert info['df_algorithm'] is None
        
        # Data should be unchanged
        pd.testing.assert_frame_equal(processed_df, noisy_df)
    
    @pytest.mark.parametrize("model_type", ["debye", "cole_cole", "havriliak_negami"])
    def test_preprocessing_different_models(self, frequency_range, synthetic_data_generator, model_type):
        """Test preprocessing on different dielectric models"""
        preprocessor = DielectricPreprocessor()
        
        # Generate data based on model type
        if model_type == "debye":
            dk, df = synthetic_data_generator.generate_debye_data(
                frequency_range, noise_level=0.1
            )
        elif model_type == "cole_cole":
            dk, df = synthetic_data_generator.generate_cole_cole_data(
                frequency_range, noise_level=0.1
            )
        elif model_type == "havriliak_negami":
            dk, df = synthetic_data_generator.generate_havriliak_negami_data(
                frequency_range, noise_level=0.1
            )
        
        test_df = pd.DataFrame({
            'Frequency_GHz': frequency_range,
            'Dk': dk,
            'Df': df
        })
        
        # Preprocess
        processed_df, info = preprocessor.preprocess(test_df, apply_smoothing=True)
        
        # Basic checks
        assert len(processed_df) == len(test_df)
        assert info['noise_metrics']['combined_noise_score'] > 0
        
        # Should work regardless of underlying dielectric model
        assert not processed_df.isnull().any().any()
        assert all(processed_df['Dk'] > 0)  # Physical constraint
        assert all(processed_df['Df'] >= 0)  # Physical constraint