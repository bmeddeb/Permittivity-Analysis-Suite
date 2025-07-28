#!/usr/bin/env python3
"""
Test suite for dielectric models using synthetic data with known parameters
This validates that models can recover their own parameters from synthetic data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from analysis import run_analysis
from models.debye_model import DebyeModel
from models.cole_cole_model import ColeColeModel
from models.havriliak_negami_model import HavriliakNegamiModel
from models.multipole_debye_model import MultiPoleDebyeModel
from models.lorentz_model import LorentzModel
from models.hybrid_model import HybridModel
from models.sarkar_model import SarkarModel


class TestModelParameterRecovery:
    """Test that models can recover known parameters from synthetic data"""
    
    def test_debye_parameter_recovery(self, clean_debye_data):
        """Test Debye model parameter recovery from synthetic data"""
        data_df, expected_params = clean_debye_data
        
        # Initialize and analyze with Debye model
        model = DebyeModel()
        result = model.analyze(data_df)
        
        # Check that fitting was successful
        assert result is not None
        assert result.get('success', False)
        assert 'fitted_params' in result
        
        # Check parameter recovery (with reasonable tolerance)
        fitted_params = result['fitted_params']
        
        # Test parameter recovery within tolerance
        assert abs(fitted_params['eps_s'] - expected_params['eps_s']) < 0.5
        assert abs(fitted_params['eps_inf'] - expected_params['eps_inf']) < 0.5
        assert abs(fitted_params['tau'] - expected_params['tau']) < 1e-9
        
        # Check that RMSE is very low for perfect data
        assert result.get('rmse', float('inf')) < 0.01
    
    def test_cole_cole_parameter_recovery(self, clean_cole_cole_data):
        """Test Cole-Cole model parameter recovery from synthetic data"""
        data_df, expected_params = clean_cole_cole_data
        
        model = ColeColeModel()
        result = model.analyze(data_df)
        
        assert result is not None
        assert result.get('success', False)
        assert 'fitted_params' in result
        
        fitted_params = result['fitted_params']
        
        # Test parameter recovery
        assert abs(fitted_params['eps_s'] - expected_params['eps_s']) < 0.5
        assert abs(fitted_params['eps_inf'] - expected_params['eps_inf']) < 0.5
        assert abs(fitted_params['tau'] - expected_params['tau']) < 1e-9
        assert abs(fitted_params['alpha'] - expected_params['alpha']) < 0.1
        
        # Check fit quality
        assert result.get('rmse', float('inf')) < 0.01
    
    def test_havriliak_negami_parameter_recovery(self, clean_havriliak_negami_data):
        """Test Havriliak-Negami model parameter recovery"""
        data_df, expected_params = clean_havriliak_negami_data
        
        model = HavriliakNegamiModel()
        result = model.analyze(data_df)
        
        assert result is not None
        assert result.get('success', False)
        assert 'fitted_params' in result
        
        fitted_params = result['fitted_params']
        
        # Test parameter recovery (HN is more challenging, so slightly looser tolerances)
        assert abs(fitted_params['eps_s'] - expected_params['eps_s']) < 1.0
        assert abs(fitted_params['eps_inf'] - expected_params['eps_inf']) < 1.0
        assert abs(fitted_params['tau'] - expected_params['tau']) < 5e-9
        assert abs(fitted_params['alpha'] - expected_params['alpha']) < 0.15
        assert abs(fitted_params['beta'] - expected_params['beta']) < 0.15
        
        # Check fit quality
        assert result.get('rmse', float('inf')) < 0.05
    
    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_multipole_debye_parameter_recovery(self, frequency_range, synthetic_data_generator, N):
        """Test Multi-pole Debye model with different numbers of poles"""
        # Generate synthetic multi-pole data
        # For simplicity, use single Debye when N=1
        if N == 1:
            dk, df = synthetic_data_generator.generate_debye_data(
                frequency_range, eps_s=10.0, eps_inf=2.0, tau=1e-9
            )
        else:
            # For N>1, generate superposition of Debye processes
            dk = np.zeros_like(frequency_range)
            df = np.zeros_like(frequency_range)
            
            # Add multiple Debye processes with different time constants
            for i in range(N):
                tau_i = 1e-9 * (10**i)  # Different time constants
                eps_s_i = 8.0 + i  # Different strengths
                dk_i, df_i = synthetic_data_generator.generate_debye_data(
                    frequency_range, eps_s=eps_s_i, eps_inf=2.0, tau=tau_i
                )
                dk += (dk_i - 2.0)  # Remove baseline
                df += df_i
            
            dk += 2.0  # Add back baseline
        
        data_df = pd.DataFrame({
            'Frequency_GHz': frequency_range,
            'Dk': dk,
            'Df': df
        })
        
        # Test the model
        model = MultiPoleDebyeModel(N=N)
        result = model.analyze(data_df)
        
        assert result is not None
        assert result.get('success', False)
        assert 'fitted_params' in result
        
        # Check that we get N poles
        fitted_params = result['fitted_params']
        assert len([k for k in fitted_params.keys() if k.startswith('tau_')]) == N
        assert len([k for k in fitted_params.keys() if k.startswith('Delta_eps_')]) == N
        
        # Check reasonable fit quality
        assert result.get('rmse', float('inf')) < 0.5  # More lenient for complex model
    
    def test_lorentz_parameter_recovery(self, frequency_range, synthetic_data_generator):
        """Test Lorentz model parameter recovery"""
        # Generate synthetic Lorentz data
        dk, df = synthetic_data_generator.generate_lorentz_data(
            frequency_range, eps_inf=2.0, f0=10.0, gamma=2.0, delta_eps=5.0
        )
        
        data_df = pd.DataFrame({
            'Frequency_GHz': frequency_range,
            'Dk': dk,
            'Df': df
        })
        
        model = LorentzModel(N=1)
        result = model.analyze(data_df)
        
        assert result is not None
        assert result.get('success', False)
        assert 'fitted_params' in result
        
        fitted_params = result['fitted_params']
        
        # Check parameter recovery (Lorentz can be challenging)
        assert abs(fitted_params['eps_inf'] - 2.0) < 1.0
        assert abs(fitted_params['f0_1'] - 10.0) < 5.0  # Frequency in GHz
        assert abs(fitted_params['gamma_1'] - 2.0) < 2.0
        
        # Check fit quality
        assert result.get('rmse', float('inf')) < 0.1


class TestModelRobustness:
    """Test model robustness with noisy data"""
    
    def test_debye_with_noise(self, noisy_debye_data):
        """Test Debye model performance with noisy data"""
        data_df, expected_params = noisy_debye_data
        
        model = DebyeModel()
        result = model.analyze(data_df)
        
        assert result is not None
        assert result.get('success', False)
        
        # Parameters should be close but not perfect due to noise
        fitted_params = result['fitted_params']
        assert abs(fitted_params['eps_s'] - expected_params['eps_s']) < 2.0  # More tolerant
        assert abs(fitted_params['eps_inf'] - expected_params['eps_inf']) < 2.0
        
        # RMSE should be reasonable but not perfect
        assert result.get('rmse', float('inf')) < 1.0
    
    @pytest.mark.parametrize("noise_level", [0.05, 0.1, 0.2])
    def test_model_noise_tolerance(self, frequency_range, synthetic_data_generator, noise_level):
        """Test how models handle different noise levels"""
        # Generate noisy Debye data
        dk, df = synthetic_data_generator.generate_debye_data(
            frequency_range, eps_s=10.0, eps_inf=2.0, tau=1e-9, noise_level=noise_level
        )
        
        data_df = pd.DataFrame({
            'Frequency_GHz': frequency_range,
            'Dk': dk,  
            'Df': df
        })
        
        model = DebyeModel()
        result = model.analyze(data_df)
        
        assert result is not None
        assert result.get('success', False)
        
        # RMSE should increase with noise level
        rmse = result.get('rmse', float('inf'))
        assert rmse < 2.0 * noise_level + 0.1  # Reasonable scaling with noise
        
        # Parameters should still be in reasonable range
        fitted_params = result['fitted_params']
        assert 5.0 < fitted_params['eps_s'] < 15.0
        assert 1.0 < fitted_params['eps_inf'] < 5.0
        assert 1e-11 < fitted_params['tau'] < 1e-7


class TestAutomaticModelSelection:
    """Test the automatic model selection functionality"""
    
    def test_auto_selection_debye_data(self, clean_debye_data):
        """Test that auto-selection correctly identifies Debye data"""
        data_df, expected_params = clean_debye_data
        
        # Run analysis in auto mode
        model_params = {'selection_method': 'balanced'}
        results = run_analysis(data_df, [], model_params, 'auto')
        
        assert 'error' not in results
        assert 'best_model_name' in results
        assert 'best_model_result' in results
        
        # Should select Debye or a simple model for Debye data
        best_model = results['best_model_name']
        assert best_model in ['debye', 'cole_cole', 'multipole_debye']  # Simple models
        
        # Should have good fit quality
        best_result = results['best_model_result']
        assert best_result.get('rmse', float('inf')) < 0.1
    
    def test_auto_selection_cole_cole_data(self, clean_cole_cole_data):
        """Test that auto-selection works with Cole-Cole data"""
        data_df, expected_params = clean_cole_cole_data
        
        model_params = {'selection_method': 'balanced'}
        results = run_analysis(data_df, [], model_params, 'auto')
        
        assert 'error' not in results
        assert 'best_model_name' in results
        
        # Should achieve good fit
        best_result = results['best_model_result']
        assert best_result.get('rmse', float('inf')) < 0.1
    
    def test_auto_compare_mode(self, clean_debye_data):
        """Test auto-compare mode returns comprehensive results"""
        data_df, expected_params = clean_debye_data
        
        model_params = {'selection_method': 'balanced'}
        results = run_analysis(data_df, [], model_params, 'auto_compare')
        
        assert 'error' not in results
        assert 'enhanced_comparison' in results
        assert 'valid_results' in results
        
        comparison = results['enhanced_comparison']
        assert 'best_model' in comparison
        assert 'comparison_data' in comparison
        
        # Should have multiple models in comparison
        valid_results = results['valid_results']
        assert len(valid_results) >= 2
        
        # All valid results should have reasonable RMSE
        for model_name, result in valid_results.items():
            if result is not None and 'rmse' in result:
                assert result['rmse'] < 1.0  # Should fit clean data well
    
    @pytest.mark.parametrize("selection_method", ["balanced", "aic_focused", "rmse_focused"])
    def test_selection_methods(self, clean_debye_data, selection_method):
        """Test different selection methods"""
        data_df, expected_params = clean_debye_data
        
        model_params = {'selection_method': selection_method}
        results = run_analysis(data_df, [], model_params, 'auto')
        
        assert 'error' not in results
        assert 'best_model_name' in results
        assert 'selection_rationale' in results
        
        # Should provide rationale for selection
        rationale = results['selection_rationale']
        assert len(rationale) > 10  # Should be descriptive
        
        # Method name should appear in rationale
        assert selection_method.replace('_', ' ').lower() in rationale.lower()


class TestModelConstraints:
    """Test that models respect physical constraints"""
    
    def test_physical_parameter_bounds(self, clean_debye_data):
        """Test that fitted parameters respect physical bounds"""
        data_df, expected_params = clean_debye_data
        
        # Test multiple models
        models = [
            DebyeModel(),
            ColeColeModel(),
            HavriliakNegamiModel()
        ]
        
        for model in models:
            result = model.analyze(data_df)
            
            if result is not None and result.get('success', False):
                params = result['fitted_params']
                
                # Physical constraints
                if 'eps_s' in params:
                    assert params['eps_s'] > 0  # Permittivity must be positive
                if 'eps_inf' in params:
                    assert params['eps_inf'] > 0
                    if 'eps_s' in params:
                        assert params['eps_inf'] < params['eps_s']  # eps_inf < eps_s
                
                if 'tau' in params:
                    assert params['tau'] > 0  # Relaxation time must be positive
                
                if 'alpha' in params:
                    assert 0 < params['alpha'] <= 1  # Cole-Cole alpha constraint
                
                if 'beta' in params:
                    assert 0 < params['beta'] <= 1  # Havriliak-Negami beta constraint
    
    def test_model_convergence(self, clean_debye_data):
        """Test that models converge consistently"""
        data_df, expected_params = clean_debye_data
        
        model = DebyeModel()
        
        # Run multiple times to check consistency
        results = []
        for _ in range(5):
            result = model.analyze(data_df)
            if result is not None and result.get('success', False):
                results.append(result)
        
        assert len(results) >= 3  # Most runs should succeed
        
        # Parameters should be consistent across runs
        if len(results) >= 2:
            params1 = results[0]['fitted_params']
            params2 = results[1]['fitted_params']
            
            for key in params1.keys():
                if key in params2:
                    # Should be very similar (within 1%)
                    relative_diff = abs(params1[key] - params2[key]) / (abs(params1[key]) + 1e-10)
                    assert relative_diff < 0.01
    
    def test_edge_case_data(self, frequency_range):
        """Test models with edge case data"""
        # Very flat data (low loss)
        flat_df = pd.DataFrame({
            'Frequency_GHz': frequency_range,
            'Dk': np.full_like(frequency_range, 4.0),
            'Df': np.full_like(frequency_range, 0.001)
        })
        
        model = DebyeModel()
        result = model.analyze(flat_df)
        
        # Should handle flat data gracefully
        assert result is not None
        # May not converge perfectly, but shouldn't crash
        
        # Very high loss data
        high_loss_df = pd.DataFrame({
            'Frequency_GHz': frequency_range,
            'Dk': np.linspace(10, 8, len(frequency_range)),
            'Df': np.linspace(0.1, 2.0, len(frequency_range))
        })
        
        result = model.analyze(high_loss_df)
        assert result is not None  # Shouldn't crash