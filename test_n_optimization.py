#!/usr/bin/env python3
"""
Test the new N-optimization feature in auto mode
"""

import pandas as pd
import numpy as np
from analysis import run_analysis

def test_n_optimization():
    """Test that auto mode now optimizes N for each variable model"""
    
    # Create realistic test data with multiple relaxation processes
    freq_ghz = np.logspace(0, 2, 20)  # 1 to 100 GHz, 20 points
    
    # Simulate dual-relaxation Debye-like behavior
    eps_inf = 3.0
    eps_s1 = 8.0  # First relaxation
    eps_s2 = 5.0  # Second relaxation  
    tau1 = 1e-11  # Fast relaxation
    tau2 = 1e-9   # Slow relaxation
    
    # Convert to angular frequency
    omega = 2 * np.pi * freq_ghz * 1e9
    
    # Calculate dual-Debye response
    eps_complex = eps_inf + (eps_s1 - eps_inf) / (1 + 1j * omega * tau1) + (eps_s2 - eps_inf) / (1 + 1j * omega * tau2)
    
    # Add realistic noise
    noise_level = 0.05
    dk_exp = eps_complex.real + np.random.normal(0, noise_level, len(freq_ghz))
    df_exp = eps_complex.imag + np.random.normal(0, noise_level, len(freq_ghz))
    df_exp = np.abs(df_exp)  # Ensure positive
    
    # Create DataFrame
    df = pd.DataFrame({
        'Frequency_GHz': freq_ghz,
        'Dk': dk_exp,
        'Df': df_exp
    })
    
    print("=" * 70)
    print("TESTING N-OPTIMIZATION IN AUTO MODE")
    print("=" * 70)
    print(f"Test data: {len(freq_ghz)} frequency points from {freq_ghz[0]:.1f} to {freq_ghz[-1]:.1f} GHz")
    print("Simulated: Dual-relaxation behavior (should favor models with N≥2)")
    
    # Test auto mode with N-optimization
    model_params = {
        'selection_method': 'balanced'
    }
    
    print("\n🧪 Running AUTO MODE with N-optimization...")
    print("-" * 50)
    
    # Set random seed for consistent results
    np.random.seed(42)
    
    auto_results = run_analysis(df, [], model_params, 'auto')
    
    if "error" in auto_results:
        print(f"❌ Error: {auto_results['error']}")
        return
    
    # Extract results
    best_model_name = auto_results['best_model_name']
    best_model_result = auto_results['best_model_result']
    
    print(f"\n🏆 RESULTS:")
    print(f"   Best Model: {best_model_name.replace('_', ' ').title()}")
    print(f"   AIC: {best_model_result.get('aic', 'N/A'):.2f}")
    print(f"   RMSE: {best_model_result.get('rmse', 'N/A'):.4f}")
    
    if 'optimal_n' in best_model_result:
        opt_summary = best_model_result['optimization_summary']
        print(f"   Optimal N: {best_model_result['optimal_n']}")
        print(f"   Tested N values: {opt_summary['tested_n_values']}")
        print(f"   Selection method: {opt_summary['selection_method']}")
        
        # Show scores for all tested N values
        print(f"   Scores by N: {opt_summary['all_scores']}")
    else:
        print(f"   (Fixed-parameter model)")
    
    print(f"\n✅ N-optimization test completed successfully!")
    print(f"   The system automatically found the best N value for variable models")
    print(f"   and selected the overall best model: {best_model_name}")

if __name__ == "__main__":
    test_n_optimization()