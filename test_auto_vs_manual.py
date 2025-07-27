#!/usr/bin/env python3
"""
Test to compare auto mode N-optimization vs manual mode results
"""

import pandas as pd
import numpy as np
from analysis import run_analysis

def create_complex_test_data():
    """Create data that should favor a complex hybrid model"""
    freq_ghz = np.logspace(0, 2, 30)  # 1 to 100 GHz, 30 points
    
    # Create complex behavior that should require higher N
    # Multiple relaxation processes + conductivity
    eps_inf = 3.0
    
    # Three relaxation processes at different frequencies
    eps_components = []
    delta_eps_list = [5.0, 3.0, 2.0]
    tau_list = [1e-11, 1e-10, 1e-9]  # Different time scales
    
    omega = 2 * np.pi * freq_ghz * 1e9
    
    eps_complex = np.full_like(freq_ghz, eps_inf, dtype=complex)
    
    for delta_eps, tau in zip(delta_eps_list, tau_list):
        eps_complex += delta_eps / (1 + 1j * omega * tau)
    
    # Add some conductivity effects
    sigma = 0.1  # S/m
    eps_0 = 8.854e-12  # F/m
    eps_complex += 1j * sigma / (omega * eps_0)
    
    # Add realistic noise
    noise_level = 0.03
    dk_exp = eps_complex.real + np.random.normal(0, noise_level * np.mean(np.abs(eps_complex.real)), len(freq_ghz))
    df_exp = eps_complex.imag + np.random.normal(0, noise_level * np.mean(np.abs(eps_complex.imag)), len(freq_ghz))
    df_exp = np.abs(df_exp)  # Ensure positive
    
    return pd.DataFrame({
        'Frequency_GHz': freq_ghz,
        'Dk': dk_exp,
        'Df': df_exp
    })

def test_manual_mode_hybrid():
    """Test manual mode with different N values for Hybrid"""
    df = create_complex_test_data()
    
    print("=" * 60)
    print("MANUAL MODE: Testing Hybrid with different N values")
    print("=" * 60)
    
    best_rmse = float('inf')
    best_n = None
    
    for n in range(1, 6):
        print(f"\n--- Testing Hybrid N={n} in Manual Mode ---")
        
        model_params = {
            'hybrid_terms': n,
            'multipole_terms': 2,
            'lorentz_terms': 2
        }
        
        results = run_analysis(df, ['hybrid'], model_params, 'manual')
        
        if results.get('hybrid') is not None:
            hybrid_result = results['hybrid']
            rmse = hybrid_result.get('rmse', float('inf'))
            aic = hybrid_result.get('aic', float('inf'))
            success = hybrid_result.get('success', False)
            
            print(f"Hybrid N={n}: AIC={aic:.2f}, RMSE={rmse:.4f}, Success={success}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_n = n
        else:
            print(f"Hybrid N={n}: Failed")
    
    print(f"\n🏆 Manual Mode Best: Hybrid N={best_n} with RMSE={best_rmse:.4f}")
    return best_n, best_rmse

def test_auto_mode():
    """Test auto mode with N-optimization"""
    df = create_complex_test_data()
    
    print(f"\n" + "=" * 60)
    print("AUTO MODE: N-optimization should find the best configuration")
    print("=" * 60)
    
    model_params = {
        'selection_method': 'balanced'
    }
    
    results = run_analysis(df, [], model_params, 'auto')
    
    if "error" in results:
        print(f"❌ Auto mode error: {results['error']}")
        return None, None, None
    
    best_model_name = results['best_model_name']
    best_model_result = results['best_model_result']
    
    print(f"\n🏆 Auto Mode Selected: {best_model_name}")
    print(f"   AIC: {best_model_result.get('aic', 'N/A'):.2f}")
    print(f"   RMSE: {best_model_result.get('rmse', 'N/A'):.4f}")
    print(f"   Success: {best_model_result.get('success', 'Unknown')}")
    
    if 'optimal_n' in best_model_result:
        print(f"   Optimal N: {best_model_result['optimal_n']}")
        print(f"   Tested N values: {best_model_result['optimization_summary']['tested_n_values']}")
    
    return best_model_name, best_model_result.get('rmse', float('inf')), best_model_result.get('optimal_n')

def main():
    """Compare manual vs auto mode results"""
    print("COMPARING MANUAL VS AUTO MODE N-OPTIMIZATION")
    print("=" * 70)
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Test manual mode
    manual_best_n, manual_best_rmse = test_manual_mode_hybrid()
    
    # Test auto mode  
    auto_best_model, auto_best_rmse, auto_optimal_n = test_auto_mode()
    
    # Compare results
    print(f"\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"Manual Mode Best:  Hybrid N={manual_best_n}, RMSE={manual_best_rmse:.4f}")
    print(f"Auto Mode Best:    {auto_best_model} N={auto_optimal_n}, RMSE={auto_best_rmse:.4f}")
    
    if auto_best_model == 'hybrid':
        if auto_optimal_n == manual_best_n:
            print(f"✅ SUCCESS: Auto mode found the same optimal N as manual mode!")
        else:
            print(f"⚠️  DIFFERENCE: Auto found N={auto_optimal_n}, manual found N={manual_best_n}")
            
        if abs(auto_best_rmse - manual_best_rmse) < 0.001:
            print(f"✅ RMSE Match: Both modes achieved similar RMSE")
        else:
            print(f"⚠️  RMSE Difference: {abs(auto_best_rmse - manual_best_rmse):.4f}")
    else:
        print(f"ℹ️  Auto mode selected a different model type: {auto_best_model}")
        if auto_best_rmse < manual_best_rmse:
            print(f"✅ Auto mode found an even better solution!")
        else:
            print(f"⚠️  Manual mode hybrid was better - auto mode may have missed it")

if __name__ == "__main__":
    main()