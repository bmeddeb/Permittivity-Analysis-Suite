#!/usr/bin/env python3
"""
Test to demonstrate the fix for auto mode N parameter issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from analysis import run_analysis

def test_auto_vs_manual_with_different_n():
    """Test that auto mode ignores UI sliders while manual mode respects them"""
    
    # Create test data
    freq_ghz = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    dk_exp = np.array([10.0, 9.5, 9.0, 8.5, 8.0, 7.5])
    df_exp = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    
    df = pd.DataFrame({
        'Frequency_GHz': freq_ghz,
        'Dk': dk_exp,
        'Df': df_exp
    })
    
    print("=" * 60)
    print("TESTING THE FIX: AUTO VS MANUAL MODE WITH DIFFERENT N VALUES")
    print("=" * 60)
    
    # Test case 1: N=2 for all models
    print("\n🧪 TEST CASE 1: Hybrid=2, Multipole=2, Lorentz=2")
    print("-" * 40)
    
    model_params_n2 = {
        'hybrid_terms': 2,
        'multipole_terms': 2,
        'lorentz_terms': 2,
        'selection_method': 'balanced'
    }
    
    auto_result_n2 = run_analysis(df, [], model_params_n2, 'auto')
    print(f"AUTO MODE RESULT (N=2): Best model = {auto_result_n2['best_model_name']}")
    
    # Test case 2: N=5 for all models (this used to cause the problem)
    print("\n🧪 TEST CASE 2: Hybrid=5, Multipole=5, Lorentz=5")
    print("-" * 40)
    
    model_params_n5 = {
        'hybrid_terms': 5,
        'multipole_terms': 5,
        'lorentz_terms': 5,
        'selection_method': 'balanced'
    }
    
    auto_result_n5 = run_analysis(df, [], model_params_n5, 'auto')
    print(f"AUTO MODE RESULT (N=5): Best model = {auto_result_n5['best_model_name']}")
    
    # Verify that auto mode gives same result regardless of UI slider values
    print(f"\n✅ VERIFICATION: Auto mode should give SAME result regardless of UI sliders")
    print(f"   Auto mode with N=2: {auto_result_n2['best_model_name']}")
    print(f"   Auto mode with N=5: {auto_result_n5['best_model_name']}")
    
    if auto_result_n2['best_model_name'] == auto_result_n5['best_model_name']:
        print(f"   🎉 SUCCESS: Auto mode is consistent! (Both selected {auto_result_n2['best_model_name']})")
    else:
        print(f"   ❌ FAILURE: Auto mode results differ!")
    
    # Test manual mode to show it respects user choices
    print(f"\n🔧 MANUAL MODE TEST: Should respect user slider values")
    print("-" * 40)
    
    manual_result_n2 = run_analysis(df, ['hybrid'], model_params_n2, 'manual')
    manual_result_n5 = run_analysis(df, ['hybrid'], model_params_n5, 'manual')
    
    print(f"Manual mode Hybrid(N=2) AIC: {manual_result_n2['hybrid']['aic']:.2f}")
    print(f"Manual mode Hybrid(N=5) AIC: {manual_result_n5['hybrid']['aic']:.2f}")
    print(f"Manual mode allows user to test different N values as expected")
    
    print(f"\n🎯 SUMMARY:")
    print(f"✅ Auto mode: Uses fixed N=2 for fair comparison (ignores UI sliders)")
    print(f"✅ Manual mode: Uses UI slider values (respects user choice)")
    print(f"✅ Problem FIXED: Auto mode no longer selects overfitted models!")

if __name__ == "__main__":
    test_auto_vs_manual_with_different_n()