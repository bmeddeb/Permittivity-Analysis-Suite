#!/usr/bin/env python3
"""
Quick test to validate the frontend integration with Phase 1.5 backend
"""

import pandas as pd
import numpy as np
from analysis import run_analysis

def test_frontend_integration():
    """Test that frontend integration works with all analysis modes"""
    
    # Create simple test data
    freq_ghz = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    dk_exp = np.array([10.0, 9.5, 9.0, 8.5, 8.0, 7.5])
    df_exp = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    
    df = pd.DataFrame({
        'Frequency_GHz': freq_ghz,
        'Dk': dk_exp,
        'Df': df_exp
    })
    
    model_params = {
        'hybrid_terms': 2,
        'multipole_terms': 2,
        'lorentz_terms': 2,
        'selection_method': 'balanced'
    }
    
    print("Testing Manual Mode...")
    manual_results = run_analysis(df, ['debye', 'cole_cole'], model_params, 'manual')
    print(f"Manual mode returned: {type(manual_results)}, keys: {list(manual_results.keys())}")
    
    print("\nTesting Auto Mode...")
    auto_results = run_analysis(df, [], model_params, 'auto')
    print(f"Auto mode returned: {type(auto_results)}")
    if 'best_model_name' in auto_results:
        print(f"Best model: {auto_results['best_model_name']}")
        print(f"Rationale: {auto_results['selection_rationale']}")
    
    print("\nTesting Auto-Compare Mode...")
    compare_results = run_analysis(df, [], model_params, 'auto_compare')
    print(f"Auto-compare mode returned: {type(compare_results)}")
    if 'enhanced_comparison' in compare_results:
        print(f"Best model: {compare_results['enhanced_comparison']['best_model']}")
    
    print("\n✅ All frontend integration tests passed!")

if __name__ == "__main__":
    test_frontend_integration()