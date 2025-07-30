# example_usage.py
"""
Example usage of the refactored Djordjevic-Sarkar model components.

This demonstrates the separation of concerns with the model definition,
analysis, and export utilities in separate modules.
"""
import os
import sys

# Add the project root to the Python path so we can find the app module
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

# Import the refactored components
from app.models.sarkar import SarkarModel
from app.models.sarkar_analyzer import SarkarModelAnalyzer
from app.utils.sarkar_export_utils import SarkarExportUtils


def load_sample_data() -> tuple:
    """Generate sample dielectric data for demonstration."""
    # Frequency range: 1 MHz to 10 GHz
    freq_ghz = np.logspace(-3, 1, 50)

    # Simulate D-S like behavior
    eps_inf = 3.0
    eps_s = 4.5
    f1 = 0.01  # 10 MHz
    f2 = 1.0  # 1 GHz
    sigma_dc = 1e-9

    # Generate "experimental" data with some noise
    omega = 2 * np.pi * freq_ghz * 1e9
    omega1 = 2 * np.pi * f1 * 1e9
    omega2 = 2 * np.pi * f2 * 1e9

    ln_ratio = np.log(omega2 / omega1)
    dk = eps_inf + (eps_s - eps_inf) * np.log((omega ** 2 + omega2 ** 2) /
                                              (omega ** 2 + omega1 ** 2)) / (2 * ln_ratio)

    eps_imag = sigma_dc / (omega * 8.854e-12) + \
               (eps_s - eps_inf) / ln_ratio * (np.arctan(omega / omega1) -
                                               np.arctan(omega / omega2))
    df = eps_imag / dk

    # Add some noise
    dk += np.random.normal(0, 0.01, len(dk))
    df += np.random.normal(0, 0.0001, len(df))

    return freq_ghz, dk, df


def demonstrate_basic_fit():
    """Demonstrate basic model fitting."""
    print("=== Basic Model Fitting ===\n")

    # Load data
    freq, dk, df = load_sample_data()

    # Create model instance
    model = SarkarModel(use_ghz=True)

    # Fit the model
    result = model.fit(freq, dk, df, weights='uniform')

    # Print basic fit report
    print(model.get_fit_report(result))

    # Get transition characteristics
    transition = model.get_transition_characteristics(result.params)
    print("\nTransition Characteristics:")
    for key, value in transition.items():
        print(f"  {key}: {value:.4g}")

    return model, result


def demonstrate_analysis():
    """Demonstrate model analysis capabilities."""
    print("\n\n=== Model Analysis ===\n")

    # Get model and fit result
    freq, dk, df = load_sample_data()
    model = SarkarModel(use_ghz=True)
    result = model.fit(freq, dk, df)

    # Create analyzer
    analyzer = SarkarModelAnalyzer(model)

    # 1. Data suitability
    suitability = analyzer.assess_data_suitability(freq, dk, df)
    print("Data Suitability Assessment:")
    print(f"  Suitable: {suitability['suitable']}")
    print(f"  Confidence: {suitability['confidence']:.1%}")
    if suitability['warnings']:
        print("  Warnings:")
        for warning in suitability['warnings']:
            print(f"    - {warning}")

    # 2. Parameter validation
    is_valid, issues = analyzer.validate_parameters(result.params)
    print(f"\nParameter Validation: {'PASS' if is_valid else 'FAIL'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")

    # 3. Kramers-Kronig validation
    kk_results = analyzer.validate_with_kramers_kronig(result)
    print(f"\nKramers-Kronig Validation:")
    print(f"  Causality: {kk_results['causality_status']}")
    if isinstance(kk_results.get('mean_relative_error'), (int, float)):
        print(f"  Mean Relative Error: {kk_results['mean_relative_error']:.2%}")
    else:
        print(f"  Mean Relative Error: {kk_results.get('mean_relative_error', 'N/A')}")

    # 4. Enhanced quality metrics
    quality_metrics = analyzer.calculate_enhanced_quality_metrics(result)
    print("\nQuality Metrics:")
    print(f"  R²: {quality_metrics['r_squared']:.4f}")
    print(f"  Overall Quality: {quality_metrics['overall_quality']:.3f}")
    print(f"  Causality Score: {quality_metrics['causality_score']:.3f}")

    # 5. Generate comprehensive report (skip if K-K failed)
    if kk_results['causality_status'] != 'ERROR':
        report = analyzer.generate_analysis_report(result)
        print("\nFull Analysis Report Available (truncated for display)")
        print(report[:500] + "...\n")
    else:
        print("\nSkipping full report due to K-K validation error")

    return analyzer


def demonstrate_model_comparison():
    """Demonstrate model comparison functionality."""
    print("\n\n=== Model Comparison ===\n")

    # Get data and model
    freq, dk, df = load_sample_data()
    model = SarkarModel(use_ghz=True)
    analyzer = SarkarModelAnalyzer(model)

    # Compare with Multi-Debye models
    comparison = analyzer.compare_with_models(
        freq, dk, df,
        models_to_compare=['MultiDebye-2', 'MultiDebye-4', 'MultiDebye-6']
    )

    print("Model Comparison Results:")
    print(f"  Best Model: {comparison['best_model']}")
    print(f"  Recommendation: {comparison['recommendation']}")

    # Show BIC values
    print("\nBIC Values:")
    print(f"  Sarkar: {comparison['sarkar_model']['bic']:.2f}")
    for model_name, data in comparison['other_models'].items():
        if 'bic' in data:
            print(f"  {model_name}: {data['bic']:.2f}")

    return comparison


def demonstrate_export_utilities():
    """Demonstrate export functionality."""
    print("\n\n=== Export Utilities ===\n")

    # Get fitted model
    freq, dk, df = load_sample_data()
    model = SarkarModel(use_ghz=True)
    result = model.fit(freq, dk, df)

    # Create export utils
    exporter = SarkarExportUtils(model)

    # 1. SPICE model generation
    spice_behavioral = exporter.generate_spice_model(result.params, format='behavioral')
    print("SPICE Behavioral Model Generated:")
    for comment in spice_behavioral['comments']:
        print(f"  {comment}")

    spice_ladder = exporter.generate_spice_model(result.params, format='ladder')
    print(f"\nSPICE Ladder Model: {spice_ladder['ladder']['n_sections']} sections")

    # 2. Touchstone S-parameters
    freq_hz = np.logspace(6, 10, 100)  # 1 MHz to 10 GHz
    s_params = exporter.to_touchstone(result.params, freq_hz, thickness_m=1e-3)

    print("\nTouchstone S-Parameters Generated:")
    print(f"  Frequency points: {len(freq_hz)}")
    print(f"  Attenuation at 1 GHz: {s_params['attenuation_dB_m'][50]:.2f} dB/m")

    # Save to file (commented out for example)
    # exporter.generate_touchstone_file(
    #     result.params, freq_hz, "ds_model.s2p", thickness_m=1e-3
    # )

    # 3. Extrapolation with uncertainty
    freq_extrap = np.logspace(-4, 2, 100)  # 100 kHz to 100 GHz
    extrap_data = exporter.extrapolate_with_uncertainty(
        result.params, freq_extrap, freq, confidence=0.95
    )

    print("\nExtrapolation with Uncertainty:")
    print(f"  Frequency range: {freq_extrap[0]:.3e} to {freq_extrap[-1]:.3e} GHz")
    if 'warning' in extrap_data:
        print(f"  Warning: {extrap_data['warning']}")

    # 4. Plot data generation
    plot_data = exporter.generate_plot_data(result, n_points=500, include_components=True)
    print("\nPlot Data Generated with:")
    print(f"  {len(plot_data['freq'])} frequency points")
    print(f"  Component breakdown included")
    print(f"  Uncertainty bands included")

    return exporter, plot_data


def demonstrate_plotting(plot_data: Dict[str, Any]):
    """Demonstrate plotting with generated data."""
    print("\n\n=== Visualization ===\n")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Dk vs frequency with fit and uncertainty
    ax1.semilogx(plot_data['freq_exp_ghz'], plot_data['dk_exp'], 'o',
                 label='Experimental', markersize=6)
    ax1.semilogx(plot_data['freq_ghz'], plot_data['dk_fit'], '-',
                 label='D-S Fit', linewidth=2)
    ax1.fill_between(plot_data['freq_ghz'],
                     plot_data['dk_uncertainty_band']['lower'],
                     plot_data['dk_uncertainty_band']['upper'],
                     alpha=0.3, label='95% Confidence')
    ax1.axvline(plot_data['f1'], color='r', linestyle='--', alpha=0.5, label='f₁')
    ax1.axvline(plot_data['f2'], color='r', linestyle='--', alpha=0.5, label='f₂')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Dk')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Real Permittivity (Dk)')

    # Plot 2: Df vs frequency
    ax2.loglog(plot_data['freq_exp_ghz'], plot_data['df_exp'], 'o',
               label='Experimental', markersize=6)
    ax2.loglog(plot_data['freq_ghz'], plot_data['df_fit'], '-',
               label='D-S Fit', linewidth=2)
    ax2.fill_between(plot_data['freq_ghz'],
                     plot_data['df_uncertainty_band']['lower'],
                     plot_data['df_uncertainty_band']['upper'],
                     alpha=0.3, label='95% Confidence')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Df')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Loss Factor (Df)')

    # Plot 3: Residuals
    ax3.semilogx(plot_data['freq_exp_ghz'], plot_data['dk_residual'], 'o-',
                 label='Dk Residual')
    ax3.semilogx(plot_data['freq_exp_ghz'], plot_data['df_residual'] * 100, 's-',
                 label='Df Residual (×100)')
    ax3.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('Residual')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Fit Residuals')

    # Plot 4: Component breakdown
    if 'components' in plot_data:
        components = plot_data['components']
        ax4.semilogx(plot_data['freq_ghz'], components['eps_inf'].real, '--',
                     label='ε∞', linewidth=2)
        ax4.semilogx(plot_data['freq_ghz'],
                     components['eps_inf'].real + components['dispersion'].real, '-',
                     label='ε∞ + Dispersion', linewidth=2)
        ax4.semilogx(plot_data['freq_ghz'], plot_data['dk_fit'], '-',
                     label='Total (with σ_DC)', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Frequency (GHz)')
        ax4.set_ylabel('Dk')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Model Components')

    plt.tight_layout()
    print("Plots generated (not displayed in this example)")

    return fig


def main():
    """Run all demonstrations."""
    print("Djordjevic-Sarkar Model - Refactored Implementation Demo")
    print("=" * 60)

    # 1. Basic fitting
    model, result = demonstrate_basic_fit()

    # 2. Analysis
    analyzer = demonstrate_analysis()

    # 3. Model comparison
    comparison = demonstrate_model_comparison()

    # 4. Export utilities
    exporter, plot_data = demonstrate_export_utilities()

    # 5. Plotting
    fig = demonstrate_plotting(plot_data)

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("\nKey Benefits of Refactored Design:")
    print("1. Separation of Concerns:")
    print("   - Model definition (djordjevic_sarkar.py)")
    print("   - Analysis and validation (sarkar_analyzer.py)")
    print("   - Export utilities (sarkar_export_utils.py)")
    print("2. Each component has a single responsibility")
    print("3. Easier to test, maintain, and extend")
    print("4. Kramers-Kronig validation integrated seamlessly")
    print("5. Enhanced error handling and validation")

    # Example of how to save results
    results_dict = {
        'model_type': 'Djordjevic-Sarkar',
        'parameters': result.params.valuesdict(),
        'quality_metrics': analyzer.calculate_enhanced_quality_metrics(result),
        'transition_characteristics': model.get_transition_characteristics(result.params),
        'data': {
            'frequency_ghz': result.freq.tolist(),
            'dk_experimental': result.dk_exp.tolist(),
            'df_experimental': result.df_exp.tolist(),
            'dk_fitted': result.dk_fit.tolist(),
            'df_fitted': result.df_fit.tolist()
        }
    }

    print("\nResults can be saved as JSON/pickle for later use")

    return model, analyzer, exporter, results_dict


if __name__ == "__main__":
    # Run the demonstration
    model, analyzer, exporter, results = main()