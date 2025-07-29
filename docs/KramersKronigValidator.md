# KramersKronigValidator User Guide

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Understanding the Parameters](#understanding-the-parameters)
- [Methods and Properties](#methods-and-properties)
- [Interpreting Results](#interpreting-results)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Theory Background](#theory-background)
- [Best Practices](#best-practices)

## Overview

The `KramersKronigValidator` class validates experimental dielectric data using the Kramers-Kronig (KK) relations. It checks whether your measured permittivity data satisfies causality requirements, which is essential for ensuring data quality and physical validity.

### Key Features
- **Automatic method selection** based on frequency grid uniformity
- **Multiple KK transform methods**: Hilbert transform and trapezoidal integration
- **Intelligent ε∞ estimation** from high-frequency data
- **Comprehensive diagnostics** and error reporting
- **Numba acceleration** for improved performance (optional)

## Quick Start

```python
import pandas as pd
from kramers_kronig_validator import KramersKronigValidator

# Load your data
data = pd.DataFrame({
    'Frequency (GHz)': [0.1, 1.0, 10.0, 100.0],
    'Dk': [10.0, 8.5, 7.0, 5.5],  # Real permittivity
    'Df': [0.5, 0.4, 0.3, 0.2]    # Loss factor
})

# Create validator and run validation
validator = KramersKronigValidator(data)
results = validator.validate()

# Check if data is causal
if validator.is_causal:
    print("✓ Data passes causality check!")
else:
    print("✗ Data fails causality check")
    
print(validator.get_report())
```

## Installation

### Dependencies
```bash
pip install numpy scipy pandas
```

### Optional Dependencies
```bash
# For performance acceleration
pip install numba

# For logging
pip install logging
```

## Basic Usage

### 1. Data Format

Your data must be in a pandas DataFrame with three required columns:
- `Frequency (GHz)`: Frequency in gigahertz (must be positive and monotonically increasing)
- `Dk`: Real part of relative permittivity (dielectric constant)
- `Df`: Dielectric loss factor (dissipation factor, must be positive)

```python
# Example data structure
data = pd.DataFrame({
    'Frequency (GHz)': freq_values,  # e.g., [0.1, 0.5, 1.0, 5.0, 10.0, ...]
    'Dk': dk_values,                 # e.g., [10.2, 9.8, 9.5, 8.9, 8.5, ...]
    'Df': df_values                  # e.g., [0.05, 0.04, 0.03, 0.02, 0.01, ...]
})
```

### 2. Creating a Validator

```python
# Default configuration
validator = KramersKronigValidator(data)

# Custom configuration
validator = KramersKronigValidator(
    data,
    method='auto',           # KK transform method
    eps_inf_method='fit',    # How to estimate ε∞
    tail_fraction=0.1,       # Fraction of data for ε∞ estimation
    window='hamming'         # Window function for Hilbert transform
)
```

### 3. Running Validation

```python
# Run with default threshold (5% mean relative error)
results = validator.validate()

# Run with custom threshold
results = validator.validate(causality_threshold=0.1)  # 10% threshold
```

## Understanding the Parameters

### Constructor Parameters

#### `method` (str, default='auto')
Selects the Kramers-Kronig transform method:
- **'auto'**: Automatically selects based on grid uniformity
  - Uniform grid → Hilbert transform
  - Non-uniform grid → Trapezoidal integration
- **'hilbert'**: Force Hilbert transform method
  - Faster for uniform grids
  - Automatically resamples if grid is non-uniform
- **'trapz'**: Force trapezoidal integration
  - Works on any grid
  - More robust but potentially slower

```python
# Force Hilbert transform
validator = KramersKronigValidator(data, method='hilbert')

# Force trapezoidal integration
validator = KramersKronigValidator(data, method='trapz')
```

#### `eps_inf_method` (str, default='fit')
Method for estimating high-frequency permittivity (ε∞):
- **'fit'**: Fits Dk vs 1/f² in the high-frequency tail
  - More accurate for data with good high-frequency coverage
  - Requires at least 3 points in the tail
- **'mean'**: Simple average of high-frequency Dk values
  - More robust for limited data
  - Less accurate if there's residual dispersion

```python
# Use fitting method
validator = KramersKronigValidator(data, eps_inf_method='fit')

# Use averaging method
validator = KramersKronigValidator(data, eps_inf_method='mean')
```

#### `eps_inf` (float, optional)
Explicitly provide ε∞ if known from other measurements:

```python
# Provide known eps_inf value
validator = KramersKronigValidator(data, eps_inf=2.2)
```

#### `tail_fraction` (float, default=0.1)
Fraction of high-frequency data used for ε∞ estimation:
- Larger values: More stable but potentially biased if dispersion continues
- Smaller values: Less biased but more sensitive to noise

```python
# Use 20% of data for eps_inf estimation
validator = KramersKronigValidator(data, tail_fraction=0.2)
```

#### `min_tail_points` (int, default=3)
Minimum number of points required in the tail for ε∞ estimation:

```python
# Require at least 5 points for eps_inf
validator = KramersKronigValidator(data, min_tail_points=5)
```

#### `window` (str, optional)
Window function for Hilbert transform to reduce edge effects:
- Options: 'hamming', 'hann', 'blackman', 'bartlett'
- None: No windowing

```python
# Use Hamming window
validator = KramersKronigValidator(data, window='hamming')

# No windowing
validator = KramersKronigValidator(data, window=None)
```

#### `resample_points` (int, optional)
Number of points for resampling when using Hilbert transform on non-uniform grids:

```python
# Resample to 2048 points
validator = KramersKronigValidator(data, resample_points=2048)
```

## Methods and Properties

### Core Methods

#### `validate(causality_threshold=0.05)`
Performs the KK validation:

```python
results = validator.validate(causality_threshold=0.05)

# Returns dictionary with:
# - 'dk_kk': KK-transformed Dk values
# - 'mean_relative_error': Average relative error
# - 'rmse': Root mean square error
# - 'causality_status': 'PASS' or 'FAIL'
```

#### `get_report()`
Returns a formatted text report:

```python
print(validator.get_report())
# Output:
# ======== Kramers-Kronig Causality Report ========
#  ▸ Causality Status:      PASS
#  ▸ Mean Relative Error:   2.31%
#  ▸ RMSE (Dk vs. Dk_KK):   0.0234
# ==================================================
```

#### `get_diagnostics()`
Returns comprehensive diagnostic information:

```python
diagnostics = validator.get_diagnostics()
# Includes:
# - grid_uniform: bool
# - num_points: int
# - freq_range_ghz: tuple
# - eps_inf: float
# - method_used: str
# - num_peaks: int
# - max_df_freq_ghz: float
# - All validation results
```

#### `to_dict()`
Returns results in standardized format for comparison with other models:

```python
results_dict = validator.to_dict()
# Includes model_name, parameters, fitting results, etc.
```

### Properties

#### `is_causal`
Boolean property indicating if data passes causality test:

```python
if validator.is_causal:
    print("Data is causal")
```

#### `relative_error`
Mean relative error from the validation:

```python
error = validator.relative_error
print(f"Mean relative error: {error:.2%}")
```

### Context Manager Support

The validator can be used as a context manager:

```python
with KramersKronigValidator(data) as validator:
    results = validator.validate()
    # Process results
# Resources automatically cleaned up
```

## Interpreting Results

### Understanding Causality Status

The validator determines causality by comparing the experimental Dk with the KK-transformed values:

- **PASS**: Mean relative error < threshold (default 5%)
  - Data is likely causal and physically valid
  - Some error is normal due to measurement noise

- **FAIL**: Mean relative error > threshold
  - Data may have causality violations
  - Could indicate measurement problems or processing errors

### Typical Error Ranges

| Data Quality | Expected Error | Status |
|--------------|----------------|---------|
| Excellent | < 2% | PASS |
| Good | 2-5% | PASS |
| Acceptable | 5-15% | PASS (adjust threshold) |
| Poor | 15-30% | FAIL |
| Problematic | > 30% | FAIL |

### Common Causes of High Errors

1. **Insufficient frequency range**: Need data extending to sufficiently high frequencies
2. **Non-physical data**: Measurement artifacts or processing errors
3. **Strong resonances**: Multiple or strong relaxations can increase numerical errors
4. **Noise**: Random measurement noise always introduces some KK error

## Advanced Usage

### 1. Optimizing for Different Data Types

#### For Broadband Data (Multiple Decades)
```python
validator = KramersKronigValidator(
    data,
    method='trapz',        # Better for non-uniform grids
    tail_fraction=0.2,     # Use more points for eps_inf
    eps_inf_method='fit'   # Fit the tail behavior
)
```

#### For Limited Frequency Range
```python
validator = KramersKronigValidator(
    data,
    eps_inf=known_value,   # Provide eps_inf if known
    method='hilbert',      # Can be more accurate
    window='hamming'       # Reduce edge effects
)
```

#### For Noisy Data
```python
validator = KramersKronigValidator(
    data,
    eps_inf_method='mean',  # More robust to noise
    tail_fraction=0.3,      # Average over more points
    window='blackman'       # Stronger windowing
)
```

### 2. Batch Processing

```python
def validate_multiple_datasets(datasets, **kwargs):
    """Validate multiple datasets with same parameters."""
    results = {}
    
    for name, data in datasets.items():
        validator = KramersKronigValidator(data, **kwargs)
        results[name] = validator.validate()
        
        print(f"\n{name}:")
        print(validator.get_report())
    
    return results

# Usage
datasets = {
    'Sample_A': data_a,
    'Sample_B': data_b,
    'Sample_C': data_c
}

all_results = validate_multiple_datasets(
    datasets,
    method='auto',
    causality_threshold=0.1
)
```

### 3. Custom Analysis Pipeline

```python
def analyze_dielectric_data(data, save_results=False):
    """Complete analysis pipeline."""
    
    # Create validator
    validator = KramersKronigValidator(data)
    
    # Get diagnostics first
    diagnostics = validator.get_diagnostics()
    
    # Check data characteristics
    if not diagnostics['grid_uniform']:
        print("Warning: Non-uniform frequency grid detected")
    
    if diagnostics['num_peaks'] > 2:
        print(f"Complex dispersion: {diagnostics['num_peaks']} peaks detected")
    
    # Validate with appropriate threshold
    if diagnostics['num_peaks'] > 1:
        threshold = 0.1  # More lenient for complex data
    else:
        threshold = 0.05  # Stricter for simple data
    
    results = validator.validate(causality_threshold=threshold)
    
    # Report results
    print(validator.get_report())
    
    # Save if requested
    if save_results:
        import json
        with open('kk_validation_results.json', 'w') as f:
            json.dump(validator.to_dict(), f, indent=2)
    
    return validator
```

### 4. Comparing Different Methods

```python
def compare_kk_methods(data):
    """Compare different KK transform methods."""
    
    methods = ['hilbert', 'trapz']
    results = {}
    
    for method in methods:
        validator = KramersKronigValidator(data, method=method)
        results[method] = validator.validate()
        
        print(f"\n{method.upper()} method:")
        print(f"  Mean error: {results[method]['mean_relative_error']:.2%}")
        print(f"  RMSE: {results[method]['rmse']:.4f}")
        print(f"  Status: {results[method]['causality_status']}")
    
    return results
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "Missing required columns" Error
```python
# Check your column names
print(data.columns.tolist())

# Rename if necessary
data = data.rename(columns={
    'freq_GHz': 'Frequency (GHz)',
    'eps_real': 'Dk',
    'loss_factor': 'Df'
})
```

#### 2. "Frequencies must be strictly increasing" Error
```python
# Sort by frequency
data = data.sort_values('Frequency (GHz)')

# Remove duplicates
data = data.drop_duplicates(subset=['Frequency (GHz)'])
```

#### 3. "Negative frequencies detected" Error
```python
# Remove negative frequencies
data = data[data['Frequency (GHz)'] > 0]
```

#### 4. High KK Errors Despite Good Data
```python
# Try providing eps_inf explicitly
eps_inf_estimate = data['Dk'].iloc[-10:].mean()
validator = KramersKronigValidator(data, eps_inf=eps_inf_estimate)

# Or increase tail fraction
validator = KramersKronigValidator(data, tail_fraction=0.3)
```

#### 5. Non-uniform Grid Warnings
```python
# Check grid uniformity
freq = data['Frequency (GHz)'].values
diffs = np.diff(freq)
is_uniform = np.allclose(diffs, diffs[0], rtol=1e-5)

if not is_uniform:
    # Use trapz method for better accuracy
    validator = KramersKronigValidator(data, method='trapz')
```

## Theory Background

### Kramers-Kronig Relations

The Kramers-Kronig relations connect the real and imaginary parts of any causal response function:

```
ε'(ω) = ε∞ + (2/π) * P ∫[0,∞] (ω'ε''(ω'))/(ω'² - ω²) dω'
```

Where:
- ε'(ω) = Real part of permittivity (Dk)
- ε''(ω) = Imaginary part of permittivity (related to Df)
- ε∞ = High-frequency limit of permittivity
- P = Cauchy principal value

### Physical Significance

Causality means the material cannot respond before being excited. Violations indicate:
- Measurement errors
- Data processing artifacts
- Non-physical behavior
- Insufficient frequency range

### Numerical Implementation

1. **Hilbert Transform Method**
   - Efficient for uniform grids
   - Uses FFT-based algorithm
   - Requires resampling for non-uniform data

2. **Trapezoidal Integration**
   - Direct numerical integration
   - Works on any grid
   - Handles singularity carefully

## Best Practices

### 1. Data Preparation
- **Frequency range**: Measure over widest possible range
- **Frequency spacing**: Use logarithmic spacing for broad ranges
- **Data quality**: Ensure smooth, noise-free measurements
- **Units**: Verify Df is positive (dissipation factor)

### 2. Parameter Selection
- Start with default parameters
- Use `method='auto'` unless you have specific requirements
- Provide `eps_inf` if known from other measurements
- Adjust `causality_threshold` based on data quality

### 3. Validation Workflow
```python
# 1. Load and inspect data
data = pd.read_csv('dielectric_data.csv')
print(f"Frequency range: {data['Frequency (GHz)'].min():.2f} - {data['Frequency (GHz)'].max():.2f} GHz")
print(f"Number of points: {len(data)}")

# 2. Create validator with appropriate settings
validator = KramersKronigValidator(
    data,
    method='auto',
    tail_fraction=0.15
)

# 3. Get diagnostics
diag = validator.get_diagnostics()
print(f"Grid uniform: {diag['grid_uniform']}")
print(f"Number of peaks: {diag['num_peaks']}")

# 4. Validate with appropriate threshold
if diag['num_peaks'] > 2:
    threshold = 0.15  # Complex data
else:
    threshold = 0.05  # Simple data

results = validator.validate(causality_threshold=threshold)

# 5. Interpret results
print(validator.get_report())
if not validator.is_causal:
    print("Consider:")
    print("- Checking measurement setup")
    print("- Extending frequency range")
    print("- Reviewing data processing")
```

### 4. Reporting Results

Always report:
- Frequency range of measurements
- Number of data points
- KK validation method used
- Mean relative error
- Causality threshold applied
- Any special parameters (eps_inf, window, etc.)

Example:
```
Kramers-Kronig validation performed on dielectric data from 0.1-100 GHz 
(50 points, logarithmic spacing). Using trapezoidal integration with 
eps_inf estimated from top 10% of frequency range. Mean relative error: 
3.2% (PASS, threshold: 5%).
```

## Conclusion

The `KramersKronigValidator` provides a robust tool for validating dielectric measurements. Remember that:

- Some KK error is normal and expected
- The goal is to detect gross violations, not achieve perfect agreement
- Adjust parameters based on your specific data characteristics
- Use the diagnostics to understand your data better

For additional support or to report issues, please refer to the project repository.