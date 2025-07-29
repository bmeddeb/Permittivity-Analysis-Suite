# BaseModel Documentation

## Overview

The `BaseModel` class provides a standardized interface for creating and fitting dielectric models to experimental complex permittivity data. It extends `lmfit.Model` with specialized functionality for handling complex-valued dielectric data, comprehensive fit quality metrics, and robust data preprocessing.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Creating a Custom Model](#creating-a-custom-model)
- [API Reference](#api-reference)
- [Weighting Methods](#weighting-methods)
- [Fit Quality Metrics](#fit-quality-metrics)
- [Data Preprocessing](#data-preprocessing)
- [Best Practices](#best-practices)
- [Examples](#examples)

## Features

- **Complex Data Handling**: Native support for complex permittivity (Dk + i*Df)
- **Automatic Parameter Initialization**: Intelligent initial parameter guessing
- **Multiple Weighting Schemes**: Uniform, relative, frequency-based, and custom weights
- **Comprehensive Metrics**: RMSE, MAPE, R², χ², and component-wise errors
- **Data Validation**: Automatic checking for common data issues
- **Preprocessing**: Built-in duplicate frequency removal
- **Type Safety**: Full type hints for better IDE support
- **Logging**: Integrated logging for debugging and monitoring

## Installation

### Dependencies

```bash
pip install numpy lmfit
```

### Optional Dependencies

```bash
# For better performance with large datasets
pip install scipy

# For logging configuration
pip install logging
```

## Quick Start

```python
from base_model import BaseModel
import numpy as np
import lmfit

# Create a simple Debye model
class DebyeModel(BaseModel):
    def __init__(self):
        super().__init__(name="Single Debye")
    
    @staticmethod
    def model_func(freq, eps_inf, eps_s, tau):
        omega = 2 * np.pi * freq * 1e9
        return eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)
    
    def create_parameters(self, freq, dk_exp, df_exp):
        params = lmfit.Parameters()
        params.add('eps_inf', value=dk_exp[-1], min=1, max=20)
        params.add('eps_s', value=dk_exp[0], min=dk_exp[-1], max=100)
        params.add('tau', value=1e-11, min=1e-15, max=1e-6)
        return params

# Fit to data
model = DebyeModel()
result = model.fit(freq_ghz, dk_data, df_data)
print(model.get_fit_report(result))
```

## Creating a Custom Model

To create a custom dielectric model, inherit from `BaseModel` and implement two abstract methods:

### 1. Define the Model Function

```python
@staticmethod
def model_func(freq, **params):
    """
    Mathematical function returning complex permittivity.
    
    Args:
        freq: Frequency array in GHz
        **params: Model parameters
        
    Returns:
        Complex permittivity array
    """
    # Your model equation here
    return complex_permittivity
```

### 2. Create Initial Parameters

```python
def create_parameters(self, freq, dk_exp, df_exp):
    """
    Create lmfit Parameters with initial guesses.
    
    Args:
        freq: Frequency array in GHz
        dk_exp: Experimental Dk data
        df_exp: Experimental Df data
        
    Returns:
        lmfit.Parameters object
    """
    params = lmfit.Parameters()
    # Add parameters with initial values and bounds
    return params
```

### Complete Example: Cole-Cole Model

```python
class ColeColeModel(BaseModel):
    def __init__(self):
        super().__init__(
            name="Cole-Cole",
            frequency_range=(0.001, 1000)  # MHz to THz
        )
    
    @staticmethod
    def model_func(freq, eps_inf, eps_s, tau, alpha):
        omega = 2 * np.pi * freq * 1e9
        return eps_inf + (eps_s - eps_inf) / (1 + (1j * omega * tau)**(1 - alpha))
    
    def create_parameters(self, freq, dk_exp, df_exp):
        params = lmfit.Parameters()
        
        # High-frequency permittivity
        params.add('eps_inf', value=dk_exp[-1], min=1, max=dk_exp[0])
        
        # Static permittivity
        params.add('eps_s', value=dk_exp[0], min=dk_exp[-1], max=100)
        
        # Relaxation time
        tau_guess = 1 / (2 * np.pi * freq[np.argmax(df_exp)] * 1e9)
        params.add('tau', value=tau_guess, min=1e-15, max=1e-6)
        
        # Distribution parameter
        params.add('alpha', value=0.1, min=0, max=0.5)
        
        return params
```

## API Reference

### Constructor

```python
BaseModel(name: Optional[str] = None,
          frequency_range: Optional[Tuple[float, float]] = None,
          **kwargs)
```

**Parameters:**
- `name`: Model name (defaults to class name)
- `frequency_range`: Valid frequency range in GHz (min, max)
- `**kwargs`: Additional arguments passed to lmfit.Model

### Core Methods

#### `fit()`

```python
fit(freq: np.ndarray,
    dk_exp: np.ndarray,
    df_exp: np.ndarray,
    params: Optional[lmfit.Parameters] = None,
    weights: Optional[Union[str, np.ndarray]] = 'uniform',
    method: str = 'leastsq',
    **kwargs) -> lmfit.model.ModelResult
```

Fit the model to experimental data.

**Parameters:**
- `freq`: Frequency array in GHz
- `dk_exp`: Experimental real permittivity (Dk)
- `df_exp`: Experimental loss factor (Df)
- `params`: Initial parameters (auto-created if None)
- `weights`: Weighting method or custom array
- `method`: Optimization algorithm ('leastsq', 'least_squares', 'differential_evolution', etc.)
- `**kwargs`: Additional minimizer arguments

**Returns:**
- `ModelResult` object containing fit results and metrics

**Example:**
```python
# Basic fit
result = model.fit(freq, dk, df)

# Weighted fit with custom parameters
params = model.create_parameters(freq, dk, df)
params['tau'].max = 1e-10  # Adjust bounds
result = model.fit(freq, dk, df, params=params, weights='relative')
```

#### `predict()`

```python
predict(freq: np.ndarray,
        params: Union[lmfit.Parameters, Dict[str, float]]) -> np.ndarray
```

Calculate model predictions at given frequencies.

**Example:**
```python
# Predict at new frequencies
new_freq = np.logspace(-1, 3, 1000)
predicted = model.predict(new_freq, result.params)
```

#### `get_fit_report()`

```python
get_fit_report(result: lmfit.model.ModelResult) -> str
```

Generate a comprehensive text report of fit results.

**Example Output:**
```
============================================================
Cole-Cole Fit Report
============================================================

Fit Quality:
  Success: True
  R²: 0.9987
  RMSE: 0.0234
  MAPE: 1.23%
  χ²: 0.0547
  χ²_reduced: 0.0137
  Max Error: 0.0891
  Dk RMSE: 0.0165
  Df RMSE: 0.0166

Fitted Parameters:
  eps_inf: 2.3456 ± 0.0123
  eps_s: 78.901 ± 0.234
  tau: 8.234e-12 ± 1.23e-13
  alpha: 0.0567 ± 0.0089

Data Info:
  Frequency range: 0.100 - 100.000 GHz
  Number of points: 50
  Degrees of freedom: 46
============================================================
```

#### `to_dict()`

```python
to_dict(result: lmfit.model.ModelResult) -> Dict[str, Any]
```

Convert fit results to dictionary for storage/serialization.

**Example:**
```python
# Save results
results_dict = model.to_dict(result)

import json
with open('fit_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
```

### Validation Methods

#### `validate_data()`

```python
validate_data(freq: np.ndarray,
              dk_exp: np.ndarray,
              df_exp: np.ndarray) -> None
```

Validates input data before fitting. Checks for:
- Matching array shapes
- NaN or infinite values
- Positive frequencies
- Non-negative Df values
- Monotonic frequency ordering

**Raises:**
- `ValueError`: If any validation check fails

### Preprocessing Methods

#### `preprocess_data()`

```python
preprocess_data(freq: np.ndarray,
                dk_exp: np.ndarray,
                df_exp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Preprocesses data before fitting. Default behavior:
- Removes duplicate frequencies by averaging corresponding dk/df values
- Can be disabled by setting `model.remove_duplicates = False`

**Override Example:**
```python
def preprocess_data(self, freq, dk_exp, df_exp):
    # Call parent preprocessing
    freq, dk_exp, df_exp = super().preprocess_data(freq, dk_exp, df_exp)
    
    # Additional custom preprocessing
    # e.g., smooth data, remove outliers, etc.
    
    return freq, dk_exp, df_exp
```

## Weighting Methods

The `fit()` method supports several weighting schemes:

### 1. Uniform Weighting (default)
```python
result = model.fit(freq, dk, df, weights='uniform')
```
- Equal weight for all data points
- Best for data with uniform uncertainty

### 2. Relative Weighting
```python
result = model.fit(freq, dk, df, weights='relative')
```
- Weights inversely proportional to data values
- Emphasizes relative errors
- Good for data spanning multiple orders of magnitude

### 3. Frequency Weighting
```python
result = model.fit(freq, dk, df, weights='frequency')
```
- Higher weight for high-frequency data
- Useful when high-frequency behavior is critical

### 4. Combined Weighting
```python
result = model.fit(freq, dk, df, weights='combined')
```
- Combines relative and frequency weighting
- Balanced approach for broadband data

### 5. Custom Weights
```python
# Custom weights for each frequency point
custom_weights = 1 / measurement_uncertainty
result = model.fit(freq, dk, df, weights=custom_weights)

# Different weights for Dk and Df
dk_weights = 1 / dk_uncertainty
df_weights = 1 / df_uncertainty
combined_weights = np.hstack([dk_weights, df_weights])
result = model.fit(freq, dk, df, weights=combined_weights)
```

## Fit Quality Metrics

The `FitQualityMetrics` dataclass provides comprehensive fit assessment:

| Metric | Description | Good Fit |
|--------|-------------|----------|
| `rmse` | Root Mean Square Error | < 0.05 |
| `mape` | Mean Absolute Percentage Error | < 5% |
| `r_squared` | Coefficient of Determination | > 0.95 |
| `chi_squared` | Chi-squared statistic | Low |
| `reduced_chi_squared` | χ²/DOF | ≈ 1.0 |
| `max_error` | Maximum absolute error | < 0.1 |
| `dk_rmse` | RMSE for Dk only | < 0.05 |
| `df_rmse` | RMSE for Df only | < 0.05 |

**Accessing Metrics:**
```python
# After fitting
metrics = result.fit_metrics

print(f"R² = {metrics.r_squared:.4f}")
print(f"RMSE = {metrics.rmse:.4g}")
print(f"MAPE = {metrics.mape:.2f}%")

# Check fit quality
if metrics.r_squared > 0.95 and metrics.mape < 5:
    print("Excellent fit!")
```

## Data Preprocessing

### Automatic Duplicate Removal

By default, duplicate frequencies are handled by averaging:

```python
# Enable/disable duplicate removal
model.remove_duplicates = True  # Default

# Duplicates are automatically handled
freq = np.array([1, 2, 2, 3, 4])
dk = np.array([10, 9, 9.2, 8, 7])
df = np.array([0.1, 0.09, 0.091, 0.08, 0.07])

# After preprocessing:
# freq = [1, 2, 3, 4]
# dk = [10, 9.1, 8, 7]  # Averaged
# df = [0.1, 0.0905, 0.08, 0.07]  # Averaged
```

### Custom Preprocessing

Override `preprocess_data()` for additional preprocessing:

```python
class MyModel(BaseModel):
    def preprocess_data(self, freq, dk_exp, df_exp):
        # Remove duplicates first
        freq, dk_exp, df_exp = super().preprocess_data(freq, dk_exp, df_exp)
        
        # Remove outliers (example)
        z_scores = np.abs((dk_exp - np.mean(dk_exp)) / np.std(dk_exp))
        mask = z_scores < 3
        
        return freq[mask], dk_exp[mask], df_exp[mask]
```

## Best Practices

### 1. Parameter Initialization

```python
def create_parameters(self, freq, dk_exp, df_exp):
    params = lmfit.Parameters()
    
    # Use data characteristics for initial guesses
    eps_inf_guess = np.mean(dk_exp[-5:])  # High-freq average
    eps_s_guess = np.mean(dk_exp[:5])     # Low-freq average
    
    # Find relaxation frequency from Df peak
    peak_idx = np.argmax(df_exp)
    f_rel = freq[peak_idx]
    tau_guess = 1 / (2 * np.pi * f_rel * 1e9)
    
    # Set reasonable bounds
    params.add('eps_inf', value=eps_inf_guess, 
               min=0.5*eps_inf_guess, max=2*eps_inf_guess)
    params.add('eps_s', value=eps_s_guess,
               min=eps_inf_guess, max=2*eps_s_guess)
    params.add('tau', value=tau_guess,
               min=tau_guess/100, max=tau_guess*100)
    
    return params
```

### 2. Model Selection

```python
# Try multiple models and compare
models = [DebyeModel(), ColeColeModel(), HavriliakNegamiModel()]
results = {}

for model in models:
    try:
        result = model.fit(freq, dk, df)
        results[model.name] = {
            'result': result,
            'aic': result.aic,
            'bic': result.bic,
            'r_squared': result.fit_metrics.r_squared
        }
    except Exception as e:
        print(f"{model.name} failed: {e}")

# Select best model by AIC
best_model = min(results.items(), key=lambda x: x[1]['aic'])
print(f"Best model: {best_model[0]}")
```

### 3. Handling Difficult Fits

```python
# For difficult fits, try different methods
methods = ['leastsq', 'least_squares', 'differential_evolution']

for method in methods:
    try:
        result = model.fit(freq, dk, df, method=method)
        if result.success:
            print(f"Success with {method}")
            break
    except:
        continue

# Use bounds and constraints
params = model.create_parameters(freq, dk, df)
params['tau'].min = 1e-12  # Tighten bounds
params['alpha'].max = 0.3  # Limit distribution

# Try with different weights
for weight_method in ['uniform', 'relative', 'frequency']:
    result = model.fit(freq, dk, df, params=params, weights=weight_method)
    print(f"{weight_method}: R² = {result.fit_metrics.r_squared:.4f}")
```

### 4. Error Analysis

```python
# Parameter uncertainties
for name in model.param_names:
    param = result.params[name]
    if param.stderr:
        rel_error = abs(param.stderr / param.value) * 100
        print(f"{name}: {param.value:.4g} ± {param.stderr:.4g} ({rel_error:.1f}%)")

# Confidence intervals
ci = result.conf_interval()
if ci:
    print("\n95% Confidence Intervals:")
    for param, bounds in ci.items():
        print(f"{param}: [{bounds[0][1]:.4g}, {bounds[-1][1]:.4g}]")
```

## Examples

### Example 1: Multi-Relaxation Model

```python
class DoubleDebyeModel(BaseModel):
    def __init__(self):
        super().__init__(name="Double Debye")
    
    @staticmethod
    def model_func(freq, eps_inf, eps_s1, tau1, eps_s2, tau2):
        omega = 2 * np.pi * freq * 1e9
        term1 = (eps_s1 - eps_s2) / (1 + 1j * omega * tau1)
        term2 = (eps_s2 - eps_inf) / (1 + 1j * omega * tau2)
        return eps_inf + term1 + term2
    
    def create_parameters(self, freq, dk_exp, df_exp):
        params = lmfit.Parameters()
        
        # Use data features to guess parameters
        eps_inf = dk_exp[-1]
        eps_s1 = dk_exp[0]
        
        # Find two peaks in Df
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(df_exp, height=0.1*np.max(df_exp))
        
        if len(peaks) >= 2:
            tau1 = 1 / (2 * np.pi * freq[peaks[0]] * 1e9)
            tau2 = 1 / (2 * np.pi * freq[peaks[1]] * 1e9)
            eps_s2 = dk_exp[peaks[0]]
        else:
            # Fallback guesses
            tau1, tau2 = 1e-10, 1e-12
            eps_s2 = (eps_s1 + eps_inf) / 2
        
        params.add('eps_inf', value=eps_inf, min=1, max=eps_s1)
        params.add('eps_s1', value=eps_s1, min=eps_inf, max=100)
        params.add('eps_s2', value=eps_s2, min=eps_inf, max=eps_s1)
        params.add('tau1', value=tau1, min=1e-15, max=1e-6)
        params.add('tau2', value=tau2, min=1e-15, max=1e-6)
        
        return params
```

### Example 2: Temperature-Dependent Fitting

```python
class TemperatureDependentModel(BaseModel):
    def __init__(self, temperature):
        self.temperature = temperature
        super().__init__(name=f"Debye @ {temperature}K")
    
    def create_parameters(self, freq, dk_exp, df_exp):
        params = super().create_parameters(freq, dk_exp, df_exp)
        
        # Temperature-dependent constraints
        if self.temperature > 300:
            params['tau'].max = 1e-11  # Faster relaxation at high T
        else:
            params['tau'].min = 1e-12  # Slower relaxation at low T
        
        return params

# Fit data at multiple temperatures
temperatures = [273, 298, 323, 348]
results = {}

for T in temperatures:
    model = TemperatureDependentModel(T)
    result = model.fit(freq, dk_data[T], df_data[T])
    results[T] = result
    
    print(f"T = {T}K: τ = {result.params['tau'].value:.3e} s")

# Analyze temperature dependence
import matplotlib.pyplot as plt

tau_values = [results[T].params['tau'].value for T in temperatures]
plt.semilogy(1000/np.array(temperatures), tau_values, 'o-')
plt.xlabel('1000/T (K⁻¹)')
plt.ylabel('τ (s)')
plt.title('Arrhenius Plot')
```

### Example 3: Batch Processing with Quality Control

```python
def batch_fit_with_qc(data_files, model_class, quality_threshold=0.95):
    """
    Batch process multiple datasets with quality control.
    """
    results = {}
    failed = []
    
    for file in data_files:
        # Load data
        data = np.loadtxt(file)
        freq, dk, df = data[:, 0], data[:, 1], data[:, 2]
        
        # Create model instance
        model = model_class()
        
        try:
            # Fit with automatic preprocessing
            result = model.fit(freq, dk, df, weights='relative')
            
            # Quality control
            if result.fit_metrics.r_squared < quality_threshold:
                print(f"Warning: Poor fit for {file} (R² = {result.fit_metrics.r_squared:.4f})")
                
                # Try different approach
                result = model.fit(freq, dk, df, 
                                 weights='combined',
                                 method='differential_evolution')
            
            results[file] = {
                'model': model,
                'result': result,
                'report': model.get_fit_report(result)
            }
            
        except Exception as e:
            print(f"Failed to fit {file}: {e}")
            failed.append(file)
    
    # Summary
    print(f"\nProcessed {len(results)} files successfully")
    print(f"Failed: {len(failed)} files")
    
    if results:
        r2_values = [r['result'].fit_metrics.r_squared for r in results.values()]
        print(f"Average R² = {np.mean(r2_values):.4f}")
    
    return results, failed
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "Model function must return complex values"
```python
# Wrong
def model_func(freq, a, b):
    return a * freq + b  # Returns real values

# Correct
def model_func(freq, a, b):
    return (a * freq + b) + 0j  # Ensure complex
```

#### 2. Poor Fit Quality
```python
# Check data quality
plt.loglog(freq, dk, 'o', label='Dk')
plt.loglog(freq, df, 's', label='Df')
plt.legend()
plt.xlabel('Frequency (GHz)')

# Try different initialization
params = model.create_parameters(freq, dk, df)
# Manually adjust initial values
params['tau'].value = 1e-11

# Use global optimization
result = model.fit(freq, dk, df, method='differential_evolution')
```

#### 3. Convergence Issues
```python
# Increase iterations
result = model.fit(freq, dk, df, max_nfev=5000)

# Use bounded optimization
result = model.fit(freq, dk, df, method='least_squares')

# Simplify model or fix some parameters
params = model.create_parameters(freq, dk, df)
params['alpha'].vary = False  # Fix parameter
```

## Advanced Topics

### Custom Residual Functions

For specialized fitting needs, override the residual calculation:

```python
class CustomModel(BaseModel):
    def fit(self, freq, dk_exp, df_exp, **kwargs):
        # Custom residual with different error weighting
        def custom_residual(params, freq, data, weights=None):
            n = len(freq)
            model_complex = self._complex_model_func(freq, **params)
            
            # Logarithmic residuals for Df
            dk_res = model_complex.real - data[:n]
            df_res = np.log10(model_complex.imag) - np.log10(data[n:])
            
            residuals = np.hstack([dk_res, df_res])
            if weights is not None:
                residuals *= np.sqrt(weights)
            
            return residuals
        
        # Continue with standard fitting...
```

### Parallel Fitting

For multiple datasets or parameter studies:

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def fit_single_dataset(args):
    freq, dk, df, model_class = args
    model = model_class()
    return model.fit(freq, dk, df)

# Parallel fitting
datasets = [(freq1, dk1, df1), (freq2, dk2, df2), ...]
args = [(f, dk, df, DebyeModel) for f, dk, df in datasets]

with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
    results = list(executor.map(fit_single_dataset, args))
```

## Conclusion

The `BaseModel` class provides a robust foundation for dielectric data analysis with:
- Comprehensive error handling and validation
- Flexible fitting options
- Detailed quality metrics
- Easy extensibility for custom models

For questions or contributions, please refer to the project repository.