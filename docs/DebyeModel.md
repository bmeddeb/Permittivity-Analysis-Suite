# Debye Model Documentation

## Table of Contents
- [Overview](#overview)
- [Theory and Physics](#theory-and-physics)
- [When to Use the Debye Model](#when-to-use-the-debye-model)
- [Mathematical Formulation](#mathematical-formulation)
- [Implementation Details](#implementation-details)
- [Usage Guide](#usage-guide)
- [Parameter Estimation Strategies](#parameter-estimation-strategies)
- [Integration with Kramers-Kronig Validation](#integration-with-kramers-kronig-validation)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Advanced Topics](#advanced-topics)
- [Examples](#examples)

## Overview

The Debye model is the fundamental model for dielectric relaxation, describing the response of an ideal dipolar material to an alternating electric field. It assumes a single, exponential relaxation process and serves as the foundation for understanding more complex dielectric behavior.

### Key Features of the Implementation

- **Single relaxation time model** with optional DC conductivity
- **Intelligent parameter initialization** from experimental data
- **Static methods** for calculating derived quantities
- **Circuit model export** for SPICE-like simulations
- **Library-agnostic plotting data** generation

## Theory and Physics

### Physical Picture

The Debye model describes the behavior of permanent dipoles in a material:

1. **Without field**: Dipoles are randomly oriented due to thermal motion
2. **With DC field**: Dipoles align with the field, reaching equilibrium
3. **With AC field**: Dipoles attempt to follow the field but lag due to finite response time

This lag between field and polarization causes:
- **Frequency-dependent permittivity**: High at low frequencies, low at high frequencies
- **Dielectric loss**: Energy dissipation due to friction during dipole rotation

### Microscopic Origin

The relaxation time τ arises from:
- **Viscous drag** on rotating molecules
- **Intermolecular interactions** hindering rotation
- **Thermal fluctuations** randomizing orientation

For spherical molecules in a viscous medium:
```
τ = 4πηa³/kT
```
Where:
- η = viscosity
- a = molecular radius
- k = Boltzmann constant
- T = temperature

## When to Use the Debye Model

### Ideal For:

1. **Pure polar liquids**
   - Water (above 10 GHz)
   - Simple alcohols
   - Acetone, acetonitrile

2. **Dilute solutions**
   - Non-interacting polar molecules
   - Low concentration electrolytes

3. **Gases**
   - Water vapor
   - Ammonia

4. **Specific temperature/frequency ranges**
   - Where a single relaxation dominates
   - Away from glass transitions

### Not Suitable For:

1. **Complex materials**
   - Polymers (use Cole-Cole or Havriliak-Negami)
   - Biological tissues (multiple relaxations)
   - Composites (interfacial polarization)

2. **Broad relaxation spectra**
   - Glass-forming liquids
   - Amorphous materials
   - Hydrogen-bonded systems

3. **Low frequencies with conductivity**
   - Unless using the conductivity-corrected version
   - Electrode polarization effects

## Mathematical Formulation

### Basic Debye Equation

The complex permittivity is given by:

```
ε*(ω) = ε∞ + (εs - ε∞)/(1 + jωτ)
```

Where:
- **ε∞**: High-frequency permittivity (optical permittivity)
- **εs**: Static (DC) permittivity
- **τ**: Relaxation time
- **ω**: Angular frequency (2πf)

### Real and Imaginary Parts

Separating into components:

```
ε'(ω) = ε∞ + (εs - ε∞)/(1 + ω²τ²)

ε''(ω) = (εs - ε∞)ωτ/(1 + ω²τ²)
```

### Key Relationships

1. **Relaxation frequency**: 
   ```
   frel = 1/(2πτ)
   ```

2. **Maximum loss**:
   ```
   ε''max = (εs - ε∞)/2  at  ω = 1/τ
   ```

3. **Loss tangent**:
   ```
   tan δ = ε''/ε' = Df
   ```

### With DC Conductivity

For materials with free charges:

```
ε*(ω) = ε∞ + (εs - ε∞)/(1 + jωτ) - j(σDC/(ωε0))
```

This adds a 1/f contribution to the imaginary part.

## Implementation Details

### Class Structure

```python
class DebyeModel(BaseModel):
    """
    Single Debye relaxation model.
    
    Parameters:
        conductivity_correction (bool): Include DC conductivity term
        name (str): Optional custom name
    """
```

### Parameter Bounds

The implementation uses physically motivated bounds:

| Parameter | Lower Bound | Upper Bound | Typical Range |
|-----------|-------------|-------------|---------------|
| ε∞ | 1.0 | εs | 1-10 |
| εs | ε∞ | 1000 | 2-100 |
| τ | 1e-15 s | 1e-3 s | 1e-12 - 1e-9 s |
| σDC | 0 | 100 S/m | 1e-6 - 1 S/m |

### Parameter Initialization Strategy

1. **ε∞**: Average of highest 10% frequency points
2. **εs**: Average of lowest 10% frequency points
3. **τ**: From loss peak frequency using fmax = 1/(2πτ)
4. **σDC**: From low-frequency slope if ε'' ∝ f^(-1)

## Usage Guide

### Basic Fitting

```python
from app.models.debye import DebyeModel
import numpy as np

# Create model
model = DebyeModel()

# Load your data
freq_ghz = np.array([...])  # Frequency in GHz
dk = np.array([...])        # Real permittivity
df = np.array([...])        # Loss factor

# Fit the model
result = model.fit(freq_ghz, dk, df)

# Check fit quality
print(model.get_fit_report(result))
```

### With DC Conductivity

```python
# For conductive samples
model = DebyeModel(conductivity_correction=True)
# or
from app.models.debye import DebyeWithConductivityModel
model = DebyeWithConductivityModel()

result = model.fit(freq_ghz, dk, df)
```

### Accessing Results

```python
# Get fitted parameters
eps_inf = result.params['eps_inf'].value
eps_s = result.params['eps_s'].value
tau = result.params['tau'].value

# Get derived quantities
f_rel = DebyeModel.get_relaxation_frequency(result.params)
delta_eps = DebyeModel.get_dielectric_strength(result.params)
eps_max = DebyeModel.get_loss_peak_height(result.params)

print(f"Relaxation frequency: {f_rel:.3f} GHz")
print(f"Dielectric strength: {delta_eps:.2f}")
print(f"Maximum loss: {eps_max:.3f}")
```

### Plotting with Plotly

```python
import plotly.graph_objects as go

# Get plot data
plot_data = model.get_plot_data(result)

# Create figure
fig = go.Figure()

# Add experimental data
fig.add_trace(go.Scatter(
    x=plot_data['experimental']['freq'],
    y=plot_data['experimental']['dk'],
    mode='markers',
    name='Dk (exp)',
    marker=dict(size=8)
))

# Add fitted curve
fig.add_trace(go.Scatter(
    x=plot_data['smooth']['freq'],
    y=plot_data['smooth']['dk'],
    mode='lines',
    name='Dk (fit)',
    line=dict(width=2)
))

# Log scale
fig.update_xaxes(type="log", title="Frequency (GHz)")
fig.update_yaxes(title="Dk")

fig.show()
```

## Parameter Estimation Strategies

### 1. Visual Inspection Method

Before fitting, examine your data:

```python
import numpy as np

# Estimate eps_inf (high-frequency plateau)
eps_inf_est = np.mean(dk[-5:])

# Estimate eps_s (low-frequency plateau)
eps_s_est = np.mean(dk[:5])

# Find loss peak
peak_idx = np.argmax(df)
f_peak = freq_ghz[peak_idx]
tau_est = 1 / (2 * np.pi * f_peak * 1e9)

print(f"Initial estimates:")
print(f"  ε∞ ≈ {eps_inf_est:.1f}")
print(f"  εs ≈ {eps_s_est:.1f}")
print(f"  τ ≈ {tau_est:.2e} s")
```

### 2. Cole-Cole Plot Analysis

The Debye model produces a perfect semicircle in the Cole-Cole plot:

```python
def check_debye_behavior(dk, df):
    """Check if data follows Debye behavior."""
    # Calculate imaginary permittivity
    eps_imag = df * dk
    
    # Fit circle to Cole-Cole plot
    from scipy.optimize import least_squares
    
    def circle_residuals(params, x, y):
        xc, yc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r
    
    # Initial guess
    x0 = (dk.max() + dk.min()) / 2
    y0 = eps_imag.max() / 2
    r0 = (dk.max() - dk.min()) / 2
    
    res = least_squares(circle_residuals, [x0, y0, r0], args=(dk, eps_imag))
    
    # Check circularity
    residual_std = np.std(res.fun)
    is_debye = residual_std < 0.05 * res.x[2]  # 5% of radius
    
    return is_debye, res.x
```

### 3. Logarithmic Derivative Method

For precise τ estimation:

```python
def estimate_tau_derivative(freq_hz, eps_imag):
    """Estimate tau using logarithmic derivative."""
    # d(ln ε'')/d(ln ω) = 1 at ω = 1/τ for Debye
    
    log_omega = np.log(2 * np.pi * freq_hz)
    log_eps = np.log(eps_imag)
    
    # Numerical derivative
    d_log_eps = np.gradient(log_eps, log_omega)
    
    # Find where derivative ≈ 1
    idx = np.argmin(np.abs(d_log_eps - 1))
    omega_c = 2 * np.pi * freq_hz[idx]
    
    return 1 / omega_c
```

## Integration with Kramers-Kronig Validation

### Pre-fitting Validation

Use KK validation to ensure data quality before fitting:

```python
from app.models.kramers_kronig_validator import KramersKronigValidator

# Validate data first
kk_validator = KramersKronigValidator(
    data_df,
    method='auto',
    causality_threshold=0.1
)

kk_results = kk_validator.validate()

if kk_validator.is_causal:
    print("Data passes causality check - proceed with Debye fitting")
    
    # Use KK-transformed data for better initial guesses
    dk_kk = kk_results['dk_kk']
    
    # Fit Debye model
    model = DebyeModel()
    result = model.fit(freq_ghz, dk, df)
else:
    print(f"Causality violation detected: {kk_results['mean_relative_error']:.2%}")
    print("Consider data preprocessing or alternative models")
```

### Using KK for Parameter Constraints

```python
def estimate_eps_inf_from_kk(kk_validator):
    """Use KK analysis to estimate eps_inf."""
    # KK validator estimates eps_inf during validation
    eps_inf_kk = kk_validator._estimate_eps_inf()
    
    # Use as constraint in Debye fitting
    model = DebyeModel()
    params = model.create_parameters(freq_ghz, dk, df)
    
    # Tighten bounds based on KK result
    params['eps_inf'].value = eps_inf_kk
    params['eps_inf'].min = 0.9 * eps_inf_kk
    params['eps_inf'].max = 1.1 * eps_inf_kk
    
    return params
```

### Post-fitting Validation

Verify that fitted model satisfies KK relations:

```python
def validate_fitted_model(model, result, freq_range_ghz=(0.001, 1000)):
    """Check if fitted Debye model satisfies KK relations."""
    
    # Generate high-resolution frequency grid
    freq_test = np.logspace(
        np.log10(freq_range_ghz[0]), 
        np.log10(freq_range_ghz[1]), 
        500
    )
    
    # Predict with fitted parameters
    eps_complex = model.predict(freq_test, result.params)
    dk_model = eps_complex.real
    df_model = eps_complex.imag / eps_complex.real
    
    # Create DataFrame for KK validation
    import pandas as pd
    model_df = pd.DataFrame({
        'Frequency (GHz)': freq_test,
        'Dk': dk_model,
        'Df': df_model
    })
    
    # Validate
    kk_validator = KramersKronigValidator(model_df)
    kk_result = kk_validator.validate(causality_threshold=0.01)
    
    print(f"Model KK validation: {kk_result['causality_status']}")
    print(f"Mean relative error: {kk_result['mean_relative_error']:.4f}")
    
    return kk_result
```

## Common Issues and Solutions

### 1. Poor High-Frequency Fit

**Symptom**: Model underestimates Dk at high frequencies

**Causes**:
- Additional fast relaxation process
- Insufficient frequency range
- Resonance effects

**Solutions**:
```python
# Fix eps_inf based on optical data
params = model.create_parameters(freq_ghz, dk, df)
params['eps_inf'].value = 2.25  # n² where n is refractive index
params['eps_inf'].vary = False

# Or allow higher eps_inf
params['eps_inf'].max = dk.min() * 1.5
```

### 2. Low-Frequency Deviation

**Symptom**: Poor fit at low frequencies, especially in loss

**Causes**:
- DC conductivity
- Electrode polarization
- Additional slow relaxation

**Solutions**:
```python
# Use conductivity model
model = DebyeModel(conductivity_correction=True)

# Or pre-process data to remove DC conductivity
def remove_dc_conductivity(freq_ghz, df, dk):
    """Remove 1/f conductivity contribution."""
    omega = 2 * np.pi * freq_ghz * 1e9
    eps_imag = df * dk
    
    # Fit power law to low-frequency data
    mask = freq_ghz < 0.1  # Below 100 MHz
    if np.sum(mask) > 3:
        p = np.polyfit(np.log10(omega[mask]), np.log10(eps_imag[mask]), 1)
        if abs(p[0] + 1) < 0.2:  # Close to -1 slope
            # Subtract conductivity
            sigma = 10**p[1] * 8.854e-12
            eps_imag_corr = eps_imag - sigma / (omega * 8.854e-12)
            df_corr = eps_imag_corr / dk
            return df_corr
    
    return df
```

### 3. Asymmetric Loss Peak

**Symptom**: Loss peak is broader than Debye prediction

**Causes**:
- Distribution of relaxation times
- Multiple overlapping relaxations
- Sample heterogeneity

**Solutions**:
```python
# Check for non-Debye behavior
def check_loss_symmetry(freq_ghz, df):
    """Check if loss peak is symmetric (Debye-like)."""
    # Find peak
    peak_idx = np.argmax(df)
    f_peak = freq_ghz[peak_idx]
    
    # Find half-maximum points
    half_max = df[peak_idx] / 2
    left_idx = np.where(df[:peak_idx] > half_max)[0]
    right_idx = np.where(df[peak_idx:] > half_max)[0] + peak_idx
    
    if len(left_idx) > 0 and len(right_idx) > 0:
        f_left = freq_ghz[left_idx[0]]
        f_right = freq_ghz[right_idx[-1]]
        
        # For Debye: f_right/f_left ≈ (f_peak/f_left)²
        ratio_actual = f_right / f_left
        ratio_debye = (f_peak / f_left) ** 2
        
        asymmetry = abs(np.log10(ratio_actual / ratio_debye))
        
        if asymmetry > 0.1:
            print(f"Non-Debye behavior detected (asymmetry: {asymmetry:.3f})")
            print("Consider Cole-Cole or Havriliak-Negami models")
            return False
    
    return True
```

### 4. Temperature-Dependent Analysis

When analyzing temperature series:

```python
def analyze_temperature_dependence(temperatures, tau_values):
    """Analyze Arrhenius or VFT behavior."""
    
    # Try Arrhenius fit: τ = τ0 exp(Ea/kT)
    from scipy.stats import linregress
    
    inv_T = 1000 / temperatures  # 1000/T for better scaling
    log_tau = np.log10(tau_values)
    
    slope, intercept, r_value, _, _ = linregress(inv_T, log_tau)
    
    if r_value**2 > 0.99:
        # Good Arrhenius fit
        Ea_eV = slope * 2.303 * 8.617e-5 * 1000  # Activation energy in eV
        tau_0 = 10**intercept
        
        print(f"Arrhenius behavior: Ea = {Ea_eV:.3f} eV, τ0 = {tau_0:.2e} s")
        return 'Arrhenius', {'Ea_eV': Ea_eV, 'tau_0': tau_0}
    else:
        print("Non-Arrhenius behavior - consider VFT or other models")
        return 'Non-Arrhenius', None
```

## Advanced Topics

### 1. Multi-Debye Decomposition

For materials with multiple Debye processes:

```python
class MultiDebyeModel:
    """Fit sum of multiple Debye relaxations."""
    
    def __init__(self, n_relaxations):
        self.n = n_relaxations
        self.models = [DebyeModel(name=f"Debye_{i+1}") for i in range(n)]
    
    def fit_sequential(self, freq_ghz, dk, df):
        """Fit Debye processes sequentially."""
        results = []
        dk_residual = dk.copy()
        df_residual = df.copy()
        
        for i, model in enumerate(self.models):
            # Fit to residual
            result = model.fit(freq_ghz, dk_residual, df_residual)
            results.append(result)
            
            # Subtract fitted component
            eps_fit = model.predict(freq_ghz, result.params)
            dk_residual -= eps_fit.real
            df_residual -= eps_fit.imag / eps_fit.real
            
        return results
```

### 2. Constrained Fitting

For known physical constraints:

```python
def fit_with_constraints(model, freq_ghz, dk, df, constraints):
    """Fit with physical constraints."""
    
    params = model.create_parameters(freq_ghz, dk, df)
    
    # Apply constraints
    if 'eps_inf_optical' in constraints:
        # From optical measurements
        n_optical = constraints['eps_inf_optical']
        params['eps_inf'].value = n_optical**2
        params['eps_inf'].vary = False
    
    if 'tau_range' in constraints:
        # From molecular size estimates
        tau_min, tau_max = constraints['tau_range']
        params['tau'].min = tau_min
        params['tau'].max = tau_max
    
    if 'eps_s_mixture' in constraints:
        # From mixture rules
        eps_s_max = constraints['eps_s_mixture']
        params['eps_s'].max = eps_s_max
    
    return model.fit(freq_ghz, dk, df, params=params)
```

### 3. Circuit Model Applications

Convert to SPICE netlist:

```python
def generate_spice_netlist(model, result, temp=300):
    """Generate SPICE netlist for fitted Debye model."""
    
    circuit = model.export_for_circuit_simulation(result.params)
    
    netlist = f"""* Debye Model Circuit
* Temperature: {temp}K

.param C_inf={circuit['C_inf_F']}
.param C_relax={circuit['C_relax_F']} 
.param R_relax={circuit['R_relax_Ohm']}

* Circuit
C1 1 2 {{C_inf}}
C2 2 0 {{C_relax}}
R1 2 0 {{R_relax}}

* Analysis
.ac dec 100 1 1e12
.end
"""
    
    return netlist
```

## Examples

### Example 1: Water at Room Temperature

```python
# Water exhibits Debye relaxation above ~10 GHz
import numpy as np
import pandas as pd

# Simulated water data
freq_ghz = np.logspace(0, 2, 50)  # 1-100 GHz
omega = 2 * np.pi * freq_ghz * 1e9

# Water parameters at 25°C
eps_s = 78.3
eps_inf = 5.2
tau = 8.27e-12  # ~19 GHz relaxation

# Generate "experimental" data
eps_complex = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)
dk = eps_complex.real + np.random.normal(0, 0.1, len(freq_ghz))
df = (eps_complex.imag / eps_complex.real) + np.random.normal(0, 0.001, len(freq_ghz))

# Fit
model = DebyeModel()
result = model.fit(freq_ghz, dk, df)

print(model.get_fit_report(result))
print(f"\nExpected τ: {tau:.2e} s")
print(f"Fitted τ:   {result.params['tau'].value:.2e} s")
```

### Example 2: Glycerol with Conductivity

```python
# Glycerol at low frequencies shows conductivity
# Create synthetic data
freq_ghz = np.logspace(-3, 1, 60)  # 1 MHz to 10 GHz
omega = 2 * np.pi * freq_ghz * 1e9

# Glycerol parameters
eps_s = 42.5
eps_inf = 3.5
tau = 2.5e-10
sigma_dc = 1e-6  # Small conductivity

# Include conductivity
eps_0 = 8.854e-12
eps_complex = eps_inf + (eps_s - eps_inf) / (1 + 1j * omega * tau)
eps_complex -= 1j * sigma_dc / (omega * eps_0)

dk = eps_complex.real
df = eps_complex.imag / eps_complex.real

# Create DataFrame
data_df = pd.DataFrame({
    'Frequency (GHz)': freq_ghz,
    'Dk': dk,
    'Df': df
})

# First, validate with KK
kk_validator = KramersKronigValidator(data_df)
kk_result = kk_validator.validate()
print(f"KK validation: {kk_result['causality_status']}")

# Fit with conductivity
model = DebyeModel(conductivity_correction=True)
result = model.fit(freq_ghz, dk, df)

print(model.get_fit_report(result))
```

### Example 3: Quality Control for Production

```python
def validate_material_batch(data_file, specifications):
    """Validate material properties against specifications."""
    
    # Load data
    data = pd.read_csv(data_file)
    freq_ghz = data['Frequency_GHz'].values
    dk = data['Dk'].values
    df = data['Df'].values
    
    # KK validation first
    kk_validator = KramersKronigValidator(data)
    kk_result = kk_validator.validate(causality_threshold=0.1)
    
    if not kk_validator.is_causal:
        return {
            'status': 'FAIL',
            'reason': 'Failed KK causality test',
            'kk_error': kk_result['mean_relative_error']
        }
    
    # Fit Debye model
    model = DebyeModel()
    
    try:
        result = model.fit(freq_ghz, dk, df)
    except Exception as e:
        return {
            'status': 'FAIL',
            'reason': f'Fitting failed: {str(e)}'
        }
    
    # Check specifications
    params = result.params
    checks = {
        'eps_s': specifications['eps_s_min'] <= params['eps_s'].value <= specifications['eps_s_max'],
        'tau': specifications['tau_min'] <= params['tau'].value <= specifications['tau_max'],
        'fit_quality': result.fit_metrics.r_squared >= specifications['min_r_squared']
    }
    
    if all(checks.values()):
        return {
            'status': 'PASS',
            'parameters': {
                'eps_s': params['eps_s'].value,
                'eps_inf': params['eps_inf'].value,
                'tau': params['tau'].value,
                'f_rel_ghz': DebyeModel.get_relaxation_frequency(params)
            },
            'quality_metrics': {
                'r_squared': result.fit_metrics.r_squared,
                'rmse': result.fit_metrics.rmse
            }
        }
    else:
        return {
            'status': 'FAIL',
            'reason': 'Out of specification',
            'failed_checks': [k for k, v in checks.items() if not v]
        }

# Usage
specs = {
    'eps_s_min': 2.0,
    'eps_s_max': 2.4,
    'tau_min': 1e-12,
    'tau_max': 1e-10,
    'min_r_squared': 0.98
}

result = validate_material_batch('batch_001.csv', specs)
print(f"Batch validation: {result['status']}")
```

## Conclusion

The Debye model, while simple, remains a cornerstone of dielectric spectroscopy. Its implementation provides:

1. **Robust fitting** with intelligent initialization
2. **Integration with KK validation** for data quality assurance
3. **Physical insight** through derived parameters
4. **Practical tools** for circuit modeling and quality control

Remember that real materials often deviate from ideal Debye behavior. Use this model as:
- A starting point for analysis
- A reference for comparing more complex models
- A tool for understanding fundamental relaxation processes

For materials showing non-Debye behavior, consider the Cole-Cole or Havriliak-Negami models, which extend the Debye framework to handle distributed relaxation times.