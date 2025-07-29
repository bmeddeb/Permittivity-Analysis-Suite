# Cole-Cole Model Documentation

## Table of Contents
- [Overview](#overview)
- [Theory and Physics](#theory-and-physics)
- [Mathematical Formulation](#mathematical-formulation)
- [When to Use Cole-Cole](#when-to-use-cole-cole)
- [Implementation Details](#implementation-details)
- [Usage Guide](#usage-guide)
- [Parameter Estimation](#parameter-estimation)
- [Distribution Analysis](#distribution-analysis)
- [Comparison with Other Models](#comparison-with-other-models)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Advanced Applications](#advanced-applications)
- [Examples](#examples)

## Overview

The Cole-Cole model extends the Debye model by introducing a symmetric distribution of relaxation times, characterized by a single broadening parameter α. This model captures the non-ideal dielectric behavior observed in many real materials where molecular interactions lead to a spectrum of relaxation times rather than a single value.

### Key Features

- **Symmetric broadening** of relaxation time distribution on log scale
- **Single parameter (α)** controls distribution width
- **Reduces to Debye** when α = 0
- **Excellent for polymers**, glasses, and biological tissues
- **Optional DC conductivity** correction

### Model Equation

```
ε*(ω) = ε∞ + (εs - ε∞) / (1 + (jωτ)^(1-α))
```

Where:
- **α**: Distribution parameter (0 ≤ α < 1)
- Other parameters same as Debye model

## Theory and Physics

### Physical Origin of Distribution

The Cole-Cole distribution arises from:

1. **Molecular Heterogeneity**
   - Polymer chain length variations
   - Different local environments
   - Conformational diversity

2. **Cooperative Effects**
   - Inter-molecular coupling
   - Collective dynamics
   - Correlation effects

3. **Structural Disorder**
   - Amorphous regions
   - Defects and impurities
   - Interface effects

### Distribution Function

The relaxation time distribution g(τ) for Cole-Cole is:

```
g(ln τ) = (sin(απ)) / (2π[cosh((1-α)ln(τ/τ₀)) + cos(απ)])
```

Key properties:
- **Symmetric** on logarithmic scale
- **Width increases** with α
- **Peak at τ₀** (central relaxation time)

### Physical Meaning of α

| α Value | Physical Interpretation | Typical Materials |
|---------|------------------------|-------------------|
| 0 | Single relaxation time (Debye) | Pure liquids |
| 0.1-0.2 | Slight broadening | Dilute polymers |
| 0.2-0.4 | Moderate distribution | Concentrated polymers |
| 0.4-0.6 | Broad distribution | Glasses near Tg |
| >0.6 | Very broad (often unphysical) | Highly heterogeneous |

## Mathematical Formulation

### Complex Permittivity Components

Real part (ε'):
```
ε'(ω) = ε∞ + (εs - ε∞) × Re[1/(1 + (jωτ)^(1-α))]
```

Imaginary part (ε''):
```
ε''(ω) = (εs - ε∞) × Im[1/(1 + (jωτ)^(1-α))]
```

### Key Relationships

1. **Peak frequency**:
   ```
   fpeak = (1/2πτ) × [sin(π/2(1+α)) / sin(απ/2(1+α))]
   ```

2. **Maximum loss**:
   ```
   ε''max ≈ (εs - ε∞)/2 × (1 - 0.377α)
   ```

3. **FWHM** (Full Width at Half Maximum):
   ```
   FWHM ≈ 1.14 + 2.18α decades
   ```

### Cole-Cole Plot

In the complex plane (ε'' vs ε'), Cole-Cole produces a **depressed semicircle**:
- Center below real axis
- Depression angle = απ/2
- Radius = (εs - ε∞)/2

## When to Use Cole-Cole

### Ideal Applications

1. **Polymers**
   - Amorphous polymers
   - Polymer solutions
   - Polymer blends
   - Semi-crystalline polymers

2. **Biological Materials**
   - Tissues (α ≈ 0.1-0.3)
   - Cell suspensions
   - Proteins in solution
   - Biomembranes

3. **Glasses and Supercooled Liquids**
   - Near glass transition
   - Fragile glass formers
   - Ionic glasses

4. **Composite Materials**
   - Polymer composites
   - Nanocomposites
   - Porous materials

### Not Recommended For

1. **Asymmetric distributions** → Use Havriliak-Negami
2. **Multiple distinct relaxations** → Use multi-Debye or multi-Cole-Cole
3. **Conductive materials at low frequencies** → Need conductivity correction
4. **Very broad distributions (α > 0.7)** → Often indicates multiple processes

## Implementation Details

### Class Structure

```python
class ColeColeModel(BaseModel):
    """
    Cole-Cole model with symmetric distribution.
    
    Parameters:
        conductivity_correction: Include DC conductivity
        constrain_alpha: Limit α to [0, 0.5]
        name: Custom model name
    """
```

### Parameter Bounds and Typical Values

| Parameter | Symbol | Typical Range | Units | Notes |
|-----------|--------|---------------|-------|-------|
| High-freq permittivity | ε∞ | 1-10 | - | Often ~n² |
| Static permittivity | εs | 2-100 | - | εs > ε∞ |
| Relaxation time | τ | 10⁻¹⁵-10⁻³ | s | Material dependent |
| Distribution | α | 0-0.5 | - | >0.5 rare |
| DC conductivity | σDC | 0-100 | S/m | If enabled |

### Initialization Strategy

The model uses sophisticated parameter estimation:

1. **α from peak width**: Analyzes FWHM to estimate broadening
2. **τ correction**: Accounts for Cole-Cole peak shift
3. **Conductivity detection**: Identifies 1/f behavior
4. **Robust peak finding**: Handles noisy data

## Usage Guide

### Basic Fitting

```python
from app.models.cole_cole import ColeColeModel
import numpy as np

# Create model
model = ColeColeModel()

# Fit data
result = model.fit(freq_ghz, dk_exp, df_exp)

# Get parameters
alpha = result.params['alpha'].value
tau = result.params['tau'].value

print(f"Distribution parameter α = {alpha:.3f}")
print(f"Central relaxation time τ = {tau:.2e} s")
```

### With Conductivity

```python
# For materials with DC conductivity
model = ColeColeModel(conductivity_correction=True)

# Or use convenience class
from app.models.cole_cole import ColeColeWithConductivityModel
model = ColeColeWithConductivityModel()

result = model.fit(freq_ghz, dk_exp, df_exp)
```

### Constraining α

```python
# Physically reasonable constraint (recommended)
model = ColeColeModel(constrain_alpha=True)  # Limits α ≤ 0.5

# Custom constraint
params = model.create_parameters(freq, dk, df)
params['alpha'].max = 0.3  # Further restriction
result = model.fit(freq, dk, df, params=params)
```

### Advanced Fitting Options

```python
# Weight by measurement uncertainty
weights = 1 / measurement_errors
result = model.fit(freq, dk, df, weights=weights)

# Use different optimization method
result = model.fit(freq, dk, df, method='differential_evolution')

# Fix known parameters
params = model.create_parameters(freq, dk, df)
params['eps_inf'].value = 2.25  # From optical data
params['eps_inf'].vary = False
result = model.fit(freq, dk, df, params=params)
```

## Parameter Estimation

### 1. Estimating α from Data

The model provides automatic α estimation from peak width:

```python
# Manual estimation for understanding
def estimate_alpha_manually(freq_ghz, df):
    """Estimate α from loss peak width."""
    # Find peak
    peak_idx = np.argmax(df)
    peak_val = df[peak_idx]
    
    # Find FWHM
    half_max = peak_val / 2
    indices = np.where(df > half_max)[0]
    
    if len(indices) > 2:
        f_low = freq_ghz[indices[0]]
        f_high = freq_ghz[indices[-1]]
        
        # Width in decades
        width_decades = np.log10(f_high / f_low)
        
        # Empirical relation
        alpha = (width_decades - 1.14) / 2.18
        return np.clip(alpha, 0, 0.5)
    
    return 0.1  # Default
```

### 2. Pre-fit Analysis

```python
def analyze_before_fitting(freq, dk, df):
    """Pre-fit analysis to guide parameter selection."""
    
    # Check for Cole-Cole behavior
    # 1. Look for symmetric broadening
    peak_idx = np.argmax(df)
    
    # Check symmetry on log scale
    log_freq = np.log10(freq)
    left_width = log_freq[peak_idx] - log_freq[0]
    right_width = log_freq[-1] - log_freq[peak_idx]
    
    if abs(left_width - right_width) < 0.5:
        print("Data shows symmetric broadening - Cole-Cole appropriate")
    else:
        print("Asymmetric broadening - consider Havriliak-Negami")
    
    # 2. Estimate parameters
    eps_inf_est = np.mean(dk[-5:])
    eps_s_est = np.mean(dk[:5])
    
    # 3. Check for conductivity
    if freq[0] < 0.01:  # If data goes below 10 MHz
        omega = 2 * np.pi * freq[:10] * 1e9
        eps_imag = df[:10] * dk[:10]
        
        # Check slope on log-log plot
        p = np.polyfit(np.log10(omega), np.log10(eps_imag), 1)
        if abs(p[0] + 1) < 0.2:
            print("Low-frequency conductivity detected")
            print("Use conductivity_correction=True")
```

### 3. Initial Guess Refinement

```python
# Refine initial parameters based on physical knowledge
def refine_parameters(model, freq, dk, df, material_info):
    """Refine parameters based on material knowledge."""
    
    params = model.create_parameters(freq, dk, df)
    
    if material_info['type'] == 'polymer':
        # Polymers typically have α = 0.1-0.4
        params['alpha'].min = 0.1
        params['alpha'].max = 0.4
        
        if material_info.get('molecular_weight'):
            # Higher MW → broader distribution
            mw = material_info['molecular_weight']
            if mw > 100000:
                params['alpha'].value = 0.3
            else:
                params['alpha'].value = 0.2
    
    elif material_info['type'] == 'biological':
        # Biological tissues: α = 0.05-0.25
        params['alpha'].value = 0.15
        params['alpha'].max = 0.25
    
    return params
```

## Distribution Analysis

### Accessing Distribution Information

```python
# After fitting
dist_params = ColeColeModel.get_distribution_parameters(result.params)

print(f"Distribution width: {dist_params['fwhm_log_decades']:.2f} decades")
print(f"Mean relaxation time: {dist_params['tau_mean']:.2e} s")
print(f"Distribution is symmetric: {dist_params['symmetric']}")
```

### Visualizing the Distribution

```python
import plotly.graph_objects as go

# Get distribution data
plot_data = model.get_plot_data(result, show_distribution=True)

if 'distribution' in plot_data:
    dist = plot_data['distribution']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dist['tau'],
        y=dist['g_tau'],
        mode='lines',
        name='g(τ)'
    ))
    
    fig.update_xaxes(type="log", title="Relaxation Time τ (s)")
    fig.update_yaxes(title="Distribution g(τ)")
    fig.update_layout(title=f"Cole-Cole Distribution (α = {dist['alpha']:.3f})")
    fig.show()
```

### Comparing Distributions

```python
def compare_alpha_effects():
    """Show how α affects the distribution."""
    import plotly.graph_objects as go
    
    tau_c = 1e-10  # Central time
    tau_array = np.logspace(-13, -7, 200)
    
    fig = go.Figure()
    
    for alpha in [0, 0.1, 0.2, 0.3, 0.4]:
        g_tau = ColeColeModel._calculate_distribution(tau_array, tau_c, alpha)
        
        fig.add_trace(go.Scatter(
            x=tau_array,
            y=g_tau,
            mode='lines',
            name=f'α = {alpha}'
        ))
    
    fig.update_xaxes(type="log", title="τ (s)")
    fig.update_yaxes(title="g(τ)")
    fig.update_layout(title="Effect of α on Distribution Width")
    fig.show()
```

## Comparison with Other Models

### Cole-Cole vs Debye

```python
def compare_with_debye(cole_cole_result):
    """Compare Cole-Cole fit with equivalent Debye."""
    
    # Get equivalent Debye parameters
    debye_equiv = ColeColeModel.estimate_debye_equivalent(cole_cole_result.params)
    
    print("Cole-Cole parameters:")
    print(f"  τ = {cole_cole_result.params['tau'].value:.2e} s")
    print(f"  α = {cole_cole_result.params['alpha'].value:.3f}")
    
    print("\nEquivalent Debye:")
    print(f"  τ_eff = {debye_equiv['tau_effective']:.2e} s")
    
    # Compare peak frequencies
    f_peak_cc = ColeColeModel.get_relaxation_frequency(cole_cole_result.params)
    f_peak_debye = 1 / (2 * np.pi * debye_equiv['tau_effective'])
    
    print(f"\nPeak frequency comparison:")
    print(f"  Cole-Cole: {f_peak_cc:.3f} GHz")
    print(f"  Debye: {f_peak_debye:.3f} GHz")
```

### Model Selection Criteria

```python
def select_model(freq, dk, df):
    """Determine if Cole-Cole is appropriate."""
    from app.models.debye import DebyeModel
    
    # Fit both models
    debye_model = DebyeModel()
    cole_cole_model = ColeColeModel()
    
    debye_result = debye_model.fit(freq, dk, df)
    cole_cole_result = cole_cole_model.fit(freq, dk, df)
    
    # Compare using AIC (Akaike Information Criterion)
    # Lower AIC is better, but penalizes extra parameters
    aic_debye = debye_result.aic
    aic_cole_cole = cole_cole_result.aic
    
    print(f"Debye AIC: {aic_debye:.1f}")
    print(f"Cole-Cole AIC: {aic_cole_cole:.1f}")
    
    # Check if α is significant
    alpha = cole_cole_result.params['alpha']
    if alpha.stderr:
        t_stat = alpha.value / alpha.stderr
        if t_stat > 2:  # Roughly 95% confidence
            print(f"α = {alpha.value:.3f} ± {alpha.stderr:.3f} is significant")
            print("Cole-Cole model is justified")
        else:
            print("α is not significantly different from 0")
            print("Debye model is sufficient")
    
    # R² comparison
    print(f"\nR² comparison:")
    print(f"Debye: {debye_result.fit_metrics.r_squared:.4f}")
    print(f"Cole-Cole: {cole_cole_result.fit_metrics.r_squared:.4f}")
```

## Common Issues and Solutions

### 1. α Approaching Upper Limit

**Problem**: α fits to maximum allowed value

**Causes**:
- Multiple overlapping relaxations
- Asymmetric distribution
- Conductivity effects

**Solutions**:
```python
# Check for multiple relaxations
def check_multiple_relaxations(freq, df):
    """Look for evidence of multiple processes."""
    from scipy.signal import find_peaks
    
    # Find all peaks
    peaks, _ = find_peaks(df, prominence=0.05*np.max(df))
    
    if len(peaks) > 1:
        print(f"Found {len(peaks)} peaks - consider multi-Cole-Cole")
        return True
    
    # Check derivative for shoulders
    d_df = np.gradient(df, np.log10(freq))
    d2_df = np.gradient(d_df, np.log10(freq))
    
    inflection_points = find_peaks(-d2_df)[0]
    if len(inflection_points) > 2:
        print("Multiple inflection points - possible hidden relaxations")
        return True
    
    return False

# If multiple relaxations suspected
if check_multiple_relaxations(freq, df):
    # Use unconstrained α to see natural fit
    model = ColeColeModel(constrain_alpha=False)
    result = model.fit(freq, dk, df)
    
    if result.params['alpha'].value > 0.5:
        print("Consider Havriliak-Negami or multi-relaxation model")
```

### 2. Poor Low-Frequency Fit

**Problem**: Model deviates at low frequencies

**Solution**: Add conductivity
```python
# Diagnose conductivity
def diagnose_low_freq_behavior(freq, dk, df):
    """Check for conductivity contribution."""
    
    # Look at lowest decade
    mask = freq < 10 * freq.min()
    
    if np.sum(mask) > 5:
        # Check ε'' ∝ f^(-1)
        omega = 2 * np.pi * freq[mask] * 1e9
        eps_imag = df[mask] * dk[mask]
        
        slope = np.polyfit(np.log10(omega), np.log10(eps_imag), 1)[0]
        
        if -1.2 < slope < -0.8:
            print("Strong conductivity contribution detected")
            print("Use conductivity_correction=True")
            
            # Estimate conductivity
            model_cond = ColeColeModel(conductivity_correction=True)
            return model_cond
    
    return ColeColeModel()
```

### 3. Convergence Issues

**Problem**: Fit doesn't converge or gives unrealistic parameters

**Solutions**:
```python
def robust_fitting(freq, dk, df):
    """Robust fitting strategy for difficult data."""
    
    model = ColeColeModel()
    
    # Strategy 1: Start with Debye, then relax
    params = model.create_parameters(freq, dk, df)
    params['alpha'].value = 0.0  # Start at Debye limit
    
    # Fit with alpha fixed
    params['alpha'].vary = False
    result_debye = model.fit(freq, dk, df, params=params)
    
    # Now release alpha
    params = result_debye.params
    params['alpha'].vary = True
    params['alpha'].min = 0.0
    params['alpha'].max = 0.5
    
    result = model.fit(freq, dk, df, params=params)
    
    # Strategy 2: Global optimization if still poor
    if result.fit_metrics.r_squared < 0.95:
        print("Trying global optimization...")
        result = model.fit(freq, dk, df, 
                          method='differential_evolution',
                          seed=42)
    
    return result
```

### 4. Temperature-Dependent Analysis

**Problem**: Analyzing T-dependent Cole-Cole parameters

**Solution**:
```python
def analyze_temperature_dependence(temperatures, results):
    """Analyze how Cole-Cole parameters vary with temperature."""
    
    tau_values = [r.params['tau'].value for r in results]
    alpha_values = [r.params['alpha'].value for r in results]
    
    # Check tau(T) - usually Arrhenius or VFT
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Arrhenius plot for tau
    ax1.semilogy(1000/temperatures, tau_values, 'o-')
    ax1.set_xlabel('1000/T (K⁻¹)')
    ax1.set_ylabel('τ (s)')
    ax1.set_title('Relaxation Time')
    
    # Temperature dependence of distribution
    ax2.plot(temperatures, alpha_values, 's-')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('α')
    ax2.set_title('Distribution Parameter')
    
    # Fit Arrhenius to tau
    from scipy.stats import linregress
    slope, intercept, r_value, _, _ = linregress(1000/temperatures, np.log10(tau_values))
    
    if r_value**2 > 0.99:
        Ea = slope * 2.303 * 8.617e-5 * 1000  # eV
        print(f"Activation energy: {Ea:.3f} eV")
    else:
        print("Non-Arrhenius behavior - check for VFT")
    
    # Check if α varies with T
    if np.std(alpha_values) / np.mean(alpha_values) > 0.1:
        print("Distribution width is temperature dependent")
        print("This indicates changing molecular dynamics")
```

## Advanced Applications

### 1. Multi-Cole-Cole Fitting

```python
def fit_double_cole_cole(freq, dk, df):
    """Fit sum of two Cole-Cole relaxations."""
    
    def double_cole_cole(freq, eps_inf, 
                        eps_s1, tau1, alpha1,
                        eps_s2, tau2, alpha2):
        omega = 2 * np.pi * freq * 1e9
        
        # First Cole-Cole
        term1 = (eps_s1 - eps_s2) / (1 + (1j * omega * tau1)**(1 - alpha1))
        
        # Second Cole-Cole
        term2 = (eps_s2 - eps_inf) / (1 + (1j * omega * tau2)**(1 - alpha2))
        
        return eps_inf + term1 + term2
    
    # Create custom model
    from lmfit import Model
    model = Model(double_cole_cole, independent_vars=['freq'])
    
    # Initial guesses (requires careful selection)
    params = model.make_params(
        eps_inf=dk[-1],
        eps_s1=dk[0],
        eps_s2=(dk[0] + dk[-1]) / 2,
        tau1=1e-9,
        tau2=1e-11,
        alpha1=0.2,
        alpha2=0.2
    )
    
    # Set bounds
    for p in ['alpha1', 'alpha2']:
        params[p].min = 0
        params[p].max = 0.5
    
    # Complex fitting
    eps_complex = dk + 1j * df * dk
    result = model.fit(eps_complex, params=params, freq=freq)
    
    return result
```

### 2. Electrode Polarization Correction

```python
def correct_electrode_polarization(freq, dk, df, cell_thickness_mm=1.0):
    """Remove electrode polarization effects."""
    
    # Fit Cole-Cole including very low frequencies
    model_full = ColeColeModel(conductivity_correction=True)
    result_full = model_full.fit(freq, dk, df)
    
    # Identify electrode polarization
    # Usually dominates below 1 kHz with large amplitude
    mask_ep = freq < 0.001  # Below 1 MHz
    
    if np.any(mask_ep) and dk[mask_ep].max() > 2 * result_full.params['eps_s'].value:
        print("Electrode polarization detected")
        
        # Fit only above 1 MHz
        mask_valid = freq > 0.001
        result_corrected = model_full.fit(
            freq[mask_valid], 
            dk[mask_valid], 
            df[mask_valid]
        )
        
        return result_corrected
    
    return result_full
```

### 3. Quality Control Applications

```python
def polymer_quality_control(data_file, specifications):
    """Quality control for polymer materials using Cole-Cole."""
    
    # Load data
    import pandas as pd
    data = pd.read_csv(data_file)
    freq = data['Frequency_GHz'].values
    dk = data['Dk'].values
    df = data['Df'].values
    
    # Fit Cole-Cole
    model = ColeColeModel()
    result = model.fit(freq, dk, df)
    
    # Check specifications
    params = result.params
    dist_params = ColeColeModel.get_distribution_parameters(params)
    
    checks = {
        'eps_s': specifications['eps_s_min'] <= params['eps_s'].value <= specifications['eps_s_max'],
        'alpha': params['alpha'].value <= specifications['alpha_max'],
        'distribution_width': dist_params['fwhm_log_decades'] <= specifications['max_width_decades'],
        'fit_quality': result.fit_metrics.r_squared >= specifications['min_r_squared']
    }
    
    # Molecular weight correlation (empirical)
    if 'target_mw' in specifications:
        # Broader distribution often indicates higher MW
        estimated_mw = specifications['mw_alpha_slope'] * params['alpha'].value + specifications['mw_alpha_intercept']
        mw_check = abs(estimated_mw - specifications['target_mw']) / specifications['target_mw'] < 0.1
        checks['molecular_weight'] = mw_check
    
    return {
        'pass': all(checks.values()),
        'parameters': {
            'eps_s': params['eps_s'].value,
            'eps_inf': params['eps_inf'].value,
            'tau': params['tau'].value,
            'alpha': params['alpha'].value,
            'distribution_width_decades': dist_params['fwhm_log_decades']
        },
        'checks': checks
    }
```

### 4. Kramers-Kronig Integration

```python
def validate_cole_cole_with_kk(freq, dk, df):
    """Validate Cole-Cole fit using Kramers-Kronig."""
    from app.models.kramers_kronig_validator import KramersKronigValidator
    import pandas as pd
    
    # First validate experimental data
    data_df = pd.DataFrame({
        'Frequency (GHz)': freq,
        'Dk': dk,
        'Df': df
    })
    
    kk_validator = KramersKronigValidator(data_df)
    kk_result = kk_validator.validate()
    
    print(f"Experimental data KK validation: {kk_result['causality_status']}")
    
    # Fit Cole-Cole
    model = ColeColeModel()
    cc_result = model.fit(freq, dk, df)
    
    # Generate model data over wider frequency range
    freq_extended = np.logspace(-3, 3, 500)  # 1 MHz to 1 THz
    eps_model = model.predict(freq_extended, cc_result.params)
    
    # Validate model
    model_df = pd.DataFrame({
        'Frequency (GHz)': freq_extended,
        'Dk': eps_model.real,
        'Df': eps_model.imag / eps_model.real
    })
    
    kk_validator_model = KramersKronigValidator(model_df)
    kk_model_result = kk_validator_model.validate(causality_threshold=0.01)
    
    print(f"Cole-Cole model KK validation: {kk_model_result['causality_status']}")
    print(f"Model mean relative error: {kk_model_result['mean_relative_error']:.6f}")
    
    return cc_result, kk_model_result
```

## Examples

### Example 1: Polymer Characterization

```python
# Simulated PMMA (poly(methyl methacrylate)) data
import numpy as np

# Generate synthetic PMMA data
freq_ghz = np.logspace(-2, 2, 60)  # 10 MHz to 100 GHz
omega = 2 * np.pi * freq_ghz * 1e9

# PMMA parameters at room temperature
eps_inf = 2.6
eps_s = 3.6
tau = 1e-10  # ~1.6 GHz
alpha = 0.25  # Moderate broadening

# Generate data with Cole-Cole
eps_complex = eps_inf + (eps_s - eps_inf) / (1 + (1j * omega * tau)**(1 - alpha))
dk = eps_complex.real + np.random.normal(0, 0.01, len(freq_ghz))
df = eps_complex.imag / eps_complex.real + np.random.normal(0, 0.0001, len(freq_ghz))

# Fit model
model = ColeColeModel()
result = model.fit(freq_ghz, dk, df)

print("PMMA Fit Results:")
print(model.get_fit_report(result))

# Analyze distribution
dist = ColeColeModel.get_distribution_parameters(result.params)
print(f"\nDistribution width: {dist['fwhm_log_decades']:.2f} decades")
print(f"This indicates molecular weight dispersity of the polymer")
```

### Example 2: Biological Tissue Analysis

```python
# Muscle tissue example
def analyze_muscle_tissue(freq_ghz, dk, df, temperature_C=37):
    """Analyze muscle tissue dielectric properties."""
    
    # Muscle typically shows Cole-Cole behavior
    model = ColeColeModel()
    result = model.fit(freq_ghz, dk, df)
    
    # Extract parameters
    params = result.params
    
    # Tissue-specific analysis
    water_content = estimate_water_content(params['eps_s'].value)
    
    print(f"Muscle Tissue Analysis at {temperature_C}°C:")
    print(f"Static permittivity: {params['eps_s'].value:.1f}")
    print(f"Distribution parameter: {params['alpha'].value:.3f}")
    print(f"Relaxation time: {params['tau'].value:.2e} s")
    print(f"Estimated water content: {water_content:.1f}%")
    
    # Check for pathological changes
    if params['alpha'].value > 0.25:
        print("Warning: Abnormally broad distribution")
        print("May indicate tissue damage or disease")
    
    return result

def estimate_water_content(eps_s, eps_s_water=78.3):
    """Estimate tissue water content from permittivity."""
    # Simple mixture model
    eps_s_protein = 3.0
    
    # Volume fraction of water
    water_fraction = (eps_s - eps_s_protein) / (eps_s_water - eps_s_protein)
    
    # Convert to weight percentage (approximate)
    water_content = water_fraction * 100
    
    return np.clip(water_content, 0, 100)
```

### Example 3: Glass Transition Studies

```python
def study_glass_transition(freq_ghz, dk_data, df_data, temperatures):
    """Analyze Cole-Cole parameters through glass transition."""
    
    results = []
    
    for i, T in enumerate(temperatures):
        model = ColeColeModel()
        result = model.fit(freq_ghz, dk_data[i], df_data[i])
        results.append({
            'T': T,
            'tau': result.params['tau'].value,
            'alpha': result.params['alpha'].value,
            'eps_s': result.params['eps_s'].value,
            'delta_eps': result.params['eps_s'].value - result.params['eps_inf'].value
        })
    
    # Convert to arrays for analysis
    import pandas as pd
    df_results = pd.DataFrame(results)
    
    # Find Tg from maximum in d(log τ)/d(1/T)
    d_log_tau = np.gradient(np.log10(df_results['tau']))
    d_inv_T = np.gradient(1/df_results['T'])
    derivative = d_log_tau / d_inv_T
    
    Tg_index = np.argmax(np.abs(derivative))
    Tg = df_results['T'].iloc[Tg_index]
    
    print(f"Glass transition temperature: {Tg:.1f} K")
    print(f"α at Tg: {df_results['alpha'].iloc[Tg_index]:.3f}")
    
    # Check for fragility
    fragility = abs(derivative[Tg_index]) / (Tg * np.log(10))
    print(f"Fragility index m: {fragility:.1f}")
    
    if fragility > 100:
        print("Fragile glass former")
    else:
        print("Strong glass former")
    
    return df_results, Tg
```

### Example 4: Composite Material Analysis

```python
def analyze_nanocomposite(freq_ghz, dk, df, filler_content):
    """Analyze polymer nanocomposite with interfacial effects."""
    
    # Fit base Cole-Cole
    model = ColeColeModel()
    result = model.fit(freq_ghz, dk, df)
    
    # Check for interfacial polarization
    if result.params['alpha'].value > 0.4:
        print("High α suggests interfacial polarization")
        print("Consider Maxwell-Wagner-Sillars effects")
        
        # Try with conductivity
        model_cond = ColeColeModel(conductivity_correction=True)
        result_cond = model_cond.fit(freq_ghz, dk, df)
        
        if result_cond.fit_metrics.r_squared > result.fit_metrics.r_squared + 0.02:
            print("Conductivity model improves fit")
            result = result_cond
    
    # Analyze filler effect
    alpha = result.params['alpha'].value
    
    # Empirical correlation for many systems
    interface_fraction = estimate_interface_fraction(alpha, filler_content)
    
    print(f"Nanocomposite Analysis:")
    print(f"Filler content: {filler_content:.1f}%")
    print(f"Distribution parameter α: {alpha:.3f}")
    print(f"Estimated interface fraction: {interface_fraction:.1f}%")
    
    return result

def estimate_interface_fraction(alpha, filler_content):
    """Estimate interfacial region from broadening."""
    # Empirical: more interface → broader distribution
    # Assumes α increases linearly with interface fraction
    
    alpha_polymer = 0.2  # Typical for neat polymer
    delta_alpha = alpha - alpha_polymer
    
    # Interface fraction proportional to filler surface area
    interface_fraction = delta_alpha * 100 / 0.3  # Calibration factor
    
    return min(interface_fraction, filler_content * 3)  # Physical limit
```

## Conclusion

The Cole-Cole model is a powerful tool for analyzing dielectric relaxation in materials with distributed relaxation times. Key advantages include:

1. **Single parameter (α)** captures distribution width
2. **Symmetric distribution** simplifies analysis
3. **Well-understood physics** relating to molecular heterogeneity
4. **Wide applicability** to polymers, glasses, and biological materials

Best practices:
- Always check if α is statistically significant vs. Debye
- Consider conductivity for low-frequency data
- Validate with Kramers-Kronig relations
- Use physical constraints when possible
- Consider Havriliak-Negami for asymmetric distributions

The implementation provides robust fitting with intelligent initialization, making it suitable for both research and quality control applications.