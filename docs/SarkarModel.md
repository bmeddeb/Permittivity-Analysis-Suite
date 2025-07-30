# Enhanced Djordjevic-Sarkar Model Documentation

## Table of Contents
- [Overview](#overview)
- [Theory and Physics](#theory-and-physics)
- [Mathematical Formulation](#mathematical-formulation)
- [Physical Interpretation](#physical-interpretation)
- [When to Use D-S Model](#when-to-use-d-s-model)
- [Implementation Details](#implementation-details)
- [Usage Guide](#usage-guide)
- [Parameter Estimation Strategies](#parameter-estimation-strategies)
- [Model Validation and Quality Assessment](#model-validation-and-quality-assessment)
- [Extrapolation and Uncertainty](#extrapolation-and-uncertainty)
- [Model Comparison and Selection](#model-comparison-and-selection)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Advanced Applications](#advanced-applications)
- [Examples](#examples)

## Overview

The Djordjevic-Sarkar (D-S) model provides a causal, wideband representation of frequency-dependent dielectric properties using a logarithmic dispersion formula. This implementation includes robust parameter estimation, comprehensive validation, and uncertainty quantification.

### Key Features

- **Logarithmic dispersion**: Smooth transition over multiple frequency decades
- **Kramers-Kronig consistent**: Guaranteed causality
- **DC conductivity**: Accurate low-frequency loss modeling
- **Robust fitting**: Advanced parameter initialization and validation
- **Uncertainty quantification**: Confidence intervals for extrapolation
- **Model comparison**: Automatic selection vs Multi-Term Debye

### Model Equations

Real part (dispersion):
```
ε'_r(ω) = ε_r,∞ + (ε_r,s - ε_r,∞)/(2ln(ω₂/ω₁)) × ln((ω² + ω₂²)/(ω² + ω₁²))
```

Imaginary part (loss):
```
ε''_r(ω) = σ_DC/(ωε₀) + (ε_r,s - ε_r,∞)/ln(ω₂/ω₁) × [arctan(ω/ω₁) - arctan(ω/ω₂)]
```

Where:
- **ε_r,∞**: High-frequency relative permittivity
- **ε_r,s**: Static (DC) relative permittivity  
- **ω₁, ω₂**: Transition angular frequencies (ω₂ > ω₁)
- **σ_DC**: DC conductivity (S/m)
- **ε₀**: Vacuum permittivity (8.854×10⁻¹² F/m)

## Theory and Physics

### Physical Origins

The D-S model captures broadband dielectric dispersion arising from:

1. **Multiple Relaxation Processes**
   - Distribution of relaxation times
   - Hierarchical molecular dynamics
   - Heterogeneous material structure

2. **Conduction Effects**
   - Ionic conductivity at low frequencies
   - Hopping conduction
   - Space charge effects

3. **Universal Dielectric Response**
   - Power-law behavior at extremes
   - Smooth transition between limits
   - No discrete relaxation times

### Logarithmic Dispersion

The logarithmic formula provides:
- **Smooth monotonic decrease** of ε' with frequency
- **Broad transition** between ε_r,s and ε_r,∞
- **No singularities** or resonances
- **Causal response** via proper ε'' formulation

### Comparison with Other Models

| Model | Parameters | Best For | Limitations |
|-------|------------|----------|-------------|
| **D-S** | 5 (ε_∞, ε_s, f₁, f₂, σ) | Wideband, smooth | No discrete features |
| **Multi-Debye** | 1+2N | Discrete processes | Many parameters |
| **Cole-Cole** | 4 (ε_∞, ε_s, τ, α) | Symmetric broadening | Single process |
| **Havriliak-Negami** | 5 (ε_∞, ε_s, τ, α, β) | Asymmetric broadening | Complex fitting |

## Mathematical Formulation

### Frequency Dependence

The model exhibits three distinct regions:

1. **Low Frequency** (ω << ω₁):
   ```
   ε'_r → ε_r,s
   ε''_r → σ_DC/(ωε₀) + const
   ```
   DC conductivity dominates loss

2. **Transition Region** (ω₁ < ω < ω₂):
   ```
   ε'_r shows logarithmic decrease
   ε''_r peaks near √(ω₁ω₂)
   ```
   Maximum dispersion occurs

3. **High Frequency** (ω >> ω₂):
   ```
   ε'_r → ε_r,∞
   ε''_r → 0
   ```
   Approaches optical limit

### Transition Characteristics

Key parameters describing the transition:

- **Center frequency**: f_c = √(f₁f₂)
- **Transition width**: log₁₀(f₂/f₁) decades
- **Dispersion strength**: Δε = ε_r,s - ε_r,∞
- **Slope at center**: -Δε/(ln(f₂/f₁)) per ln(f)

### Kramers-Kronig Relations

The model satisfies causality:
```
ε'(ω) - ε_∞ = (2/π) × P.V. ∫₀^∞ [ω'ε''(ω')/(ω'² - ω²)] dω'
```

The analytical formula for ε'' ensures this relationship holds exactly.

## Physical Interpretation

### Parameter Meanings

| Parameter | Physical Meaning | Typical Range | Units |
|-----------|-----------------|---------------|-------|
| ε_r,∞ | Electronic polarization | 2-5 | - |
| ε_r,s | Total polarization | 3-15 | - |
| f₁ | Start of dispersion | MHz-GHz | Hz |
| f₂ | End of dispersion | GHz-THz | Hz |
| σ_DC | Bulk conductivity | 10⁻¹²-10⁻³ | S/m |

### Material Signatures

Different materials show characteristic parameter ranges:

1. **FR-4 PCB**
   - ε_r,s ≈ 4.5, ε_r,∞ ≈ 4.0
   - f₁ ≈ 1 MHz, f₂ ≈ 10 GHz
   - σ_DC ≈ 10⁻⁹ S/m

2. **Low-Loss Microwave Substrates**
   - ε_r,s ≈ 3.5, ε_r,∞ ≈ 3.2
   - f₁ ≈ 100 MHz, f₂ ≈ 100 GHz
   - σ_DC < 10⁻¹⁰ S/m

3. **High-K Ceramics**
   - ε_r,s ≈ 10-100, ε_r,∞ ≈ 5-20
   - Wide f₂/f₁ ratio (>10⁴)
   - Variable σ_DC

### Temperature Dependence

Parameters typically vary with temperature:
- **ε_r,s**: Decreases with T (usually)
- **σ_DC**: Arrhenius behavior (exponential)
- **f₁, f₂**: Shift to higher frequencies
- **Δε**: May increase or decrease

## When to Use D-S Model

### Ideal Applications

1. **PCB Substrate Characterization**
   - Signal integrity analysis
   - Wideband material models
   - Manufacturing QC
   - Lot-to-lot comparison

2. **Microwave Materials**
   - Antenna substrates
   - RF/microwave circuits
   - Radome materials
   - Absorber design

3. **Cable Dielectrics**
   - Coaxial cables
   - Twisted pairs
   - Power cables
   - Fiber optic jackets

4. **Simulation Models**
   - EM simulators (HFSS, CST)
   - Circuit simulators (ADS, AWR)
   - Signal integrity tools
   - Time-domain analysis

### Diagnostic Signs for D-S

Use D-S model when observing:
- **Smooth monotonic decrease** of Dk with frequency
- **No resonance peaks** in the loss
- **Wide frequency range** (>3 decades)
- **Need for causal model** in simulation

### When Other Models Are Better

Don't use D-S if:
- **Sharp resonances** present (use Lorentz)
- **Discrete relaxations** visible (use Multi-Debye)
- **Limited frequency range** (<2 decades)
- **Non-monotonic Dk** behavior

## Implementation Details

### Class Structure

```python
class SarkarModel(BaseModel):
    """
    D-S model with validation and uncertainty.
    
    Attributes:
        use_ghz: Frequency units (True=GHz, False=Hz)
        freq_range: Data frequency range (set during fit)
        eps0: Vacuum permittivity constant
    """
```

### Parameter Bounds and Constraints

| Parameter | Default Bounds | Constraints | Notes |
|-----------|---------------|-------------|-------|
| ε_r,∞ | [1.0, 20.0] | < ε_r,s | High-freq limit |
| ε_r,s | [ε_r,∞+0.1, 50.0] | > ε_r,∞ | Static limit |
| f₁ | [0.01f_min, f_max] | < f₂/2 | Lower transition |
| f₂ | [2f₁, 100f_max] | > 2f₁ | Upper transition |
| σ_DC | [0, 10⁻²] | ≥ 0 | DC conductivity |

### Robust Parameter Initialization

The implementation uses:

1. **Median-based estimation** for outlier resistance
2. **Derivative analysis** to find transition region
3. **Low-frequency Df** to estimate σ_DC
4. **Adaptive bounds** based on data range

## Usage Guide

### Basic Fitting

```python
from app.models.sarkar import SarkarModel

# Create model
model = SarkarModel(use_ghz=True)

# Fit data
result = model.fit(freq_ghz, dk_exp, df_exp)

# Check validation
if result.validation['is_valid']:
    print("Parameters are physically valid")
else:
    print(f"Issues: {result.validation['issues']}")

# Get transition characteristics
trans = model.get_transition_characteristics(result.params)
print(f"Transition: {trans['f1_ghz']:.2f} - {trans['f2_ghz']:.2f} GHz")
print(f"Dispersion: {trans['delta_eps']:.3f} ({trans['relative_dispersion']:.1%})")
```

### Model Suitability Assessment

```python
# Check before fitting
suitability = SarkarModel.assess_model_suitability(freq, dk, df)

print(f"Suitability: {suitability['confidence']:.0%}")
if not suitability['suitable']:
    print("Reasons:", suitability['reasons'])
    print("Try:", suitability['alternatives'])

# Warnings indicate potential issues
for warning in suitability['warnings']:
    print(f"Warning: {warning}")
```

### Quality Metrics

```python
# Get comprehensive quality assessment
quality = model.calculate_quality_metrics(result)

print(f"R²: {quality['r_squared']:.4f}")
print(f"RMSE: {quality['rmse']:.6f}")
print(f"Smoothness (Dk): {quality['dk_smoothness']:.4f}")
print(f"Parameter stability: {quality['parameter_stability']:.2f}")
print(f"Physical consistency: {quality['physical_consistency']:.2f}")
print(f"Overall quality: {quality['overall_quality']:.2f}")

# Good fit criteria
if quality['overall_quality'] > 0.8:
    print("Excellent fit quality")
elif quality['overall_quality'] > 0.6:
    print("Good fit quality")
else:
    print("Poor fit - check data or try different model")
```

### Extrapolation with Uncertainty

```python
# Extrapolate beyond measurement range
freq_extrap = np.logspace(-3, 3, 1000)  # 1 MHz to 1 THz
extrap = model.extrapolate_with_uncertainty(
    result.params, 
    freq_extrap,
    confidence=0.95  # 95% confidence intervals
)

# Plot with uncertainty bands
plt.figure(figsize=(10, 6))
plt.fill_between(extrap['frequency'], 
                 extrap['dk_lower'], 
                 extrap['dk_upper'],
                 alpha=0.3, label='95% CI')
plt.loglog(extrap['frequency'], extrap['dk'], 'b-', label='Model')
plt.loglog(freq, dk_exp, 'ko', label='Data')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Dk')
plt.legend()

# Check extrapolation warnings
if 'warning' in extrap:
    print(f"⚠️ {extrap['warning']}")
```

## Parameter Estimation Strategies

### 1. Smart Initialization

```python
def analyze_data_for_initialization(freq, dk, df):
    """Pre-fit analysis for better initialization."""
    
    # Find transition region using derivative
    dk_smooth = np.convolve(dk, np.ones(3)/3, mode='valid')
    d_dk = -np.gradient(dk_smooth)
    
    # Peak in derivative indicates center of transition
    peak_idx = np.argmax(d_dk) + 1
    f_center = freq[peak_idx]
    
    print(f"Estimated transition center: {f_center:.3f} GHz")
    
    # Estimate transition width from 10-90% points
    dk_norm = (dk - dk[-1]) / (dk[0] - dk[-1])
    idx_10 = np.argmin(np.abs(dk_norm - 0.9))
    idx_90 = np.argmin(np.abs(dk_norm - 0.1))
    
    width_decades = np.log10(freq[idx_90] / freq[idx_10])
    print(f"Transition width: {width_decades:.1f} decades")
    
    # Suggest f1, f2
    f1_suggest = f_center / 10**(width_decades/2)
    f2_suggest = f_center * 10**(width_decades/2)
    
    return {
        'f_center': f_center,
        'f1': f1_suggest,
        'f2': f2_suggest,
        'width_decades': width_decades
    }
```

### 2. Constrained Fitting

```python
# Fix permittivity limits if known
result = model.fit(freq, dk, df, 
                   fix_eps_inf=3.0,  # Known high-freq value
                   fix_eps_s=4.5)    # Known static value

# Or create custom constraints
params = model.create_parameters(freq, dk, df)

# Constrain transition width
params.add('width_decades', expr='log10(f2/f1)', max=3.0)

# Ensure minimum dispersion
params.add('delta_eps', expr='eps_r_s - eps_r_inf', min=0.2)

result = model.fit(freq, dk, df, params=params)
```

### 3. Sequential Refinement

```python
def refine_fit_sequentially(model, freq, dk, df):
    """Improve fit through sequential refinement."""
    
    # Step 1: Fit without conductivity
    params = model.create_parameters(freq, dk, df)
    params['sigma_dc'].vary = False
    result1 = model.fit(freq, dk, df, params=params)
    
    # Step 2: Add conductivity with good initial guess
    params = result1.params
    params['sigma_dc'].vary = True
    
    # Estimate from low-freq residuals
    eps_model = model.predict(freq[:5], params)
    df_residual = df[:5] - eps_model.imag / eps_model.real
    sigma_est = np.mean(df_residual * dk[:5] * 2*np.pi*freq[:5]*1e9 * model.eps0)
    params['sigma_dc'].value = max(0, sigma_est)
    
    # Final fit
    result2 = model.fit(freq, dk, df, params=params)
    
    print(f"R² improvement: {result1.fit_metrics.r_squared:.4f} → "
          f"{result2.fit_metrics.r_squared:.4f}")
    
    return result2
```

## Model Validation and Quality Assessment

### Parameter Validation

```python
# Automatic validation during fit
result = model.fit(freq, dk, df)

# Manual validation
is_valid, issues = model.validate_parameters(result.params)

if not is_valid:
    print("Parameter issues found:")
    for issue in issues:
        print(f"  - {issue}")
    
    # Common fixes
    if "f2/f1 = ... is too small" in str(issues):
        print("→ Try wider frequency range or different model")
    
    if "Dispersion ... > 90%" in str(issues):
        print("→ Check for measurement errors or additional processes")
```

### Quality Score Interpretation

The overall quality score (0-1) combines:
- **30%** R² (goodness of fit)
- **30%** MAPE (accuracy)
- **20%** Physical consistency
- **10%** Parameter stability
- **10%** Extrapolation quality

Interpretation:
- **>0.85**: Excellent - suitable for precision applications
- **0.70-0.85**: Good - suitable for most applications
- **0.50-0.70**: Acceptable - check for improvements
- **<0.50**: Poor - reconsider model choice

### Smoothness Metrics

```python
quality = model.calculate_quality_metrics(result)

# Smoothness values (lower is better)
if quality['dk_smoothness'] > 0.01:
    print("Dk fit may be too noisy")
    print("Consider:")
    print("- Smoothing input data")
    print("- Reducing number of parameters")
    print("- Adding regularization")

if quality['df_smoothness'] > 0.1:
    print("Df fit shows oscillations")
    print("Check for:")
    print("- Measurement noise")
    print("- Inadequate model")
```

## Extrapolation and Uncertainty

### Understanding Uncertainty Growth

Uncertainty increases with distance from measured data:
```
Uncertainty = Base × (1 + 0.1 × decades_from_data)
```

Example:
- Within data range: ±1%
- 1 decade beyond: ±11%
- 2 decades beyond: ±21%

### Practical Extrapolation Limits

```python
def assess_extrapolation_reliability(model, result, target_freq):
    """Check if extrapolation is reliable."""
    
    data_min, data_max = model.freq_range
    
    # Calculate extrapolation distance
    if target_freq < data_min:
        decades = np.log10(data_min / target_freq)
        direction = "below"
    elif target_freq > data_max:
        decades = np.log10(target_freq / data_max)
        direction = "above"
    else:
        decades = 0
        direction = "within"
    
    # Reliability assessment
    if decades == 0:
        reliability = "Excellent (within data)"
    elif decades < 1:
        reliability = "Good (<1 decade)"
    elif decades < 2:
        reliability = "Fair (1-2 decades)"
    else:
        reliability = "Poor (>2 decades)"
    
    print(f"Extrapolating {decades:.1f} decades {direction} data")
    print(f"Reliability: {reliability}")
    
    # Get specific uncertainty
    extrap = model.extrapolate_with_uncertainty(
        result.params, 
        np.array([target_freq])
    )
    
    uncertainty_pct = extrap['dk_uncertainty'][0] / extrap['dk'][0] * 100
    print(f"Expected uncertainty: ±{uncertainty_pct:.1f}%")
    
    return reliability, uncertainty_pct
```

### Confidence Intervals

```python
# 95% confidence intervals (default)
extrap_95 = model.extrapolate_with_uncertainty(
    result.params, freq_extrap, confidence=0.95
)

# 99% confidence intervals (more conservative)
extrap_99 = model.extrapolate_with_uncertainty(
    result.params, freq_extrap, confidence=0.99
)

# Compare intervals
print(f"At 100 GHz:")
idx = np.argmin(np.abs(freq_extrap - 100))
print(f"  Dk = {extrap_95['dk'][idx]:.3f}")
print(f"  95% CI: [{extrap_95['dk_lower'][idx]:.3f}, "
      f"{extrap_95['dk_upper'][idx]:.3f}]")
print(f"  99% CI: [{extrap_99['dk_lower'][idx]:.3f}, "
      f"{extrap_99['dk_upper'][idx]:.3f}]")
```

## Model Comparison and Selection

### D-S vs Multi-Term Debye

```python
# Automatic comparison
comparison = SarkarModel.compare_with_multi_debye(freq, dk, df)

print(f"Best Multi-Debye: {comparison['best_multi_debye_terms']} terms")
print(f"BIC difference: {comparison['ds_vs_md_bic_difference']:.1f}")
print(f"Recommendation: {comparison['recommendation']}")

# Detailed comparison
ds_quality = comparison['ds_model']['quality']
print(f"\nD-S Model:")
print(f"  Parameters: 5")
print(f"  R²: {ds_quality['r_squared']:.4f}")
print(f"  Overall quality: {ds_quality['overall_quality']:.2f}")

for n_terms, md_data in comparison['multi_debye'].items():
    print(f"\nMulti-Debye ({n_terms} terms):")
    print(f"  Parameters: {md_data['n_params']}")
    print(f"  R²: {md_data['r_squared']:.4f}")
    print(f"  BIC: {md_data['bic']:.1f}")
```

### Model Selection Guidelines

Choose based on BIC difference (ΔBIC = BIC_DS - BIC_MD):
- **ΔBIC < -10**: D-S strongly preferred
- **ΔBIC < -2**: D-S preferred
- **|ΔBIC| < 2**: Models comparable
- **ΔBIC > 2**: Multi-Debye preferred
- **ΔBIC > 10**: Multi-Debye strongly preferred

Additional considerations:
- **Extrapolation**: D-S better for wideband
- **Physical meaning**: D-S has clearer parameters
- **Simulation**: D-S guaranteed causal
- **Flexibility**: Multi-Debye can fit complex shapes

## Common Issues and Solutions

### 1. Poor Low-Frequency Fit

**Problem**: Model doesn't fit low-frequency Df well

**Solution**:
```python
# Check if conductivity is needed
if freq[0] < 0.01:  # Below 10 MHz
    # Ensure sigma_dc is varying
    params = model.create_parameters(freq, dk, df)
    if not params['sigma_dc'].vary:
        print("Enabling conductivity fitting")
        params['sigma_dc'].vary = True
        params['sigma_dc'].min = 1e-12
        params['sigma_dc'].max = 1e-6
    
    # If still poor, check for electrode effects
    low_freq_df = df[:5]
    if np.all(np.diff(low_freq_df) > 0):
        print("Increasing Df at low freq - possible electrode polarization")
        print("Consider removing lowest frequencies")
```

### 2. Non-Monotonic Dk

**Problem**: Dk increases with frequency in some regions

**Solution**:
```python
# Detect non-monotonic behavior
dk_diff = np.diff(dk)
if np.any(dk_diff > 0):
    # Find problem regions
    problem_idx = np.where(dk_diff > 0)[0]
    problem_freqs = freq[problem_idx]
    
    print(f"Non-monotonic Dk at: {problem_freqs} GHz")
    
    # Options:
    # 1. Smooth data
    from scipy.ndimage import gaussian_filter1d
    dk_smooth = gaussian_filter1d(dk, sigma=1)
    
    # 2. Remove outliers
    dk_cleaned = dk.copy()
    for idx in problem_idx:
        dk_cleaned[idx] = (dk_cleaned[idx-1] + dk_cleaned[idx+1]) / 2
    
    # 3. Consider different model
    print("Non-monotonic Dk suggests:")
    print("- Measurement errors")
    print("- Resonance effects (try Hybrid model)")
    print("- Multiple materials (try effective medium)")
```

### 3. Convergence Issues

**Problem**: Fit doesn't converge or gives unrealistic parameters

**Solution**:
```python
def robust_fitting_strategy(model, freq, dk, df):
    """Try multiple fitting strategies."""
    
    strategies = [
        {'method': 'leastsq', 'max_nfev': 1000},
        {'method': 'least_squares', 'ftol': 1e-10},
        {'method': 'differential_evolution', 'seed': 42}
    ]
    
    best_result = None
    best_quality = 0
    
    for i, strategy in enumerate(strategies):
        print(f"\nTrying strategy {i+1}: {strategy['method']}")
        
        try:
            result = model.fit(freq, dk, df, **strategy)
            quality = model.calculate_quality_metrics(result)
            
            if quality['overall_quality'] > best_quality:
                best_quality = quality['overall_quality']
                best_result = result
                print(f"  Success! Quality: {quality['overall_quality']:.3f}")
        except Exception as e:
            print(f"  Failed: {e}")
    
    if best_result is None:
        print("\nAll strategies failed - check data quality")
    
    return best_result
```

### 4. Unrealistic Transition Width

**Problem**: f₂/f₁ ratio is extremely large or small

**Solution**:
```python
# Add constraints on transition width
params = model.create_parameters(freq, dk, df)

# Limit to reasonable range (2-4 decades)
min_ratio = 100   # 2 decades
max_ratio = 10000 # 4 decades

params['f2'].min = params['f1'].value * min_ratio
params['f2'].max = params['f1'].value * max_ratio

# Or fix ratio if known
if material_type == "FR4":
    params.add('f_ratio_fixed', value=1000, vary=False)
    params['f2'].expr = 'f1 * f_ratio_fixed'

result = model.fit(freq, dk, df, params=params)
```

## Advanced Applications

### 1. Temperature-Dependent Analysis

```python

def analyze_temperature_series(temps, freq, dk_data, df_data):
    """Extract temperature dependence of D-S parameters."""
    
    model = SarkarModel()
    results = []
    
    for i, T in enumerate(temps):
        print(f"\nFitting T = {T} K")
        result = model.fit(freq, dk_data[i], df_data[i])
        
        if result.validation['is_valid']:
            p = result.params.valuesdict()
            results.append({
                'T': T,
                'eps_inf': p['eps_r_inf'],
                'eps_s': p['eps_r_s'],
                'f1': p['f1'],
                'f2': p['f2'],
                'sigma_dc': p['sigma_dc'],
                'delta_eps': p['eps_r_s'] - p['eps_r_inf']
            })
    
    # Analyze trends
    df_results = pd.DataFrame(results)
    
    # Arrhenius plot for conductivity
    if df_results['sigma_dc'].min() > 0:
        plt.figure(figsize=(8, 6))
        plt.semilogy(1000/df_results['T'], df_results['sigma_dc'], 'bo-')
        plt.xlabel('1000/T (K⁻¹)')
        plt.ylabel('σ_DC (S/m)')
        plt.title('Arrhenius Plot - DC Conductivity')
        plt.grid(True)
        
        # Fit activation energy
        x = 1/df_results['T']
        y = np.log(df_results['sigma_dc'])
        slope, intercept = np.polyfit(x, y, 1)
        E_a = -slope * 8.617e-5  # eV
        print(f"Activation energy: {E_a:.3f} eV")
    
    # Temperature coefficient of permittivity
    tc_eps = np.polyfit(df_results['T'], df_results['eps_s'], 1)[0]
    print(f"Temperature coefficient: {tc_eps*1e6:.1f} ppm/K")
    
    return df_results
```

### 2. PCB Manufacturing QC

```python
def pcb_quality_control(freq, dk, df, material_spec):
    """Automated QC for PCB materials."""
    
    model = SarkarModel()
    
    # Fit model
    result = model.fit(freq, dk, df)
    
    # Extract key parameters at specification frequencies
    spec_freqs = material_spec['test_frequencies']  # e.g., [1, 10] GHz
    eps_at_spec = model.predict(spec_freqs, result.params)
    
    qc_report = {
        'lot_id': material_spec['lot_id'],
        'date': pd.Timestamp.now(),
        'pass': True,
        'parameters': result.params.valuesdict(),
        'issues': []
    }
    
    # Check Dk at spec frequencies
    for i, f_spec in enumerate(spec_freqs):
        dk_measured = eps_at_spec[i].real
        dk_nominal = material_spec['dk_nominal'][i]
        dk_tol = material_spec['dk_tolerance'][i]
        
        deviation = abs(dk_measured - dk_nominal) / dk_nominal * 100
        
        if deviation > dk_tol:
            qc_report['pass'] = False
            qc_report['issues'].append(
                f"Dk at {f_spec} GHz: {dk_measured:.3f} "
                f"(spec: {dk_nominal:.3f} ± {dk_tol}%)"
            )
    
    # Check Df
    for i, f_spec in enumerate(spec_freqs):
        df_measured = eps_at_spec[i].imag / eps_at_spec[i].real
        df_max = material_spec['df_max'][i]
        
        if df_measured > df_max:
            qc_report['pass'] = False
            qc_report['issues'].append(
                f"Df at {f_spec} GHz: {df_measured:.4f} "
                f"(max: {df_max:.4f})"
            )
    
    # Model quality check
    quality = model.calculate_quality_metrics(result)
    if quality['overall_quality'] < 0.8:
        qc_report['issues'].append(
            f"Poor model fit quality: {quality['overall_quality']:.2f}"
        )
    
    # Generate report
    print(f"\nQC Report for Lot {qc_report['lot_id']}")
    print(f"Status: {'PASS' if qc_report['pass'] else 'FAIL'}")
    
    if qc_report['issues']:
        print("\nIssues found:")
        for issue in qc_report['issues']:
            print(f"  - {issue}")
    
    return qc_report
```

### 3. Signal Integrity Modeling

```python
def create_si_model(freq, dk, df, trace_geometry):
    """Create signal integrity model from measurements."""
    
    model = SarkarModel()
    result = model.fit(freq, dk, df)
    
    # Verify causality
    df_val = pd.DataFrame({
        'Frequency (GHz)': freq,
        'Dk': result.dk_fit,
        'Df': result.df_fit
    })
    
    with KramersKronigValidator(df_val) as kk:
        kk_result = kk.validate()
        print(f"Causality check: {kk_result['causality_status']}")
    
    # Generate S-parameters for transmission line
    freq_si = np.logspace(np.log10(0.01), np.log10(50), 1000)  # 10 MHz - 50 GHz
    
    s_params = model.to_touchstone(
        result.params,
        freq_si * 1e9,  # Convert to Hz
        thickness_m=trace_geometry['substrate_thickness'],
        z0=50
    )
    
    # Calculate key SI metrics
    alpha_dB_inch = s_params['attenuation_dB_m'] * 0.0254
    tpd = 1e12 / (3e8 / np.sqrt(eps_at_1ghz.real))  # ps/inch
    
    print(f"\nSignal Integrity Parameters:")
    print(f"  Loss at 10 GHz: {alpha_dB_inch[500]:.2f} dB/inch")
    print(f"  Propagation delay: {tpd:.1f} ps/inch")
    
    # Create SPICE model
    spice = model.generate_spice_model(result.params)
    
    return {
        's_parameters': s_params,
        'spice_model': spice,
        'dk_model': result.params,
        'si_metrics': {
            'loss_10ghz_db_inch': alpha_dB_inch[500],
            'prop_delay_ps_inch': tpd
        }
    }
```

### 4. Batch Processing and Database

```python
def process_material_database(database_path, output_path):
    """Process multiple materials and build parameter database."""
    
    import sqlite3
    
    model = SarkarModel()
    
    # Create results database
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ds_parameters (
            material_id TEXT PRIMARY KEY,
            material_name TEXT,
            eps_inf REAL,
            eps_s REAL,
            f1_ghz REAL,
            f2_ghz REAL,
            sigma_dc REAL,
            r_squared REAL,
            quality_score REAL,
            measurement_date TEXT
        )
    ''')
    
    # Process each material
    materials = load_material_database(database_path)
    
    for mat_id, mat_data in materials.items():
        print(f"\nProcessing {mat_data['name']}...")
        
        try:
            # Fit model
            result = model.fit(
                mat_data['frequency'],
                mat_data['dk'],
                mat_data['df']
            )
            
            # Calculate quality
            quality = model.calculate_quality_metrics(result)
            
            # Store results
            p = result.params.valuesdict()
            cursor.execute('''
                INSERT OR REPLACE INTO ds_parameters 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                mat_id,
                mat_data['name'],
                p['eps_r_inf'],
                p['eps_r_s'],
                p['f1'],
                p['f2'],
                p['sigma_dc'],
                result.fit_metrics.r_squared,
                quality['overall_quality'],
                mat_data['date']
            ))
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"\nProcessed {len(materials)} materials")
    print(f"Database saved to: {output_path}")
```

## Examples

### Example 1: FR-4 PCB Characterization

```python
import numpy as np
import matplotlib.pyplot as plt

# Typical FR-4 data
freq_ghz = np.logspace(-3, 1.5, 50)  # 1 MHz to 30 GHz
# Simulated FR-4 response
dk_fr4 = 4.5 - 0.3 * np.log10(1 + freq_ghz/0.01)
df_fr4 = 0.02 * (1 + 0.5 * freq_ghz**0.3) / (1 + freq_ghz)

# Add realistic noise
np.random.seed(42)
dk_fr4 += np.random.normal(0, 0.01, len(dk_fr4))
df_fr4 += np.random.normal(0, 0.0005, len(df_fr4))

# Create and fit model
model = SarkarModel()

# Check suitability
suitability = model.assess_model_suitability(freq_ghz, dk_fr4, df_fr4)
print(f"Model suitability: {suitability['confidence']:.0%}")

# Fit with validation
result = model.fit(freq_ghz, dk_fr4, df_fr4)

print("\nFR-4 Analysis Results:")
print("=" * 50)

# Parameters
trans = model.get_transition_characteristics(result.params)
print(f"εr,∞ = {result.params['eps_r_inf'].value:.3f}")
print(f"εr,s = {result.params['eps_r_s'].value:.3f}")
print(f"Transition: {trans['f1_ghz']:.3f} - {trans['f2_ghz']:.3f} GHz")
print(f"σ_DC = {result.params['sigma_dc'].value:.2e} S/m")

# Quality assessment
quality = model.calculate_quality_metrics(result)
print(f"\nFit Quality:")
print(f"  R² = {quality['r_squared']:.4f}")
print(f"  Overall = {quality['overall_quality']:.2f}")

# Validation
if result.validation['is_valid']:
    print("\n✓ Parameters physically valid")
else:
    print("\n✗ Validation issues:", result.validation['issues'])

# Plot with extrapolation
plot_data = model.get_plot_data(result, show_uncertainty=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Dk plot with uncertainty
ax1.fill_between(plot_data['freq'], 
                 plot_data['dk_lower'], 
                 plot_data['dk_upper'],
                 alpha=0.2, color='blue', label='95% CI')
ax1.semilogx(plot_data['freq'], plot_data['dk_fit'], 'b-', label='D-S Model')
ax1.semilogx(freq_ghz, dk_fr4, 'ko', markersize=4, label='Data')
ax1.axvline(trans['f1_ghz'], color='r', linestyle='--', alpha=0.5)
ax1.axvline(trans['f2_ghz'], color='r', linestyle='--', alpha=0.5)
ax1.set_ylabel('Dk')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('FR-4 Dielectric Properties')

# Df plot
ax2.loglog(plot_data['freq'], plot_data['df_fit'], 'b-', label='D-S Model')
ax2.loglog(freq_ghz, df_fr4, 'ko', markersize=4, label='Data')
ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('Df')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Example 2: Model Comparison Study

```python
# Compare D-S with Multi-Debye for a low-loss substrate
freq_ghz = np.logspace(-2, 2, 80)  # 10 MHz to 100 GHz

# Generate test data with known D-S parameters
true_params = {
    'eps_r_inf': 3.0,
    'eps_r_s': 3.5,
    'f1': 0.1,  # 100 MHz
    'f2': 100,  # 100 GHz
    'sigma_dc': 1e-10
}

# Generate synthetic data
model_true = SarkarModel()
eps_true = model_true.model_func(freq_ghz, **true_params)
dk_true = eps_true.real
df_true = eps_true.imag / eps_true.real

# Add noise
dk_meas = dk_true + np.random.normal(0, 0.005, len(dk_true))
df_meas = df_true + np.random.normal(0, 0.0001, len(df_true))

# Compare models
comparison = SarkarModel.compare_with_multi_debye(
    freq_ghz, dk_meas, df_meas
)

print("Model Comparison Results:")
print("=" * 50)

# D-S results
ds_r2 = comparison['ds_model']['quality']['r_squared']
ds_params = comparison['ds_model']['result'].params

print(f"\nD-S Model (5 parameters):")
print(f"  R² = {ds_r2:.4f}")
print(f"  BIC = {comparison['ds_model']['bic']:.1f}")
print(f"  εr,∞ = {ds_params['eps_r_inf'].value:.3f} "
      f"(true: {true_params['eps_r_inf']:.3f})")
print(f"  εr,s = {ds_params['eps_r_s'].value:.3f} "
      f"(true: {true_params['eps_r_s']:.3f})")

# Best Multi-Debye
best_n = comparison['best_multi_debye_terms']
best_md = comparison['multi_debye'][best_n]

print(f"\nBest Multi-Debye ({best_n} terms, {1+best_n} parameters):")
print(f"  R² = {best_md['r_squared']:.4f}")
print(f"  BIC = {best_md['bic']:.1f}")

# Recommendation
print(f"\nBIC difference: {comparison['ds_vs_md_bic_difference']:.1f}")
print(f"Recommendation: {comparison['recommendation']}")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Dk comparison
ax1.semilogx(freq_ghz, dk_meas, 'ko', markersize=3, label='Data')
ax1.semilogx(freq_ghz, 
             comparison['ds_model']['result'].dk_fit, 
             'b-', linewidth=2, label='D-S')
ax1.semilogx(freq_ghz, 
             best_md['result'].dk_fit, 
             'r--', linewidth=2, label=f'Multi-Debye ({best_n})')
ax1.set_xlabel('Frequency (GHz)')
ax1.set_ylabel('Dk')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Residuals
ds_resid = (comparison['ds_model']['result'].dk_fit - dk_meas) * 1000
md_resid = (best_md['result'].dk_fit - dk_meas) * 1000

ax2.semilogx(freq_ghz, ds_resid, 'b-', label='D-S')
ax2.semilogx(freq_ghz, md_resid, 'r--', label=f'Multi-Debye ({best_n})')
ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('Dk Residuals (×10³)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('D-S vs Multi-Debye Model Comparison')
plt.tight_layout()
plt.show()
```

### Example 3: Temperature Series Analysis

```python
# Temperature-dependent measurements
temps = np.array([233, 253, 273, 293, 313, 333, 353])  # K
freq_ghz = np.logspace(-2, 1.5, 40)  # 10 MHz to 30 GHz

# Simulate temperature-dependent data
dk_data = []
df_data = []

for T in temps:
    # Temperature-dependent parameters
    eps_inf = 3.8 - 0.0005 * (T - 293)
    eps_s = 4.3 - 0.001 * (T - 293)
    f1 = 0.01 * np.exp((T - 293) / 50)  # Activated process
    f2 = 10 * np.exp((T - 293) / 100)
    sigma_dc = 1e-10 * np.exp(-0.5 * 11600 / T)  # 0.5 eV activation
    
    # Generate data
    eps_model = model.model_func(freq_ghz, 
                                eps_r_inf=eps_inf,
                                eps_r_s=eps_s,
                                f1=f1,
                                f2=f2,
                                sigma_dc=sigma_dc)
    
    dk_data.append(eps_model.real + np.random.normal(0, 0.005, len(freq_ghz)))
    df_data.append(eps_model.imag/eps_model.real + 
                   np.random.normal(0, 0.0001, len(freq_ghz)))

# Analyze series
df_results = analyze_temperature_series(temps, freq_ghz, dk_data, df_data)

# Plot parameter evolution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Static permittivity
ax = axes[0, 0]
ax.plot(temps, df_results['eps_s'], 'bo-', markersize=8)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('εr,s')
ax.grid(True, alpha=0.3)
ax.set_title('Static Permittivity')

# Dispersion strength
ax = axes[0, 1]
ax.plot(temps, df_results['delta_eps'], 'ro-', markersize=8)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Δε')
ax.grid(True, alpha=0.3)
ax.set_title('Dispersion Strength')

# Transition frequencies
ax = axes[1, 0]
ax.semilogy(temps, df_results['f1'], 'go-', markersize=8, label='f₁')
ax.semilogy(temps, df_results['f2'], 'mo-', markersize=8, label='f₂')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Frequency (GHz)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Transition Frequencies')

# DC conductivity (Arrhenius)
ax = axes[1, 1]
ax.semilogy(1000/temps, df_results['sigma_dc'], 'ko-', markersize=8)
ax.set_xlabel('1000/T (K⁻¹)')
ax.set_ylabel('σ_DC (S/m)')
ax.grid(True, alpha=0.3)
ax.set_title('Arrhenius Plot - Conductivity')

# Add linear fit
x = 1000/temps
y = np.log(df_results['sigma_dc'])
p = np.polyfit(x, y, 1)
ax.semilogy(x, np.exp(np.polyval(p, x)), 'r--', 
            label=f'Ea = {-p[0]*0.0862:.3f} eV')
ax.legend()

plt.tight_layout()
plt.show()

print("\nTemperature Analysis Summary:")
print(f"Temperature coefficient of εr,s: "
      f"{np.polyfit(temps, df_results['eps_s'], 1)[0]*1e6:.0f} ppm/K")
print(f"Conductivity activation energy: {-p[0]*0.0862:.3f} eV")
```

### Example 4: High-Speed Digital Application

```python
# High-speed digital signal integrity analysis
# 56 Gbps PAM4 application

# Load measurement data for high-speed substrate
freq_ghz = np.logspace(-2, 2, 100)  # 10 MHz to 100 GHz
# (Load actual dk, df data here)

# Fit model with emphasis on high-frequency accuracy
model = SarkarModel()

# Weight high frequencies more
weights = np.ones(len(freq_ghz))
weights[freq_ghz > 10] *= 2  # Double weight above 10 GHz

result = model.fit(freq_ghz, dk, df, weights=weights)

# Signal integrity analysis
print("High-Speed Digital SI Analysis")
print("=" * 50)

# Key frequencies for 56 Gbps PAM4
nyquist_freq = 28  # GHz
f_knee = 0.35 * 56  # 19.6 GHz

# Evaluate at key frequencies
key_freqs = np.array([1, 10, nyquist_freq, 50])
eps_at_keys = model.predict(key_freqs, result.params)

print("\nDielectric Properties at Key Frequencies:")
print(f"{'Freq (GHz)':>10} {'Dk':>8} {'Df':>8} {'Loss (dB/in)':>12}")
print("-" * 40)

for i, f in enumerate(key_freqs):
    dk = eps_at_keys[i].real
    df = eps_at_keys[i].imag / dk
    # Loss in dB/inch
    alpha = 2.3 * f * np.sqrt(dk) * df  # Approximate
    print(f"{f:>10.1f} {dk:>8.3f} {df:>8.4f} {alpha:>12.2f}")

# Generate touchstone for 2-inch trace
trace_length = 2 * 0.0254  # 2 inches in meters
substrate_thickness = 0.1e-3  # 100 um

s_params = model.to_touchstone(
    result.params,
    np.logspace(7, 11, 1000),  # 10 MHz to 100 GHz in Hz
    thickness_m=substrate_thickness,
    z0=50
)

# Find -3dB bandwidth
s21_db = 20 * np.log10(np.abs(s_params['s21']))
idx_3db = np.argmin(np.abs(s21_db + 3))
bw_3db = s_params['frequency_hz'][idx_3db] / 1e9

print(f"\n2-inch Trace Performance:")
print(f"  3-dB Bandwidth: {bw_3db:.1f} GHz")
print(f"  Loss at Nyquist ({nyquist_freq} GHz): "
      f"{s21_db[np.argmin(np.abs(s_params['frequency_hz']/1e9 - nyquist_freq))]:.1f} dB")

# Eye diagram impact estimate
loss_nyquist = -s21_db[np.argmin(np.abs(s_params['frequency_hz']/1e9 - nyquist_freq))]
eye_closure_pct = (1 - 10**(-loss_nyquist/20)) * 100

print(f"  Estimated eye closure: {eye_closure_pct:.0f}%")

# Save S-parameters
with open('substrate_2inch.s2p', 'w') as f:
    f.write('# MHz S MA R 50\n')
    f.write('! 2-inch trace on characterized substrate\n')
    for i in range(len(s_params['frequency_hz'])):
        f_mhz = s_params['frequency_hz'][i] / 1e6
        s11 = s_params['s11'][i]
        s21 = s_params['s21'][i]
        f.write(f'{f_mhz:.3f} {np.abs(s11):.6f} {np.angle(s11, deg=True):.3f} ')
        f.write(f'{np.abs(s21):.6f} {np.angle(s21, deg=True):.3f} ')
        f.write(f'{np.abs(s21):.6f} {np.angle(s21, deg=True):.3f} ')
        f.write(f'{np.abs(s11):.6f} {np.angle(s11, deg=True):.3f}\n')

print(f"\nS-parameters saved to: substrate_2inch.s2p")
```

### Example 5: Automated Report Generation

```python
def generate_material_report(freq, dk, df, material_info):
    """Generate comprehensive characterization report."""
    
    from datetime import datetime
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Fit model
    model = SarkarModel()
    result = model.fit(freq, dk, df)
    
    # Create PDF report
    pdf_file = f"{material_info['name']}_report_{datetime.now():%Y%m%d}.pdf"
    
    with PdfPages(pdf_file) as pdf:
        # Page 1: Summary
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(f"Dielectric Characterization Report\n{material_info['name']}", 
                    fontsize=16, fontweight='bold')
        
        # Add text summary
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Get all results
        trans = model.get_transition_characteristics(result.params)
        quality = model.calculate_quality_metrics(result)
        validation = result.validation
        
        report_text = f"""
Material: {material_info['name']}
Date: {datetime.now():%Y-%m-%d %H:%M}
Operator: {material_info.get('operator', 'Unknown')}

MEASUREMENT SUMMARY
Frequency Range: {freq.min():.3f} - {freq.max():.3f} GHz
Number of Points: {len(freq)}
Temperature: {material_info.get('temperature', 25)}°C

D-S MODEL PARAMETERS
εr,∞ = {result.params['eps_r_inf'].value:.3f} ± {result.params['eps_r_inf'].stderr or 0:.3f}
εr,s = {result.params['eps_r_s'].value:.3f} ± {result.params['eps_r_s'].stderr or 0:.3f}
f₁ = {trans['f1_ghz']:.3f} GHz
f₂ = {trans['f2_ghz']:.3f} GHz
σ_DC = {result.params['sigma_dc'].value:.2e} S/m

TRANSITION CHARACTERISTICS
Center Frequency: {trans['f_center_ghz']:.2f} GHz
Dispersion Strength: {trans['delta_eps']:.3f} ({trans['relative_dispersion']:.1%})
Transition Width: {trans['transition_width_decades']:.1f} decades
Slope: {trans['slope_per_decade']:.3f} per decade

FIT QUALITY
R² = {quality['r_squared']:.4f}
RMSE = {quality['rmse']:.6f}
Overall Quality Score = {quality['overall_quality']:.2f}
Physical Consistency = {'PASS' if validation['is_valid'] else 'FAIL'}

EXTRAPOLATION LIMITS
Reliable to: {freq.min()/10:.3f} - {freq.max()*10:.3f} GHz
Caution beyond: {freq.min()/100:.3f} - {freq.max()*100:.3f} GHz
"""
        
        ax.text(0.1, 0.9, report_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Plots
        plot_data = model.get_plot_data(result, show_uncertainty=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle(f"{material_info['name']} - Measurement Plots", 
                    fontsize=14, fontweight='bold')
        
        # Dk vs frequency
        ax = axes[0, 0]
        ax.semilogx(freq, dk, 'ko', markersize=4, label='Measured')
        ax.semilogx(plot_data['freq'], plot_data['dk_fit'], 'b-', label='D-S Fit')
        if 'dk_lower' in plot_data:
            ax.fill_between(plot_data['freq'], 
                           plot_data['dk_lower'], 
                           plot_data['dk_upper'],
                           alpha=0.2, color='blue')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Dk')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Dielectric Constant')
        
        # Df vs frequency
        ax = axes[0, 1]
        ax.loglog(freq, df, 'ko', markersize=4, label='Measured')
        ax.loglog(plot_data['freq'], plot_data['df_fit'], 'b-', label='D-S Fit')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Df')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Loss Factor')
        
        # Residuals
        ax = axes[1, 0]
        dk_resid = (result.dk_fit - dk) / dk * 100
        df_resid = (result.df_fit - df) / df * 100
        
        ax.semilogx(freq, dk_resid, 'bo-', label='Dk', markersize=4)
        ax.semilogx(freq, df_resid, 'ro-', label='Df', markersize=4)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Relative Error (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Fit Residuals')
        ax.set_ylim(-10, 10)
        
        # Cole-Cole plot
        ax = axes[1, 1]
        eps_complex = dk + 1j * df * dk
        ax.plot(eps_complex.real, eps_complex.imag, 'ko', markersize=4, label='Measured')
        eps_fit = plot_data['eps_real'] + 1j * plot_data['eps_imag']
        ax.plot(eps_fit.real, eps_fit.imag, 'b-', label='D-S Fit')
        ax.set_xlabel("ε'")
        ax.set_ylabel('ε"')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Cole-Cole Plot')
        ax.axis('equal')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Report saved to: {pdf_file}")
    
    return pdf_file

# Example usage
material_info = {
    'name': 'RO4003C',
    'operator': 'J. Smith',
    'temperature': 23,
    'humidity': 45
}

report_file = generate_material_report(freq_ghz, dk, df, material_info)
```

## Troubleshooting Guide

### Common Error Messages

1. **"D-S model may not be suitable"**
   - Check data quality (noise, outliers)
   - Verify monotonic Dk behavior
   - Consider alternative models if resonances present

2. **"Parameter validation issues"**
   - Review parameter bounds
   - Check for unrealistic values
   - May indicate poor data quality

3. **"Extrapolating X decades beyond data"**
   - Limit extrapolation to 1-2 decades
   - Use uncertainty bands
   - Collect data over wider range if needed

4. **"Poor fit quality"**
   - Try different weighting schemes
   - Check for missing processes
   - Consider Multi-Debye model

### Performance Optimization

For large datasets or batch processing:

```python
# Enable parallel processing for batch fits
from multiprocessing import Pool

def process_single_material(args):
    freq, dk, df, mat_id = args
    model = SarkarModel()
    try:
        result = model.fit(freq, dk, df)
        return (mat_id, result)
    except:
        return (mat_id, None)

# Process multiple materials in parallel
with Pool(processes=4) as pool:
    results = pool.map(process_single_material, material_list)
```

### Integration with Measurement Systems

```python
# Example: Keysight VNA integration
def process_vna_data(touchstone_file):
    """Process S-parameter data from VNA."""
    import skrf as rf
    
    # Load touchstone file
    network = rf.Network(touchstone_file)
    
    # Extract material properties (assuming Nicolson-Ross-Weir)
    # This is simplified - actual implementation depends on fixture
    thickness = 0.001  # 1mm sample
    
    # Convert S-params to epsilon
    freq_ghz = network.frequency.f / 1e9
    # ... (NRW algorithm implementation)
    
    # Fit D-S model
    model = SarkarModel()
    result = model.fit(freq_ghz, dk_extracted, df_extracted)
    
    return result
```

## API Reference

### Core Methods

#### `SarkarModel(name=None, use_ghz=True)`
- **name**: Custom model name (default: "Enhanced-DS")
- **use_ghz**: Frequency units (True=GHz, False=Hz)

#### `fit(freq, dk_exp, df_exp, **kwargs)`
- **freq**: Frequency array
- **dk_exp**: Experimental Dk values
- **df_exp**: Experimental Df values
- **Returns**: ModelResult with validation and quality metrics

#### `assess_model_suitability(freq, dk_exp, df_exp)`
- **Static method**
- **Returns**: Dict with confidence score and recommendations

#### `calculate_quality_metrics(result)`
- **result**: ModelResult from fit
- **Returns**: Dict with comprehensive quality scores

#### `extrapolate_with_uncertainty(params, freq_extrap, confidence=0.95)`
- **params**: Model parameters
- **freq_extrap**: Extrapolation frequencies
- **confidence**: Confidence level (0.95 or 0.99)
- **Returns**: Dict with values and uncertainty bands

### Utility Methods

#### `compare_with_multi_debye(freq, dk_exp, df_exp)`
- **Class method**
- **Returns**: Comparison results and recommendation

#### `get_transition_characteristics(params)`
- **params**: Model parameters
- **Returns**: Dict with transition analysis

#### `generate_spice_model(params, z0=50.0)`
- **params**: Model parameters
- **z0**: Reference impedance
- **Returns**: SPICE model elements

#### `to_touchstone(params, freq_hz, thickness_m, z0=50.0)`
- **params**: Model parameters
- **freq_hz**: Frequency in Hz
- **thickness_m**: Material thickness
- **z0**: Reference impedance
- **Returns**: S-parameter dictionary

## References

1. **Original D-S Model**:
   - Djordjevic, A.R., and Sarkar, T.K., "Closed-form formulas for frequency-dependent resistance and inductance per unit length of microstrip and strip transmission lines," IEEE Trans. Microwave Theory Tech., vol. 42, pp. 241-248, Feb. 1994.

2. **Causality and KK Relations**:
   - Bode, H.W., "Network Analysis and Feedback Amplifier Design," Van Nostrand, 1945.
   - Hilbert, D., "Grundzüge einer allgemeinen Theorie der linearen Integralgleichungen," 1912.

3. **PCB Material Characterization**:
   - IPC-TM-650 Test Methods Manual
   - "Signal Integrity: Applied Electromagnetics and Professional Practice," S. H. Hall and H. L. Heck, 2022.

4. **Model Selection**:
   - Akaike, H., "Information theory and an extension of the maximum likelihood principle," 1973.
   - Schwarz, G., "Estimating the dimension of a model," Annals of Statistics, 1978.

## Version History

- **v1.0.0**: Initial implementation
  - Robust parameter estimation
  - Validation framework
  - Quality metrics
  
- **v1.1.0**: Added features (planned)
  - GPU acceleration for batch processing
  - Automated report generation
 

## Conclusion

The Enhanced Djordjevic-Sarkar model provides a robust, production-ready tool for wideband dielectric characterization. Key advantages:

1. **Robust Implementation**: Smart initialization, validation, and error handling
2. **Uncertainty Quantification**: Confidence intervals for extrapolation
3. **Model Selection**: Automatic comparison with alternatives
4. **Quality Metrics**: Comprehensive assessment of fit quality
5. **Practical Tools**: S-parameters, SPICE models, reports

Best practices:
- Always check model suitability before fitting
- Validate parameters for physical consistency
- Use quality metrics to assess reliability
- Consider uncertainty when extrapolating
- Compare with Multi-Debye for validation

The implementation is suitable for research, development, quality control, and signal integrity applications across the electronics industry.