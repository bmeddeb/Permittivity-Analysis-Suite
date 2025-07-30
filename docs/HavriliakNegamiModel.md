# Havriliak-Negami Model Documentation

## Table of Contents
- [Overview](#overview)
- [Theory and Physics](#theory-and-physics)
- [Mathematical Formulation](#mathematical-formulation)
- [Special Cases and Model Hierarchy](#special-cases-and-model-hierarchy)
- [When to Use Havriliak-Negami](#when-to-use-havriliak-negami)
- [Implementation Details](#implementation-details)
- [Usage Guide](#usage-guide)
- [Parameter Estimation Strategies](#parameter-estimation-strategies)
- [Model Selection and Comparison](#model-selection-and-comparison)
- [Distribution Analysis](#distribution-analysis)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Advanced Applications](#advanced-applications)
- [Examples](#examples)

## Overview

The Havriliak-Negami (H-N) model is the most general empirical model for dielectric relaxation, combining both symmetric and asymmetric broadening of the relaxation time distribution. It provides a flexible framework that encompasses all other common relaxation models as special cases.

### Key Features

- **Two shape parameters**: α (symmetric) and β (asymmetric broadening)
- **Universal model**: Reduces to Debye, Cole-Cole, and Davidson-Cole
- **Asymmetric distributions**: Captures complex relaxation behavior
- **Automatic model identification**: Detects when simpler models suffice
- **Physical constraints**: Optional α×β ≤ 1 constraint

### Model Equation

```
ε*(ω) = ε∞ + (εs - ε∞) / (1 + (jωτ)^α)^β
```

Where:
- **α**: Symmetric broadening parameter (0 < α ≤ 1)
- **β**: Asymmetric broadening parameter (0 < β ≤ 1)
- Other parameters as in Debye model

## Theory and Physics

### Physical Origins

The H-N model captures complex relaxation processes arising from:

1. **Cooperative Dynamics**
   - Many-body interactions
   - Hierarchical relaxation
   - Dynamic heterogeneity

2. **Structural Complexity**
   - Multiple length scales
   - Fractal-like environments
   - Constrained geometries

3. **Combined Effects**
   - Both intra- and inter-molecular interactions
   - Temperature-dependent coupling
   - Non-exponential decay channels

### Distribution Interpretation

The H-N model represents a **asymmetric distribution** of relaxation times:

- **α < 1**: Broadens the distribution symmetrically (like Cole-Cole)
- **β < 1**: Introduces asymmetry, cutting off high-frequency tail
- **α×β**: Total broadening factor

### Physical Meaning of Parameters

| Parameter | Physical Effect | Observable Impact |
|-----------|----------------|-------------------|
| α → 1 | Less cooperative dynamics | Narrower distribution |
| α → 0 | Highly cooperative | Very broad, symmetric |
| β → 1 | Symmetric relaxation | No high-frequency cutoff |
| β → 0 | Strong asymmetry | Sharp high-frequency cutoff |

## Mathematical Formulation

### Complex Permittivity

The real and imaginary parts are:

```
ε'(ω) = ε∞ + (εs - ε∞) × Re[(1 + (jωτ)^α)^(-β)]
ε''(ω) = (εs - ε∞) × Im[(1 + (jωτ)^α)^(-β)]
```

### Key Relationships

1. **Peak frequency** (requires numerical calculation):
   ```
   fpeak ≠ 1/(2πτ) in general
   Shift depends on both α and β
   ```

2. **Loss peak characteristics**:
   - Height decreases with decreasing α and β
   - Asymmetry increases with decreasing β
   - Width increases with decreasing α

3. **Limiting behaviors**:
   - Low frequency: ε' → εs, ε'' ∝ ω^α
   - High frequency: ε' → ε∞, ε'' ∝ ω^(-αβ)

### Cole-Cole Plot

In the complex plane (ε'' vs ε'), H-N produces:
- **Asymmetric arc** (not a semicircle)
- Depression angle on low-frequency side: απ/2
- Slope at high-frequency side: βπ/2
- Center typically below real axis

## Special Cases and Model Hierarchy

The H-N model encompasses all common relaxation models:

```
Havriliak-Negami (α, β)
    │
    ├─ α = 1, β = 1 → Debye (single relaxation time)
    │
    ├─ α < 1, β = 1 → Cole-Cole (symmetric broadening)
    │
    ├─ α = 1, β < 1 → Davidson-Cole (asymmetric broadening)
    │
    └─ α < 1, β < 1 → Full H-N (general case)
```

### Model Identification

The implementation automatically identifies which special case best describes your data:

```python
model_type = HavriliakNegamiModel.identify_model_type(result.params)
# Returns: "Debye", "Cole-Cole", "Davidson-Cole", or "Havriliak-Negami"
```

## When to Use Havriliak-Negami

### Ideal Applications

1. **Complex Polymers**
   - Polymer blends
   - Block copolymers
   - Crosslinked networks
   - Polymer nanocomposites

2. **Glass-Forming Systems**
   - Supercooled liquids
   - Fragile glass formers
   - Ionic glasses
   - Metallic glasses

3. **Hydrogen-Bonded Systems**
   - Alcohols and polyols
   - Water-containing systems
   - Proteins and polysaccharides
   - Pharmaceutical compounds

4. **Heterogeneous Materials**
   - Porous materials
   - Composite systems
   - Biological tissues
   - Geological materials

### Diagnostic Signs for H-N

Use H-N when you observe:
- **Asymmetric loss peaks** on log-frequency scale
- **Non-circular Cole-Cole plots**
- **Poor fits** with Cole-Cole or Davidson-Cole alone
- **Different slopes** in low and high-frequency wings

### When Simpler Models Suffice

Don't use H-N if:
- **Debye fits well** (R² > 0.99)
- **Cole-Cole captures the data** (symmetric broadening only)
- **α or β converge to 1** during fitting
- **Limited frequency range** prevents parameter resolution

## Implementation Details

### Class Structure

```python
class HavriliakNegamiModel(BaseModel):
    """
    Most general empirical relaxation model.
    
    Parameters:
        conductivity_correction: Include DC conductivity
        constrain_parameters: Apply α×β ≤ 1 constraint
        name: Custom model name
    """
```

### Parameter Bounds and Constraints

| Parameter | Default Bounds | Physical Constraint | Notes |
|-----------|---------------|-------------------|-------|
| α | (0.01, 1.0) | Must be positive | Avoid α = 0 (numerical) |
| β | (0.01, 1.0) | Must be positive | β = 0 is non-physical |
| α×β | - | ≤ 1 (optional) | Ensures causality |

### Advanced Parameter Estimation

The implementation uses sophisticated shape analysis:

1. **Asymmetry detection**: Compares left/right peak widths
2. **Width analysis**: Total FWHM indicates α
3. **Asymmetry mapping**: Maps to β parameter
4. **Peak correction**: Accounts for H-N peak shift

## Usage Guide

### Basic Fitting

```python
from app.models.havriliak_negami import HavriliakNegamiModel

# Create model
model = HavriliakNegamiModel()

# Fit data
result = model.fit(freq_ghz, dk_exp, df_exp)

# Check which model it reduces to
model_type = HavriliakNegamiModel.identify_model_type(result.params)
print(f"Data best described by: {model_type}")

# Get parameters
if model_type == "Havriliak-Negami":
    alpha = result.params['alpha'].value
    beta = result.params['beta'].value
    print(f"α = {alpha:.3f}, β = {beta:.3f}")
```

### With Physical Constraints

```python
# Enforce α×β ≤ 1 constraint
model = HavriliakNegamiModel(constrain_parameters=True)

# This ensures causality and prevents overfitting
result = model.fit(freq_ghz, dk_exp, df_exp)

# Check constraint
alpha_beta = result.params['alpha'].value * result.params['beta'].value
print(f"α×β = {alpha_beta:.3f} (should be ≤ 1)")
```

### Constrained Fitting Options

```python
# Force symmetric H-N (α = β)
result = model.fit_constrained(
    freq_ghz, dk_exp, df_exp, 
    constraint='alpha_equals_beta'
)

# This reduces the parameter space and can improve stability
```

### With Conductivity

```python
# For conductive samples
model = HavriliakNegamiModel(conductivity_correction=True)

# Or use convenience class
from app.models.havriliak_negami import HavriliakNegamiWithConductivityModel
model = HavriliakNegamiWithConductivityModel()
```

## Parameter Estimation Strategies

### 1. Visual Pre-Analysis

```python
def analyze_peak_shape(freq, df):
    """Pre-fit analysis to determine if H-N is needed."""
    
    # Find peak
    peak_idx = np.argmax(df)
    peak_freq = freq[peak_idx]
    peak_val = df[peak_idx]
    
    # Analyze asymmetry
    half_max = peak_val / 2
    
    # Find frequencies at half maximum
    left_mask = df[:peak_idx] >= half_max
    right_mask = df[peak_idx:] >= half_max
    
    if np.any(left_mask) and np.any(right_mask):
        f_left = freq[left_mask][0]
        f_right = freq[right_mask][-1]
        
        # Calculate asymmetry on log scale
        log_left = np.log10(peak_freq / f_left)
        log_right = np.log10(f_right / peak_freq)
        
        asymmetry = (log_right - log_left) / (log_right + log_left)
        
        print(f"Peak asymmetry: {asymmetry:.3f}")
        
        if abs(asymmetry) < 0.05:
            print("Symmetric peak - Consider Cole-Cole")
        elif abs(asymmetry) > 0.2:
            print("Highly asymmetric - H-N recommended")
        else:
            print("Moderate asymmetry - Try Davidson-Cole first")
        
        return asymmetry
```

### 2. Sequential Model Fitting

```python
def hierarchical_model_selection(freq, dk, df):
    """Fit models in order of complexity."""
    from app.models.debye import DebyeModel
    from app.models.cole_cole import ColeColeModel
    
    models = {
        'Debye': DebyeModel(),
        'Cole-Cole': ColeColeModel(),
        'H-N': HavriliakNegamiModel()
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            result = model.fit(freq, dk, df)
            results[name] = {
                'result': result,
                'aic': result.aic,
                'bic': result.bic,
                'r_squared': result.fit_metrics.r_squared
            }
            
            # Check if simpler model is sufficient
            if name == 'Debye' and result.fit_metrics.r_squared > 0.99:
                print("Debye model is sufficient")
                break
                
        except Exception as e:
            print(f"{name} fitting failed: {e}")
    
    # Select best model using AIC
    best_model = min(results.items(), 
                     key=lambda x: x[1]['aic'])
    
    print(f"Best model by AIC: {best_model[0]}")
    return results
```

### 3. Parameter Space Exploration

```python
def explore_alpha_beta_space(model, freq, dk, df):
    """Map the α-β parameter space for H-N model."""
    
    # Create grid
    alphas = np.linspace(0.1, 1.0, 10)
    betas = np.linspace(0.1, 1.0, 10)
    
    chi_squared = np.zeros((len(alphas), len(betas)))
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Fix α and β
            params = model.create_parameters(freq, dk, df)
            params['alpha'].value = alpha
            params['alpha'].vary = False
            params['beta'].value = beta
            params['beta'].vary = False
            
            # Fit only other parameters
            try:
                result = model.fit(freq, dk, df, params=params)
                chi_squared[i, j] = result.chisqr
            except:
                chi_squared[i, j] = np.inf
    
    # Find minimum
    min_idx = np.unravel_index(np.argmin(chi_squared), chi_squared.shape)
    best_alpha = alphas[min_idx[0]]
    best_beta = betas[min_idx[1]]
    
    print(f"Best α = {best_alpha:.2f}, β = {best_beta:.2f}")
    
    return chi_squared, alphas, betas
```

## Model Selection and Comparison

### Comparing with Special Cases

```python
def compare_with_special_cases(hn_result):
    """Compare H-N fit with its special cases."""
    
    # Get equivalent parameters
    special_cases = HavriliakNegamiModel.estimate_special_cases(hn_result.params)
    
    print("H-N Parameters:")
    print(f"  α = {hn_result.params['alpha'].value:.3f}")
    print(f"  β = {hn_result.params['beta'].value:.3f}")
    print(f"  τ = {hn_result.params['tau'].value:.2e} s")
    
    print("\nEquivalent Special Cases:")
    for model_name, params in special_cases.items():
        print(f"\n{model_name}:")
        for param, value in params.items():
            if isinstance(value, float):
                print(f"  {param} = {value:.3e}")
```

### Statistical Model Selection

```python
def statistical_model_comparison(freq, dk, df):
    """Use statistical criteria for model selection."""
    
    # Fit H-N model
    hn_model = HavriliakNegamiModel()
    hn_result = hn_model.fit(freq, dk, df)
    
    # Check parameter significance
    alpha = hn_result.params['alpha']
    beta = hn_result.params['beta']
    
    # t-statistics
    if alpha.stderr and beta.stderr:
        t_alpha = abs(1 - alpha.value) / alpha.stderr
        t_beta = abs(1 - beta.value) / beta.stderr
        
        print(f"α significance: t = {t_alpha:.2f}")
        print(f"β significance: t = {t_beta:.2f}")
        
        # Rule of thumb: t > 2 indicates significance
        if t_alpha < 2 and t_beta < 2:
            print("Neither α nor β significantly different from 1")
            print("→ Debye model recommended")
        elif t_alpha > 2 and t_beta < 2:
            print("Only α significant")
            print("→ Cole-Cole model recommended")
        elif t_alpha < 2 and t_beta > 2:
            print("Only β significant")
            print("→ Davidson-Cole model recommended")
        else:
            print("Both α and β significant")
            print("→ Full H-N model justified")
    
    return hn_result
```

## Distribution Analysis

### Understanding H-N Distribution

The H-N distribution is **asymmetric** and complex:

```python
def analyze_hn_distribution(result):
    """Analyze the relaxation time distribution."""
    
    # Get distribution parameters
    dist_params = HavriliakNegamiModel.get_distribution_parameters(result.params)
    
    print("Distribution Characteristics:")
    print(f"  Most probable τ: {dist_params['tau_peak']:.2e} s")
    print(f"  Mean τ: {dist_params['tau_mean']:.2e} s")
    print(f"  H-N characteristic τ: {dist_params['tau_hn']:.2e} s")
    print(f"  Asymmetry parameter: {dist_params['asymmetry']:.3f}")
    
    # Get full distribution
    plot_data = model.get_plot_data(result, show_distribution=True)
    
    if 'distribution' in plot_data:
        tau = plot_data['distribution']['tau']
        g_tau = plot_data['distribution']['g_tau']
        
        # Find width at half maximum
        half_max = np.max(g_tau) / 2
        above_half = tau[g_tau > half_max]
        
        if len(above_half) > 0:
            width_decades = np.log10(above_half[-1] / above_half[0])
            print(f"  Distribution width: {width_decades:.2f} decades")
```

### Visualizing Parameter Effects

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_parameter_effects():
    """Show how α and β affect the spectrum."""
    
    freq = np.logspace(-2, 4, 100)  # 10 mHz to 10 kHz
    
    # Fixed parameters
    eps_inf = 2.0
    eps_s = 10.0
    tau = 1e-9
    
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=['α effect (β=1)', 'β effect (α=0.8)',
                                      'Cole-Cole plot', 'Distribution'])
    
    # Effect of α (Cole-Cole like)
    for alpha in [1.0, 0.8, 0.6, 0.4]:
        params = {'eps_inf': eps_inf, 'eps_s': eps_s, 
                  'tau': tau, 'alpha': alpha, 'beta': 1.0}
        eps_c = HavriliakNegamiModel.model_func(freq, **params)
        
        fig.add_trace(go.Scatter(x=freq, y=eps_c.imag,
                                name=f'α={alpha}',
                                mode='lines'), row=1, col=1)
    
    # Effect of β (Davidson-Cole like)
    for beta in [1.0, 0.8, 0.6, 0.4]:
        params = {'eps_inf': eps_inf, 'eps_s': eps_s,
                  'tau': tau, 'alpha': 0.8, 'beta': beta}
        eps_c = HavriliakNegamiModel.model_func(freq, **params)
        
        fig.add_trace(go.Scatter(x=freq, y=eps_c.imag,
                                name=f'β={beta}',
                                mode='lines'), row=1, col=2)
    
    # Update axes
    fig.update_xaxes(type="log", title="Frequency (Hz)")
    fig.update_yaxes(title="ε''")
    
    return fig
```

## Common Issues and Solutions

### 1. Parameter Correlation

**Problem**: α and β are often highly correlated

**Solution**:
```python
def reduce_correlation(model, freq, dk, df):
    """Strategies to reduce α-β correlation."""
    
    # Strategy 1: Fix one parameter initially
    params = model.create_parameters(freq, dk, df)
    
    # First fit with β=1 (Cole-Cole)
    params['beta'].value = 1.0
    params['beta'].vary = False
    result_cc = model.fit(freq, dk, df, params=params)
    
    # Then release β
    params = result_cc.params
    params['beta'].vary = True
    params['beta'].min = 0.1
    params['beta'].max = 1.0
    
    result_final = model.fit(freq, dk, df, params=params)
    
    # Check correlation
    try:
        correl = result_final.params['alpha'].correl['beta']
        print(f"α-β correlation: {correl:.3f}")
        
        if abs(correl) > 0.95:
            print("Warning: High correlation - results may be unstable")
    except:
        pass
    
    return result_final
```

### 2. Convergence to Boundaries

**Problem**: α or β converge to 0 or 1

**Solution**:
```python
def handle_boundary_convergence(result):
    """Check and handle parameter boundary issues."""
    
    alpha = result.params['alpha'].value
    beta = result.params['beta'].value
    
    warnings = []
    
    if alpha < 0.05:
        warnings.append("α near lower bound - check data quality")
    elif alpha > 0.95:
        warnings.append("α near 1 - consider simpler model")
        
    if beta < 0.05:
        warnings.append("β near lower bound - check high-frequency data")
    elif beta > 0.95:
        warnings.append("β near 1 - consider Cole-Cole model")
    
    if warnings:
        print("Parameter warnings:")
        for w in warnings:
            print(f"  - {w}")
    
    # Suggest alternatives
    if alpha > 0.95 and beta > 0.95:
        print("\nRecommendation: Use Debye model")
    elif alpha < 0.95 and beta > 0.95:
        print("\nRecommendation: Use Cole-Cole model")
    elif alpha > 0.95 and beta < 0.95:
        print("\nRecommendation: Use Davidson-Cole model")
```

### 3. Limited Frequency Range

**Problem**: Narrow frequency range prevents parameter resolution

**Solution**:
```python
def assess_frequency_coverage(freq, df):
    """Check if frequency range is sufficient for H-N."""
    
    # Find peak
    peak_idx = np.argmax(df)
    f_peak = freq[peak_idx]
    
    # Check coverage
    decades_below = np.log10(f_peak / freq[0])
    decades_above = np.log10(freq[-1] / f_peak)
    
    print(f"Peak at {f_peak:.3f} GHz")
    print(f"Coverage: {decades_below:.1f} decades below, {decades_above:.1f} above")
    
    if decades_below < 1 or decades_above < 1:
        print("Warning: Limited frequency range")
        print("- May not resolve asymmetry properly")
        print("- Consider constraining parameters")
        
        if decades_below < decades_above:
            print("- Better high-frequency coverage: β more reliable than α")
        else:
            print("- Better low-frequency coverage: α more reliable than β")
    
    return decades_below, decades_above
```

### 4. Non-Physical Results

**Problem**: Fit gives non-physical parameter combinations

**Solution**:
```python
def validate_physical_consistency(result):
    """Check physical consistency of H-N parameters."""
    
    params = result.params
    alpha = params['alpha'].value
    beta = params['beta'].value
    tau = params['tau'].value
    
    # Physical checks
    checks = {
        'Causality': alpha * beta <= 1.0,
        'Relaxation time reasonable': 1e-15 < tau < 1e-3,
        'Loss peak exists': alpha > 0 and beta > 0,
        'Finite zero-frequency limit': alpha * beta > 0
    }
    
    print("Physical consistency checks:")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    
    if not all(checks.values()):
        print("\nRecommendations:")
        if not checks['Causality']:
            print("- Enable constrain_parameters=True")
        if not checks['Relaxation time reasonable']:
            print("- Check frequency units (should be GHz)")
```

## Advanced Applications

### 1. Temperature-Dependent H-N Analysis

```python
def analyze_temperature_series(temperatures, freq, dk_data, df_data):
    """Analyze H-N parameters vs temperature."""
    
    results = []
    model = HavriliakNegamiModel()
    
    for i, T in enumerate(temperatures):
        result = model.fit(freq, dk_data[i], df_data[i])
        
        results.append({
            'T': T,
            'tau': result.params['tau'].value,
            'alpha': result.params['alpha'].value,
            'beta': result.params['beta'].value,
            'model_type': HavriliakNegamiModel.identify_model_type(result.params)
        })
    
    # Analyze trends
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Check for model transitions
    model_changes = df['model_type'].ne(df['model_type'].shift()).sum()
    if model_changes > 1:
        print(f"Model type changes {model_changes} times with temperature")
        print("This indicates changing relaxation mechanisms")
    
    # Fragility analysis from tau(T)
    if len(temperatures) > 5:
        # VFT fit to tau
        from scipy.optimize import curve_fit
        
        def vft(T, tau0, B, T0):
            return tau0 * np.exp(B / (T - T0))
        
        try:
            popt, _ = curve_fit(vft, df['T'], df['tau'], 
                               p0=[1e-14, 1000, df['T'].min()-50])
            
            # Fragility at Tg
            Tg_idx = np.argmax(df['tau'] > 100)  # τ = 100s definition
            if Tg_idx > 0:
                Tg = df['T'].iloc[Tg_idx]
                m = popt[1] / (Tg - popt[2])**2 / np.log(10)
                print(f"Fragility index m = {m:.1f}")
                
                if m > 100:
                    print("Fragile behavior - expect strong T-dependence of α, β")
                else:
                    print("Strong behavior - α, β may be T-independent")
        except:
            print("VFT fit failed - try Arrhenius")
    
    return df
```

### 2. Molecular Dynamics Interpretation

```python
def interpret_molecular_dynamics(result, material_info):
    """Relate H-N parameters to molecular dynamics."""
    
    alpha = result.params['alpha'].value
    beta = result.params['beta'].value
    tau = result.params['tau'].value
    
    print(f"Molecular Dynamics Interpretation for {material_info['name']}:")
    
    # Coupling parameter
    n_coupling = 1 - alpha
    print(f"\nCoupling parameter n = {n_coupling:.3f}")
    if n_coupling < 0.3:
        print("  → Weak intermolecular coupling")
    elif n_coupling < 0.7:
        print("  → Moderate coupling")
    else:
        print("  → Strong cooperative dynamics")
    
    # Asymmetry interpretation
    if material_info['type'] == 'polymer':
        if beta < 0.8:
            print(f"\nAsymmetry (β = {beta:.3f}):")
            print("  → Indicates constrained segmental motion")
            print("  → Possible entanglement effects")
            print("  → May reflect broad molecular weight distribution")
    
    elif material_info['type'] == 'glass_former':
        # Relate to fragility
        fragility_indicator = (1 - alpha) * (1 - beta)
        print(f"\nFragility indicator: {fragility_indicator:.3f}")
        if fragility_indicator > 0.5:
            print("  → Fragile glass former")
            print("  → Strong T-dependence expected")
    
    # Time scale analysis
    f_peak = HavriliakNegamiModel.get_relaxation_frequency(result.params)
    print(f"\nTime scales:")
    print(f"  Peak frequency: {f_peak:.3e} Hz")
    print(f"  Most probable τ: {1/(2*np.pi*f_peak):.3e} s")
    
    # Activation energy estimate (if multiple temperatures available)
    if 'activation_energy' in material_info:
        Ea = material_info['activation_energy']  # in eV
        attempt_freq = 1e13  # Hz, typical
        expected_tau = (1/attempt_freq) * np.exp(Ea * 11600 / material_info['temperature'])
        
        print(f"\nActivation analysis:")
        print(f"  Expected τ (from Ea): {expected_tau:.3e} s")
        print(f"  Actual τ: {tau:.3e} s")
        
        if tau > expected_tau * 10:
            print("  → Cooperative effects slow down relaxation")
```

### 3. Composite Material Analysis

```python
def analyze_composite_relaxation(freq, dk, df, composite_info):
    """Analyze relaxation in composite materials."""
    
    model = HavriliakNegamiModel()
    result = model.fit(freq, dk, df)
    
    # Check for Maxwell-Wagner-Sillars (MWS) relaxation
    tau = result.params['tau'].value
    f_mws = 1 / (2 * np.pi * tau)
    
    # Estimate MWS frequency from composite properties
    if 'matrix_permittivity' in composite_info:
        eps_matrix = composite_info['matrix_permittivity']
        eps_filler = composite_info['filler_permittivity']
        phi = composite_info['filler_fraction']
        
        # Simple MWS estimate
        f_mws_theory = composite_info['matrix_conductivity'] / (
            2 * np.pi * 8.854e-12 * eps_matrix
        )
        
        print(f"MWS Analysis:")
        print(f"  Measured relaxation: {f_mws:.3e} Hz")
        print(f"  Theoretical MWS: {f_mws_theory:.3e} Hz")
        
        if 0.1 < f_mws/f_mws_theory < 10:
            print("  → Likely interfacial polarization")
            print("  → H-N shape reflects interface distribution")
            
            # Interpret shape parameters
            alpha = result.params['alpha'].value
            beta = result.params['beta'].value
            
            if alpha < 0.8:
                print(f"  → α = {alpha:.3f} indicates distributed interface properties")
            if beta < 0.8:
                print(f"  → β = {beta:.3f} suggests asymmetric charge accumulation")
    
    return result
```

### 4. Multi-Process Deconvolution

```python
def deconvolve_multiple_processes(freq, dk, df, n_processes=2):
    """Attempt to separate multiple H-N processes."""
    
    def multi_hn_model(freq, *params):
        """Sum of multiple H-N processes."""
        n = n_processes
        eps_inf = params[0]
        
        eps_total = eps_inf + 0j
        
        for i in range(n):
            idx = 1 + i * 4
            delta_eps = params[idx]
            tau = params[idx + 1]
            alpha = params[idx + 2]
            beta = params[idx + 3]
            
            omega = 2 * np.pi * freq * 1e9
            eps_total += delta_eps / ((1 + (1j * omega * tau)**alpha)**beta)
        
        return eps_total
    
    # Initial guess based on peak analysis
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(df, prominence=0.1*np.max(df))
    
    if len(peaks) >= n_processes:
        print(f"Found {len(peaks)} peaks, attempting {n_processes}-process fit")
        
        # Set up parameters
        from lmfit import Model, Parameters
        
        model = Model(multi_hn_model, independent_vars=['freq'])
        params = Parameters()
        
        # Global parameters
        params.add('eps_inf', value=dk[-1], min=1, max=dk[0])
        
        # Per-process parameters
        for i in range(n_processes):
            prefix = f'p{i+1}_'
            
            if i < len(peaks):
                peak_idx = peaks[i]
                tau_guess = 1 / (2 * np.pi * freq[peak_idx] * 1e9)
                delta_eps_guess = (dk[0] - dk[-1]) / n_processes
            else:
                tau_guess = 10**(-(9 + i))
                delta_eps_guess = 1.0
            
            params.add(f'{prefix}delta_eps', value=delta_eps_guess, min=0)
            params.add(f'{prefix}tau', value=tau_guess, min=1e-15, max=1e-3)
            params.add(f'{prefix}alpha', value=0.8, min=0.1, max=1.0)
            params.add(f'{prefix}beta', value=0.8, min=0.1, max=1.0)
        
        # Fit complex data
        eps_complex = dk + 1j * df * dk
        result = model.fit(eps_complex, params=params, freq=freq)
        
        return result
    else:
        print(f"Only {len(peaks)} peaks found, single H-N may be sufficient")
        return None
```

## Examples

### Example 1: Polymer Glass Analysis

```python
# Example: Polycarbonate near Tg
import numpy as np

# Simulated data
freq_ghz = np.logspace(-3, 3, 80)  # 1 MHz to 1 THz
T = 423  # 150°C, near Tg

# Generate H-N data
omega = 2 * np.pi * freq_ghz * 1e9
eps_inf = 2.8
eps_s = 3.2
tau = 1e-6  # Slow near Tg
alpha = 0.45  # Broad distribution
beta = 0.6   # Asymmetric

eps_complex = eps_inf + (eps_s - eps_inf) / ((1 + (1j * omega * tau)**alpha)**beta)
dk = eps_complex.real + np.random.normal(0, 0.01, len(freq_ghz))
df = eps_complex.imag / eps_complex.real + np.random.normal(0, 0.0001, len(freq_ghz))

# Fit and analyze
model = HavriliakNegamiModel()
result = model.fit(freq_ghz, dk, df)

print("Polycarbonate Analysis:")
print(model.get_fit_report(result))

# Interpret
model_type = HavriliakNegamiModel.identify_model_type(result.params)
print(f"\nModel type: {model_type}")

if model_type == "Havriliak-Negami":
    print("\nInterpretation:")
    print("- Broad asymmetric distribution indicates cooperative segmental dynamics")
    print("- α < 0.5 suggests strong intermolecular coupling")
    print("- β < 1 reflects constraints from neighboring chains")
```

### Example 2: Pharmaceutical Formulation

```python
def analyze_drug_formulation(freq_ghz, dk, df, drug_info):
    """Analyze pharmaceutical formulation stability."""
    
    model = HavriliakNegamiModel()
    result = model.fit(freq_ghz, dk, df)
    
    # Get parameters
    params = result.params
    tau = params['tau'].value
    alpha = params['alpha'].value
    beta = params['beta'].value
    
    # Molecular mobility assessment
    print(f"Drug: {drug_info['name']}")
    print(f"Excipient: {drug_info['excipient']}")
    
    # Relaxation time at storage temperature
    T_storage = 298  # 25°C
    T_measure = drug_info['measurement_temp']
    
    # Estimate tau at storage (assume Arrhenius)
    if 'activation_energy' in drug_info:
        Ea = drug_info['activation_energy']
        tau_storage = tau * np.exp(Ea * (1/T_storage - 1/T_measure) / 8.314e-3)
        
        print(f"\nMolecular mobility at storage:")
        print(f"  τ at {T_storage}K: {tau_storage:.2e} s")
        
        if tau_storage < 100:
            print("  ⚠ Warning: High molecular mobility")
            print("  → Risk of crystallization")
            print("  → Consider different excipient")
        else:
            print("  ✓ Good stability expected")
    
    # Distribution width indicates heterogeneity
    if alpha < 0.7 or beta < 0.7:
        print("\nBroad relaxation distribution detected:")
        print("  → Indicates molecular-level mixing")
        print("  → Good for preventing crystallization")
    
    return result

# Example usage
drug_info = {
    'name': 'Indomethacin',
    'excipient': 'PVP K30',
    'measurement_temp': 353,  # 80°C
    'activation_energy': 80  # kJ/mol
}

result = analyze_drug_formulation(freq_ghz, dk, df, drug_info)
```

### Example 3: Biological Tissue Characterization

```python
def characterize_tissue(freq_ghz, dk, df, tissue_type):
    """Analyze biological tissue dielectric properties."""
    
    # Fit H-N model
    model = HavriliakNegamiModel(conductivity_correction=True)
    result = model.fit(freq_ghz, dk, df)
    
    # Tissue-specific analysis
    print(f"Tissue Type: {tissue_type}")
    
    # Multiple relaxation processes in tissues
    # β dispersion: 1-100 MHz (cellular)
    # γ dispersion: >1 GHz (water)
    
    params = result.params
    f_peak = HavriliakNegamiModel.get_relaxation_frequency(params)
    
    if 0.001 < f_peak < 0.1:  # β dispersion
        print("\nβ-dispersion detected (cellular structures)")
        print(f"  Peak at {f_peak*1000:.1f} MHz")
        
        # Relate to cell properties
        alpha = params['alpha'].value
        beta_param = params['beta'].value
        
        if alpha < 0.8:
            print(f"  α = {alpha:.3f} indicates cell size distribution")
        if beta_param < 0.9:
            print(f"  β = {beta_param:.3f} suggests membrane heterogeneity")
            
    elif f_peak > 1:  # γ dispersion
        print("\nγ-dispersion detected (tissue water)")
        print(f"  Peak at {f_peak:.1f} GHz")
        
        # Estimate water content
        delta_eps = params['eps_s'].value - params['eps_inf'].value
        water_fraction = delta_eps / 75  # Approximate
        
        print(f"  Estimated water content: {water_fraction*100:.1f}%")
    
    # Check for pathological changes
    dist_params = HavriliakNegamiModel.get_distribution_parameters(params)
    
    if dist_params['width_parameter'] > 1.2:
        print("\n⚠ Abnormally broad distribution")
        print("  May indicate:")
        print("  - Tissue damage")
        print("  - Inflammation")
        print("  - Tumor presence")
    
    return result
```

### Example 4: Quality Control with KK Validation

```python
def validate_hn_fit_with_kk(freq, dk, df):
    """Validate H-N fit using Kramers-Kronig relations."""
    from app.models.kramers_kronig_validator import KramersKronigValidator
    import pandas as pd
    
    # First, validate experimental data
    exp_df = pd.DataFrame({
        'Frequency (GHz)': freq,
        'Dk': dk,
        'Df': df
    })
    
    kk_validator = KramersKronigValidator(exp_df)
    kk_exp = kk_validator.validate()
    
    print(f"Experimental data KK: {kk_exp['causality_status']}")
    
    # Fit H-N model
    model = HavriliakNegamiModel(constrain_parameters=True)
    result = model.fit(freq, dk, df)
    
    # Check if constraint is satisfied
    alpha_beta = result.params['alpha'].value * result.params['beta'].value
    print(f"\nα×β = {alpha_beta:.3f} (constraint: ≤ 1)")
    
    # Generate model over extended range
    freq_extended = np.logspace(-6, 6, 500)
    eps_model = model.predict(freq_extended, result.params)
    
    # Validate model
    model_df = pd.DataFrame({
        'Frequency (GHz)': freq_extended,
        'Dk': eps_model.real,
        'Df': eps_model.imag / eps_model.real
    })
    
    kk_model = KramersKronigValidator(model_df).validate()
    
    print(f"H-N model KK: {kk_model['causality_status']}")
    print(f"Model error: {kk_model['mean_relative_error']:.6f}")
    
    if kk_model['mean_relative_error'] > 0.001:
        print("\nWarning: Model shows KK violations")
        print("Consider:")
        print("- Enabling parameter constraints")
        print("- Checking for additional processes")
        print("- Extending frequency range")
    
    return result, kk_model
```

## Conclusion

The Havriliak-Negami model provides the most flexible framework for analyzing complex dielectric relaxation. Its key advantages:

1. **Universal applicability**: Can fit any empirical relaxation
2. **Automatic simplification**: Identifies when simpler models suffice
3. **Physical insight**: Parameters relate to molecular dynamics
4. **Asymmetric distributions**: Captures real material complexity

Best practices:
- Start with simpler models and progress to H-N if needed
- Use statistical criteria for model selection
- Apply physical constraints (α×β ≤ 1) for causality
- Validate with Kramers-Kronig relations
- Interpret parameters in context of material physics

The implementation provides robust fitting with sophisticated initialization and comprehensive analysis tools, making it suitable for research and industrial applications across polymers, pharmaceuticals, biological materials, and composites.