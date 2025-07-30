# Hybrid Debye-Lorentz Model Documentation

## Table of Contents
- [Overview](#overview)
- [Theory and Physics](#theory-and-physics)
- [Mathematical Formulation](#mathematical-formulation)
- [Physical Interpretation](#physical-interpretation)
- [When to Use Hybrid Model](#when-to-use-hybrid-model)
- [Implementation Details](#implementation-details)
- [Usage Guide](#usage-guide)
- [Parameter Estimation Strategies](#parameter-estimation-strategies)
- [Process Separation and Analysis](#process-separation-and-analysis)
- [Model Selection and Optimization](#model-selection-and-optimization)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Advanced Applications](#advanced-applications)
- [Examples](#examples)

## Overview

The Hybrid Debye-Lorentz model combines relaxation and resonance mechanisms in a single framework, capturing both low-frequency dispersive behavior and high-frequency resonant features. This unified approach is essential for materials exhibiting multiple polarization mechanisms across wide frequency ranges.

### Key Features

- **Dual mechanisms**: Debye relaxation + Lorentz resonance
- **Automatic detection**: Identifies optimal number of each process
- **Frequency separation**: Natural division between mechanisms
- **Component analysis**: Individual process contributions
- **Adaptive fitting**: Self-optimizing configuration

### Model Equation

```
ε*(ω) = ε∞ + Σᵢ[Δεᵢ/(1+jωτᵢ)] + Σₖ[fₖω₀ₖ²/(ω₀ₖ²-ω²-jγₖω)]
```

Where:
- **Debye terms**: Δεᵢ (strength), τᵢ (relaxation time)
- **Lorentz terms**: fₖ (strength), ω₀ₖ (resonance), γₖ (damping)
- **ε∞**: High-frequency permittivity (shared)

## Theory and Physics

### Physical Origins

The hybrid model captures two fundamental classes of polarization:

1. **Relaxation Processes (Debye)**
   - Dipolar reorientation
   - Ionic conduction
   - Interfacial polarization (Maxwell-Wagner)
   - Space charge accumulation
   - Typically dominant < 1 GHz

2. **Resonance Processes (Lorentz)**
   - Lattice vibrations (phonons)
   - Electronic transitions
   - Molecular vibrations
   - Plasma oscillations
   - Typically dominant > 1 GHz

### Frequency Hierarchy

Different mechanisms dominate at different frequencies:

| Frequency Range | Mechanism | Physical Process |
|----------------|-----------|------------------|
| Hz - kHz | Interfacial | Space charge, electrodes |
| kHz - MHz | Dipolar | Molecular reorientation |
| MHz - GHz | Ionic | Ion hopping, conduction |
| GHz - THz | Vibrational | Phonons, molecular modes |
| THz - PHz | Electronic | Interband transitions |

### Crossover Behavior

The transition between relaxation and resonance regimes often occurs around 1-10 GHz:
- **Below**: Relaxation dominates, ε'' ∝ 1/ω
- **Above**: Resonance dominates, sharp peaks in ε''
- **Crossover**: Both contribute significantly

## Mathematical Formulation

### Complex Permittivity

The total permittivity is the sum of all contributions:

```
ε*(ω) = ε∞ + ε*_Debye(ω) + ε*_Lorentz(ω)
```

#### Debye Contribution

For N_D Debye processes:
```
ε*_Debye(ω) = Σᵢ₌₁^{N_D} [Δεᵢ/(1 + jωτᵢ)]
```

Real part:
```
ε'_Debye = Σᵢ [Δεᵢ/(1 + ω²τᵢ²)]
```

Imaginary part:
```
ε''_Debye = Σᵢ [Δεᵢωτᵢ/(1 + ω²τᵢ²)]
```

#### Lorentz Contribution

For N_L Lorentz oscillators:
```
ε*_Lorentz(ω) = Σₖ₌₁^{N_L} [fₖω₀ₖ²/(ω₀ₖ² - ω² - jγₖω)]
```

Real part:
```
ε'_Lorentz = Σₖ [fₖω₀ₖ²(ω₀ₖ² - ω²)/((ω₀ₖ² - ω²)² + γₖ²ω²)]
```

Imaginary part:
```
ε''_Lorentz = Σₖ [fₖω₀ₖ²γₖω/((ω₀ₖ² - ω²)² + γₖ²ω²)]
```

### Limiting Behaviors

1. **Low frequency** (ω → 0):
   ```
   ε' → ε_static = ε∞ + ΣΔεᵢ + Σfₖ
   ε'' → Σ(Δεᵢωτᵢ) (linear in ω)
   ```

2. **High frequency** (ω → ∞):
   ```
   ε' → ε∞
   ε'' → 0
   ```

3. **Near relaxation** (ω ≈ 1/τᵢ):
   - Broad peak in ε''
   - Smooth dispersion in ε'

4. **Near resonance** (ω ≈ ω₀ₖ):
   - Sharp peak in ε''
   - Anomalous dispersion in ε'

## Physical Interpretation

### Process Identification

The model parameters directly relate to physical mechanisms:

| Parameter | Physical Meaning | Typical Values |
|-----------|-----------------|----------------|
| Δεᵢ | Dipole moment × concentration | 0.1 - 100 |
| τᵢ | Molecular correlation time | 10⁻¹² - 10⁻⁶ s |
| fₖ | Oscillator strength ∝ N·q²/m | 0.01 - 10 |
| ω₀ₖ | Natural frequency | 1 - 1000 GHz |
| γₖ | Damping rate | 0.01 - 10 GHz |

### Energy Considerations

- **Relaxation**: Thermal activation over barriers
  ```
  τ = τ₀ exp(E_a/k_B T)
  ```

- **Resonance**: Quantum mechanical transitions
  ```
  ℏω₀ = E_upper - E_lower
  ```

### Material Signatures

Different materials show characteristic hybrid behavior:

1. **Semiconductors**: Ionic conduction + phonons
2. **Biological tissues**: Cellular relaxation + water resonance
3. **Polymers**: Segmental motion + vibrational modes
4. **Composites**: Interfacial polarization + filler resonances

## When to Use Hybrid Model

### Ideal Applications

1. **Wide-Band Spectroscopy**
   - Measurements spanning MHz to THz
   - Comprehensive material characterization
   - Broadband device design
   - EMC/EMI analysis

2. **Complex Materials**
   - Semiconductor devices with parasitics
   - Hydrated biomaterials
   - Filled polymers and composites
   - Ionic conductors with lattice modes

3. **Temperature Studies**
   - Activation of different mechanisms
   - Phase transition analysis
   - Process separation by temperature
   - Thermal stability assessment

4. **Quality Control**
   - Multiple parameter fingerprinting
   - Defect identification
   - Composition analysis
   - Process monitoring

### Diagnostic Signs for Hybrid Model

Use the hybrid model when observing:
- **Multiple slopes** in log(ε'') vs log(f)
- **Broad low-frequency dispersion** AND **sharp high-frequency peaks**
- **Poor fits** with pure Debye or pure Lorentz
- **Temperature-dependent mechanism crossover**

### When Simpler Models Suffice

Don't use hybrid model if:
- **Single mechanism dominates** entire range
- **Limited frequency range** (< 3 decades)
- **Excellent fit** with Debye or Lorentz alone
- **Parameters strongly correlated** (indicates overparameterization)

## Implementation Details

### Class Structure

```python
class HybridDebyeLorentzModel(BaseModel):
    """
    Combines Debye relaxation and Lorentz resonance.
    
    Parameters:
        n_debye: Number of relaxation processes
        n_lorentz: Number of resonance processes
        auto_detect: Automatic process identification
        constrain_parameters: Physical constraints
        shared_eps_inf: Single high-frequency limit
    """
```

### Parameter Organization

| Process | Parameters | Count | Total |
|---------|-----------|-------|-------|
| Global | ε∞ | 1 | 1 |
| Debye | Δε, τ | 2 × N_D | 2N_D |
| Lorentz | f, ω₀, γ | 3 × N_L | 3N_L |
| **Total** | | | 1 + 2N_D + 3N_L |

### Detection Algorithm

The auto-detection uses frequency-based separation:

1. **Define transition frequency** (default: 1 GHz)
2. **Analyze low frequencies** for relaxation:
   - Monotonic decrease in ε''
   - Peaks in Df
   - Slope changes in log-log plot
3. **Analyze high frequencies** for resonance:
   - Sharp peaks in ε''
   - Anomalous dispersion in ε'
4. **Limit complexity** (max 3 Debye, 5 Lorentz)

## Usage Guide

### Basic Hybrid Fitting

```python
from app.models.hybrid_debye_lorentz import HybridDebyeLorentzModel

# Create model with known configuration
model = HybridDebyeLorentzModel(n_debye=2, n_lorentz=1)
result = model.fit(freq_ghz, dk_exp, df_exp)

# Print summary
print(f"Model: {model.name}")
print(f"Success: {result.success}")
print(f"R²: {result.fit_metrics.r_squared:.4f}")

# Access parameters
for i in range(2):
    print(f"\nDebye process {i+1}:")
    print(f"  Δε = {result.params[f'delta_eps_{i+1}'].value:.3f}")
    print(f"  τ = {result.params[f'tau_{i+1}'].value:.2e} s")

print(f"\nLorentz oscillator 1:")
print(f"  f = {result.params['f_1'].value:.3f}")
print(f"  ω₀ = {result.params['omega0_1'].value:.2f} GHz")
print(f"  γ = {result.params['gamma_1'].value:.2f} GHz")
```

### Automatic Detection

```python
# Let model determine configuration
model = HybridDebyeLorentzModel(auto_detect=True)
result = model.fit(freq_ghz, dk_exp, df_exp)

print(f"Detected configuration:")
print(f"  {model.n_debye} Debye processes")
print(f"  {model.n_lorentz} Lorentz oscillators")

# Get characteristic frequencies
char_freqs = model.get_characteristic_frequencies(result.params)
print("\nCharacteristic frequencies:")
for f_relax in char_freqs['debye_relaxation']:
    print(f"  Relaxation: {f_relax:.3f} GHz")
for f_res in char_freqs['lorentz_resonance']:
    print(f"  Resonance: {f_res:.3f} GHz")
```

### Component Analysis

```python
# Get individual contributions
contributions = model.get_process_contributions(result.params, freq_ghz)

# Plot components
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Real part
ax1.loglog(freq_ghz, contributions['eps_inf'].real, 'k--', label='ε∞')
ax1.loglog(freq_ghz, contributions['debye_total'].real, 'b-', label='Debye')
ax1.loglog(freq_ghz, contributions['lorentz_total'].real, 'r-', label='Lorentz')
ax1.set_xlabel('Frequency (GHz)')
ax1.set_ylabel("ε'")
ax1.legend()

# Imaginary part
ax2.loglog(freq_ghz, contributions['debye_total'].imag, 'b-', label='Debye')
ax2.loglog(freq_ghz, contributions['lorentz_total'].imag, 'r-', label='Lorentz')
ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('ε"')
ax2.legend()
```

### Adaptive Optimization

```python
from app.models.hybrid_debye_lorentz import AdaptiveHybridModel

# Find optimal configuration
adaptive = AdaptiveHybridModel(max_debye=3, max_lorentz=5)
result = adaptive.fit(freq_ghz, dk_exp, df_exp)

print(f"Optimal configuration found:")
print(f"  {adaptive.n_debye} Debye + {adaptive.n_lorentz} Lorentz")
print(f"  AIC: {result.aic:.1f}")

# Get simplification suggestions
suggestions = adaptive.suggest_model_simplification(result)
print(f"\nModel suggestions:\n{suggestions}")
```

## Parameter Estimation Strategies

### 1. Frequency-Based Initialization

```python
def smart_hybrid_initialization(freq, dk, df, transition_freq=1.0):
    """Initialize parameters based on frequency regions."""
    
    # Separate frequency regions
    low_mask = freq < transition_freq
    high_mask = freq >= transition_freq
    
    # Initialize containers
    debye_params = []
    lorentz_params = []
    
    # Analyze low-frequency region
    if np.any(low_mask):
        freq_low = freq[low_mask]
        dk_low = dk[low_mask]
        df_low = df[low_mask]
        
        # Estimate static permittivity
        eps_s = dk_low[0]
        eps_inf_est = dk[-1]  # High-frequency limit
        
        # Find relaxation frequency from Df peak
        if len(df_low) > 3:
            peak_idx = np.argmax(df_low)
            f_relax = freq_low[peak_idx]
            tau = 1 / (2 * np.pi * f_relax * 1e9)
            
            debye_params.append({
                'delta_eps': eps_s - eps_inf_est,
                'tau': tau
            })
    
    # Analyze high-frequency region
    if np.any(high_mask):
        freq_high = freq[high_mask]
        eps_imag_high = df[high_mask] * dk[high_mask]
        
        # Find resonance peaks
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(eps_imag_high, 
                                     prominence=0.1*np.max(eps_imag_high))
        
        for peak_idx in peaks:
            f_res = freq_high[peak_idx]
            
            # Estimate parameters from peak
            lorentz_params.append({
                'f': eps_imag_high[peak_idx] * f_res / (2 * np.pi),
                'omega0': f_res,
                'gamma': 0.1 * f_res  # Initial guess
            })
    
    return {
        'eps_inf': eps_inf_est,
        'debye': debye_params,
        'lorentz': lorentz_params
    }
```

### 2. Sequential Fitting Strategy

```python
def sequential_hybrid_fitting(model, freq, dk, df):
    """Fit Debye and Lorentz components sequentially."""
    
    # Step 1: Fit low-frequency data with Debye only
    low_mask = freq < model.transition_freq
    if np.any(low_mask):
        from app.models.debye import DebyeModel
        
        debye_model = DebyeModel(n_terms=model.n_debye)
        debye_result = debye_model.fit(freq[low_mask], 
                                      dk[low_mask], 
                                      df[low_mask])
        
        # Extract Debye parameters
        debye_params = {}
        for i in range(model.n_debye):
            debye_params[f'delta_eps_{i+1}'] = debye_result.params[f'delta_eps_{i+1}'].value
            debye_params[f'tau_{i+1}'] = debye_result.params[f'tau_{i+1}'].value
    
    # Step 2: Subtract Debye contribution from high-frequency data
    high_mask = freq >= model.transition_freq
    if np.any(high_mask):
        # Calculate Debye contribution at high frequencies
        debye_contrib = np.zeros_like(freq[high_mask], dtype=complex)
        for i in range(model.n_debye):
            omega = 2 * np.pi * freq[high_mask] * 1e9
            tau = debye_params[f'tau_{i+1}']
            delta_eps = debye_params[f'delta_eps_{i+1}']
            debye_contrib += delta_eps / (1 + 1j * omega * tau)
        
        # Residual for Lorentz fitting
        dk_residual = dk[high_mask] - debye_contrib.real
        df_residual = (df[high_mask] * dk[high_mask] - debye_contrib.imag) / dk_residual
        
        # Fit Lorentz to residual
        from app.models.lorentz import LorentzOscillatorModel
        
        lorentz_model = LorentzOscillatorModel(n_oscillators=model.n_lorentz)
        lorentz_result = lorentz_model.fit(freq[high_mask], 
                                          dk_residual, 
                                          df_residual)
    
    # Step 3: Combined refinement
    params = model.create_parameters(freq, dk, df)
    
    # Set initial values from sequential fits
    params['eps_inf'].value = debye_result.params['eps_inf'].value
    
    for i in range(model.n_debye):
        params[f'delta_eps_{i+1}'].value = debye_params[f'delta_eps_{i+1}']
        params[f'tau_{i+1}'].value = debye_params[f'tau_{i+1}']
    
    for i in range(model.n_lorentz):
        params[f'f_{i+1}'].value = lorentz_result.params[f'f_{i+1}'].value
        params[f'omega0_{i+1}'].value = lorentz_result.params[f'omega0_{i+1}'].value
        params[f'gamma_{i+1}'].value = lorentz_result.params[f'gamma_{i+1}'].value
    
    # Final combined fit
    result = model.fit(freq, dk, df, params=params)
    
    return result
```

### 3. Temperature-Dependent Analysis

```python
def analyze_hybrid_temperature_dependence(temps, freq, dk_data, df_data):
    """Track mechanism evolution with temperature."""
    
    model = HybridDebyeLorentzModel(auto_detect=True)
    results = []
    
    for i, T in enumerate(temps):
        result = model.fit(freq, dk_data[i], df_data[i])
        
        # Analyze regional dominance
        regions = model.analyze_frequency_regions(result)
        
        results.append({
            'T': T,
            'n_debye': model.n_debye,
            'n_lorentz': model.n_lorentz,
            'debye_dominance_low': regions['low']['debye_dominance'],
            'lorentz_dominance_high': regions['high']['lorentz_dominance'],
            'params': result.params
        })
    
    # Analyze trends
    import pandas as pd
    df_results = pd.DataFrame(results)
    
    # Check for mechanism transitions
    print("Temperature-dependent mechanism analysis:")
    
    # Plot dominance vs temperature
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['T'], df_results['debye_dominance_low'], 
             'b-o', label='Debye (low freq)')
    plt.plot(df_results['T'], df_results['lorentz_dominance_high'], 
             'r-s', label='Lorentz (high freq)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Mechanism Dominance')
    plt.legend()
    plt.title('Mechanism Evolution with Temperature')
    
    # Identify transitions
    for i in range(1, len(temps)):
        if df_results['n_debye'].iloc[i] != df_results['n_debye'].iloc[i-1]:
            print(f"Debye process change at T = {temps[i]} K")
        if df_results['n_lorentz'].iloc[i] != df_results['n_lorentz'].iloc[i-1]:
            print(f"Lorentz process change at T = {temps[i]} K")
    
    return df_results
```

## Process Separation and Analysis

### Mechanism Dominance Mapping

```python
def map_mechanism_dominance(result, n_freq_points=100):
    """Create frequency map of mechanism dominance."""
    
    model = result.model
    freq_map = np.logspace(np.log10(result.freq.min()), 
                          np.log10(result.freq.max()), 
                          n_freq_points)
    
    # Calculate contributions at each frequency
    contributions = model.get_process_contributions(result.params, freq_map)
    
    # Calculate dominance metrics
    debye_power = np.abs(contributions['debye_total'])**2
    lorentz_power = np.abs(contributions['lorentz_total'])**2
    total_power = debye_power + lorentz_power + 1e-10
    
    dominance_map = {
        'frequency': freq_map,
        'debye_fraction': debye_power / total_power,
        'lorentz_fraction': lorentz_power / total_power,
        'crossover_freq': None
    }
    
    # Find crossover frequency
    diff = dominance_map['debye_fraction'] - dominance_map['lorentz_fraction']
    zero_crossings = np.where(np.diff(np.sign(diff)))[0]
    
    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        f_cross = freq_map[idx]
        dominance_map['crossover_freq'] = f_cross
        print(f"Mechanism crossover at {f_cross:.3f} GHz")
    
    return dominance_map
```

### Individual Process Analysis

```python
def analyze_individual_processes(result):
    """Detailed analysis of each process."""
    
    model = result.model
    analysis = {
        'debye': [],
        'lorentz': []
    }
    
    # Analyze Debye processes
    for i in range(model.n_debye):
        delta_eps = result.params[f'delta_eps_{i+1}'].value
        tau = result.params[f'tau_{i+1}'].value
        f_relax = 1 / (2 * np.pi * tau) / 1e9
        
        # Calculate contribution to static permittivity
        static_contribution = delta_eps / (result.params['eps_s'].value - 
                                         result.params['eps_inf'].value)
        
        analysis['debye'].append({
            'index': i + 1,
            'delta_eps': delta_eps,
            'tau': tau,
            'relaxation_freq_ghz': f_relax,
            'static_contribution': static_contribution,
            'activation_energy_estimate': None  # Requires T-dependent data
        })
    
    # Analyze Lorentz oscillators
    for i in range(model.n_lorentz):
        f = result.params[f'f_{i+1}'].value
        omega0 = result.params[f'omega0_{i+1}'].value
        gamma = result.params[f'gamma_{i+1}'].value
        
        Q = omega0 / gamma
        
        # Peak characteristics
        if gamma < omega0:
            omega_peak = omega0 * np.sqrt(1 - (gamma/(2*omega0))**2)
        else:
            omega_peak = 0  # Overdamped
        
        analysis['lorentz'].append({
            'index': i + 1,
            'strength': f,
            'resonance_ghz': omega0,
            'damping_ghz': gamma,
            'q_factor': Q,
            'peak_freq_ghz': omega_peak,
            'overdamped': gamma >= omega0
        })
    
    return analysis
```

## Model Selection and Optimization

### AIC-Based Configuration Selection

```python
def optimize_hybrid_configuration(freq, dk, df, max_debye=4, max_lorentz=6):
    """Find optimal n_debye and n_lorentz using information criteria."""
    
    results = {}
    
    # Try different configurations
    for n_d in range(0, max_debye + 1):
        for n_l in range(0, max_lorentz + 1):
            if n_d == 0 and n_l == 0:
                continue
            
            try:
                model = HybridDebyeLorentzModel(n_debye=n_d, n_lorentz=n_l)
                result = model.fit(freq, dk, df)
                
                # Calculate information criteria
                n_params = 1 + 2*n_d + 3*n_l
                n_data = 2 * len(freq)
                
                aic = 2 * n_params + n_data * np.log(result.chisqr / n_data)
                bic = n_params * np.log(n_data) + n_data * np.log(result.chisqr / n_data)
                
                # Adjusted R² to penalize complexity
                r2_adj = 1 - (1 - result.fit_metrics.r_squared) * (n_data - 1) / (n_data - n_params - 1)
                
                results[(n_d, n_l)] = {
                    'result': result,
                    'aic': aic,
                    'bic': bic,
                    'r2_adj': r2_adj,
                    'rmse': result.fit_metrics.rmse
                }
                
            except Exception as e:
                print(f"Fit failed for ({n_d}D, {n_l}L): {e}")
    
    # Find optimal by different criteria
    best_aic = min(results.items(), key=lambda x: x[1]['aic'])
    best_bic = min(results.items(), key=lambda x: x[1]['bic'])
    best_r2 = max(results.items(), key=lambda x: x[1]['r2_adj'])
    
    print("Optimization results:")
    print(f"Best AIC: {best_aic[0]} (AIC = {best_aic[1]['aic']:.1f})")
    print(f"Best BIC: {best_bic[0]} (BIC = {best_bic[1]['bic']:.1f})")
    print(f"Best R²adj: {best_r2[0]} (R²adj = {best_r2[1]['r2_adj']:.4f})")
    
    # Create comparison plot
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # AIC vs complexity
    complexities = [1 + 2*k[0] + 3*k[1] for k in results.keys()]
    aics = [v['aic'] for v in results.values()]
    
    ax1.scatter(complexities, aics)
    ax1.set_xlabel('Number of Parameters')
    ax1.set_ylabel('AIC')
    ax1.set_title('Model Complexity vs AIC')
    
    # Configuration heatmap
    n_d_vals = sorted(set(k[0] for k in results.keys()))
    n_l_vals = sorted(set(k[1] for k in results.keys()))
    
    aic_matrix = np.full((len(n_d_vals), len(n_l_vals)), np.nan)
    
    for (n_d, n_l), vals in results.items():
        i = n_d_vals.index(n_d)
        j = n_l_vals.index(n_l)
        aic_matrix[i, j] = vals['aic']
    
    im = ax2.imshow(aic_matrix, aspect='auto', origin='lower')
    ax2.set_xticks(range(len(n_l_vals)))
    ax2.set_xticklabels(n_l_vals)
    ax2.set_yticks(range(len(n_d_vals)))
    ax2.set_yticklabels(n_d_vals)
    ax2.set_xlabel('Number of Lorentz')
    ax2.set_ylabel('Number of Debye')
    ax2.set_title('AIC Heatmap')
    plt.colorbar(im, ax=ax2)
    
    return results
```

### Cross-Validation for Model Selection

```python
def cross_validate_hybrid_model(freq, dk, df, n_folds=5):
    """K-fold cross-validation for configuration selection."""
    from sklearn.model_selection import KFold
    
    # Prepare data
    n_points = len(freq)
    indices = np.arange(n_points)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Test configurations
    configs = [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)]
    cv_results = {}
    
    for n_d, n_l in configs:
        fold_errors = []
        
        for train_idx, test_idx in kf.split(indices):
            # Split data
            freq_train = freq[train_idx]
            dk_train = dk[train_idx]
            df_train = df[train_idx]
            
            freq_test = freq[test_idx]
            dk_test = dk[test_idx]
            df_test = df[test_idx]
            
            try:
                # Train model
                model = HybridDebyeLorentzModel(n_debye=n_d, n_lorentz=n_l)
                result = model.fit(freq_train, dk_train, df_train)
                
                # Test prediction
                eps_pred = model.predict(freq_test, result.params)
                dk_pred = eps_pred.real
                df_pred = eps_pred.imag / eps_pred.real
                
                # Calculate test error
                rmse = np.sqrt(np.mean((dk_test - dk_pred)**2 + 
                                      (df_test - df_pred)**2))
                fold_errors.append(rmse)
                
            except:
                fold_errors.append(np.inf)
        
        cv_results[(n_d, n_l)] = {
            'mean_rmse': np.mean(fold_errors),
            'std_rmse': np.std(fold_errors)
        }
    
    # Find best configuration
    best_config = min(cv_results.items(), key=lambda x: x[1]['mean_rmse'])
    
    print(f"Cross-validation results:")
    for config, scores in sorted(cv_results.items()):
        print(f"  {config}: RMSE = {scores['mean_rmse']:.4f} ± {scores['std_rmse']:.4f}")
    
    print(f"\nBest configuration: {best_config[0]}")
    
    return cv_results
```

## Common Issues and Solutions

### 1. Process Overlap

**Problem**: Debye and Lorentz processes overlap in frequency

**Solution**:
```python
def handle_process_overlap(model, freq, dk, df):
    """Strategies for overlapping processes."""
    
    # Check for overlap
    result_initial = model.fit(freq, dk, df)
    char_freqs = model.get_characteristic_frequencies(result_initial.params)
    
    # Find maximum Debye frequency
    max_debye_freq = max(char_freqs['debye_relaxation']) if char_freqs['debye_relaxation'] else 0
    
    # Find minimum Lorentz frequency
    min_lorentz_freq = min(char_freqs['lorentz_resonance']) if char_freqs['lorentz_resonance'] else np.inf
    
    if max_debye_freq > 0.1 * min_lorentz_freq:
        print(f"Warning: Process overlap detected")
        print(f"  Max Debye: {max_debye_freq:.3f} GHz")
        print(f"  Min Lorentz: {min_lorentz_freq:.3f} GHz")
        
        # Strategy 1: Adjust transition frequency
        new_transition = np.sqrt(max_debye_freq * min_lorentz_freq)
        model.transition_freq = new_transition
        print(f"  Adjusted transition frequency to {new_transition:.3f} GHz")
        
        # Strategy 2: Use constraints
        params = model.create_parameters(freq, dk, df)
        
        # Constrain Debye to low frequencies
        for i in range(model.n_debye):
            f_max = 1 / (2 * np.pi * params[f'tau_{i+1}'].min) / 1e9
            if f_max > new_transition:
                params[f'tau_{i+1}'].min = 1 / (2 * np.pi * new_transition * 1e9)
        
        # Constrain Lorentz to high frequencies
        for i in range(model.n_lorentz):
            params[f'omega0_{i+1}'].min = new_transition
        
        # Refit with constraints
        result_constrained = model.fit(freq, dk, df, params=params)
        
        return result_constrained
    
    return result_initial
```

### 2. Parameter Correlation

**Problem**: Strong correlation between Debye and Lorentz parameters

**Solution**:
```python
def reduce_parameter_correlation(model, freq, dk, df):
    """Reduce correlation through intelligent fitting."""
    
    # Strategy 1: Fix eps_inf from high-frequency limit
    eps_inf_est = np.mean(dk[-5:])  # Average last 5 points
    
    params = model.create_parameters(freq, dk, df)
    params['eps_inf'].value = eps_inf_est
    params['eps_inf'].vary = False
    
    # Initial fit with fixed eps_inf
    result_fixed = model.fit(freq, dk, df, params=params)
    
    # Strategy 2: Sequential relaxation
    params = result_fixed.params
    
    # First, fix Lorentz and optimize Debye
    for i in range(model.n_lorentz):
        params[f'f_{i+1}'].vary = False
        params[f'omega0_{i+1}'].vary = False
        params[f'gamma_{i+1}'].vary = False
    
    result_debye = model.fit(freq, dk, df, params=params)
    
    # Then, fix Debye and optimize Lorentz
    params = result_debye.params
    for i in range(model.n_debye):
        params[f'delta_eps_{i+1}'].vary = False
        params[f'tau_{i+1}'].vary = False
    
    for i in range(model.n_lorentz):
        params[f'f_{i+1}'].vary = True
        params[f'omega0_{i+1}'].vary = True
        params[f'gamma_{i+1}'].vary = True
    
    result_lorentz = model.fit(freq, dk, df, params=params)
    
    # Final joint optimization
    params = result_lorentz.params
    for param in params:
        params[param].vary = True
    
    params['eps_inf'].vary = True  # Release eps_inf
    result_final = model.fit(freq, dk, df, params=params)
    
    # Check improvement
    print(f"Correlation reduction:")
    print(f"  Initial R²: {result_fixed.fit_metrics.r_squared:.4f}")
    print(f"  Final R²: {result_final.fit_metrics.r_squared:.4f}")
    
    return result_final
```

### 3. Insufficient Frequency Coverage

**Problem**: Limited data in either relaxation or resonance region

**Solution**:
```python
def handle_limited_coverage(model, freq, dk, df):
    """Adapt model to available frequency range."""
    
    # Analyze coverage
    f_min, f_max = freq.min(), freq.max()
    decades = np.log10(f_max / f_min)
    
    print(f"Frequency coverage: {f_min:.3e} - {f_max:.3e} GHz ({decades:.1f} decades)")
    
    # Check coverage below and above transition
    low_coverage = np.sum(freq < model.transition_freq)
    high_coverage = np.sum(freq >= model.transition_freq)
    
    if low_coverage < 10:
        print("Warning: Limited low-frequency coverage")
        print("  - Debye parameters may be poorly constrained")
        print("  - Consider fixing n_debye = 1 or 0")
        
        if low_coverage < 5:
            # Insufficient data for Debye
            model.n_debye = 0
            model._update_param_names()
            print("  - Set n_debye = 0")
    
    if high_coverage < 10:
        print("Warning: Limited high-frequency coverage")
        print("  - Lorentz parameters may be poorly constrained")
        print("  - Consider reducing n_lorentz")
        
        if high_coverage < 5:
            # Insufficient data for Lorentz
            model.n_lorentz = 0
            model._update_param_names()
            print("  - Set n_lorentz = 0")
    
    # Adjust transition frequency if needed
    if low_coverage > 0 and high_coverage > 0:
        # Put transition at geometric mean of coverage
        f_low_max = freq[freq < model.transition_freq].max()
        f_high_min = freq[freq >= model.transition_freq].min()
        
        new_transition = np.sqrt(f_low_max * f_high_min)
        model.transition_freq = new_transition
        print(f"  - Adjusted transition to {new_transition:.3f} GHz")
    
    return model.fit(freq, dk, df)
```

### 4. Convergence Issues

**Problem**: Fit fails to converge due to model complexity

**Solution**:
```python
def robust_hybrid_fitting(model, freq, dk, df, max_attempts=3):
    """Robust fitting with fallback strategies."""
    
    strategies = [
        {'method': 'leastsq', 'ftol': 1e-8, 'xtol': 1e-8},
        {'method': 'least_squares', 'ftol': 1e-6, 'xtol': 1e-6},
        {'method': 'differential_evolution', 'maxiter': 1000}
    ]
    
    best_result = None
    best_r2 = -np.inf
    
    for attempt, strategy in enumerate(strategies):
        print(f"\nAttempt {attempt + 1}: {strategy['method']}")
        
        try:
            # Create fresh parameters
            params = model.create_parameters(freq, dk, df)
            
            # Add some noise to avoid local minima
            if attempt > 0:
                for param in params:
                    if params[param].vary:
                        params[param].value *= (1 + 0.1 * np.random.randn())
            
            # Fit with current strategy
            result = model.fit(freq, dk, df, params=params, **strategy)
            
            if result.success and result.fit_metrics.r_squared > best_r2:
                best_result = result
                best_r2 = result.fit_metrics.r_squared
                print(f"  Success! R² = {best_r2:.4f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    if best_result is None:
        print("\nAll strategies failed. Trying simplified model...")
        
        # Reduce complexity
        if model.n_debye > 1:
            model.n_debye = 1
        if model.n_lorentz > 1:
            model.n_lorentz = 1
        
        model._update_param_names()
        
        # Try once more with simplified model
        best_result = model.fit(freq, dk, df)
    
    return best_result
```

### 5. Physical Inconsistencies

**Problem**: Fitted parameters violate physical constraints

**Solution**:
```python
def enforce_physical_consistency(result):
    """Check and enforce physical parameter constraints."""
    
    model = result.model
    params = result.params
    issues = []
    
    # Check global parameters
    if params['eps_inf'].value < 1:
        issues.append("eps_inf < 1 (non-physical)")
    
    # Check Debye parameters
    eps_static = params['eps_inf'].value
    for i in range(model.n_debye):
        delta_eps = params[f'delta_eps_{i+1}'].value
        tau = params[f'tau_{i+1}'].value
        
        eps_static += delta_eps
        
        if delta_eps < 0:
            issues.append(f"delta_eps_{i+1} < 0 (non-physical)")
        
        if tau < 1e-15 or tau > 1:
            issues.append(f"tau_{i+1} = {tau:.2e} s (unrealistic)")
    
    # Check Lorentz parameters
    for i in range(model.n_lorentz):
        f = params[f'f_{i+1}'].value
        omega0 = params[f'omega0_{i+1}'].value
        gamma = params[f'gamma_{i+1}'].value
        
        if f < 0:
            issues.append(f"f_{i+1} < 0 (non-physical)")
        
        Q = omega0 / gamma
        if Q < 0.5:
            issues.append(f"Q_{i+1} = {Q:.2f} (overdamped)")
    
    # Check static permittivity
    if eps_static < params['eps_inf'].value:
        issues.append("eps_static < eps_inf (violation of causality)")
    
    if issues:
        print("Physical consistency issues:")
        for issue in issues:
            print(f"  - {issue}")
        
        # Suggest fixes
        print("\nSuggested remedies:")
        print("  1. Enable constrain_parameters=True")
        print("  2. Check data quality and units")
        print("  3. Reduce model complexity")
        print("  4. Use bounded optimization methods")
    else:
        print("✓ All parameters physically consistent")
    
    # Validate with KK relations
    from app.models.kramers_kronig_validator import KramersKronigValidator
    import pandas as pd
    
    eps_model = model.predict(result.freq, params)
    kk_df = pd.DataFrame({
        'Frequency (GHz)': result.freq,
        'Dk': eps_model.real,
        'Df': eps_model.imag / eps_model.real
    })
    
    with KramersKronigValidator(kk_df) as validator:
        kk_result = validator.validate()
        print(f"\nKramers-Kronig validation: {kk_result['causality_status']}")
        if kk_result['mean_relative_error'] > 0.05:
            issues.append("Failed causality check")
    
    return issues
```

## Advanced Applications

### 1. Semiconductor Device Modeling

```python
def model_semiconductor_device(freq, dk, df, device_info):
    """Model semiconductor with carriers and lattice vibrations."""
    
    # Expected processes:
    # - Low freq: Carrier relaxation, trapping
    # - High freq: Optical phonons
    
    model = HybridDebyeLorentzModel(auto_detect=True)
    result = model.fit(freq, dk, df)
    
    # Interpret parameters
    print(f"Semiconductor: {device_info['material']}")
    print(f"Doping: {device_info['doping_cm3']:.2e} cm⁻³")
    
    # Analyze Debye processes (carriers)
    for i in range(model.n_debye):
        tau = result.params[f'tau_{i+1}'].value
        mobility = device_info['q'] * tau / device_info['m_eff']
        
        print(f"\nCarrier process {i+1}:")
        print(f"  Relaxation time: {tau:.2e} s")
        print(f"  Estimated mobility: {mobility:.1f} cm²/Vs")
        
        # Identify process type
        if tau > 1e-9:
            print("  Type: Likely trap-related")
        elif tau > 1e-12:
            print("  Type: Carrier scattering")
        else:
            print("  Type: Plasma oscillation")
    
    # Analyze Lorentz processes (phonons)
    for i in range(model.n_lorentz):
        omega0 = result.params[f'omega0_{i+1}'].value
        
        print(f"\nPhonon mode {i+1}:")
        print(f"  Frequency: {omega0:.1f} GHz ({omega0/0.03:.1f} cm⁻¹)")
        
        # Compare with known phonons
        if device_info['material'] == 'GaAs':
            if 8.0 < omega0 < 8.5:
                print("  Assignment: TO phonon")
            elif 8.7 < omega0 < 9.2:
                print("  Assignment: LO phonon")
        elif device_info['material'] == 'Si':
            if 15.5 < omega0 < 16.0:
                print("  Assignment: Optical phonon")
    
    # Calculate key device parameters
    eps_static = result.params['eps_inf'].value
    for i in range(model.n_debye):
        eps_static += result.params[f'delta_eps_{i+1}'].value
    
    # Plasma frequency
    f_plasma = 9e3 * np.sqrt(device_info['doping_cm3'] / 
                             (device_info['m_eff'] * eps_static))
    
    print(f"\nDevice parameters:")
    print(f"  Static permittivity: {eps_static:.2f}")
    print(f"  Plasma frequency: {f_plasma:.1f} GHz")
    
    return result
```

### 2. Biological Tissue Characterization

```python
def analyze_biological_tissue(freq, dk, df, tissue_type):
    """Comprehensive analysis of tissue dielectric properties."""
    
    # Expected:
    # - Multiple Debye: α (MHz), β (100 MHz), δ (GHz) dispersions  
    # - Lorentz: Water resonance (>10 GHz)
    
    model = HybridDebyeLorentzModel(n_debye=3, n_lorentz=1)
    result = model.fit(freq, dk, df)
    
    print(f"Tissue type: {tissue_type}")
    
    # Identify dispersions
    dispersions = {
        'alpha': {'range': (0.001, 0.1), 'mechanism': 'ionic diffusion'},
        'beta': {'range': (0.1, 100), 'mechanism': 'cellular membranes'},
        'delta': {'range': (0.1, 10), 'mechanism': 'proteins'},
        'gamma': {'range': (10, 100), 'mechanism': 'water'}
    }
    
    # Assign Debye processes
    debye_assignments = []
    for i in range(model.n_debye):
        f_relax = 1 / (2 * np.pi * result.params[f'tau_{i+1}'].value) / 1e9
        
        for name, props in dispersions.items():
            if props['range'][0] < f_relax < props['range'][1]:
                debye_assignments.append({
                    'process': i + 1,
                    'dispersion': name,
                    'frequency': f_relax,
                    'mechanism': props['mechanism']
                })
                break
    
    # Print assignments
    print("\nDispersion analysis:")
    for assignment in debye_assignments:
        print(f"  Debye {assignment['process']}: "
              f"{assignment['dispersion']}-dispersion at "
              f"{assignment['frequency']:.3f} GHz "
              f"({assignment['mechanism']})")
    
    # Water content from gamma dispersion
    if model.n_lorentz > 0:
        f_water = result.params['omega0_1'].value
        if f_water > 10:  # GHz
            # Estimate water content from relaxation strength
            delta_eps_water = result.params['f_1'].value
            water_fraction = delta_eps_water / 80  # Approximate
            
            print(f"\nWater analysis:")
            print(f"  Resonance: {f_water:.1f} GHz")
            print(f"  Estimated water content: {water_fraction*100:.1f}%")
    
    # Tissue state assessment
    beta_strength = 0
    for assignment in debye_assignments:
        if assignment['dispersion'] == 'beta':
            idx = assignment['process'] - 1
            beta_strength = result.params[f'delta_eps_{idx+1}'].value
    
    if beta_strength > 0:
        print(f"\nCellular integrity indicator: {beta_strength:.1f}")
        if beta_strength < 10:
            print("  Status: Possible membrane damage")
        else:
            print("  Status: Normal cellular structure")
    
    return result, debye_assignments
```

### 3. Composite Material Engineering

```python
def design_composite_response(target_properties, material_database):
    """Design composite with target hybrid response."""
    
    # Target: specific permittivity vs frequency
    freq_target = target_properties['frequency']
    eps_target = target_properties['permittivity']
    
    # Available materials
    matrix_options = material_database['matrices']
    filler_options = material_database['fillers']
    
    best_design = None
    best_error = np.inf
    
    for matrix in matrix_options:
        for filler in filler_options:
            for volume_fraction in np.linspace(0.1, 0.4, 10):
                # Predict composite response
                # Matrix: Debye relaxation
                # Filler: Lorentz resonance
                # Interface: Additional Debye
                
                n_debye = 2  # Matrix + interface
                n_lorentz = len(filler['resonances'])
                
                # Build model parameters
                params = {
                    'eps_inf': matrix['eps_inf'],
                    # Matrix relaxation
                    'delta_eps_1': matrix['delta_eps'] * (1 - volume_fraction),
                    'tau_1': matrix['tau'],
                    # Interface relaxation (Maxwell-Wagner)
                    'delta_eps_2': 2 * volume_fraction * (filler['eps'] - matrix['eps']) / 3,
                    'tau_2': matrix['eps'] * 8.854e-12 / matrix['conductivity']
                }
                
                # Add filler resonances
                for i, resonance in enumerate(filler['resonances']):
                    params[f'f_{i+1}'] = resonance['strength'] * volume_fraction
                    params[f'omega0_{i+1}'] = resonance['frequency']
                    params[f'gamma_{i+1}'] = resonance['damping']
                
                # Calculate response
                model = HybridDebyeLorentzModel(n_debye=n_debye, n_lorentz=n_lorentz)
                eps_composite = model.model_func(freq_target, **params)
                
                # Calculate error
                error = np.mean(np.abs(eps_composite - eps_target)**2)
                
                if error < best_error:
                    best_error = error
                    best_design = {
                        'matrix': matrix['name'],
                        'filler': filler['name'],
                        'volume_fraction': volume_fraction,
                        'predicted_response': eps_composite,
                        'model_params': params
                    }
    
    print("Optimal composite design:")
    print(f"  Matrix: {best_design['matrix']}")
    print(f"  Filler: {best_design['filler']}")
    print(f"  Volume fraction: {best_design['volume_fraction']:.1%}")
    print(f"  Design error: {best_error:.4f}")
    
    return best_design
```

### 4. Temperature-Dependent Process Evolution

```python
def track_mechanism_evolution(temps, freq, dk_data, df_data):
    """Monitor how mechanisms evolve with temperature."""
    
    evolution = {
        'temperature': temps,
        'debye_count': [],
        'lorentz_count': [],
        'crossover_freq': [],
        'activation_energies': {}
    }
    
    # Fit at each temperature
    for i, T in enumerate(temps):
        model = AdaptiveHybridModel()
        result = model.fit(freq, dk_data[i], df_data[i])
        
        evolution['debye_count'].append(model.n_debye)
        evolution['lorentz_count'].append(model.n_lorentz)
        
        # Find crossover frequency
        dominance = map_mechanism_dominance(result)
        evolution['crossover_freq'].append(dominance['crossover_freq'])
        
        # Store parameters for activation energy
        for j in range(model.n_debye):
            key = f'tau_{j+1}'
            if key not in evolution['activation_energies']:
                evolution['activation_energies'][key] = []
            evolution['activation_energies'][key].append(
                result.params[key].value
            )
    
    # Calculate activation energies
    k_B = 8.617e-5  # eV/K
    
    for param, tau_values in evolution['activation_energies'].items():
        if len(tau_values) > 3:
            # Arrhenius fit
            ln_tau = np.log(tau_values)
            inv_T = 1 / temps[:len(tau_values)]
            
            from scipy.stats import linregress
            slope, intercept, r, _, _ = linregress(inv_T, ln_tau)
            
            E_a = slope * k_B
            tau_0 = np.exp(intercept)
            
            print(f"\n{param} activation:")
            print(f"  E_a = {E_a:.3f} eV")
            print(f"  τ₀ = {tau_0:.2e} s")
            print(f"  R² = {r**2:.3f}")
    
    # Plot mechanism evolution
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Process count vs T
    ax = axes[0, 0]
    ax.plot(temps, evolution['debye_count'], 'b-o', label='Debye')
    ax.plot(temps, evolution['lorentz_count'], 'r-s', label='Lorentz')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Number of Processes')
    ax.legend()
    ax.set_title('Process Count Evolution')
    
    # Crossover frequency vs T
    ax = axes[0, 1]
    crossover_clean = [f for f in evolution['crossover_freq'] if f is not None]
    temps_clean = [T for T, f in zip(temps, evolution['crossover_freq']) if f is not None]
    if crossover_clean:
        ax.semilogy(temps_clean, crossover_clean, 'g-^')
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Crossover Frequency (GHz)')
        ax.set_title('Mechanism Crossover')
    
    # Activation plot
    ax = axes[1, 0]
    for param, tau_values in evolution['activation_energies'].items():
        if len(tau_values) > 3:
            ax.semilogy(1000/temps[:len(tau_values)], tau_values, 
                       'o-', label=param)
    ax.set_xlabel('1000/T (K⁻¹)')
    ax.set_ylabel('τ (s)')
    ax.set_title('Arrhenius Plot')
    ax.legend()
    
    # Phase diagram
    ax = axes[1, 1]
    # Create 2D representation of dominant mechanism
    freq_grid = np.logspace(-3, 3, 50)
    T_grid = np.linspace(temps.min(), temps.max(), 50)
    
    dominance_grid = np.zeros((50, 50))
    
    # This would need interpolation of the evolution data
    # Simplified version shown
    ax.imshow(dominance_grid, aspect='auto', origin='lower',
             extent=[temps.min(), temps.max(), -3, 3])
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('log₁₀(Frequency/GHz)')
    ax.set_title('Mechanism Dominance Map')
    
    plt.tight_layout()
    
    return evolution
```

## Examples

### Example 1: Silicon Semiconductor

```python
# Silicon with carriers and phonons
import numpy as np

# Frequency range: kHz to THz
freq_ghz = np.logspace(-6, 3, 150)

# Silicon parameters
si_info = {
    'material': 'Si',
    'doping_cm3': 1e16,  # n-type
    'mobility': 1400,    # cm²/Vs
    'm_eff': 0.26,       # Effective mass
    'q': 1.602e-19       # Elementary charge
}

# Generate synthetic data (example)
# Low freq: Free carrier response
# High freq: Optical phonon at ~15.5 THz

# [Load or generate dk, df data]

# Fit with hybrid model
model = HybridDebyeLorentzModel(auto_detect=True)
result = model.fit(freq_ghz, dk, df)

print("Silicon Analysis:")
print(f"Configuration: {model.n_debye} Debye + {model.n_lorentz} Lorentz")

# Detailed analysis
result_analysis = model_semiconductor_device(freq_ghz, dk, df, si_info)

# Visualize components
plot_data = model.get_plot_data(result, show_components=True)

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Real part with components
ax1.loglog(freq_ghz, dk, 'ko', label='Experimental', markersize=4)
ax1.loglog(plot_data['freq'], plot_data['dk_fit'], 'k-', label='Total fit')
ax1.loglog(plot_data['freq'], 
          plot_data['components']['eps_inf'] + 
          plot_data['components']['debye']['total_real'], 
          'b--', label='Carrier (Debye)')
ax1.loglog(plot_data['freq'], 
          plot_data['components']['lorentz']['total_real'], 
          'r--', label='Phonon (Lorentz)')
ax1.set_ylabel("Dk")
ax1.legend()
ax1.set_title('Silicon: Carriers + Phonons')

# Imaginary part
ax2.loglog(freq_ghz, df * dk, 'ko', markersize=4)
ax2.loglog(plot_data['freq'], 
          plot_data['df_fit'] * plot_data['dk_fit'], 'k-')
ax2.loglog(plot_data['freq'], 
          plot_data['components']['debye']['total_imag'], 
          'b--', label='Carrier loss')
ax2.loglog(plot_data['freq'], 
          plot_data['components']['lorentz']['total_imag'], 
          'r--', label='Phonon absorption')
ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('ε"')
ax2.legend()

plt.tight_layout()
```

### Example 2: Hydrated Protein

```python
def analyze_protein_hydration(freq_ghz, dk, df, protein_info):
    """Analyze protein with hydration shell dynamics."""
    
    # Expected processes:
    # Debye 1: Protein tumbling (MHz)
    # Debye 2: Hydration shell (GHz)
    # Debye 3: Bulk water (10 GHz)
    # Lorentz: Water librational mode (THz)
    
    model = HybridDebyeLorentzModel(n_debye=3, n_lorentz=1)
    result = model.fit(freq_ghz, dk, df)
    
    print(f"Protein: {protein_info['name']}")
    print(f"Concentration: {protein_info['concentration']} mg/mL")
    print(f"pH: {protein_info['pH']}")
    
    # Analyze each relaxation
    char_freqs = model.get_characteristic_frequencies(result.params)
    
    # Sort relaxations by frequency
    relax_data = []
    for i, f_relax in enumerate(char_freqs['debye_relaxation']):
        relax_data.append({
            'index': i + 1,
            'frequency': f_relax,
            'delta_eps': result.params[f'delta_eps_{i+1}'].value,
            'tau': result.params[f'tau_{i+1}'].value
        })
    
    relax_data.sort(key=lambda x: x['frequency'])
    
    # Assign processes
    for relax in relax_data:
        f = relax['frequency']
        
        if f < 0.01:  # < 10 MHz
            print(f"\nProcess {relax['index']}: Protein tumbling")
            # Estimate molecular weight from tau
            eta = 1e-3  # Water viscosity (Pa·s)
            k_B = 1.38e-23
            T = 298
            
            # Stokes-Einstein for rotation
            V_h = relax['tau'] * k_B * T / (6 * eta)
            MW_est = V_h * 1e27 * 0.73 / 6.02e23  # Rough estimate
            
            print(f"  Frequency: {f*1000:.2f} MHz")
            print(f"  Estimated MW: {MW_est/1000:.0f} kDa")
            
        elif 0.1 < f < 5:  # 100 MHz - 5 GHz
            print(f"\nProcess {relax['index']}: Hydration shell")
            print(f"  Frequency: {f:.2f} GHz")
            print(f"  Hydration number: ~{relax['delta_eps']/3:.0f}")
            
        elif f > 5:  # > 5 GHz
            print(f"\nProcess {relax['index']}: Bulk water")
            print(f"  Frequency: {f:.1f} GHz")
            
            # Compare with pure water
            f_water_pure = 17  # GHz at 25°C
            shift = f - f_water_pure
            print(f"  Shift from pure water: {shift:+.1f} GHz")
    
    # Analyze water libration (Lorentz)
    if model.n_lorentz > 0:
        f_lib = result.params['omega0_1'].value
        print(f"\nWater libration mode: {f_lib:.0f} GHz")
        
        if f_lib > 100:
            print("  Type: Hindered rotation")
        else:
            print("  Type: Collective mode")
    
    # Calculate hydration properties
    total_water = sum(r['delta_eps'] for r in relax_data[1:])  # Exclude protein
    bound_water = relax_data[1]['delta_eps'] if len(relax_data) > 1 else 0
    
    print(f"\nHydration analysis:")
    print(f"  Total water contribution: Δε = {total_water:.1f}")
    print(f"  Bound/total ratio: {bound_water/total_water:.2f}")
    
    return result

# Example usage
protein_info = {
    'name': 'Lysozyme',
    'concentration': 50,  # mg/mL
    'pH': 7.0,
    'MW': 14300  # Da
}

result = analyze_protein_hydration(freq_ghz, dk, df, protein_info)