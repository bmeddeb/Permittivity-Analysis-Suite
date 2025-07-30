# Lorentz Oscillator Model Documentation

## Table of Contents
- [Overview](#overview)
- [Theory and Physics](#theory-and-physics)
- [Mathematical Formulation](#mathematical-formulation)
- [Physical Interpretation](#physical-interpretation)
- [When to Use Lorentz Model](#when-to-use-lorentz-model)
- [Implementation Details](#implementation-details)
- [Usage Guide](#usage-guide)
- [Parameter Estimation Strategies](#parameter-estimation-strategies)
- [Multi-Oscillator Analysis](#multi-oscillator-analysis)
- [Optical Properties Calculation](#optical-properties-calculation)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Advanced Applications](#advanced-applications)
- [Examples](#examples)

## Overview

The Lorentz oscillator model describes resonant polarization mechanisms in materials, where bound charges oscillate at characteristic frequencies. It's fundamental for understanding optical, infrared, and terahertz responses of materials.

### Key Features

- **Multiple oscillators**: Captures complex spectra with multiple resonances
- **Automatic detection**: Identifies number of oscillators from data
- **Resonance analysis**: Full characterization of each oscillator
- **Optical properties**: Calculates refractive index and absorption
- **Physical constraints**: Ensures causality and stability

### Model Equation

```
ε*(ω) = ε∞ + Σᵢ [fᵢ·ω₀ᵢ² / (ω₀ᵢ² - ω² - jγᵢω)]
```

Where:
- **fᵢ**: Oscillator strength (dimensionless)
- **ω₀ᵢ**: Resonance frequency (rad/s)
- **γᵢ**: Damping coefficient (rad/s)
- **ε∞**: High-frequency permittivity

## Theory and Physics

### Physical Origins

Lorentz oscillators arise from:

1. **Electronic Transitions**
   - Interband transitions in semiconductors
   - Excitons in quantum wells
   - Color centers in crystals
   - π→π* transitions in organics

2. **Vibrational Modes**
   - Optical phonons in crystals
   - Molecular vibrations (IR active)
   - Lattice resonances
   - Local modes from defects

3. **Collective Excitations**
   - Plasmons in metals/semiconductors
   - Polaritons (photon-phonon coupling)
   - Magnons in magnetic materials
   - Surface phonon polaritons

### Classical Oscillator Model

The Lorentz model derives from Newton's equation for a bound charge:

```
m(d²x/dt²) + mγ(dx/dt) + mω₀²x = qE(t)
```

This gives:
- **Resonance** at ω₀ (natural frequency)
- **Damping** γ (energy dissipation)
- **Response** proportional to q²/m (oscillator strength)

### Key Physics Concepts

| Phenomenon | Frequency Range | Typical Materials |
|------------|----------------|-------------------|
| Plasma oscillations | THz-UV | Metals, doped semiconductors |
| Phonons (optical) | THz-IR | Ionic crystals, polar materials |
| Molecular vibrations | IR | Organic molecules, polymers |
| Electronic transitions | Vis-UV | Semiconductors, molecules |
| Excitons | Vis-NIR | Quantum structures |

## Mathematical Formulation

### Complex Permittivity

The frequency-dependent permittivity:

```
ε*(ω) = ε' - jε''
```

Real part (dispersion):
```
ε'(ω) = ε∞ + Σᵢ [fᵢ·ω₀ᵢ²(ω₀ᵢ² - ω²) / ((ω₀ᵢ² - ω²)² + γᵢ²ω²)]
```

Imaginary part (absorption):
```
ε''(ω) = Σᵢ [fᵢ·ω₀ᵢ²·γᵢω / ((ω₀ᵢ² - ω²)² + γᵢ²ω²)]
```

### Resonance Characteristics

At resonance (ω = ω₀):
- **Peak ε''**: fᵢω₀ᵢ/γᵢ
- **FWHM**: γᵢ (in angular frequency)
- **Quality factor**: Q = ω₀ᵢ/γᵢ

### Limiting Behaviors

1. **Low frequency** (ω << ω₀):
   ```
   ε' → ε∞ + Σᵢfᵢ (static contribution)
   ε'' → 0
   ```

2. **High frequency** (ω >> ω₀):
   ```
   ε' → ε∞ - Σᵢ(fᵢω₀ᵢ²/ω²) (plasma-like)
   ε'' → Σᵢ(fᵢω₀ᵢ²γᵢ/ω³)
   ```

3. **Resonance** (ω ≈ ω₀):
   - Anomalous dispersion in ε'
   - Maximum absorption in ε''

## Physical Interpretation

### Oscillator Strength

The oscillator strength fᵢ relates to:
```
fᵢ = (Nᵢq²)/(ε₀mᵢω₀ᵢ²)
```

Where:
- Nᵢ: Number density of oscillators
- q: Charge
- mᵢ: Effective mass

### Quality Factor

Q = ω₀/γ indicates:
- **Q > 100**: Sharp resonance (low loss)
- **Q = 10-100**: Moderate broadening
- **Q < 10**: Heavily damped (overdamped if Q < 0.5)

### Sum Rules

The oscillators satisfy:
```
Σᵢfᵢ = ωₚ²/ω₀² (f-sum rule)
```

Where ωₚ is the plasma frequency.

## When to Use Lorentz Model

### Ideal Applications

1. **Optical Materials**
   - Dielectric thin films
   - Optical crystals (quartz, sapphire)
   - Glass and transparent ceramics
   - Photonic crystals

2. **Semiconductors**
   - Band edge absorption
   - Excitonic features
   - Impurity/defect states
   - Quantum wells/dots

3. **Metamaterials**
   - Split-ring resonators
   - Plasmonic structures
   - Negative index materials
   - Metasurfaces

4. **Spectroscopic Analysis**
   - FTIR of molecular crystals
   - THz spectroscopy
   - Ellipsometry data
   - Reflectance/transmittance

### Diagnostic Signs for Lorentz

Use Lorentz when observing:
- **Sharp peaks** in ε'' or absorption
- **Anomalous dispersion** (S-shaped) in ε'
- **Symmetric peaks** on linear frequency scale
- **Well-defined resonances** with clear frequencies

### When Other Models Are Better

Don't use Lorentz for:
- **Relaxation processes** (use Debye/Cole-Cole)
- **Broad, asymmetric features** (use Havriliak-Negami)
- **Conductivity-dominated** response (add DC term)
- **Continuous distributions** (use other approaches)

## Implementation Details

### Class Structure

```python
class LorentzOscillatorModel(BaseModel):
    """
    Lorentz oscillator model for resonant polarization.
    
    Parameters:
        n_oscillators: Number of resonances
        auto_detect: Automatically find oscillators
        constrain_parameters: Apply physical constraints
        name: Custom model name
    """
```

### Parameter Bounds

| Parameter | Symbol | Typical Range | Units | Constraints |
|-----------|--------|---------------|-------|-------------|
| ε∞ | eps_inf | 1-20 | - | > 1 |
| fᵢ | f_i | 0.001-1000 | - | > 0 |
| ω₀ᵢ | omega0_i | 0.1-10⁶ | GHz | > 0 |
| γᵢ | gamma_i | 0.001-ω₀ᵢ | GHz | > 0, < ω₀ᵢ |

### Oscillator Detection Algorithm

The auto-detection uses:
1. **Peak finding** in ε''
2. **Prominence analysis** for significance
3. **Anomalous dispersion** detection in ε'
4. **Second derivative** analysis
5. **Practical limit** of 5 oscillators

## Usage Guide

### Basic Single Oscillator

```python
from app.models.lorentz import LorentzOscillatorModel

# Single resonance
model = LorentzOscillatorModel(n_oscillators=1)
result = model.fit(freq_ghz, dk_exp, df_exp)

# Get parameters
f = result.params['f_1'].value
omega0 = result.params['omega0_1'].value
gamma = result.params['gamma_1'].value
Q = omega0 / gamma

print(f"Resonance at {omega0:.2f} GHz with Q = {Q:.1f}")
```

### Automatic Detection

```python
# Let model find oscillators
model = LorentzOscillatorModel(auto_detect=True)
result = model.fit(freq_ghz, dk_exp, df_exp)

# Check what was found
n_found = model.n_oscillators
print(f"Found {n_found} oscillators")

# Get spectral properties
spectral = model.get_spectral_properties(result.params)
print(f"Frequency range: {spectral['frequency_range']} GHz")
```

### Multi-Oscillator System

```python
# Three-oscillator system
model = LorentzOscillatorModel(n_oscillators=3)
result = model.fit(freq_ghz, dk_exp, df_exp)

# Analyze each oscillator
for i in range(3):
    props = model.get_oscillator_properties(result.params, i+1)
    print(f"\nOscillator {i+1}:")
    print(f"  Frequency: {props['resonance_freq_GHz']:.2f} GHz")
    print(f"  Q factor: {props['quality_factor']:.1f}")
    print(f"  Strength: {props['oscillator_strength']:.3f}")
    print(f"  Peak absorption: {props['max_absorption']:.3f}")
```

### Visualization with Components

```python
# Get plot data with individual contributions
plot_data = model.get_plot_data(result, show_components=True)

# Access components
if 'components' in plot_data:
    freq_smooth = plot_data['components']['freq']
    
    for osc in plot_data['components']['oscillators']:
        idx = osc['index']
        eps_real = osc['eps_real']
        eps_imag = osc['eps_imag']
        
        # Plot individual oscillator contribution
        plt.plot(freq_smooth, eps_imag, 
                label=f'Oscillator {idx}')
```

## Parameter Estimation Strategies

### 1. Peak-Based Initialization

```python
def estimate_lorentz_parameters(freq, dk, df):
    """Smart parameter estimation from peaks."""
    
    # Calculate loss
    eps_imag = df * dk
    
    # Find peaks
    from scipy.signal import find_peaks, peak_widths
    peaks, properties = find_peaks(eps_imag, 
                                 prominence=0.1*np.max(eps_imag))
    
    oscillators = []
    
    for peak_idx in peaks:
        # Frequency
        f0 = freq[peak_idx]
        
        # Height gives strength estimate
        peak_height = eps_imag[peak_idx]
        f_est = peak_height * f0 / (2 * np.pi)
        
        # Width gives damping
        widths = peak_widths(eps_imag, [peak_idx], rel_height=0.5)
        if widths[0][0] > 0:
            width_idx = widths[0][0]
            width_freq = freq[int(peak_idx + width_idx/2)] - freq[int(peak_idx - width_idx/2)]
            gamma_est = width_freq
        else:
            gamma_est = 0.1 * f0
        
        oscillators.append({
            'f0': f0,
            'strength': f_est,
            'damping': gamma_est,
            'Q': f0 / gamma_est
        })
    
    return oscillators
```

### 2. Kramers-Kronig Constrained Fitting

```python
def fit_with_kk_constraint(model, freq, dk, df):
    """Ensure Kramers-Kronig consistency using the KramersKronigValidator."""
    from app.models.kramers_kronig_validator import KramersKronigValidator
    import pandas as pd
    
    # Initial fit
    result = model.fit(freq, dk, df)
    
    # Validate the fitted model using KK
    eps_model = model.predict(freq, result.params)
    
    # Create DataFrame for validator
    model_df = pd.DataFrame({
        'Frequency (GHz)': freq,
        'Dk': eps_model.real,
        'Df': eps_model.imag / eps_model.real
    })
    
    # Run KK validation
    validator = KramersKronigValidator(model_df)
    kk_results = validator.validate()
    
    if kk_results['causality_status'] == 'FAIL':
        print(f"Warning: KK violation detected (error: {kk_results['mean_relative_error']:.2%})")
        # Could implement parameter adjustment here
    
    return result, kk_results
```

### 3. Temperature Series Analysis

```python
def analyze_temperature_dependence(temps, freq, dk_data, df_data):
    """Extract temperature trends of oscillators."""
    
    model = LorentzOscillatorModel(n_oscillators=2)
    results = []
    
    for i, T in enumerate(temps):
        result = model.fit(freq, dk_data[i], df_data[i])
        
        # Extract oscillator parameters
        for j in range(2):
            osc_data = {
                'T': T,
                'osc': j+1,
                'f0': result.params[f'omega0_{j+1}'].value,
                'gamma': result.params[f'gamma_{j+1}'].value,
                'strength': result.params[f'f_{j+1}'].value
            }
            results.append(osc_data)
    
    # Analyze trends
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Frequency shift with temperature
    for osc in [1, 2]:
        osc_df = df[df['osc'] == osc]
        
        # Linear fit to frequency
        from scipy import stats
        slope, intercept, r, p, se = stats.linregress(osc_df['T'], osc_df['f0'])
        
        print(f"Oscillator {osc}:")
        print(f"  Frequency shift: {slope*1000:.2f} MHz/K")
        print(f"  Thermal expansion coefficient: {-slope/intercept:.2e} K⁻¹")
```

## Multi-Oscillator Analysis

### Coupling Between Oscillators

```python
def analyze_oscillator_coupling(result):
    """Check for coupling between oscillators."""
    
    n_osc = result.model.n_oscillators
    
    if n_osc < 2:
        return
    
    # Extract frequencies and strengths
    frequencies = []
    strengths = []
    
    for i in range(n_osc):
        frequencies.append(result.params[f'omega0_{i+1}'].value)
        strengths.append(result.params[f'f_{i+1}'].value)
    
    # Check frequency spacing
    frequencies = np.sort(frequencies)
    spacings = np.diff(frequencies)
    min_spacing = np.min(spacings)
    
    # Coupling criterion
    for i in range(len(spacings)):
        f1, f2 = frequencies[i], frequencies[i+1]
        gamma1 = result.params[f'gamma_{i+1}'].value
        gamma2 = result.params[f'gamma_{i+2}'].value
        
        overlap = (gamma1 + gamma2) / (2 * (f2 - f1))
        
        if overlap > 0.5:
            print(f"Strong coupling between oscillators {i+1} and {i+2}")
            print(f"  Frequency separation: {f2-f1:.2f} GHz")
            print(f"  Overlap parameter: {overlap:.2f}")
            print("  Consider coupled oscillator model")
```

### Mode Assignment

```python
def assign_oscillator_modes(result, material_info):
    """Assign physical modes to oscillators."""
    
    assignments = []
    
    for i in range(result.model.n_oscillators):
        props = LorentzOscillatorModel.get_oscillator_properties(result.params, i+1)
        f0 = props['resonance_freq_GHz']
        
        # Database of known modes
        if material_info['type'] == 'semiconductor':
            if 0.1 < f0 < 1:
                mode = "Acoustic phonon overtone"
            elif 1 < f0 < 20:
                mode = "Optical phonon (TO/LO)"
            elif 20 < f0 < 100:
                mode = "High-frequency optical mode"
            elif f0 > 1000:
                mode = "Electronic transition"
            else:
                mode = "Unknown"
                
        elif material_info['type'] == 'organic':
            if 20 < f0 < 120:
                mode = "Lattice/intermolecular"
            elif 800 < f0 < 1200:
                mode = "C-H bending"
            elif 1200 < f0 < 1800:
                mode = "C=C/C=O stretching"
            elif 2800 < f0 < 3200:
                mode = "C-H stretching"
            else:
                mode = "Unknown"
        
        assignments.append({
            'oscillator': i+1,
            'frequency': f0,
            'assignment': mode,
            'Q_factor': props['quality_factor']
        })
    
    return pd.DataFrame(assignments)
```

## Optical Properties Calculation

### Refractive Index and Absorption

```python
def calculate_optical_constants(result, wavelengths_nm):
    """Calculate n, k, and other optical properties."""
    
    model = result.model
    optical = model.estimate_optical_properties(result.params, wavelengths_nm)
    
    # Create comprehensive report
    print("Optical Properties:")
    print(f"{'λ (nm)':>8} {'n':>8} {'k':>8} {'α (cm⁻¹)':>10} {'R':>8}")
    print("-" * 50)
    
    for i, wl in enumerate(wavelengths_nm):
        n = optical['n'][i]
        k = optical['k'][i]
        alpha = optical['absorption_coeff_per_cm'][i]
        R = optical['reflectivity'][i]
        
        print(f"{wl:8.0f} {n:8.3f} {k:8.3f} {alpha:10.2e} {R:8.3f}")
    
    return optical
```

### Sellmeier Equation Conversion

```python
def convert_to_sellmeier(result):
    """Convert Lorentz parameters to Sellmeier form."""
    
    # Sellmeier: n² = A + Σ(Bᵢλ²/(λ² - Cᵢ²))
    
    c = 299792458  # m/s
    params = result.params
    
    # Background term
    A = params['eps_inf'].value
    
    # Oscillator terms
    terms = []
    for i in range(result.model.n_oscillators):
        f = params[f'f_{i+1}'].value
        f0_ghz = params[f'omega0_{i+1}'].value
        
        # Convert to wavelength
        lambda0_um = c / (f0_ghz * 1e9) * 1e6
        
        # Sellmeier coefficients
        B = f * params['eps_inf'].value
        C = lambda0_um
        
        terms.append({'B': B, 'C': C})
    
    # Generate Sellmeier formula
    formula = f"n² = {A:.3f}"
    for i, term in enumerate(terms):
        formula += f" + {term['B']:.3f}λ²/(λ² - {term['C']:.3f}²)"
    
    print("Sellmeier equation:")
    print(formula)
    
    return A, terms
```

### Dispersion Analysis

```python
def analyze_dispersion(result, wavelength_range=(400, 1000)):
    """Analyze dispersion properties."""
    
    # Calculate dn/dλ
    wavelengths = np.linspace(*wavelength_range, 100)
    optical = result.model.estimate_optical_properties(result.params, wavelengths)
    n = optical['n']
    
    # Numerical derivative
    dn_dlambda = np.gradient(n, wavelengths)
    
    # Find zero-dispersion wavelengths
    zero_disp_idx = np.where(np.diff(np.sign(dn_dlambda)))[0]
    
    if len(zero_disp_idx) > 0:
        print("Zero-dispersion wavelengths:")
        for idx in zero_disp_idx:
            print(f"  {wavelengths[idx]:.1f} nm")
    
    # Abbe number at reference wavelength
    n_d = np.interp(587.6, wavelengths, n)  # d-line
    n_F = np.interp(486.1, wavelengths, n)  # F-line  
    n_C = np.interp(656.3, wavelengths, n)  # C-line
    
    abbe = (n_d - 1) / (n_F - n_C)
    print(f"\nAbbe number: {abbe:.1f}")
    
    # Group velocity dispersion
    c = 3e8  # m/s
    omega = 2 * np.pi * c / (wavelengths * 1e-9)
    
    # GVD = d²n/dω²
    d2n_domega2 = np.gradient(np.gradient(n, omega), omega)
    
    # Convert to fs²/mm
    GVD = d2n_domega2 * 1e-15 * 1e3
    
    return {
        'wavelengths': wavelengths,
        'n': n,
        'dn_dlambda': dn_dlambda,
        'GVD_fs2_per_mm': GVD,
        'abbe_number': abbe
    }
```

## Common Issues and Solutions

### 1. Overlapping Oscillators

**Problem**: Oscillators too close to resolve

**Solution**:
```python
def handle_overlapping_oscillators(model, freq, dk, df):
    """Strategies for overlapping resonances."""
    
    # Try different approaches
    approaches = []
    
    # 1. Constrain oscillator number
    for n_osc in [1, 2, 3]:
        model_n = LorentzOscillatorModel(n_oscillators=n_osc)
        result = model_n.fit(freq, dk, df)
        approaches.append({
            'n_osc': n_osc,
            'result': result,
            'aic': result.aic
        })
    
    # 2. Fix frequency ratios
    model_constrained = LorentzOscillatorModel(n_oscillators=2)
    params = model_constrained.create_parameters(freq, dk, df)
    
    # Enforce minimum separation
    params.add('freq_ratio', expr='omega0_2 / omega0_1', min=1.5)
    
    result_constrained = model_constrained.fit(freq, dk, df, params=params)
    
    # 3. Use prior knowledge
    if 'expected_modes' in material_info:
        result_fixed = model.fit_with_constraints(
            freq, dk, df,
            known_resonances=material_info['expected_modes']
        )
    
    # Compare approaches
    best = min(approaches, key=lambda x: x['aic'])
    print(f"Best approach: {best['n_osc']} oscillators (AIC={best['aic']:.1f})")
    
    return best['result']
```

### 2. Weak Oscillators

**Problem**: Some oscillators have very small strength

**Solution**:
```python
def identify_significant_oscillators(result, threshold=0.01):
    """Filter out insignificant oscillators."""
    
    # Get total oscillator strength
    total_strength = LorentzOscillatorModel.get_total_oscillator_strength(result.params)
    
    significant = []
    weak = []
    
    for i in range(result.model.n_oscillators):
        f = result.params[f'f_{i+1}'].value
        relative_strength = f / total_strength
        
        if relative_strength > threshold:
            significant.append(i+1)
        else:
            weak.append(i+1)
    
    if weak:
        print(f"Weak oscillators detected: {weak}")
        print("Consider reducing model complexity")
        
        # Refit with fewer oscillators
        n_sig = len(significant)
        if n_sig < result.model.n_oscillators:
            model_reduced = LorentzOscillatorModel(n_oscillators=n_sig)
            result_reduced = model_reduced.fit(freq, dk, df)
            
            print(f"Reduced model R²: {result_reduced.fit_metrics.r_squared:.4f}")
            return result_reduced
    
    return result
```

### 3. High-Frequency Artifacts

**Problem**: Spurious oscillators at measurement limits

**Solution**:
```python
def remove_edge_artifacts(freq, dk, df, edge_fraction=0.1):
    """Remove potential artifacts near frequency edges."""
    
    # Identify edge regions
    n_points = len(freq)
    edge_points = int(n_points * edge_fraction)
    
    # Check for peaks near edges
    eps_imag = df * dk
    peaks, _ = find_peaks(eps_imag)
    
    edge_peaks = []
    for peak in peaks:
        if peak < edge_points or peak > n_points - edge_points:
            edge_peaks.append(peak)
    
    if edge_peaks:
        print(f"Warning: Peaks detected near frequency edges")
        print("These may be measurement artifacts")
        
        # Option 1: Trim data
        trim_idx = slice(edge_points, -edge_points)
        freq_trim = freq[trim_idx]
        dk_trim = dk[trim_idx]
        df_trim = df[trim_idx]
        
        # Option 2: Weight fitting
        weights = np.ones_like(freq)
        weights[:edge_points] = 0.5
        weights[-edge_points:] = 0.5
        
        # Refit with weights
        model = LorentzOscillatorModel(auto_detect=True)
        result = model.fit(freq_trim, dk_trim, df_trim)
        
        return result, freq_trim
    
    return None, freq

from scipy.signal import find_peaks
```

### 4. Overdamped Oscillators

**Problem**: γ > ω₀ (overdamped regime)

**Solution**:
```python
def handle_overdamped_oscillators(result):
    """Check and handle overdamped oscillators."""
    
    overdamped = []
    
    for i in range(result.model.n_oscillators):
        props = LorentzOscillatorModel.get_oscillator_properties(result.params, i+1)
        
        if props['overdamped']:
            overdamped.append(i+1)
            print(f"Oscillator {i+1} is overdamped:")
            print(f"  ω₀ = {props['resonance_freq_GHz']:.2f} GHz")
            print(f"  γ = {props['damping_GHz']:.2f} GHz")
            print(f"  Q = {props['quality_factor']:.2f}")
    
    if overdamped:
        print("\nRecommendations for overdamped oscillators:")
        print("1. Check if this is physically expected")
        print("2. Consider relaxation models (Debye/Cole-Cole)")
        print("3. May indicate measurement issues")
        
        # Try alternative model
        from app.models.cole_davidson import ColeDavidsonModel
        alt_model = ColeDavidsonModel()
        alt_result = alt_model.fit(freq, dk, df)
        
        print(f"\nAlternative model R²: {alt_result.fit_metrics.r_squared:.4f}")
        
        if alt_result.fit_metrics.r_squared > result.fit_metrics.r_squared:
            print("Relaxation model provides better fit")
            return alt_result
    
    return result
```

### 5. Causality Violations

**Problem**: Model fails Kramers-Kronig test

**Solution**:
```python
def ensure_causality(model, freq, dk, df):
    """Ensure model satisfies causality via KK relations."""
    from app.models.kramers_kronig_validator import KramersKronigValidator
    import pandas as pd
    
    # Initial fit
    result = model.fit(freq, dk, df)
    
    # Check causality
    eps_model = model.predict(freq, result.params)
    model_df = pd.DataFrame({
        'Frequency (GHz)': freq,
        'Dk': eps_model.real,
        'Df': eps_model.imag / eps_model.real
    })
    
    with KramersKronigValidator(model_df) as validator:
        kk_results = validator.validate()
        
        if kk_results['causality_status'] == 'FAIL':
            print(f"Causality violation: {kk_results['mean_relative_error']:.2%} error")
            
            # Strategy 1: Extend frequency range
            print("\nExtending frequency range for KK validation...")
            freq_extended = np.logspace(np.log10(freq[0])-1, np.log10(freq[-1])+1, 500)
            eps_extended = model.predict(freq_extended, result.params)
            
            extended_df = pd.DataFrame({
                'Frequency (GHz)': freq_extended,
                'Dk': eps_extended.real,
                'Df': eps_extended.imag / eps_extended.real
            })
            
            validator_ext = KramersKronigValidator(extended_df)
            kk_ext = validator_ext.validate()
            
            if kk_ext['causality_status'] == 'PASS':
                print("✓ Model is causal over extended range")
            else:
                print("✗ Causality issues persist")
                print("\nPossible solutions:")
                print("- Add missing oscillators")
                print("- Include conductivity term")
                print("- Check for measurement errors")
    
    return result, kk_results
```

## Advanced Applications

### 1. Phonon-Polariton Analysis

```python
def analyze_phonon_polaritons(result, material_params):
    """Analyze phonon-polariton dispersion."""
    
    # Extract TO and LO frequencies
    oscillators = []
    for i in range(result.model.n_oscillators):
        props = LorentzOscillatorModel.get_oscillator_properties(result.params, i+1)
        oscillators.append(props)
    
    # Sort by frequency
    oscillators.sort(key=lambda x: x['resonance_freq_GHz'])
    
    # Identify TO-LO pairs
    print("Phonon Analysis:")
    
    eps_s = result.params['eps_s'].value
    eps_inf = result.params['eps_inf'].value
    
    for i, osc in enumerate(oscillators):
        f_TO = osc['resonance_freq_GHz']
        
        # Lyddane-Sachs-Teller relation
        f_LO = f_TO * np.sqrt(eps_s / eps_inf)
        
        print(f"\nMode {i+1}:")
        print(f"  TO frequency: {f_TO:.2f} GHz")
        print(f"  LO frequency: {f_LO:.2f} GHz")
        print(f"  TO-LO splitting: {f_LO - f_TO:.2f} GHz")
        
        # Polariton gap
        print(f"  Reststrahlen band: {f_TO:.2f} - {f_LO:.2f} GHz")
        
        # Calculate polariton dispersion
        k_values = np.linspace(0, 1000, 100)  # cm⁻¹
        omega = np.zeros((len(k_values), 2))  # Upper and lower branches
        
        for j, k in enumerate(k_values):
            # Solve polariton dispersion relation
            # ω² = c²k²/ε(ω)
            # This requires numerical solution
            pass
    
    return oscillators
```

### 2. Metamaterial Design

```python
def design_metamaterial_response(target_spectrum, constraints):
    """Design Lorentz oscillators for target response."""
    
    freq = target_spectrum['frequency']
    target_n = target_spectrum['n']
    target_k = target_spectrum['k']
    
    # Convert to permittivity
    target_eps = (target_n + 1j*target_k)**2
    
    # Optimization problem
    from scipy.optimize import differential_evolution
    
    def objective(params, n_osc):
        """Minimize difference from target."""
        # Unpack parameters
        eps_inf = params[0]
        
        model_params = {'eps_inf': eps_inf}
        for i in range(n_osc):
            idx = 1 + i*3
            model_params[f'f_{i+1}'] = params[idx]
            model_params[f'omega0_{i+1}'] = params[idx+1]
            model_params[f'gamma_{i+1}'] = params[idx+2]
        
        # Calculate model response
        eps_model = LorentzOscillatorModel.model_func(freq, **model_params)
        
        # Weighted error
        error = np.sum(np.abs(eps_model - target_eps)**2)
        
        # Add constraints
        if 'bandwidth' in constraints:
            bw_penalty = 0
            for i in range(n_osc):
                gamma = params[1 + i*3 + 2]
                if gamma > constraints['bandwidth']:
                    bw_penalty += (gamma - constraints['bandwidth'])**2
            error += 100 * bw_penalty
        
        return error
    
    # Try different numbers of oscillators
    best_design = None
    best_error = np.inf
    
    for n_osc in range(1, 5):
        # Parameter bounds
        n_params = 1 + 3*n_osc
        bounds = [(1, 10)]  # eps_inf
        
        for i in range(n_osc):
            bounds.extend([
                (0.01, 100),    # f
                (freq[0], freq[-1]),  # omega0
                (0.001, 10)     # gamma
            ])
        
        # Optimize
        result = differential_evolution(
            lambda p: objective(p, n_osc),
            bounds,
            maxiter=1000
        )
        
        if result.fun < best_error:
            best_error = result.fun
            best_design = {
                'n_oscillators': n_osc,
                'parameters': result.x,
                'error': result.fun
            }
    
    print(f"Optimal design: {best_design['n_oscillators']} oscillators")
    print(f"Design error: {best_design['error']:.6f}")
    
    return best_design
```

### 3. Thin Film Analysis

```python
def analyze_thin_film_oscillators(result, film_thickness_nm):
    """Relate oscillators to thin film properties."""
    
    # Check for thickness-dependent modes
    c = 299792458  # m/s
    n_eff = np.sqrt(result.params['eps_inf'].value)
    
    # Fabry-Perot modes
    m_values = np.arange(1, 10)
    fp_frequencies = m_values * c / (2 * n_eff * film_thickness_nm * 1e-9) / 1e9  # GHz
    
    print(f"Expected Fabry-Perot modes for {film_thickness_nm} nm film:")
    for m, f_fp in zip(m_values, fp_frequencies):
        print(f"  m={m}: {f_fp:.2f} GHz")
    
    # Match with observed oscillators
    for i in range(result.model.n_oscillators):
        f_obs = result.params[f'omega0_{i+1}'].value
        
        # Check if matches FP mode
        for m, f_fp in zip(m_values, fp_frequencies):
            if abs(f_obs - f_fp) / f_fp < 0.1:
                print(f"\nOscillator {i+1} matches FP mode m={m}")
                print(f"  Observed: {f_obs:.2f} GHz")
                print(f"  Expected: {f_fp:.2f} GHz")
                break
    
    # Surface phonon polaritons
    if film_thickness_nm < 100:
        print("\nThin film regime - check for surface modes")
        
        # Dispersion relation for surface modes
        # Requires solving: ε₁ + ε₂*coth(kd) = 0
        # where ε₁, ε₂ are film and substrate permittivities
```

### 4. Machine Learning Integration

```python
def ml_oscillator_identification(freq, dk, df, material_database):
    """Use ML to identify oscillator origins."""
    
    # Extract features
    model = LorentzOscillatorModel(auto_detect=True)
    result = model.fit(freq, dk, df)
    
    features = []
    for i in range(result.model.n_oscillators):
        props = LorentzOscillatorModel.get_oscillator_properties(result.params, i+1)
        features.extend([
            props['resonance_freq_GHz'],
            props['quality_factor'],
            props['oscillator_strength']
        ])
    
    # Pad to fixed size
    max_oscillators = 5
    while len(features) < max_oscillators * 3:
        features.extend([0, 0, 0])
    
    # Load pre-trained model
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    clf = joblib.load('oscillator_classifier.pkl')
    
    # Predict oscillator types
    predictions = clf.predict([features])
    
    # Map to physical origins
    oscillator_types = {
        0: "Optical phonon",
        1: "Electronic transition", 
        2: "Plasmon",
        3: "Defect state",
        4: "Surface mode"
    }
    
    print("ML-predicted oscillator assignments:")
    for i in range(result.model.n_oscillators):
        pred_type = predictions[i]
        print(f"  Oscillator {i+1}: {oscillator_types.get(pred_type, 'Unknown')}")
```

## Examples

### Example 1: Semiconductor Optical Response

```python
# GaAs optical phonons and electronic transitions
import numpy as np

# Experimental data range: THz to visible
freq_ghz = np.logspace(1, 6, 200)  # 10 GHz to 1 PHz

# Generate synthetic data with known features
# TO phonon at 8.0 THz
# Electronic transition at 430 THz (1.77 eV bandgap)

# [Generate dk, df data...]

# Fit with auto-detection
model = LorentzOscillatorModel(auto_detect=True)
result = model.fit(freq_ghz, dk, df)

print("GaAs Analysis:")
print(f"Detected {model.n_oscillators} oscillators")

# Analyze each oscillator
spectral = model.get_spectral_properties(result.params)
for i, osc in enumerate(spectral['oscillators']):
    f0 = osc['resonance_freq_GHz']
    
    if 7000 < f0 < 9000:
        print(f"\nOscillator {i+1}: TO phonon")
        print(f"  Frequency: {f0/1000:.2f} THz")
        
        # Calculate LO frequency
        eps_s = result.params['eps_s'].value
        eps_inf = result.params['eps_inf'].value
        f_LO = f0 * np.sqrt(eps_s / eps_inf) / 1000
        print(f"  LO frequency: {f_LO:.2f} THz")
        
    elif f0 > 400000:
        print(f"\nOscillator {i+1}: Electronic transition")
        energy_eV = 4.136e-15 * f0 * 1e9
        print(f"  Energy: {energy_eV:.3f} eV")
        print(f"  Wavelength: {1240/energy_eV:.0f} nm")

# Calculate optical properties at 1550 nm
optical = model.estimate_optical_properties(result.params, 1550)
print(f"\nAt 1550 nm: n = {optical['n'][0]:.3f}, k = {optical['k'][0]:.3e}")
```

### Example 2: Metamaterial Resonator

```python
def analyze_split_ring_resonator(freq_ghz, dk, df, geometry):
    """Analyze metamaterial split-ring resonator response."""
    
    # Fit Lorentz model
    model = LorentzOscillatorModel(n_oscillators=1)
    result = model.fit(freq_ghz, dk, df)
    
    # Extract LC circuit parameters
    f0 = result.params['omega0_1'].value
    gamma = result.params['gamma_1'].value
    f_strength = result.params['f_1'].value
    
    print(f"Split-Ring Resonator Analysis:")
    print(f"  Resonance: {f0:.2f} GHz")
    print(f"  Q factor: {f0/gamma:.1f}")
    
    # Relate to geometry
    # SRR resonance: f = c/(2π√(LC))
    # L ∝ μ₀ × ring area / gap
    # C ∝ ε₀ × gap area / gap width
    
    if 'ring_radius' in geometry:
        r = geometry['ring_radius']  # mm
        gap = geometry['gap_width']  # mm
        
        # Estimate circuit parameters
        L_est = 1.26e-6 * np.pi * r**2 / gap  # nH
        C_est = 8.85e-12 * gap * 0.1 / gap  # pF
        
        f_circuit = 1 / (2 * np.pi * np.sqrt(L_est * C_est * 1e-21)) / 1e9
        
        print(f"\nCircuit model estimate: {f_circuit:.2f} GHz")
        print(f"  Inductance: {L_est:.2f} nH")
        print(f"  Capacitance: {C_est:.3f} pF")
    
    # Negative index band
    eps_real = model.predict(freq_ghz, result.params).real
    mu_real = 1 - f_strength / (1 - (freq_ghz/f0)**2)  # Approximate
    
    n_eff = np.sqrt(eps_real * mu_real)
    negative_band = freq_ghz[(eps_real < 0) & (mu_real < 0)]
    
    if len(negative_band) > 0:
        print(f"\nNegative index band: {negative_band[0]:.2f} - {negative_band[-1]:.2f} GHz")
    
    return result
```

### Example 3: Polymer Film IR Spectroscopy

```python
# Analyze polymer vibrational modes
def analyze_polymer_ir_spectrum(freq_ghz, dk, df, polymer_type):
    """Full IR analysis of polymer film."""
    
    # Fit multi-oscillator model
    model = LorentzOscillatorModel(auto_detect=True, max_oscillators=10)
    result = model.fit(freq_ghz, dk, df)
    
    print(f"{polymer_type} IR Spectrum Analysis")
    print(f"Found {model.n_oscillators} vibrational modes")
    
    # Convert to wavenumbers for IR convention
    c = 2.998e10  # cm/s
    
    # Mode assignment based on frequency
    mode_assignments = {
        (800, 1300): "C-O stretching, C-H bending",
        (1300, 1500): "CH2/CH3 deformation",
        (1500, 1700): "C=C stretching, aromatic",
        (1700, 1800): "C=O stretching",
        (2800, 3000): "C-H stretching (aliphatic)",
        (3000, 3100): "C-H stretching (aromatic)",
        (3200, 3600): "O-H stretching"
    }
    
    # Analyze each mode
    for i in range(model.n_oscillators):
        props = model.get_oscillator_properties(result.params, i+1)
        f_ghz = props['resonance_freq_GHz']
        wavenumber = f_ghz * 1e9 / c
        
        print(f"\nMode {i+1}:")
        print(f"  Frequency: {f_ghz:.1f} GHz ({wavenumber:.0f} cm⁻¹)")
        print(f"  Strength: {props['oscillator_strength']:.3f}")
        print(f"  Width: {props['damping_GHz']:.1f} GHz")
        
        # Assign mode
        for (wn_min, wn_max), assignment in mode_assignments.items():
            if wn_min < wavenumber < wn_max:
                print(f"  Assignment: {assignment}")
                break
    
    # Calculate integrated absorption
    total_absorption = 0
    for i in range(model.n_oscillators):
        f = result.params[f'f_{i+1}'].value
        omega0 = result.params[f'omega0_{i+1}'].value
        
        # Integrated absorption ∝ f * omega0
        total_absorption += f * omega0
    
    print(f"\nTotal integrated absorption: {total_absorption:.2f}")
    
    # Crystallinity estimate (for semi-crystalline polymers)
    if polymer_type in ['polyethylene', 'polypropylene', 'nylon']:
        # Ratio of sharp to broad peaks indicates crystallinity
        crystalline_modes = []
        amorphous_modes = []
        
        for i in range(model.n_oscillators):
            Q = result.params[f'omega0_{i+1}'].value / result.params[f'gamma_{i+1}'].value
            if Q > 20:
                crystalline_modes.append(i)
            else:
                amorphous_modes.append(i)
        
        if crystalline_modes:
            print(f"\nCrystallinity indicators:")
            print(f"  Crystalline modes: {len(crystalline_modes)}")
            print(f"  Amorphous modes: {len(amorphous_modes)}")
    
    return result

# Example usage
polymer_result = analyze_polymer_ir_spectrum(freq_ghz, dk, df, "polyethylene")
```

### Example 4: Quality Control Application

```python
def quality_control_lorentz(freq_ghz, dk, df, reference_params, tolerances):
    """QC using Lorentz parameters as fingerprint."""
    
    # Fit sample
    model = LorentzOscillatorModel(n_oscillators=len(reference_params['oscillators']))
    result = model.fit(freq_ghz, dk, df)
    
    print("Quality Control Analysis")
    print("-" * 50)
    
    qc_pass = True
    
    # Check each oscillator
    for i, ref_osc in enumerate(reference_params['oscillators']):
        sample_props = model.get_oscillator_properties(result.params, i+1)
        
        # Compare frequencies
        f_ref = ref_osc['frequency']
        f_sample = sample_props['resonance_freq_GHz']
        f_deviation = abs(f_sample - f_ref) / f_ref * 100
        
        # Compare strengths
        s_ref = ref_osc['strength']
        s_sample = sample_props['oscillator_strength']
        s_deviation = abs(s_sample - s_ref) / s_ref * 100
        
        # Compare Q factors
        q_ref = ref_osc['q_factor']
        q_sample = sample_props['quality_factor']
        q_deviation = abs(q_sample - q_ref) / q_ref * 100
        
        print(f"\nOscillator {i+1}:")
        print(f"  Frequency: {f_sample:.2f} GHz (ref: {f_ref:.2f}, dev: {f_deviation:.1f}%)")
        print(f"  Strength: {s_sample:.3f} (ref: {s_ref:.3f}, dev: {s_deviation:.1f}%)")
        print(f"  Q factor: {q_sample:.1f} (ref: {q_ref:.1f}, dev: {q_deviation:.1f}%)")
        
        # Check tolerances
        if f_deviation > tolerances['frequency_percent']:
            print(f"  ❌ Frequency out of tolerance")
            qc_pass = False
        if s_deviation > tolerances['strength_percent']:
            print(f"  ❌ Strength out of tolerance")
            qc_pass = False
        if q_deviation > tolerances['q_factor_percent']:
            print(f"  ❌ Q factor out of tolerance")
            qc_pass = False
    
    # Overall metrics
    eps_inf_ref = reference_params['eps_inf']
    eps_inf_sample = result.params['eps_inf'].value
    eps_deviation = abs(eps_inf_sample - eps_inf_ref) / eps_inf_ref * 100
    
    print(f"\nHigh-frequency permittivity: {eps_inf_sample:.3f}")
    print(f"  Reference: {eps_inf_ref:.3f}, Deviation: {eps_deviation:.1f}%")
    
    if eps_deviation > tolerances['eps_inf_percent']:
        print(f"  ❌ ε∞ out of tolerance")
        qc_pass = False
    
    # Final verdict
    print("\n" + "="*50)
    if qc_pass:
        print("✅ SAMPLE PASSES QC")
    else:
        print("❌ SAMPLE FAILS QC")
        
        # Suggest possible causes
        print("\nPossible issues:")
        if f_deviation > tolerances['frequency_percent']:
            print("- Composition variation")
            print("- Stress/strain in material")
        if s_deviation > tolerances['strength_percent']:
            print("- Density/concentration variation")
            print("- Defect concentration")
        if q_deviation > tolerances['q_factor_percent']:
            print("- Impurities or defects")
            print("- Processing conditions")
    
    return qc_pass, result

# Example QC parameters
reference = {
    'eps_inf': 2.25,
    'oscillators': [
        {'frequency': 8.5, 'strength': 0.15, 'q_factor': 25},
        {'frequency': 12.3, 'strength': 0.08, 'q_factor': 18}
    ]
}

tolerances = {
    'frequency_percent': 2.0,
    'strength_percent': 10.0,
    'q_factor_percent': 15.0,
    'eps_inf_percent': 5.0
}

qc_result = quality_control_lorentz(freq_ghz, dk, df, reference, tolerances)
```

## Conclusion

The Lorentz oscillator model is essential for analyzing resonant dielectric responses. Key advantages:

1. **Physical basis**: Direct connection to oscillating charges
2. **Spectroscopic tool**: Identifies and characterizes resonances
3. **Optical design**: Enables refractive index engineering
4. **Material fingerprinting**: Unique spectral signatures
5. **Wide applicability**: From THz to UV frequencies

Best practices:
- Use auto-detection for unknown materials
- Validate oscillator assignments with material knowledge
- Check for overdamping (consider relaxation models if Q < 1)
- Apply Kramers-Kronig relations for consistency
- Consider coupling for closely-spaced oscillators

The implementation provides robust fitting with intelligent initialization, comprehensive analysis capabilities, and tools for optical property calculation, making it suitable for research, development, and quality control applications across photonics, materials science, and spectroscopy.