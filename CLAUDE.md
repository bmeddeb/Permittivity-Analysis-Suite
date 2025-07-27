# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Running the Application
```bash
python app.py
```
The main Dash web application runs on http://127.0.0.1:8050

### Package Management
This project uses `uv` for package management:
```bash
uv sync                    # Install dependencies from uv.lock
uv add <package>          # Add a new dependency
uv remove <package>       # Remove a dependency
```

### Installation (Alternative)
```bash
pip install -r requirements.txt
```

## Architecture Overview

This is a **Dash web application** for dielectric permittivity analysis with the following structure:

### Core Components
- **`app.py`** - Main application entry point with Flask server and Dash app initialization
- **`layout.py`** - UI layout definition using Dash components
- **`callbacks.py`** - Interactive callback functions for the web interface
- **`plots.py`** - Plotly visualization functions

### Models Package (`models/`)
The application implements multiple dielectric models for fitting experimental data:

- **`base_model.py`** - BaseModel class with common fitting utilities and bounds checking
- **`debye_model.py`** - Single relaxation Debye model
- **`multipole_debye_model.py`** - Multiple Debye relaxation terms
- **`cole_cole_model.py`** - Symmetric broadening distribution model  
- **`cole_davidson_model.py`** - Asymmetric broadening distribution model
- **`havriliak_negami_model.py`** - General distribution model (α and β factors)
- **`lorentz_model.py`** - Resonant oscillator model for high frequencies
- **`hybrid_model.py`** - Combined Debye-Lorentz model
- **`sarkar_model.py`** - Debye model with conductivity term
- **`kk_model.py`** - Kramers-Kronig causality checking
- **`kk_check.py`** - Model-free causality verification

### Utilities (`utils/`)
- **`fitting_utils.py`** - Common fitting algorithms and optimization helpers
- **`helpers.py`** - General utility functions

### Key Features
- **CSV Data Upload** - Expects columns: Frequency_GHz, Dk (real permittivity), Df (loss factor)
- **Model Fitting** - Multiple dielectric models with bounded optimization using scipy.optimize.least_squares
- **Causality Verification** - Kramers-Kronig relations checking for data validation
- **Interactive Visualization** - Real-time plotting with Plotly and Bootstrap styling

### Model Fitting Architecture
All models inherit from `BaseModel` which provides:
- **lmfit-based parameter handling** with automatic bounds checking
- **AIC/BIC calculation** for statistical model comparison 
- **Parameter uncertainties** from covariance matrix
- **Consistent interface** with `create_parameters()` and `model_function()` methods

**Important**: Variable-complexity models require N to be specified at initialization:
```python
# Correct usage for variable models
MultiPoleDebyeModel(N=3).analyze(df)    # 3 Debye terms
LorentzModel(N=2).analyze(df)           # 2 oscillators  
HybridModel(N=2).analyze(df)            # 2 hybrid terms

# Simple models use default constructor
DebyeModel().analyze(df)
ColeColeModel().analyze(df)
```

The application supports broadband dielectric analysis from low-frequency relaxation (Debye) to high-frequency resonance effects (Lorentz), with automated model selection via AIC/BIC comparison.