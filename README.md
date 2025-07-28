# Permittivity Analysis Suite

A comprehensive Dash web application for analyzing dielectric permittivity data with **advanced preprocessing**, intelligent noise analysis, and automated model fitting workflows. Features scientific-grade spline-based smoothing algorithms with professional visualization and complete transparency.

## 🎯 Supported Models

 **Simple Debye Model** – Single relaxation time analysis  
 **Multi-Pole Debye Model** – Multiple relaxation terms  
 **Cole-Cole Model** – Symmetric broadening distribution  
 **Cole-Davidson Model** – Asymmetric broadening distribution  
 **Havriliak-Negami Model** – General distribution with α and β factors  
 **Lorentz Oscillator Model** – Resonant behavior at high frequencies  
 **Sarkar Model** – Debye model with conductivity terms  
 **Hybrid Debye-Lorentz Model** – Combined relaxation and resonance  
 **Kramers-Kronig Causality Check** – Model-free validation of measured data  

---

## 🚀 Features

### 🔧 **Advanced Data Preprocessing**
- **Intelligent Noise Analysis**: Automatic assessment using spectral SNR, roughness metrics, and derivative analysis
- **Scientific Smoothing Algorithms**: Spline-based methods (interpolating, smoothing, PCHIP) with automatic parameter tuning
- **Robust Outlier Handling**: LOWESS, Savitzky-Golay, Gaussian, and median filtering options
- **Smart Algorithm Selection**: Hybrid selection combining rule-based and quality-based approaches
- **Before/After Visualization**: Interactive comparison plots showing preprocessing impact
- **Full Manual Control**: Algorithm-specific parameter adjustment with real-time preview
- **Quality Score Visualization**: Visual metrics showing data improvement and noise reduction

### 📊 **Model Analysis & Fitting**
- **Smart Upload**: CSV files with columns **Frequency_GHz, Dk, Df**
- **Intelligent Analysis**: Auto-select best model or compare all models
- **Interactive Configuration**: Off-canvas panel with analysis settings
- **Real-time Visualization**: Interactive Plotly charts with model comparison
- **Data Export**: Download fitted data as CSV for single-model analysis
- **Loading Indicators**: Visual feedback during analysis processing
- **Statistical Validation**: AIC/BIC model comparison and parameter uncertainties

---

## 📦 Installation & Setup

### 🍎 macOS Setup

#### Option 1: Automated Setup (Recommended)
```bash
git clone https://github.com/bmeddeb/Permittivity-Analysis-Suite.git
cd Permittivity-Analysis-Suite
chmod +x run.sh
./run.sh
```

The `run.sh` script will automatically:
-  Check and install `uv` package manager if needed
-  Create Python virtual environment
-  Install all dependencies
-  Launch the application

#### Option 2: Manual Setup (macOS)
```bash
git clone https://github.com/bmeddeb/Permittivity-Analysis-Suite.git
cd permittivity_app

# Using uv (recommended)
uv venv
source .venv/bin/activate
uv sync
python app.py

# Or using pip
pip install -r requirements.txt
python app.py
```

### 🪟 Windows Setup

The `run.sh` script is macOS/Linux specific. Windows users should set up manually:

```cmd
git clone https://github.com/bmeddeb/Permittivity-Analysis-Suite.git
cd Permittivity-Analysis-Suite

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

**Alternative with conda:**
```cmd
git clone https://github.com/bmeddeb/Permittivity-Analysis-Suite.git
cd Permittivity-Analysis-Suite

# Create conda environment
conda create -n permittivity python=3.9
conda activate permittivity

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

---

## ▶️ Access the Application
Open your browser at **http://127.0.0.1:8050**

### Analysis Modes
- **Auto-Select Best Model**: Automatically chooses optimal model based on AIC/RMSE
- **Auto + Compare All Models**: Shows detailed comparison of all models
- **Manual Model Selection**: Choose specific models to analyze

### Preprocessing Workflow
1. **Upload Data**: System automatically analyzes noise characteristics
2. **Configure Preprocessing**: 
   - **Auto Mode** (Recommended): Intelligent algorithm selection using hybrid approach
   - **Manual Mode**: Choose specific algorithm with fine-tuned parameters
   - **No Preprocessing**: Skip smoothing for high-quality data
3. **Preview Results**: View quality assessment, recommendations, and algorithm info
4. **Visualize Impact**: See before/after comparison and detailed noise analysis
5. **Run Analysis**: Preprocessed data flows seamlessly into model fitting

#### Advanced Visualization Features
- **Before/After Comparison**: 4-panel plots showing original vs processed data side-by-side
- **Noise Analysis Plots**: Spectral analysis and derivative visualization for quality assessment  
- **Impact Metrics**: Visual progress bars and quality scores showing preprocessing effectiveness
- **Algorithm Information**: Interactive tooltips with detailed algorithm descriptions and use cases

---

## 🔬 Preprocessing Algorithms

### Spline-Based Methods (Primary Tier)
- **🔬 Interpolating Spline**: Passes exactly through data points - ideal for clean, high-quality data
- **🌊 Smoothing Spline**: Adaptive cubic spline using scientific s = m×σ² parameter rule
- **📈 PCHIP**: Piecewise Cubic Hermite interpolation - shape-preserving for monotonic data
- **🛡️ LOWESS**: Locally weighted regression - robust against outliers and measurement artifacts

### General-Purpose Methods (Secondary Tier)
- **📊 Savitzky-Golay**: Polynomial smoothing filter that preserves peaks and spectral features
- **🔲 Gaussian Filter**: Uniform convolution smoothing with adjustable bandwidth
- **🔧 Median Filter**: Non-linear filter for spike removal and impulse noise suppression

### Noise Analysis Metrics

#### Domain-Specific Metrics
- **Spectral SNR**: FFT-based signal-to-noise ratio in frequency domain
- **Roughness Metric**: Second-order differences normalized by first differences
- **Derivative Variance**: Variance of first derivatives for smoothness assessment

#### Quality Assessment
- **Overall Noise Score**: Composite metric (0-1 scale) combining all analysis methods
- **Local Variance Ratio**: Non-stationarity detection using windowed variance analysis
- **Quality Recommendations**: Algorithm suggestions based on data characteristics

---

## 📄 Data Format

### Input CSV Requirements
Your CSV file should contain exactly three columns:

| Frequency_GHz | Dk   | Df   |
|---------------|------|------|
| 1.0           | 4.2  | 0.015 |
| 2.0           | 4.1  | 0.017 |
| 5.0           | 4.0  | 0.020 |

- **Frequency_GHz**: Frequency values in GHz
- **Dk**: Real part of permittivity (dielectric constant)
- **Df**: Imaginary part of permittivity (loss factor)

### Export Functionality
- **CSV Download**: Export fitted model data when single model is selected
- **Filename Format**: `{original_filename}_{model_name}.csv`
- **Data Includes**: Frequency_GHz, fitted Dk, and fitted Df values

### Preprocessing Output Example
When preprocessing is applied, you'll see detailed feedback:

```
✅ Data Quality: Good (Noise Score: 0.32)
🔧 Preprocessing: Smoothing Spline Applied
💡 Recommendations:
  • Data quality is good - moderate smoothing applied
  • Spline parameters optimized using s = m×σ² rule
📊 Quality Score: 68% improvement
```

---

## 🛠 Tech Stack

### Core Framework
- **Dash / Plotly** – Interactive web interface and advanced visualization
- **Flask** – Backend web server and session management
- **Bootstrap** – Responsive UI components and modern styling

### Scientific Computing
- **SciPy** – Advanced signal processing, spline interpolation, and filtering algorithms
- **NumPy** – High-performance numerical computing and array operations
- **Pandas** – Data manipulation and analysis
- **statsmodels** – LOWESS regression and statistical smoothing methods
- **lmfit** – Non-linear least-squares fitting with parameter bounds and uncertainties

### Development Tools
- **uv** – Fast Python package management and dependency resolution

---

## 👤 Authors
- Ben Meddeb (Main Developer)

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation
If you use this tool in academic work, please cite:

```
Meddeb, B. (2025). Permittivity Analysis Suite: A tool for dielectric permittivity modeling and causality verification.
```
