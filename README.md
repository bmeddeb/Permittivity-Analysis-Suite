# Permittivity Analysis Suite

A comprehensive Dash web application for analyzing dielectric permittivity data with advanced modeling capabilities and automated analysis workflows.

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

---

## 🛠 Tech Stack
- **Dash / Plotly** – Interactive web interface and visualization
- **Flask** – Backend web server
- **SciPy / NumPy / Pandas** – Scientific computing and data processing
- **lmfit** – Non-linear least-squares fitting with parameter bounds
- **Bootstrap** – Responsive UI components
- **uv** – Fast Python package management

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
Meddeb, B. (2025). Permittivity Analysis Suite: A Dash-based tool for dielectric permittivity modeling and causality verification.
```
