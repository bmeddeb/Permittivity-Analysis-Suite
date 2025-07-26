# Permittivity Analysis Suite

A Dash web application for analyzing dielectric permittivity data.

It supports:

✅ **Debye Model Fitting** – Extracts permittivity parameters from experimental data  
✅ **Hybrid Debye-Lorentz Model** – Advanced fitting with multiple relaxation terms  
✅ **Kramers-Kronig Causality Check** – Model-free validation of measured data  

---

## 🚀 Features
- Upload CSV files with columns: **Frequency (GHz), Dk, Df**
- Fit **Debye** and **Hybrid Debye-Lorentz** models to experimental data
- Verify **causality** using **Kramers-Kronig relations**
- Interactive plots and pass/fail summary badges

---

## 📦 Installation
```bash
git clone <repo-url>
cd permittivity_app
pip install -r requirements.txt
```

---

## ▶️ Run the App
```bash
python app.py
```
Open your browser at **http://127.0.0.1:8050**

---

## 📄 CSV Format
| Frequency_GHz | Dk   | Df   |
|---------------|------|------|
| 1.0           | 4.2  | 0.015 |
| 2.0           | 4.1  | 0.017 |

---

## 🛠 Tech Stack
- **Dash / Plotly** – Interactive UI & charts
- **Flask** – Backend server
- **SciPy / NumPy / Pandas** – Data processing & optimization

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
