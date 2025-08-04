# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

- **Start development server**: `python run.py` (starts Flask with debug mode)
- **Linting**: `ruff check` (check code quality)
- **Formatting**: `ruff format` (auto-format code)
- **Testing**: `pytest` (run test suite)
- **Database migrations**: `flask db init`, `flask db migrate`, `flask db upgrade`
- **Clean Python cache**: `pyclean .` (removes __pycache__ directories)

## Project Architecture

This is a **Flask web application** with **Dash integration** for scientific data analysis of dielectric spectroscopy measurements.

### Core Components

1. **Flask Application Factory** (`app/__init__.py`)
   - SQLAlchemy ORM with SQLite database
   - Flask-Login authentication system  
   - Flask-Migrate for database versioning
   - Integrated Dash app for interactive dashboards

2. **Database Models** (`app/auth/models.py`)
   - **User**: Authentication with username/email/password
   - **Trial**: Represents uploaded dielectric measurement datasets with data fingerprinting
   - **MaterialData**: Individual frequency/dielectric constant/dissipation factor measurements
   - **FittedModel**: Stores fitting results with parameters and quality metrics

3. **Dielectric Analysis Engine** (`app/models/dielectric_fitter.py`)
   - Core scientific computation class for fitting dielectric spectroscopy data
   - **Supported Models**: Debye (single/multi-pole), Cole-Cole, Havriliak-Negami, Djordjevic-Sarkar
   - Features: parameter estimation, model comparison using AIC/BIC, Kramers-Kronig validation
   - Uses **LMFit** for robust parameter fitting with bounds and constraints

4. **Web Interface Structure**
   - **Authentication**: Login/register system (`app/auth/`)
   - **Main routes**: Data upload, trial management (`app/main/`)
   - **Dashboard**: Interactive Dash/Plotly visualization (`app/dashboard/`)
     - Side-by-side layout with plot and parameter controls for real-time feedback
     - Compact header with trial and model selection
     - Frequency display for omega parameters (f1, f2 in GHz)
   - **Static assets**: Bootstrap 5 CSS, custom JS (`app/static/`)
   - **Templates**: Jinja2 HTML templates (`app/templates/`)

### Key Features

- **Multi-user system** with session-based authentication
- **CSV data upload** with duplicate detection via SHA256 fingerprinting
- **Interactive scientific visualization** using Plotly.js
- **Multiple dielectric fitting models** with parameter estimation and model comparison
- **Real-time parameter adjustment** with live preview of fitted curves
- **Data export capabilities** for fitted models and results

### Dependencies

- **Web Framework**: Flask 3.1.1 with Dash 3.1.1 for interactive components
- **Scientific Stack**: NumPy, SciPy, Pandas for data handling; LMFit for curve fitting
- **Visualization**: Matplotlib, Plotly for charts and graphs
- **Performance**: Numba for numerical acceleration of computations
- **Database**: SQLAlchemy with SQLite (development), Flask-Migrate for schema management
- **Authentication**: Flask-Login, Flask-WTF for secure user management

## Development Notes

- **Package Manager**: Uses UV for dependency management (see `uv.lock`)
- **Database**: SQLite stored in `instance/` directory (gitignored)
- **Documentation**: Internal docs are gitignored (`docs/` in .gitignore)
- **Migration Function**: `run.py` includes `migrate_existing_trials()` for handling existing data when schema changes
- **No existing tests**: `tests/` directory is empty - tests should be added for new features
- **Code Quality**: Uses Ruff for both linting and formatting (configured in pyproject.toml)

## File Structure Context

- `app/auth/`: User authentication and database models
- `app/main/`: Primary application routes and views  
- `app/models/`: Scientific computation engine (DielectricFitter class)
- `app/dashboard/`: Dash integration for interactive data visualization
- `app/static/`: Frontend assets (CSS, JavaScript)
- `app/templates/`: Jinja2 HTML templates
- `instance/`: Flask instance folder (database, config files)
- `run.py`: Application entry point with development server and migration utilities