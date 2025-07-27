from models.debye_model import DebyeModel
from models.hybrid_model import HybridModel
from models.multipole_debye_model import MultiPoleDebyeModel
from models.cole_cole_model import ColeColeModel
from models.cole_davidson_model import ColeDavidsonModel
from models.havriliak_negami_model import HavriliakNegamiModel
from models.lorentz_model import LorentzModel
from models.sarkar_model import SarkarModel
from models.kk_model import KKModel
import numpy as np


def compute_rmse(result, df):
    freq_exp = df.iloc[:, 0].values
    dk_exp = df.iloc[:, 1].values
    df_exp = df.iloc[:, 2].values  # Also include Df in RMSE calculation

    if "eps_fit" not in result:
        return None

    if "freq" in result:
        freq_fit = result["freq"]
    elif "freq_ghz" in result:
        freq_fit = result["freq_ghz"]
    else:
        return None

    freq_fit = np.array(freq_fit)
    eps_fit = np.array(result["eps_fit"])
    eps_fit_real = eps_fit.real
    eps_fit_imag = eps_fit.imag

    # Remove NaNs
    mask = (~np.isnan(freq_fit) & ~np.isnan(eps_fit_real) &
            ~np.isnan(eps_fit_imag))
    freq_fit = freq_fit[mask]
    eps_fit_real = eps_fit_real[mask]
    eps_fit_imag = eps_fit_imag[mask]

    if len(freq_fit) == 0:
        return None

    # Sort by frequency
    sort_idx = np.argsort(freq_fit)
    freq_fit = freq_fit[sort_idx]
    eps_fit_real = eps_fit_real[sort_idx]
    eps_fit_imag = eps_fit_imag[sort_idx]

    # Interpolate both real and imaginary parts
    dk_interp = np.interp(freq_exp, freq_fit, eps_fit_real)
    df_interp = np.interp(freq_exp, freq_fit, eps_fit_imag)

    # Calculate RMSE for both components
    rmse_dk = np.sqrt(np.mean((dk_interp - dk_exp) ** 2))
    rmse_df = np.sqrt(np.mean((df_interp - df_exp) ** 2))

    # Combined RMSE (you can adjust weighting if needed)
    rmse_total = np.sqrt(rmse_dk ** 2 + rmse_df ** 2)

    return {
        'rmse': rmse_total,
        'rmse_dk': rmse_dk,
        'rmse_df': rmse_df
    }


def run_analysis(df, selected_models, model_params):
    """
    Updated to handle model_params dictionary instead of single n_terms

    Args:
        df: DataFrame with experimental data
        selected_models: List of model names
        model_params: Dictionary with keys like:
            - 'hybrid_terms': int
            - 'multipole_terms': int
            - 'lorentz_terms': int
    """
    results = {}

    # Extract parameters with defaults
    hybrid_terms = model_params.get('hybrid_terms', 2)
    multipole_terms = model_params.get('multipole_terms', 3)
    lorentz_terms = model_params.get('lorentz_terms', 2)

    print(f"Using parameters: Hybrid={hybrid_terms}, Multipole={multipole_terms}, Lorentz={lorentz_terms}")

    if "debye" in selected_models:
        print("Running Debye model...")
        results["debye"] = DebyeModel().analyze(df)

    if "multipole_debye" in selected_models:
        print(f"Running Multipole Debye model with N={multipole_terms}...")
        results["multipole_debye"] = MultiPoleDebyeModel().analyze(df, N=multipole_terms)

    if "cole_cole" in selected_models:
        print("Running Cole-Cole model...")
        results["cole_cole"] = ColeColeModel().analyze(df)

    if "cole_davidson" in selected_models:
        print("Running Cole-Davidson model...")
        results["cole_davidson"] = ColeDavidsonModel().analyze(df)

    if "havriliak_negami" in selected_models:
        print("Running Havriliak-Negami model...")
        results["havriliak_negami"] = HavriliakNegamiModel().analyze(df)

    if "lorentz" in selected_models:
        print(f"Running Lorentz model with N={lorentz_terms}...")
        results["lorentz"] = LorentzModel().analyze(df, N=lorentz_terms)

    if "sarkar" in selected_models:
        print("Running Sarkar model...")
        results["sarkar"] = SarkarModel().analyze(df)

    if "hybrid" in selected_models:
        print(f"Running Hybrid model with N={hybrid_terms}...")
        results["hybrid"] = HybridModel().analyze(df, N=hybrid_terms)

    if "kk" in selected_models:
        print("Running KK model...")
        results["kk"] = KKModel().analyze(df)

    # Compute RMSE for each model that has eps_fit
    for key, res in results.items():
        print(f"Computing RMSE for {key}...")
        rmse_result = compute_rmse(res, df)
        if rmse_result is not None:
            if isinstance(rmse_result, dict):
                # New format with separate RMSE components
                res.update(rmse_result)
            else:
                # Legacy format (single RMSE value)
                res["rmse"] = rmse_result

        print(f"{key} RMSE: {res.get('rmse', 'N/A')}")

    return results