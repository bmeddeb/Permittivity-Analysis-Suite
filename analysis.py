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

    if "eps_fit" not in result:
        return None

    if "freq" in result:
        freq_fit = result["freq"]
    elif "freq_ghz" in result:
        freq_fit = result["freq_ghz"]
    else:
        return None

    freq_fit = np.array(freq_fit)
    eps_fit_real = np.array(result["eps_fit"].real)

    # Remove NaNs
    mask = ~np.isnan(freq_fit) & ~np.isnan(eps_fit_real)
    freq_fit = freq_fit[mask]
    eps_fit_real = eps_fit_real[mask]

    if len(freq_fit) == 0:
        return None

    # Sort by frequency
    sort_idx = np.argsort(freq_fit)
    freq_fit = freq_fit[sort_idx]
    eps_fit_real = eps_fit_real[sort_idx]

    eps_interp = np.interp(freq_exp, freq_fit, eps_fit_real)
    return np.sqrt(np.mean((eps_interp - dk_exp) ** 2))






def run_analysis(df, selected_models, n_terms):
    results = {}
    dk_exp = df.iloc[:, 1].values

    if "debye" in selected_models:
        results["debye"] = DebyeModel().analyze(df)

    if "multipole_debye" in selected_models:
        results["multipole_debye"] = MultiPoleDebyeModel().analyze(df, N=n_terms)

    if "cole_cole" in selected_models:
        results["cole_cole"] = ColeColeModel().analyze(df)

    if "cole_davidson" in selected_models:
        results["cole_davidson"] = ColeDavidsonModel().analyze(df)

    if "havriliak_negami" in selected_models:
        results["havriliak_negami"] = HavriliakNegamiModel().analyze(df)

    if "lorentz" in selected_models:
        results["lorentz"] = LorentzModel().analyze(df, N=2)

    if "sarkar" in selected_models:
        results["sarkar"] = SarkarModel().analyze(df)

    if "hybrid" in selected_models:
        results["hybrid"] = HybridModel().analyze(df, N=n_terms)

    if "kk" in selected_models:
        results["kk"] = KKModel().analyze(df)

    # Compute RMSE for each model that has eps_fit
    for key, res in results.items():
        rmse = compute_rmse(res, df)
        if rmse is not None:
            res["rmse"] = rmse

    return results
