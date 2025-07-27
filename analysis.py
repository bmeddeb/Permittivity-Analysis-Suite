from models.debye_model import DebyeModel
from models.hybrid_model import HybridModel
from models.multipole_debye_model import MultiPoleDebyeModel
from models.cole_cole_model import ColeColeModel
from models.cole_davidson_model import ColeDavidsonModel
from models.havriliak_negami_model import HavriliakNegamiModel
from models.lorentz_model import LorentzModel
from models.sarkar_model import SarkarModel
from models.kk_model import KKModel

def run_analysis(df, selected_models, n_terms):
    results = {}

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

    return results
