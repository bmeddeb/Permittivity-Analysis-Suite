from models.debye_model import DebyeModel
from models.kk_check import kk_causality_check

def run_analysis(df, selected_models, n_terms):
    results = {}
    if "debye" in selected_models:
        results["debye"] = DebyeModel().analyze(df)
    if "kk" in selected_models:
        results["kk"] = kk_causality_check(df)
    return results
