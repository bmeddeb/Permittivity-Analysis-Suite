# sarkar_explorer.py
import numpy as np
import pandas as pd
import lmfit
from scipy.optimize import differential_evolution

from models.sarkar_model import SarkarModel
from models.base_model import BaseModel


def sensitivity_analysis(model, df, param_name, span=0.2, steps=50):
    """
    Vary one parameter around its fitted value and compute RMSE.

    Returns:
        pandas.DataFrame with columns [param_name, 'rmse']
    """
    # Run initial fit
    result = model.analyze(df)
    p_opt = result['params_fit']  # [eps_s, eps_inf, f_p]
    names = ['eps_s', 'eps_inf', 'f_p']
    idx = names.index(param_name)
    center = p_opt[idx]

    # Create sweep grid
    grid = np.linspace(center * (1 - span), center * (1 + span), steps)
    rmse_vals = []
    for val in grid:
        params = result['lmfit_result'].params.copy()
        params[param_name].value = val
        eps_fit = model.model_function(params, result['freq_ghz'])
        comps = model.calculate_rmse_components(eps_fit, result['dk_exp'], result['df_exp'])
        rmse_vals.append(comps['rmse'])

    return pd.DataFrame({param_name: grid, 'rmse': rmse_vals})


def grid_search_fp(model, df, f_range, n_points=30):
    """
    Coarse grid search over f_p in log-space to seed local fits.

    Returns:
        (DataFrame with ['f_p', 'rmse'], best_f_p, best_rmse)
    """
    freq = df.iloc[:, 0].values
    dk = df.iloc[:, 1].values
    dfe = df.iloc[:, 2].values

    logf = np.linspace(np.log10(f_range[0]), np.log10(f_range[1]), n_points)
    f_p_vals = 10 ** logf
    rmse_vals = []

    for f_p in f_p_vals:
        params = model.create_parameters(freq, dk, dfe)
        params['f_p'].value = f_p
        res = lmfit.minimize(model.residual_function, params, args=(freq, dk, dfe))
        eps_fit = model.model_function(res.params, freq)
        comps = model.calculate_rmse_components(eps_fit, dk, dfe)
        rmse_vals.append(comps['rmse'])

    df_grid = pd.DataFrame({'f_p': f_p_vals, 'rmse': rmse_vals})
    best_idx = df_grid['rmse'].idxmin()
    return df_grid, df_grid.at[best_idx, 'f_p'], df_grid.at[best_idx, 'rmse']


def multi_start_fit(model, df, n_starts=20, method='leastsq'):
    """
    Run multiple local fits from random starting points within bounds.

    Returns:
        (best_result_dict, pandas.DataFrame with each start's params + rmse)
    """
    freq = df.iloc[:, 0].values
    dk = df.iloc[:, 1].values
    dfe = df.iloc[:, 2].values
    params_template = model.create_parameters(freq, dk, dfe)

    records = []
    best_rmse = np.inf
    best_result = None

    for _ in range(n_starts):
        # random initial values
        p_rand = params_template.copy()
        for name, par in p_rand.items():
            if not par.expr:
                par.value = np.random.uniform(par.min, par.max)

        res = lmfit.minimize(model.residual_function, p_rand, args=(freq, dk, dfe), method=method)
        eps_fit = model.model_function(res.params, freq)
        comps = model.calculate_rmse_components(eps_fit, dk, dfe)
        rmse = comps['rmse']

        # record
        rec = {**{f: p_rand[f].value for f in p_rand if not p_rand[f].expr}, 'rmse': rmse}
        records.append(rec)

        if rmse < best_rmse:
            best_rmse = rmse
            best_result = {
                'result': res,
                'rmse': rmse,
                'params': {n: p_rand[n].value for n in p_rand}
            }

    df_multi = pd.DataFrame(records)
    return best_result, df_multi


def global_optimize(model, df):
    """
    Use a global optimizer (differential evolution) over parameter bounds.

    Returns:
        (best_result_dict, pandas.DataFrame single-row with best params + rmse)
    """
    freq = df.iloc[:, 0].values
    dk = df.iloc[:, 1].values
    dfe = df.iloc[:, 2].values
    params0 = model.create_parameters(freq, dk, dfe)

    bounds = []
    varnames = []
    for name, par in params0.items():
        if not par.expr:
            bounds.append((par.min, par.max))
            varnames.append(name)

    def obj(x):
        for name, val in zip(varnames, x):
            params0[name].value = val
        eps = model.model_function(params0, freq)
        return np.sqrt(np.mean((np.real(eps) - dk)**2 + (np.imag(eps) - dfe)**2))

    res_de = differential_evolution(obj, bounds)
    for name, val in zip(varnames, res_de.x):
        params0[name].value = val

    lm_res = lmfit.minimize(model.residual_function, params0, args=(freq, dk, dfe))
    eps_fit = model.model_function(lm_res.params, freq)
    comps = model.calculate_rmse_components(eps_fit, dk, dfe)

    rec = {n: params0[n].value for n in params0 if not params0[n].expr}
    rec['rmse'] = comps['rmse']
    df_global = pd.DataFrame([rec])

    best_result = {
        'result': lm_res,
        'rmse': comps['rmse'],
        'params': rec
    }
    return best_result, df_global


def explore_sarkar(df):
    """
    Orchestrate an automated exploration of Sarkar fits.

    Returns:
        dict of DataFrames and best_method info for Dash consumption.
    """
    model = SarkarModel()

    # Initial fit
    initial = model.analyze(df)

    # Sensitivity sweep
    df_sens = sensitivity_analysis(model, df, 'f_p')

    # Grid search over f_p
    df_grid, best_fp, best_fp_rmse = grid_search_fp(model, df, (np.min(df.iloc[:, 0]), np.max(df.iloc[:, 0])))

    # Multi-start local fits
    best_local, df_multi = multi_start_fit(model, df)

    # Global optimization
    best_global, df_global = global_optimize(model, df)

    # Compare methods
    scores = {
        'initial': initial.get('cost', np.nan),
        'grid_search': best_fp_rmse,
        'local': best_local['rmse'],
        'global': best_global['rmse']
    }
    df_scores = pd.DataFrame(list(scores.items()), columns=['method', 'rmse'])
    best_method = df_scores.loc[df_scores['rmse'].idxmin(), 'method']

    return {
        'initial_result': initial,
        'sensitivity': df_sens,
        'gridsearch': df_grid,
        'multistart': df_multi,
        'global': df_global,
        'scores': df_scores,
        'best_method': best_method
    }
