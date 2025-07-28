import numpy as np
import lmfit
from scipy.optimize import differential_evolution, basinhopping
import matplotlib.pyplot as plt

from models.sarkar_model import SarkarModel
from models.base_model import BaseModel


def sensitivity_analysis(model, df, param_name, span=0.2, steps=50):
    """
    Vary one parameter around its fitted value and compute RMSE
    """
    # Extract fitted params
    result = model.analyze(df)
    p_opt = result['params_fit']  # [eps_s, eps_inf, f_p]
    names = ['eps_s', 'eps_inf', 'f_p']
    i = names.index(param_name)
    p0 = p_opt[i]

    grid = np.linspace(p0 * (1-span), p0 * (1+span), steps)
    rmse_vals = []
    for val in grid:
        params = result['lmfit_result'].params.copy()
        params[name].value = val
        eps_fit = model.model_function(params, result['freq_ghz'])
        comps = model.calculate_rmse_components(eps_fit, result['dk_exp'], result['df_exp'])
        rmse_vals.append(comps['rmse'])

    plt.figure()
    plt.plot(grid, rmse_vals)
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.title(f'Sensitivity: {param_name}')
    plt.show()
    return grid, np.array(rmse_vals)


def grid_search_fp(model, df, f_range, n_points=30):
    """
    Coarse grid search over f_p in log-space to seed local fits
    """
    freq = df.iloc[:,0].values
    dk = df.iloc[:,1].values
    df_exp = df.iloc[:,2].values

    logf = np.linspace(np.log10(f_range[0]), np.log10(f_range[1]), n_points)
    scores = []
    for lf in logf:
        p0 = model.create_parameters(freq, dk, df_exp)
        p0['f_p'].value = 10**lf
        res = lmfit.minimize(model.residual_function, p0, args=(freq, dk, df_exp))
        comps = model.calculate_rmse_components(res.eval(params=res.params, freq=freq), dk, df_exp)
        scores.append(comps['rmse'])

    plt.figure()
    plt.plot(10**logf, scores)
    plt.xscale('log')
    plt.xlabel('f_p (GHz)')
    plt.ylabel('RMSE')
    plt.title('Grid search over f_p')
    plt.show()
    best_idx = np.argmin(scores)
    return 10**logf[best_idx], scores[best_idx]


def multi_start_fit(model, df, n_starts=20, method='leastsq'):
    """
    Run multiple local fits from random starting points within bounds
    """
    freq, dk, dfe = df.iloc[:,0].values, df.iloc[:,1].values, df.iloc[:,2].values
    params_template = model.create_parameters(freq, dk, dfe)
    bounds = [(p.min, p.max) for p in params_template.values()]
    best = None
    for _ in range(n_starts):
        p0 = []
        for name, p in params_template.items():
            if not p.expr:
                p0.append(np.random.uniform(p.min, p.max))
        # build new params
        p_rand = params_template.copy()
        for (name, param), val in zip(params_template.items(), p0):
            if not param.expr:
                p_rand[name].value = val
        res = lmfit.minimize(model.residual_function, p_rand, args=(freq, dk, dfe), method=method)
        comps = model.calculate_rmse_components(res.eval(params=res.params, freq=freq), dk, dfe)
        score = comps['rmse']
        if best is None or score < best['rmse']:
            best = {'result': res, 'rmse': score}
    return best


def global_optimize(model, df):
    """
    Use a global optimizer (differential evolution) over parameter bounds
    """
    freq, dk, dfe = df.iloc[:,0].values, df.iloc[:,1].values, df.iloc[:,2].values
    params0 = model.create_parameters(freq, dk, dfe)
    bounds = []
    varnames = []
    for name, par in params0.items():
        if not par.expr:
            bounds.append((par.min, par.max))
            varnames.append(name)

    def obj(x):
        ps = params0.copy()
        for name, val in zip(varnames, x):
            ps[name].value = val
        eps_fit = model.model_function(ps, freq)
        return np.sqrt(np.mean((np.real(eps_fit)-dk)**2 + (np.imag(eps_fit)-dfe)**2))

    result = differential_evolution(obj, bounds)
    # map back optimized params
    for name, val in zip(varnames, result.x):
        params0[name].value = val
    lm_res = lmfit.minimize(model.residual_function, params0, args=(freq, dk, dfe))
    comps = model.calculate_rmse_components(lm_res.eval(params=lm_res.params, freq=freq), dk, dfe)
    return lm_res, comps


def explore_sarkar(df):
    """
    Orchestrate an automated exploration of Sarkar fits
    Returns the best fit result and diagnostic data
    """
    model = SarkarModel()

    # 1. Initial fit
    initial = model.analyze(df)

    # 2. Sensitivity (optional plots)
    # grid_eps_s, rmse_eps_s = sensitivity_analysis(model, df, 'eps_s')
    # grid_fp, rmse_fp = sensitivity_analysis(model, df, 'f_p')

    # 3. Grid search over f_p
    best_fp, fp_score = grid_search_fp(model, df, (np.min(df.iloc[:,0]), np.max(df.iloc[:,0])))
    print(f"Best fp from grid: {best_fp:.3f} GHz (RMSE={fp_score:.4f})")

    # 4. Multi-start local fits
    best_local = multi_start_fit(model, df)
    print(f"Best local rmse: {best_local['rmse']:.4f}")

    # 5. Global optimization
    global_res, global_comps = global_optimize(model, df)
    print(f"Global rmse: {global_comps['rmse']:.4f}")

    # 6. Compare and pick best
    candidates = [
        ('initial', initial),
        ('local', best_local['result']),
        ('global', global_res)
    ]
    # Compute RMSE for each
    scores = {}
    for name, r in candidates:
        if hasattr(r, 'params'):
            eps = model.model_function(r.params, df.iloc[:,0].values)
            comps = model.calculate_rmse_components(eps, df.iloc[:,1].values, df.iloc[:,2].values)
            scores[name] = comps['rmse']

    best_method = min(scores, key=scores.get)
    print(f"🏆 Best method: {best_method} (RMSE={scores[best_method]:.4f})")

    return {
        'initial': initial,
        'best_fp': best_fp,
        'best_local': best_local,
        'global': {'result': global_res, **global_comps},
        'scores': scores,
        'best_method': best_method
    }
