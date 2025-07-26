import numpy as np
from scipy.optimize import least_squares
from utils import get_numeric_data
from utils.fitting_utils import init_hybrid_params, get_hybrid_bounds
from .base_model import BaseModel

class HybridModel(BaseModel):
    def hybrid_eval(self, params, freq, N):
        delta_eps = params[:N]
        f_k = params[N:2*N]
        sigma_k = params[2*N:3*N]
        eps_inf = params[-1]
        F = freq.reshape(-1, 1)
        FK = f_k.reshape(1, -1)
        debye_terms = delta_eps/(1+1j*(F/FK))
        lorentz_terms = 1j*F.flatten()*(sigma_k/(F.flatten()**2 + FK.flatten()**2))
        return eps_inf + np.sum(debye_terms, axis=1) + np.sum(lorentz_terms, axis=1)

    def objective(self, params, freq, eps_exp, N):
        eps_fit = self.hybrid_eval(params, freq, N)
        return np.concatenate([
            np.real(eps_fit)-np.real(eps_exp),
            np.imag(eps_fit)-np.imag(eps_exp)
        ])

    def analyze(self, df, N=5):
        freq_ghz, dk, df_loss = get_numeric_data(df)
        freq = freq_ghz*1e9
        eps_exp = dk - 1j*dk*df_loss
        p0 = init_hybrid_params(freq, dk, N)
        lb, ub = get_hybrid_bounds(N)
        res = least_squares(self.objective, p0, bounds=(lb, ub), args=(freq, eps_exp, N), method="trf", max_nfev=5000)
        eps_fit = self.hybrid_eval(res.x, freq, N)
        return {
            "freq": freq_ghz,
            "dk_fit": np.real(eps_fit),
            "df_fit": np.imag(eps_fit),
            "eps_fit": eps_fit,
            "params_fit": res.x,
            "success": res.success,
            "cost": res.cost,
            "dk_exp": dk
        }
