import numpy as np
from scipy.optimize import least_squares
from utils import get_numeric_data
from .base_model import BaseModel

class MultiPoleDebyeModel(BaseModel):
    def model(self, params, freq, N):
        delta_eps = params[:N]
        tau = params[N:2*N]
        eps_inf = params[-1]
        F = freq.reshape(-1, 1)
        terms = delta_eps / (1 + 1j * 2 * np.pi * F * tau)
        return eps_inf + np.sum(terms, axis=1)

    def objective(self, params, freq, eps_exp, N):
        eps_fit = self.model(params, freq, N)
        return np.concatenate([np.real(eps_fit)-np.real(eps_exp),
                               np.imag(eps_fit)-np.imag(eps_exp)])

    def analyze(self, df, N=3):
        freq_ghz, dk, df_loss = get_numeric_data(df)
        freq = freq_ghz * 1e9
        eps_exp = dk - 1j*dk*df_loss
        delta_eps0 = np.ones(N) * (np.max(dk)-np.min(dk))/N
        tau0 = np.ones(N) / (2*np.pi*np.mean(freq))
        p0 = np.concatenate([delta_eps0, tau0, [np.min(dk)]])
        res = least_squares(self.objective, p0, args=(freq, eps_exp, N))
        eps_fit = self.model(res.x, freq, N)
        return {"freq": freq_ghz, "eps_fit": eps_fit, "params_fit": res.x, "dk_exp": dk}
