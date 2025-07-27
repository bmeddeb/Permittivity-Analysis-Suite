import numpy as np
from scipy.optimize import least_squares
from utils import get_numeric_data
from .base_model import BaseModel

class LorentzModel(BaseModel):
    def model(self, params, freq, N):
        f = params[:N]
        w0 = params[N:2*N]
        gamma = params[2*N:3*N]
        eps_inf = params[-1]
        w = 2*np.pi*freq.reshape(-1, 1)
        terms = f / (w0**2 - w**2 - 1j*gamma*w)
        return eps_inf + np.sum(terms, axis=1)

    def objective(self, params, freq, eps_exp, N):
        eps_fit = self.model(params, freq, N)
        return np.concatenate([np.real(eps_fit)-np.real(eps_exp),
                               np.imag(eps_fit)-np.imag(eps_exp)])

    def analyze(self, df, N=2):
        freq_ghz, dk, df_loss = get_numeric_data(df)
        freq = freq_ghz*1e9
        eps_exp = dk - 1j*dk*df_loss
        f0 = np.ones(N)*0.5
        w0_0 = np.ones(N)*2*np.pi*np.mean(freq)
        gamma0 = np.ones(N)*1e9
        p0 = np.concatenate([f0, w0_0, gamma0, [np.min(dk)]])
        res = least_squares(self.objective, p0, args=(freq, eps_exp, N))
        eps_fit = self.model(res.x, freq, N)
        return {"freq": freq_ghz, "eps_fit": eps_fit, "params_fit": res.x, "dk_exp": dk}
