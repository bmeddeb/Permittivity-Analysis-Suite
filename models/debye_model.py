import numpy as np
from scipy.optimize import least_squares
from .base_model import BaseModel
from utils import get_numeric_data

class DebyeModel(BaseModel):
    def model(self, params, freq):
        delta_eps, tau, eps_inf = params
        return eps_inf + delta_eps/(1+1j*2*np.pi*freq*tau)

    def objective(self, params, freq, eps_exp):
        eps_fit = self.model(params, freq)
        return np.concatenate([
            np.real(eps_fit)-np.real(eps_exp),
            np.imag(eps_fit)-np.imag(eps_exp)
        ])

    def analyze(self, df):
        freq_ghz, dk, df_loss = get_numeric_data(df)
        freq = freq_ghz * 1e9
        eps_exp = dk - 1j*dk*df_loss

        eps_inf = np.min(np.real(eps_exp))
        delta_eps = np.max(np.real(eps_exp))-eps_inf
        tau = 1/(2*np.pi*np.mean(freq))

        res = least_squares(self.objective, [delta_eps,tau,eps_inf], args=(freq,eps_exp))
        eps_fit = self.model(res.x,freq)

        return {"freq_ghz": freq_ghz, "eps_fit": eps_fit, "params_fit": res.x}
