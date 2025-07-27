import numpy as np
from scipy.optimize import least_squares
from utils import get_numeric_data
from .base_model import BaseModel

class ColeDavidsonModel(BaseModel):
    def model(self, params, freq):
        delta_eps, tau, beta, eps_inf = params
        return eps_inf + delta_eps / (1 + 1j*2*np.pi*freq*tau)**beta

    def objective(self, params, freq, eps_exp):
        eps_fit = self.model(params, freq)
        return np.concatenate([np.real(eps_fit)-np.real(eps_exp),
                               np.imag(eps_fit)-np.imag(eps_exp)])

    def analyze(self, df):
        freq_ghz, dk, df_loss = get_numeric_data(df)
        freq = freq_ghz*1e9
        eps_exp = dk - 1j*dk*df_loss
        delta_eps0 = np.max(dk)-np.min(dk)
        tau0 = 1/(2*np.pi*np.mean(freq))
        beta0 = 0.8
        p0 = [delta_eps0, tau0, beta0, np.min(dk)]
        res = least_squares(self.objective, p0, args=(freq, eps_exp))
        eps_fit = self.model(res.x, freq)
        return {"freq": freq_ghz, "eps_fit": eps_fit, "params_fit": res.x, "dk_exp": dk}
