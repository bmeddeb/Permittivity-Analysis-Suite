import numpy as np
from scipy.optimize import least_squares
from utils.helpers import get_numeric_data
from .base_model import BaseModel

class SarkarModel(BaseModel):
    def model(self, params, freq):
        eps_s, eps_inf, f_p = params
        return eps_inf + (eps_s - eps_inf) / (1 + 1j * (freq / f_p))

    def objective(self, params, freq, dk_exp, df_exp):
        eps_fit = self.model(params, freq)
        return np.concatenate([
            np.real(eps_fit) - dk_exp,
            np.imag(eps_fit) - df_exp
        ])

    def analyze(self, df):
        freq_ghz, dk_exp, df_exp = get_numeric_data(df)
        # Initial guess
        p0 = [5.0, 3.0, 1.0]  # [eps_s, eps_inf, f_p] in GHz

        res = least_squares(
            self.objective,
            p0,
            args=(freq_ghz, dk_exp, df_exp),
            bounds=([1.0, 1.0, 1e-3], [20.0, 20.0, 1e3]),
            method="trf"
        )

        eps_fit = self.model(res.x, freq_ghz)
        dk_fit = np.real(eps_fit)
        df_fit = np.imag(eps_fit)

        return {
            "freq": freq_ghz,
            "eps_fit": eps_fit,
            "dk_fit": dk_fit,
            "df_fit": df_fit,
            "params_fit": res.x,
            "success": res.success,
            "cost": res.cost
        }
