# models/kk_check.py
import numpy as np
from utils.helpers import get_numeric_data

def kk_real_from_imag(freq, eps_imag, eps_inf):
    N = len(freq)
    eps_real_kk = np.zeros(N)
    omega_all = 2*np.pi*freq
    d_omega = np.gradient(omega_all)
    for i in range(N):
        omega = omega_all[i]
        integrand = (omega_all*eps_imag)/(omega_all**2 - omega**2 + 1e-30)
        integrand[i] = 0
        principal_value = (2/np.pi)*np.sum(integrand*d_omega)
        eps_real_kk[i] = eps_inf + principal_value
    return eps_real_kk

def kk_causality_check(df):
    freq_ghz, dk, df_loss = get_numeric_data(df)
    freq = freq_ghz*1e9
    eps_real_meas = dk
    eps_imag_meas = dk*df_loss

    num_tail = max(3, round(0.1*len(freq)))
    eps_inf = np.mean(eps_real_meas[-num_tail:])

    eps_real_kk_full = kk_real_from_imag(freq, eps_imag_meas, eps_inf)
    rel_error_full = np.abs(eps_real_meas-eps_real_kk_full)/eps_real_meas
    mean_err_full = np.mean(rel_error_full)

    f_cut = 40e9
    idx_partial = freq <= f_cut
    freq_partial = freq[idx_partial]
    eps_real_partial = eps_real_meas[idx_partial]
    eps_imag_partial = eps_imag_meas[idx_partial]

    eps_real_kk_partial = kk_real_from_imag(freq_partial, eps_imag_partial, eps_inf)
    rel_error_partial = np.abs(eps_real_partial-eps_real_kk_partial)/eps_real_partial
    mean_err_partial = np.mean(rel_error_partial)

    return {
        "freq_ghz": freq_ghz,
        "eps_real_meas": eps_real_meas,
        "eps_real_kk_full": eps_real_kk_full,
        "mean_err_full": mean_err_full,
        "freq_partial": freq_partial/1e9,
        "eps_real_partial": eps_real_partial,
        "eps_real_kk_partial": eps_real_kk_partial,
        "mean_err_partial": mean_err_partial,
        "eps_inf": eps_inf
    }
