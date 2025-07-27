import numpy as np

def init_hybrid_params(freq, dk, N):
    min_freq, max_freq = np.min(freq), np.max(freq)
    delta_eps = (np.max(dk) - np.min(dk)) / N
    f_k = np.logspace(np.log10(min_freq), np.log10(max_freq), N)
    sigma_k = np.ones(N) * 0.01
    eps_inf = np.min(dk)
    return np.concatenate([np.ones(N)*delta_eps, f_k, sigma_k, [eps_inf]])

def get_hybrid_bounds(N):
    lb = np.concatenate([
        np.zeros(N),
        np.ones(N)*1e3,
        np.zeros(N),
        [1]
    ])
    ub = np.concatenate([
        np.ones(N)*10,
        np.ones(N)*1e12,
        np.ones(N)*10,
        [20]
    ])
    return lb, ub
