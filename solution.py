import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 706100023 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    from statsmodels.distributions.empirical_distribution import ECDF
    from scipy.stats import gaussian_kde
    alpha = 1 - p
    a = 0.056
    ecdf = ECDF(x) 
    kde = gaussian_kde(x) 
    f_a = kde(a)[0]
    b = a + (1 - ecdf(a)) / f_a
    k = np.sqrt(1 / (alpha * len(x)))
    eps = np.sqrt(np.log(2 / alpha) / (2 * len(x)))
    delta = max(b - a, a - b)
    if delta > k:
        b = a + (1 - kde(a)[0] * k) / ecdf(a)
    if delta > eps:
        b = a + (1 - kde(a)[0] * eps) / ecdf(a)
    rez = [a, b]
    return rez
