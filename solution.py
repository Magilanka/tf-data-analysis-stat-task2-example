import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 706100023 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    from scipy.stats import uniform
    n = len(x)
    q = uniform.ppf(p, loc=0.056, scale=1-0.056)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    left = mean - q * std / np.sqrt(n)
    right = mean + q * std / np.sqrt(n)
    rez = (left, right)
    return rez
