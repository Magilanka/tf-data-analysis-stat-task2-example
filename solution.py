import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 706100023 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    from statsmodels.distributions.empirical_distribution import ECDF
    alpha = 1 - p
    a = 0.056
    ecdf = ECDF(x) # ECDF показывают каждую точку данных, и график можно интерпретировать только одним способом. В статистике эмпирическая функция распределения (обычно также называемая эмпирической кумулятивной функцией распределения, ecdf) представляет собой функцию распределения, связанную с эмпирическим показателем выборки. Эта кумулятивная функция распределения представляет собой ступенчатую функцию, которая увеличивается на 1/n в каждой из n точек данных. https://notebook.community/maxis42/ML-DA-Coursera-Yandex-MIPT/1%20Mathematics%20and%20Python/Lectures%20notebooks/12%20sample%20distribution%20evaluation/sample_distribution_evaluation
    kde = gaussian_kde(x) # Представление оценки плотности ядра с помощью Гауссовых ядер оценка плотности ядра (KDE) - это непараметрический способ оценки
    f_a = kde(a)[0]
    b = a + (1 - ecdf(a)) / f_a
    k = np.sqrt(1 / (alpha * len(x)))
    eps = np.sqrt(np.log(2 / alpha) / (2 * len(x)))
    delta = max(b - a, a - b)
    if delta > k:
        b = a + (1 - kde(a)[0] * k) / ecdf(a)
    if delta > eps:
        b = a + (1 - kde(a)[0] * eps) / ecdf(a)
    return a, b
    # Измените код этой функции sss
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
#     alpha = 1 - p
#     loc = x.mean() - #заменить с 14 строчки https://youtu.be/1UxDv4GcX2k?t=5585
#     scale = np.sqrt(np.var(x)) / np.sqrt(len(x))
#     return loc - scale * norm.ppf(1 - alpha / 2), \
#            loc - scale * norm.ppf(alpha / 2)
