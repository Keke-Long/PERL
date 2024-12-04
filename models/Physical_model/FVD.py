import numpy as np

def FVD(arg, vi, delta_v, delta_d):
    alpha, lamda, v_0, b, beta = arg
    V_star = v_0 * (np.tanh(delta_d / b - beta) - np.tanh(-beta))
    ahat = alpha * (V_star - vi) + lamda * delta_v
    return ahat
