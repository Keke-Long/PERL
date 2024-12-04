import numpy as np

def IDM(arg, vi, delta_v, delta_d):
    vf, A, b, s0, T = arg
    s_star = s0 + np.max([0, vi*T + (vi * delta_v) / (2 * (A*b) ** 0.5)], axis=0)
    #print('A*b',A*b)
    epsilon = 1e-5
    ahat = A*(1 - (vi/vf)**4 - (s_star/(delta_d+epsilon))**2)
    return ahat