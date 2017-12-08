__author__ = 'Derek Qi'
'''
Very simple Black scholes option pricer and implied vol calculator
'''

import numpy as np
from scipy.stats import norm
from scipy.optimize import newton

def BSPrice(S, K, r, q, T, sigma, type='C'):
    d1 = ( np.log(S/K) + (r - q + sigma**2 / 2) * T ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == 'C':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return - S * np.exp(-q * T) * norm.cdf(-d1) + K * np.exp(-r * T) * norm.cdf(-d2)


def BSDelta(S, K, r, q, T, sigma, type='C'):
    dS = S * 1e-4
    return (BSPrice(S+dS, K, r, q, T, sigma, type) - BSPrice(S-dS, K, r, q, T, sigma, type)) / (2*dS)


def BSImpVol(price, S, K, r, q, T, type='C'):
    def pricer(sigma):
        return BSPrice(S, K, r, q, T, sigma, type) - price
    impvol = newton(pricer, 0.3, tol=1e-6)
    return impvol
    
