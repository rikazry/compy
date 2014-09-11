import numpy as np
import scipy.stats as st
"""
shortfall risk: the probability that a portfolio value or return will fall below a threshold over a given time period

Roy's safety-first criterion: optimal portfolio minimizes shortfall risk

maximizes SFRatio
"""

def sfratio(rp_mean, rp_std, rth):
    return np.divide(rp_mean - rth,rp_std)

def sharpe(rp_mean, rp_std, rf):
    return np.divide(rp_mean - rf,rp_std)


