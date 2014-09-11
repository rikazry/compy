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

def cml(rm_mean, rm_std, rp_std, rf):
    """
    Capital Market Line
    """
    rp_mean = rf + sharpe(rm_mean, rm_std, rf) * rp_std
    return rp_mean

def sml(rm_mean, rf, beta):
    """
    Security Market Line
    Capital Asset Pricing Model
    """
    rp_mean = rf + beta * (rm_mean - rf)
    return rp_mean

def ja(rm_mean, rp_mean, rp_beta, rf):
    """
    Jensen's alpha
    percentage portfolio return above SML with same beta
    """
    return rp_mean - sml(rm_mean, rf, rp_beta)

def m2(rm_mean, rm_std, rp_mean, rp_std, rf):
    """
    excess return on a portfolio, constructed by
    taking a leveraged position s.t. has the same total risk as market index
    """
    return (rp_mean - rf) * np.divide(rm_std, rp_std) - (rm_mean - rf)

def treynor(rp_mean, rp_beta, rf):
    """
    excess returns per unit of systematic risk
    """
    return np.divide(rp_mean - rf, rp_beta)

def beta(rel, rm_std, rp_std = None, corr = True):
    """
    sensitivity of an asset's return to the return on the market index
    systematic risk
    """
    if corr:
        return rel*np.divide(rp_std, rm_std)
    else:
        return np.divide(rel, np.power(rm_std, 2))
