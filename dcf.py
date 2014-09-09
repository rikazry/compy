"""
NPV: net present value
IRR: internal rate of return

NPV decision rule vs. IRR decision rule
NPV method assumes the reinvestment of a project's cash flows at the opportunity cost of capital
IRR method assumes that the reinvestment rate is the IRR
select the project with the greatest NPV when the IRR and NPV rules provide conflicting decisions
"""

import numpy as np
from scipy.optimize import newton

def npv(r, cf0, *args):
    from tvm import _sumgs
    npv = cf0
    i = 1
    for arg in args:
        if isinstance(arg, tuple):
            lag, _cf, frq = arg
            cf = _sumgs(r = r, n = frq, a1 = _cf)
            i += (lag + frq - 1)
        else:
            cf = arg
        npv += cf/np.power(1+r, i)
        i += 1
    return npv

def irr(cf0, *args):
    def func(r):
        return npv(r, cf0, *args)
    return newton(func, 0.05)

def hpy2bey(mat, hpy):
    """bond equivalent yield: semiannual discount rate
    """
    return 2*(np.power(1+hpy, 0.5/mat)-1)

def bdy2mmy(t, bdy):
    return hpy2mmy(t, bdy2hpy(t, bdy))

def mmy2eay(t, mmy):
    return hpy2eay(t, mmy2hpy(t, mmy))

def bdy2hpy(t, bdy):
    """
    HPY = D/(F-D) = 1/(F/D -1), BDY = D/F * 360/t
    """
    return 1/(360/(t*bdy) - 1)

def hpy2mmy(t, hpy):
    return hpy*360/t

def mmy2hpy(t, mmy):
    return mmy*t/360.

def hpy2eay(t, hpy):
    return np.power(1+hpy, 365./t) - 1

def hpr2ccr(hpr):
    return np.log(1+hpr)

def ccr2hpr(ccr):
    return np.exp(ccr) - 1

def _bdy(t,D,F,*args):
    """bank discount yield
    
    parameters
    ----------
    D: dollar discount, which is equal to the difference between the face value and purchase price
    F: face value
    t: number of days remainning until maturity
    360: bank convention of number of days in a year

    returns
    -------
    rbd: annualized yield on a bank discont basis
    """
    cf = 0
    for arg in args:
        cf += arg
    return ((D+cf)/F)*(360/t)

def _mmy(t, P0, F, *args):
    """money market yield
    """
    return hpy2mmy(t, _hpr(P0, F, *args))

def _eay(t, P0, F, *args):
    """effective annual yield
    """
    return hpy2eay(t, _hpr(P0,F, *args))

def _hpr(begin_value, end_value, *args):
    """holding period return
    """
    cf = 0
    for arg in args:
        cf += arg
    return (end_value - begin_value + cf)/begin_value
