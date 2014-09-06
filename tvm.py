
"""
PV and FV of single cash flow, annuity ,a series of uneven cash flows

N = number of compounding periods
1_Y = interest rate per compounding period
PV = present value
FV = future value
PMT = annuity payments or constant perodic cash flow

interest rate:
    required rate of return
    discount rate
    opportunity cost

required interest rate:
    nominal risk-free rate (e.g. T-bill):
      real risk-free rate
      + expected inflation rate
    + default risk premium
    + liquidity premium
    + maturity risk premium
"""
import numpy as np

class FI(object):
   
    def ear(self, nom_rate, num_comp = 1, cc = False):
        return _ear(nom_rate, num_comp = 1, cc = False)

    def fv(self, pv, _y, n):
        return _fv(pv, _y, n)

    def pv(self, fv, _y, n):
        return _pv(fv, _y, n)

    def fvf(self, _y, n):
        return _fvf(_y, n)

def _ear(nom_rate, num_comp = 1, cc = False):
    """
    effective annual rate

    Parameters
    ----------
    nom_rate: norminal rate
    num_comp: number of compounding periods per year
    cc: continuous compounding
    """
    if cc:
        ear = np.exp(nom_rate) - 1
    elif num_comp == 1:
        ear = nom_rate
    else:
        prd_rate = nom_rate / num_comp
        ear = _fvf(prd_rate, num_comp) - 1
    return ear

def _fv(pv, _y, n):
    """
    future value of a single sum

    Parameters
    ----------
    pv: present value
    _y: rate of return per compounding period
    n: total number of compounding periods
    """
    return pv*_fvf(_y, n)

def _pv(fv, _y, n):
    return fv/fvf(_y, n)

def _fvf(_y, n):
    """
    future value factor
    """
    return np.power(1 + _y, n)
