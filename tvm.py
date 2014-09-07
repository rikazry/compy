
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

class Perpetuity(FixIncome):
    """
    eg. British concul bonds, preferred stocks
    pv = pmt / r
    """
    def cpt_pv(self, pmt, r):
        self.pmt = pmt
        self.pv = _sumgs(r = 1/(1+r), a1 = pmt/(1+r), series = True) 
        return self.pv

    def cpt_r(self, pmt, pv):
        self.pmt = pmt
        self.pv = pv
        return pmt/pv

class Annuity(FixIncome):
    def __init__(self, ordinary = True):
        """
        ordinary annuities -> type1, annuity due -> type0
        """
        if ordinary:
            self.type = 1
        else:
            self.type = 0
    
    def cpt_value(self, mat, pmt, r, lag0 = 1, lag = 0):
        self.maturity = mat
        self.pmt = pmt
        self.fv = _sumgs(a1 = pmt, n = mat, r = 1+r)
        # adjust fv by 1 year for annuity due
        if self.type == 0:
            self.fv = _fv(pv = self.fv, _y = r, n = 1)
        # calculate pv according to first payment time
        self.pv = _pv(fv = self.fv, _y = r, n = mat + lag0 - 1)
        # adjust fv by cashing out time
        if lag:
            self.fv = _fv(pv = self.fv, _y = r, n = lag)
        return self.fv, self.pv

class FixIncome(object):
   
    def cpt_ear(self, nom_rate, num_comp = 1, cc = False):
        return _ear(nom_rate, num_comp = 1, cc = False)

    def cpt_fv(self, pv, _y, n):
        return _fv(pv, _y, n)

    def cpt_pv(self, fv, _y, n):
        return _pv(fv, _y, n)

    def cpt_fvf(self, _y, n):
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

def _sumgs(r, n = 0, a1 = 1, series = False):
    """
    sum of geometric sequence
    """
    if series:
        return a1/(1-r)
    else:
        return a1*(1-np.power(r,n))/(1-r)

