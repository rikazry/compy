
"""
PV and FV of single cash flow, annuity ,a series of uneven cash flows

N = number of compounding periods
1_Y = interest rate per compounding period
PV = present value
FV = future value
PMT = annuity payments or constant perodic cash flow

interest rate:
    *rate of return* required in equilibrium for a particular investment 
    *discount rate* of future cash flows
    *opportunity cost* of consuming now rather than saving and investing

required interest rate:
    nominal risk-free rate (e.g. T-bill):
      real risk-free rate
      + expected inflation rate
    + default risk premium
    + liquidity premium
    + maturity risk premium
"""
import numpy as np
import pandas as pd
from scipy.optimize import newton

class FixIncome(object):
   
    def cpt_ear(self, nom_rate, num_comp = 1, cc = False):
        return _ear(nom_rate, num_comp = 1, cc = False)

    def cpt_fv(self, pv, _y, n):
        return _fv(pv, _y, n)

    def cpt_pv(self, fv, _y, n):
        return _pv(fv, _y, n)

    def cpt_fvf(self, _y, n):
        return _fvf(_y, n)

class Perpetuity(FixIncome):
    """
    annuity with infinite lives
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
    """
    ordinary annuity cash flows occur at the end of each time period
    annuity due cash flows occur at the beginning
    """
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

    def cpt_r(self, mat, pmt, value, fv = True):
        self.maturity = mat
        self.pmt = pmt
        if fv:
            self.fv = value
            def func(r):
                return _sumgs(a1 = pmt, n = mat, r = 1+r) - value
        else:
            self.pv = value
            def func(r):
                return _sumgs(a1 = pmt, n = mat, r = 1/(1+r)) - value*(1+r)
        r = newton(func, 0.05)
        return r

    def cpt_mat_fv(self, pmt, fv, r):
        self.pmt = pmt
        self.fv = fv
        self.maturity = _ngs(r = 1 + r, s = fv, a1 = pmt)
        return self.maturity

    def cpt_mat(self, pmt, pv, r):
        self.pmt = pmt
        self.pv = pv
        self.maturity = _ngs(r = 1/(1+r), s = pv*(1+r), a1 = pmt)
        return self.maturity

    def cpt_pmt_fv(self, mat, fv, r):
        self.pmt = self.cpt_pmt(mat, self.cpt_pv(fv = fv, _y = r, n = mat), r)
        return self.pmt

    def cpt_pmt(self, mat, pv, r):
        """
        loan payment and amortization schedule calculation
        """
        self.maturity = mat
        self.pv = pv
        self.pmt = _a1gs(r = 1/(1+r), s = pv*(1+r), n = mat)
        self.r = r
        return self.pmt

    def cpt_amt(self):
        """
        amortization schedule construction
        on top of cpt_pmt
        """
        label = ['begin_balance','payment','interest','principal','end_balance']
        df = pd.DataFrame(np.zeros((self.maturity, len(label))), columns = label)
        balance = self.pv
        n = self.maturity
        for i in range(n):
            df.ix[i,0] = balance
            df.ix[i,1] = self.pmt
            df.ix[i,2] = balance*self.r
            df.ix[i,3] = self.pmt - df.ix[i,2]
            balance -= df.ix[i, 3]
            df.ix[i,4] = balance
        err = df.ix[n-1, 4]
        df.ix[n-1, [1,3]] += (err, err)
        df.ix[n-1, 4] -= err
        self.amt = df
        return self.amt
    
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

def _a1gs(r, s, n = 0, series = False):
    if series:
        return s*(1-r)
    else:
        return s*(1-r)/(1-np.power(r,n))

def _ngs(r, s, a1 = 1):
    return np.log(1 - s*(1-r)/a1) / np.log(r)
