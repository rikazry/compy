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

def hpr(begin_value, end_value, *args):
    """holding period return
    """
    cf = 0
    for arg in args:
        cf += arg
    return (end_value - begin_value + cf)/begin_value
