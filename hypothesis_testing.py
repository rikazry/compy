import numpy as np
import scipy.stats as st
"""
Monte Carlo is often applied to:
    * value complex securities
    * simulate the pnl from a trading strategy
    * calculate estimates of var
    * simulate pension fund assets and liabilities over time to examine the variablity of the difference between the 2
    * value portfolios of assets that have non-normal returns distrubution

pro:
    what if

con:
    cannot provide better conclusion than the assumption
    statistical rather than analytical
"""
"""
CLT:
    sample mean ~ normal(porpulation mean, population variance / n)
    (n >= 30)

standard error of sample mean:
    standard deviation of the distribution of sample means

desirable statistical properties of an estimator:
    unbiasedness
    efficiency ( smaller variance than other unbiased estimators)
    cosistency ( accuracy increases as as sample size increases)

t-dist has fatter tail when degree of freedom is smaller

criteria for selecting the appropriate test statistics
population                      smallsample     bigsample
normal + known variance         z-statistic     z-statistic
normal + unknown variance       t-statistic     t-statistic more conservative
nonnormal + known variance      N/A             z-statistic
nonnormal + unknown variance    N/A             t-statistic more conservative
"""
def test_1samp_raw(a, popmean, sig = None, popstd = False, normal = True, tail = 2, thresz = 30):
    n = len(a)
    avg = np.mean(a)
    if popstd:
        std = popstd
    else:
        std = np.std(a, ddof = 1)
    return test_1samp(a = avg, popmean = popmean, std = std, n = n, sig = sig, popstd = popstd, normal = normal, tail = tail, thresz = thresz)

def test_1samp(a, popmean, std, n, sig = None, popstd = False, normal = True, tail = 2, thresz = 30):
    if n < thresz and not normal:
        raise Exception("Test for non-normal small sample not applicable")
    d = a - popmean
    denom = np.divide(std, np.sqrt(float(n)))
    x = np.divide(d, denom)
    if popstd:
        p = pscore(x = x, tail = tail, test = 'z')
    else:
        df = n - 1
        p = pscore(x = x, tail = tail, df = df)
    return decirule(x, p, sig)

def decirule(x, p, sig = None):
    if sig:
        if p < sig:
            return x, p, 'reject'
        else:
            return x, p, 'cannot reject'
    else:
        return x, p

def pscore(x, tail = 2, df = None, df2 = None, test = 't'):
    if tail == 2:
        x = np.abs(x)
    if not df:
        print "z test"
        dist = st.norm
        args = x,
    elif test == 't':
        print "t test"
        dist = st.t
        args = x, df
    elif test == 'chi2':
        print "chi square test"
        dist = st.chi2
        args = x, df
    elif test == 'f':
        print "f test"
        dist = st.f
        args = x, df, df2
    if tail == 2:
        print "2 sided"
        p = min(dist.sf(*args)*2, 1)
    if tail == -1:
        print "left tail"
        p = dist.cdf(*args)
    if tail == 1:
        print "right tail"
        p = dist.sf(*args)
    return p

def test_corr(corr, n, sig = None, tail = 2):
    df = n - 2
    x = np.divide(corr * np.sqrt(df), np.sqrt(1 - corr**2))
    p = pscore(x = x, tail = tail, df = df)
    return decirule(x, p, sig)

def test_ind_raw(a1, a2, dmean = 0, sig = None, eqvar = True, tail = 2):
    n1 = len(a1)
    n2 = len(a2)
    avg1 = np.mean(a1)
    avg2 = np.mean(a2)
    var1 = np.var(a1, ddof = 1)
    var2 = np.var(a2, ddof = 2)
    return test_ind(a1 = avg1, a2 = avg2, var1 = var1, var2 = var2, n1 = n1, n2 = n2, dmean = dmean, sig = sig, eqvar = eqvar, tail = tail)

def test_ind(a1, a2, var1, var2, n1, n2, dmean = 0, sig = None, eqvar = True, tail = 2):
    d = a1 - a2 - dmean
    if eqvar:
        # pooled variance
        df = n1 + n2 - 2
        varp = np.divide(((n1 - 1) * var1 + (n2 - 1) * var2), float(df))
        denom = np.sqrt(np.divide(varp, float(n1)) + np.divide(varp, float(n2)))
    else:
        # Welch's test
        vn1 = np.divide(var1, float(n1))
        vn2 = np.divide(var2, float(n2))
        varp = vn1 + vn2
        df = np.divide(np.power(varp, 2), np.divide(np.power(vn1, 2), n1-1) + np.divide(np.power(vn2, 2), n2-2))
        denom = np.sqrt(varp)
    x = np.divide(d, denom)
    p = pscore(x = x, tail = tail, df = df)
    return decirule(x, p, sig)

def test_rel_raw(a1, a2, dmean = 0, sig = None, tail = 2, thresz = 30):
    n = len(a1)
    if n != len(b2):
        raise ValueError('input arrays with different lengths')
    d = (a1 - a2).astype(np.float64)
    avg = np.mean(d)
    std = np.std(d, ddof = 1)
    return test_1samp(a = avg, popmean = dmean, std = std, n = n, sig = sig, popstd = False, normal = True, tail = tail, thresz = thresz)

def test_chi2_raw(a, popvar, sig = None, tail = 2):
    n = len(a)
    var = np.var(a, ddof = 1)
    return test_chi2(var = var, popvar = popvar, n = n, sig = sig, tail = tail)

def test_chi2(var, popvar, n, sig = None, tail = 2):
    df = n - 1
    x = np.divide(float(df)*var, popvar)
    p = pscore(x = x, tail = tail, df = df, test = 'chi2')
    return decirule(x, p, sig)

def test_f_raw(a1, a2, sig = None, tail = 1):
    n1, n2 = len(a1), len(a2)
    var1, var2 = np.var(a1, ddof = 1), np.var(a2, ddof = 1)
    return test_f(var1 = var1, var2 = var2, n1 = n1, n2 = n2, sig = sig, tail = tail)

def test_f(var1, var2, n1, n2, sig = None, tail = 1):
    df1, df2 = n1 - 1, n2 - 1
    if var1 > var2:
        vn, vd, dfn, dfd = var1, var2, df1, df2
    else:
        vn, vd, dfn, dfd = var2, var1, df2, df1
    x = np.divide(vn, vd)
    p = pscore(x = x, tail = tail, df = dfn, df2 = dfd, test = 'f')
    return decirule(x, p, sig)
"""
data-mining bias:
    out-of-sample test to avoid

sample selection bias:(nonrandomness due to lack of availability)
    survivorship bias: most common form of sample selection bias

look-ahead bias: (eg.end-of-year book values available 60days after fiscal year)

time-period bias:
    too short: reflect phenomena specific to that time period
    too long: covers a fundamental change in relation of inflation/unemployment
"""
