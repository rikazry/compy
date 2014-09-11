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
def test_1sample_raw(a, popmean, sig = None, popstd = False, normal = True, tail = 2, thresz = 30):
    n = len(a)
    avg = np.mean(a)
    if popstd:
        std = popstd
    else:
        std = np.std(a, ddof = 1)
    return test_1samp(a = avg, popmean = popmean, std = std, sampsz = n, sig = sig, popstd = popstd, normal = normal, tail = tail, thresz = thresz)

def test_1samp(a, popmean, std, sampsz, sig = None, popstd = False, normal = True, tail = 2, thresz = 30):
    if sampsz < thresz and not normal:
        raise Exception("Test for non-normal small sample not applicable")
    d = a - popmean
    denom = np.divide(std, np.sqrt(float(sampsz)))
    x = np.divide(d, denom)
    if popstd:
        print "z statistical test"
        if tail == 2:
            print "2 tail"
            p = st.norm.sf(np.abs(x))*2
        if tail == -1:
            print "left tail"
            p = st.norm.cdf(x)
        if tail == 1:
            print "right tail"
            p = st.norm.sf(x)
    else:
        print "t statistical test"
        df = sampsz - 1
        if tail == 2:
            print "2 tail"
            p = st.t.sf(np.abs(x), df)*2
        if tail == -1:
            print "left tail"
            p = st.t.cdf(x, df)
        if tail == 1:
            print "right tail"
            p = st.t.sf(x, df)
    if sig:
        if p < sig:
            return x, p, 'reject'
        else:
            return x, p, 'cannot reject'
    else:
        return x, p
 
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
