
"""
shortfall risk: the probability that a portfolio value or return will fall below a threshold over a given time period

Roy's safety-first criterion: optimal portfolio minimizes shortfall risk

maximizes SFRatio
"""

def sfratio(rp_mean, rp_std, rth):
    return (rp_mean - rth)/rp_std

def sharpe(rp_mean, rp_std, rf):
    return (rp_mean - rf)/rp_std

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
