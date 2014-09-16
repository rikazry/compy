"""
General Concepts:

ideal / optimal predictor f(x) = E(Y|X=x) with regard to mean-squared prediction error:
    minimizes E[(Y - g(X))^2 | X = x] over all functions g at all points X = x

    for any estimate g of f:
        E[(Y - g(X))^2 | X = x] = [f(x) - g(X)]^2 + Var(Y - f(x))
                                = reducible error + irreducible error
                                = Var(g(x)) + Bias(g(x))^2 + Var(Y-f(x))

    More complicated models typically have lower bias at the cost of higher variance. 
    This has an unclear effect on Reducible Error and no effect on Irreducible Error.

    a naive way of estimating f: nearest neighbor averaging
        problem: curse of dimensionality when p > 4, nearest neighbors tend to be far away

Interpretability-Flexibility trade-off of prediction models:
    interpretability decreases and flexibility increases:
        Subset Selection, Lasso
        Least Squares
        Generalized Additive Models, Trees
        Bagging, Boosting
        Support Vector Machines

Bias-Variance Trade-off of prediction models:
    flexibility increases -> variance increases -> bias decreases
    training set MSE may be biased toward more overfit models
    choosing flexibility based on test set MSE amounts to a bias-variance trade-off


Linear Regression:

Assumptions:
    * A linear relationship exists between the dependent and independent variable
    * No perfect multicollinearity for multiple regressors: corr(X_i, X_j) != 1 or -1
    * The independent variable is uncorrelated with the residuals
    * The expected value of the residual tearm is zero: E(e)=0
    * The variance of the residual term is constant, not heteroskedastic: E(e_i^2)=const
    * The residual term is independently distributed, no autocorrelation: E(e_i * e_j) = 0 for i != j
    * The residual term is normally distributed

Ordinary Least Square approach: minimizes Sum of Squared Errors (residual)
    min f(b) = min (Y-Xb)'(Y-Xb)
    df/db = -2X'Y + 2X'Xb = 0 -> (X'X)b = X'Y
    beta = cov(X,Y)/Var(X)
    intercept = mean(Y) - beta*mean(X)
        regression line passes through a point w/ coordinates equal to the mean
        of the independent and dependent variables

Accuracy Assessing:
    Standard Errors to compute Confidence Interval and conduct Hyothesis Testing:
        SE(beta)^2 = Var(Y-f(x)) / SumSquaredDeviation(X)
        SE(intercept)^2 = Var(Y-f(x)) * ( 1/n + mean(X)^2 / SumSquaredDeviation(X) )
        df fot t-dist is n-1-k for k regressors
    ANOVA:
        * Total Sum of Squares (SST): 
            measures the total variation in the dependent variable
            SST = Var(Y) * (n-1)
            df = n - 1
        * Regression Sum of Squares (RSS): 
            measures the variation in the dependent variable that is explained by the independent variable
            df = 1
        * Sum of Squared Errors (SSE): 
            measures the unexplained variantion in the dependent variables
            df = n-2
                Standard Deviation of Regression Error (SEE):
                    SEE^2 = SSE / df
        * SST = RSS + SSE
          total variation = explained variation + unexplained variation

    R Square:
        R^2 = RSS/SST
        equals to squared correlation coefficient if only 1 regressor
    F Stats:
        whether at least 1 independent variable explains a significant portion
        of the variation of the dependent variable
        F = (RSS / k) / (SSE / (n-1-k))
        F-test is always one-tailed, tests all independent variables as a group

Limitations:
    * Parameter Instability: Linear relationships can change over time.
    * If the assumptions underlying regression analysis do hot hold, the interpretation and tests of hypotheses may not be valid. 
    * Even if the regression model accurately reflects the historical relationship, its usefulness in investment analysis will be limited if  other market participants are also aware of act on this evidence.
"""

import numpy as np
import pandas as pd
import scipy
import statsmodels.formula.api as sa
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import EqPython.datamaster.eq_datamanager as eq

EQ = eq.EQ('mysql_db_test.conf')
aapl = EQ.download_ticker_df('AAPL','1998-1-1')
amzn = EQ.download_ticker_df('AMZN','1998-1-1')
aapl_adj = aapl['Adj Close']
amzn_adj = amzn['Adj Close']

plt.scatter(aapl_adj, amzn_adj)
plt.show()

result = sa.OLS(amzn_adj, aapl_adj).fit()
print result.summary()
