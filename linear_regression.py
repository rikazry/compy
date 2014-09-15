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
    * The variance of the residual term is constant for all observations: E(e_i^2)=const
    * The residual term is independently distributed: E(e_i * e_j) = 0 for i != j
    * The residual term is normally distributed

Ordinary Least Square approach: minimizes Residual Sum of Squares
    min f(b) = min (Y-Xb)'(Y-Xb)
    df/db = -2X'Y + 2X'Xb = 0 -> (X'X)b = X'Y
    beta = cov(X,Y)/Var(X)
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
