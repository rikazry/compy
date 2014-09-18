"""
=================
General Concepts:
=================

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

==================
Linear Regression:
==================

Assumptions:
    * A linear relationship exists between the dependent and independent variables
    * No perfect multicollinearity for multiple regressors: corr(X_i, X_j) != 1 or -1
        ideal scenario is when predictors are uncorrelated: a balanced design
        problems of correlation:
            variance of all coefficients tends to increase
                -> TypeII error
            interpretations of effects of changing 1 variable become hazardous
        detection:
            no t-test is significant + f-test significant + high R^2
        correction:
            omit one or more correlated independent variables:
                e.g. stepwise
    * The independent variable is uncorrelated with the residuals
    * The expected value of the residual tearm is zero: E(e)=0
    * The variance of the residual term is constant across observations, not heteroskedastic: E(e_i^2)=const
        2 types:
            unconditional heteroskedasticity:
                not related to the level (value) of independent variables
                usually causes no major problems
            conditional heteroskedasticity:
                conditional on the independent variables
                effects:
                    standard errors are usually unreliable estimates
                codfficient estimates aren't affected
                t-test is therefore affected
                f-tst is also unreliable
        detection:
            examine scatter plots of residuals
            Breusch-Pagan chi-square test
                BP chi2 test = n * R_resid^2 with df = k
                    R_resid^2 = R^2 from a second regression of the squared residuals
        correction:
            calculate robust standard errors (White-corrected standard errors) for t test
            generalized least squares
    * The residual term is independently distributed, no autocorrelation: E(e_i * e_j) = 0 for i != j
        2 types:
            positive serial correlation
                effect: underestimated standard errors
                    -> TypeI error in t-test and f-test
            negative serial correlation
        detection:
            residual plot
            Durbin-Watson statistic
                DW = sum of squared adjacent residual difference / SSE 
                    DW -> 2*(1-r) with large sample size
                    r = correlation of adjacent residuals
                    homoskedastic and no serial correlation: DW -> 2
                    positive serial correlation: DW < 2
                    negative serial correlation: DW > 2
        correction:
            adjust coefficient standard errors: Hansen method
            improve the specification of the model:
                explicitly incorporate the time-series nature (e.g., include a sesonal term)
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
            df = k 
        * Sum of Squared Errors (SSE): 
            measures the unexplained variantion in the dependent variables
            df = n - k - 1
                Standard Deviation of Regression Error (SEE):
                    SEE^2 = SSE / df
        * SST = RSS + SSE
          total variation = explained variation + unexplained variation
    R Square:
        R^2 = RSS/SST
        equals to squared correlation coefficient if only 1 regressor
    Adjusted R Square:
        R^2 almost always increases as variables are added to the model
            even if the marginal contribution of the new variables is not significant
        R_a^2 = 1 - (1-R^2) * ((n-1) / (n-k-1))
    F Stats:
        whether at least 1 independent variable explains a significant portion
        of the variation of the dependent variable
        F = (RSS / k) / (SSE / (n-1-k))
        F-test is always one-tailed, tests all independent variables as a group

Model Specification:
    Misspecification:
        Functional form can be misspecified:
            Important variables are omitted
            Variables should be transformed
            Data is improperly pooled
        Explanatory variables are correlated with the error term in TS models:
            A lagged dependent variable is used as an independent variable
            A function of the dependent variable is used as an independent variable: forecasting the past
            Independent variables are measured with error
        Other TS misspecification that result in nonstationarity
    Variable Selection:
        All Subsets/ Best Subsets Regression:
            Choose between all based on some criterion that balances training error with model size
        Forward Selection:
            1) null model
            2) fit k simple linear regresions and add the variable with lowest SSE
            3) add 1 more variable with lowest 2-variable SSE
            4) stopping rule:
                eg. when all remaining variables have a p-value above a threshold
        Backward Selection:
            1) all variables
            2) remove the variable with the largest p-value
            3) remove 1 more variable with the largest p-value in this (k-1)-fit
            4) stopping rule

Model Assessment:
    Is the model correctly specified?
    Are individual coeefficients statistically significant? (t-test)
    Is model statistically significant? (F-test)
    Is heteroskedasticity present?
        If yes, is it conditional? (Breusch-Pagan Chi-square test)
            If yes, correct with Wite-corrected standard errors
    Is serial correlation present? (Durbin-Watson test)
        If yes, correct with Hansen method to adjust standard errors
    Does model have significant multicollinearity?
        If yes, drop one of the corelated variables

Limitations:
    * Parameter Instability: Linear relationships can change over time.
    * If the assumptions underlying regression analysis do hot hold, the interpretation and tests of hypotheses may not be valid. 
    * Even if the regression model accurately reflects the historical relationship, its usefulness in investment analysis will be limited if  other market participants are also aware of act on this evidence.

Generalizations of Linear Model:
        Classification problems:
            logistic regression
            support vector machines
        Non-linearity:
            kernel smoothing
            splines
            generalized additive models
            nearest neighbor
        Interactions (also capture non-linearities):
            Tree-based methods
            bagging
            random forests
            boosting
        Regularized fitting:
            Ridge regression
            lasso
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
