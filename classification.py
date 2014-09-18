
"""
====================
Logistic Regression:
====================
2 classes:
    p(X) = exp{LM} / (1 + exp{LM})
    log odds/ logit transformation of p(X): log( p(X) / (1-p(X)) ) = LM
multinomial regression (>2 classes):
    p(Y=k|X) = exp{LM for class k} / sum of exp{LM for each class}

Maximum Likelihood Approach: 
    l = product of p(x_i)|{y_i == 1} * product of ( 1 - p(x_i) )|{y_i == 0}
Case-control sampling:
    slope parameters are accurately estimated
    interseption parameter needs a transformation:
        b0 += ( logit transform of prior prob - logit transform of case/(case+control) )
    sampling more controls than cases reduces the variance of parameter estimates
        but after a ratio of 5/1 the variance reduction flattens out
vs. LDA
    same expression
    LR uses conditional likelihood based on P(Y|X): discriminative learning
    LDA uses full likelihood based on P(X,Y): generative learning
    practice results are often similar
    LR can also fit quandratic boundaries by explicitly including quadratic term

======================
Discriminant Analysis:
======================

Bayes theorem for classification:
    P(Y=k | X=x) = prior prob of k * density for X=x in k / sum of numerator for each class
LDA
    Discriminant Functions:
        assume density for X=x in each class with Gaussian density
            of shared covariance matrix and class specified mean
        parameter estimation:
            prior prob of k: empirical frequency
            mean_k: empirical average for X in k
            variance: sum of squared deviation for each class / (n-K)
                    = sum of[ (n_k - 1) * var_k / (n-K) ]for each class
        conditional probability of classification can be calculated with Bayes
        Finding the largest conditional prob equivalent to largest discriminant score:
            discriminant score = x * mean_k / variance - mean_k^2 / (2*variance) + log(prio prob of k)
            which is a linear function of x 
                decision boundary for k = 2 and prior prob = .5:
                    x = average of 2 means of density of X
        dicriminant score -> conditional probability:
            P(Y=k | X=x) = exp{score(x) for k} / sum of exp{score(x) for each class}
    Advantage of LDA:
        more stable than logistic regression when:
            classes are well-seperated
            sample size is small and X is approximately normal in each class
        provides low-dim view of data with more than 2 classes
Quadratic Discriminant Analysis
    Discriminant Functions:
        instead of shared covariance matrix in LDA
            assume different covariance matrix in each class
Naive Bayes
    Discriminant Function:
        assume density function in each class conditional independent
            therefore diagonal covariance matrix for each class
    Advantage:
        useful when p is large and QDA, LDA break down
        can use for qualitative variable, just replace pdf with pmf

+++++++++++
Comparison:
+++++++++++

Logistic Regression:
    K = 2
LDA:
    small n
    well seperated classes
    Gaussian assumptions are resonable
    K > 2
Naive Bayes:
    very large p
"""
