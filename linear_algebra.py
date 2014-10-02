import numpy as np
from sympy import pprint, symbols, Eq
from sympy.abc import sigma, rho, X, P
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, eye, ones

def demo_eigen_number():
    m = Matrix(3,3,[2,1,0,0,2,1,0,0,2])
    pprint(m)
    print "\nany n*n matrix has n complex eigen values, counted with multiplications\n"\
            "since the characteritic polynomial is n-dim\n"\
            "and Fundamental Theorem of Algebra gives us exactly n roots:"
    pprint(m.charpoly().as_expr())
    print"\neach distinct eigen value as at least 1 at most multiplication eigen vectors\n"\
            "but it's possible to have less than multiplication number of eigen vectors\n"\
            "the given matrix is an example with only 1 eigen vector:\n"
    pprint(m.eigenvects())
    print "to se this for any vector v we can compute M * v:\n"
    v = IndexedBase('v')
    vec = Matrix(m.rows, 1, lambda i, j: v[i+1])
    pprint(vec)
    pprint(m*vec)

def demo_corr_bound(n):
    def corr(i, j):
        if i == j:
            return 1
        else:
            return rho
    Rho = Matrix(n, n, corr)
    print "what's the bound of rho to make below a correlation matrix?\n"
    pprint(Rho)
    print "\ncan be viewed as the sum of 2 matrices Rho = A + B:\n"
    A = (1-rho)*eye(n)
    B = rho*ones(n,n)
    pprint(A)
    pprint(B)
    print "\nthe eigen value and its dimention of first matrix A is:"
    pprint(A.eigenvects())
    print "\nas for the seconde matrix B\n"\
            "it's product of any vector v:"
    v = IndexedBase('v')
    vec = Matrix(n, 1, lambda i, j: v[i+1])
    pprint(vec)
    pprint(B*vec)
    print "\nin order for it's to equal to a linear transform of v\n"\
            "we can see its eigen values, vectors and dimentions are:"
    pprint(B.eigenvects())
    print "\nfor any eigen vector of B v, we have Rho*v :\n"
    pprint(Rho*vec)
    print "\nwhich means v is also an eigen vector of Rho,\n"\
            "the eigen values, vectors and dimentions of Rho are:\n"
    pprint(Rho.eigenvects())
    print "\nsince have no negative eigen values <=> positive semidefinite:\n"\
            "the boundaries for rho are: [1/(%d-1),1]" %n

def demo_vcv2corr():
    vcv = Matrix(3,3, [1,.36,-1.44,.36,4,.8,-1.44,.8,9])
    pprint(vcv)
    corr, D = vcv2corr(vcv)
    pprint(D)
    pprint(corr)

def vcv2corr(vcv):
    n = vcv.rows
    m = vcv.cols
    if n != m:
        raise ValueError('variance mattrix has to be square matrix')
    def var(i, j):
        if i == j:
            return vcv[i, j]**.5
        else:
            return 0
    D = Matrix(n, n, var)
    corr = D.inv() * vcv * D.inv()
    return corr, D

def demo_corr_positive_semidefinite(n):
    print 'First prove that any variance-covariance matrix is positive semidefinite:\n' 
    Sigma = demo_vcv_positive_semidefinite(n)
    print '\nNow prove same conclusion also holds for correlation matrix:\n'\
            'for any correlation matrix Rho:\n'
    rho = IndexedBase('corr')
    def corr(i, j):
        if i == j:
            return 1
        else:
            return rho[i+1, j+1]
    Rho = Matrix(n, n, corr)
    pprint (Rho)
    print '\n we have D * Rho * D = Sigma\n'\
            'where D is a diagonal matrix with variances of each asset:'
    D = vcv2corr(Sigma)[1]
    pprint(D)
    pprint(Eq(D*Rho*D, Sigma))
    print "D's inverse:"
    pprint(D.inv())
    print '\nfor any non-zero vector w:'
    w = IndexedBase('w')
    weight = Matrix(n, 1, lambda i, j: w[i+1])
    pprint(weight)
    print "\nw' * Rho * w = w' * D.inv() * D * Rho * D * D.inv() * w = (w' * D.inv()) * Sigma * (D.inv() * w):\n"\
            "let c = D.inv() * w, then c' = w'* D.inv():"
    pprint(D.inv() * weight)
    pprint(weight.T * D.inv())
    print "\nw' * Rho * w = c' * Sigma * c is proven non-negative"

def demo_vcv_positive_semidefinite(n):
    sig = IndexedBase('cov')
    Sigma = Matrix(n, n, lambda i, j: sig[i+1, j+1])
    print 'Target\nprove any covaraince matrix Sigma is positive semidefinite:\n'
    pprint(Sigma)
    print '(is variance-covariance matrix of assets:)'
    x = IndexedBase('X')
    asset = Matrix(n, 1, lambda i, j: x[i+1])
    pprint(asset)
    print '\nfor any non-zero vector c:'
    c = IndexedBase('c')
    weight = Matrix(n, 1, lambda i, j: c[i+1])
    pprint(weight)
    print 'view it as the weight como of a portfolio of above assets\n'\
            " consider portfolio P = c' * X:" 
    pprint(weight.T * asset) 
    print "\nthen the total variance of P var(P) = c' * Sigma * c:"
    pprint(weight.T * Sigma * weight)
    print '\nwhich can never be negative as it is an asset variance'
    return Sigma
