import numpy as np

#augmented coefficient matrix: adjoin the constant vector b to the coefficient matrix A
#coefficient matrix nonsingular/ rows or columns linearly independent/ determinant != 0 => a system of n linear equations in n unknowns has unique solution
#coefficient matrix singular: a system of n linear equations in n unknowns has no solution/ infinite solutions
#matrix condition number cond(A) = ||A||*||A^(-1)||, matrix is well-conditioned if cond(A) is close to unity, cond(A) reaches infinity for a singular matrix, conditioness affects stableness of system therefore numerical solution of ill-conditioned system might introduce large error

def solve_elimin_Gauss(A, b):
    """ 
    solving linear system using Gauss Elimination Method numerically
    please pass in A.copy() and b.copy() if don't want func to write to A,b
    please define A, b with detype = float for accuracy
    eg. A = numpy.identity(3,float), b = numpy.array(range(3),float)
    """
    n = len(b)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if A[j, i] != 0.0:
                lam = A[j, i]/A[i, i]
                A[j, i+1:n] = A[j, i+1:n] - lam*A[i, i+1:n]
                b[j] = b[j] - lam*b[i]
    for i in range(n-1, -1, -1):
        b[i] = (b[i] - np.dot(A[i, i+1:n], b[i+1: n]))/A[i, i]
    return b

def solve_LUdecomp_Doolittle(A, b):
    """
    solving linear system using Doolittle's Method LU Decomposition
    """
    n = len(b)
    LUdecomp_Doolittle(A)
    for i in range(1, n):
        b[i] = b[i] - np.dot(A[i, 0:i], b[0: i])
    for i in range(n-1, -1, -1):
        b[i] = (b[i] - np.dot(A[i, i+1:n], b[i+1: n]))/A[i, i]
    return b

def LUdecomp_Doolittle(A):
    """LU decomposition using Doolittle's Method
       L = array([[1,   0,   0],
                  [l_21,1,   0],
                  [l_31,l_32,1]])
       U = array([[u_11,u_12,u_13],
                  [0,   u_22,u_23],
                  [0,   0,   u_33]])
       A = array([[u_11,     u_12,                 u_13],
                  [l_21*u_11,l_21*u_12 + u_22,     l_21*u_13 + u_23],
                  [l_31*u_11,l_31*u_12 + l_32*u_22,l_31*u_13 + l_32*u_23 + u33]])
    """
    n = len(A)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if A[j, i] != 0.0:
                lam = A[j, i]/A[i, i]
                A[j, i+1:n] = A[j, i+1:n] - lam*A[i, i+1:n]
                A[j, i] = lam
    return A
