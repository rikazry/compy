from numpy import dot

#augmented coefficient matrix: adjoin the constant vector b to the coefficient matrix A
#coefficient matrix nonsingular/ rows or columns linearly independent/ determinant != 0 => a system of n linear equations in n unknowns has unique solution
#coefficient matrix singular: a system of n linear equations in n unknowns has no solution/ infinite solutions
#matrix condition number cond(A) = ||A||*||A^(-1)||, matrix is well-conditioned if cond(A) is close to unity, cond(A) reaches infinity for a singular matrix, conditioness affects stableness of system therefore numerical solution of ill-conditioned system might introduce large error

def Elimin_Gauss(A, b):
    """ 
    pseudo code
    solving linear system using Gauss Elimination Method numerically
    """
    n = len(b)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if A[j, i] != 0.0:
                lam = A[j, i]/A[i, i]
                A[j, i+1:n] = A[j, i+1:n] - lam*A[i, i+1:n]
                b[j] = b[j] - lam*b[i]
    for i in range(n-1, -1, -1):
        b[i] = (b[i] - dot(A[i, i+1:n], b[i+1: n]))/A[i, i]
    return b
