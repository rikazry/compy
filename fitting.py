from sympy.polys.specialpolys import interpolating_poly
from sympy.abc import x
from sympy.core import Add, Mul

def inter_poly(X, Y, symbol = x, algo = 'Lr', expand = True):
    """
    takes numpy array
    """
    n = X.size
    if algo == 'Lr':
        poly = interpolating_poly(n, symbol, X, Y)
    elif algo == 'Nt':
        poly = interpolating_poly_Newton(n, symbol, X, Y)
    poly = poly.expand() if expand == True else 0
    return poly

def interpolating_poly_Newton(n, symbol, X,Y):
    """
    polynomial interpolation using Newton's Methos
    """
    def coef(X, Y):
        """
        calculate coefficients a[i]s of polynomial
        """
        a = Y.copy()
        for i in range(1, n):
            a[i:n] = (a[i:n] - a[i-1])/(X[i:n] - X[i-1])
        return a
    
    a = coef(X, Y)
    poly = a[n-1]
    for i in range(1, n):
        poly = a[n-i-1] + (symbol - X[n-i-1])*poly
    return poly

