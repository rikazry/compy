from sympy.polys.specialpolys import interpolating_poly
from sympy.abc import x

def inter_poly(X, Y, symbol = x, algo = 'Lr', simplify = True):
    """
    polynomial interpolation using Lagrange's Methods
    takes numpy array
    """
    n = X.size
    if algo == 'Lr':
        poly = interpolating_poly(n, symbol, X, Y)
    elif algo == 'Nt':
        poly = interpolating_poly_Newton(n, symbol, X, Y)
    elif algo == 'Nv':
        poly = interpolating_poly_Neville(n, symbol, X, Y)
    if simplify == True:
        poly = poly.expand() 
    return poly

def interpolating_poly_Newton(n, symbol, X,Y):
    """
    polynomial interpolation using Newton's Methods
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

def interpolating_poly_Neville(n, symbol, X, Y):
    """
    polynomial interpolation using Neville's Methods
    currently only support numerical result, not symbolic result
    """
    poly = Y.copy()
    for i in range(1, n):
        poly[0:n-i] = ((symbol - X[i:n])*poly[0:n-i]+(X[0:n-i] - symbol)*poly[1:n-i+1])/(X[0:n-i] - X[i:n])
    return poly[0]
