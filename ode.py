from sympy import symbols, Function, Eq, pprint, solve, dsolve, integrate, Integral

x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)

def demo_separable_equation():
    separable_equation(1, y*(1-y), f(x)*(1-f(x)))

def separable_equation(g, h, hf = None):
    """
    dy/dx = g(x)*h(y)
    dy/h(y) = g(x)*dx
    """
    dy, dx = symbols('dy, dx')
    print '\nODE to solve:'
    pprint(Eq(dy/dx, g*h))
    pprint(Eq(dy/h, g*dx))
    print '\nintegrate both sides:'
    LHS, RHS = symbols('LHS, RHS')
    pprint(Eq(LHS, Integral(1/h,y)))
    H = integrate(1/h,y)
    pprint(Eq(LHS, H))
    pprint(Eq(RHS,Integral(g,x)))
    G = integrate(g, x)
    pprint(Eq(RHS, G))
    C = symbols('C')
    print '\nsolving LHS = RHS + C...'
    eq = Eq(H,G+C)
    pprint(eq)
    pprint(solve(eq, y))
    if hf:
        print '\nsolving ODE directly ...'
        pprint(dsolve(f(x).diff(x)-g*hf, f(x)))

def demo_non_homo_linear():
    non_homo_linear(1,2,(-4,1),(4,0))

def non_homo_linear(rhs, *cds):
    char_func, eq, rs = homo_linear(*cds)
    eq -= rhs
    print('\nsolving non-homogeneous linear equation...\nresult:')
    rs = dsolve(eq)
    pprint(rs)
    return char_func, eq, rs

def homo_linear(*cds):
    char_func = 0
    eq = 0
    for cd in cds:
        try:
            c, d = cd
        except TypeError:
            c, d = 1, cd
        char_func += c * y**d
        eq += c * f(x).diff(x, d)
    print('\nODE:')
    pprint(eq)
    print('\nhomogeneous characteristic function:')
    pprint(char_func)
    print('\nsolving characteristic function...\nresult:')
    pprint(solve(char_func))
    print('\nsolving homogeneous linear equation...\nresult:')
    rs = dsolve(eq)
    pprint(rs)
    return char_func, eq, rs
