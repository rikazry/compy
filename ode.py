from sympy import symbols, Function, pprint, solve, dsolve

x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)


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
