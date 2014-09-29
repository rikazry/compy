import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import quad
from sympy import integrate, init_printing, pprint, Integral, summation,\
symbols, factorial, tan, cos, log
from sympy.tensor import IndexedBase, Idx
from sympy.abc import x, y, z

def i_to_the_i():
    a = 1j**1j
    print 'direct computation: 1j**1j:', a
    print '\nEuler Formula:\n'\
          'exp(i*pi/2) = cos(pi/2) + i*sin(pi/2) = i\n'\
          'i^i = ( exp(i*pi/2) )^i = exp(-pi/2)\n'
    b = np.exp(-np.pi/2)
    print 'derived computation: np.exp(-np.pi/2):', b
    print '\n compare result equal'
    return a==b

def e_to_pi_compare_pi_to_e():
    x1 = np.exp(np.pi)
    x2 = np.pi**np.e
    a = ( x1>x2 )
    print 'e^pi = ', x1
    print 'pi^e = ', x2
    print 'direct comparison e^pi > pi^e ?:', a
    print '\n e^pi ? pi^e\n'\
          '<=> pi*ln(e) ? e*ln(pi)\n'\
          '<=> ln(e)/e ? ln(pi)/pi\n'\
          'f(x) = ln(x)/x\n'\
          "f'(x) = ( 1-ln(x) )/( x^2 )\n"\
          "f'(e) = 0\n"\
          "f'(x) < 0 for x > e, e.g. x = pi\n"\
          "=> f(e) > f(pi)\n"\
          "=> e^pi > pi\n"
    b = True
    print 'derived comparison e^pi > pi^e ?:', b
    print '\n compare result equal'
    return a==b

def inf_sqrt_2(n):
    puzzle = 'calculate: sqrt(2+sqrt(2+(sqrt(2+...'
    solution = '\nFirst prove convergence:\n'\
          'let x_0 = sqrt(2), x_(n+1) = sqrt(2+x_n)\n'\
          '\n1) sequence is upperbounded:\n'\
          'x_0 < 2 => x_n < 2=sqrt(4) for any n\n'\
          '2) sequence is increasing:\n'\
          '    x_(n+1) - x_n > 0\n'\
          '<=> x_(n+1)^2 - x_n^2 > 0 since x_n > 0 for any n\n'\
          '<=> 2 + x_n - x_n^2 > 0\n'\
          '<=> (2 - x_n)*(1 + x_n) >0\n'\
          'Therefore sequnce converges to a limit l\n'\
          '\nNext calculate l,s.t.\n'\
          'l = sqrt(2+l) => l^2 - l - 2 = 0 => (l-2)*(l+1) = 0\n'\
          '=> l = 2\n'
    def x_n(n):
        x = np.sqrt(2)
        for i in range(n):
            x = np.sqrt(x+2)
        return x
    return inf_sequence(n, puzzle, solution, x_n)

def inf_power_equal2(n):
    puzzle = 'find x: x^x^x^... = 2'
    solution = '\nIf x eixist, then x^2 = 2 => x = sqrt(2)\n'\
          'let x_0 = sqrt(2), x_(n+1) = sqrt(2)^x_n\n'\
          'First prove the sequence with x=sqrt(2) actually converges\n'\
          '\n1) sequence is upperbounded:\n'\
          'x_0 < 2 => x_n < 2 for any n\n'\
          '2) sequence is increasing:\n'\
          'x_0 < x_1 + induction => x_n = sqrt(2)^x_(n-1) < sqrt(2)^x_n = x_(n+1)\n'\
          'Therefore sequence converges to a limit l\n'\
          '\nNext calculate l,s.t.\n'\
          'l = sqrt(2)^l => l^(1/l) = 2^(1/2)\n'\
          "f(x) = x^(1/x) with f'(x) = f(x)*(1-lnx)/(x^2)\n"\
          "f'(x) = 0 => x = e => root for f(l) = f(2) is l1=2 or l2>3\n"\
          "upperbound of x_n => l=2\n"
    def x_n(n):
        x = np.sqrt(2)
        for i in range(n):
            x = np.sqrt(2)**x
        return x
    return inf_sequence(n, puzzle, solution, x_n)

def inf_series_convergence(n, n0=2):
    def quad_n_1(n):
        return quad_n(n, integrand_1) + integrand_1(n)
    def quad_n_2(n):
        return quad_n(n, integrand_2) + integrand_2(n)
    def quad_n_3(n):
        return quad_n(n, integrand_3) + integrand_3(n0)
    def x_n_1(n):
        return x_n(n, integrand_1)
    def x_n_2(n):
        return x_n(n, integrand_2)
    def x_n_3(n):
        return x_n(n, integrand_3)
    def quad_n(n, integrand):
        return quad(integrand, n0, n)[0] 
    def x_n(n, integrand):
        x = 0.
        for i in range(n0, n+1):
            x += integrand(i)
        return x
    def integrand_1(x):
        return 1./x
    def integrand_2(x):
        return 1./(x*np.log(x))
    def integrand_3(x):
        return 1./(x**2)
    t, x1 = inf_sequence(n, '1/k', 'series', x_n_1, n0)
    x2 = inf_sequence(n, '1/kln(k)', 'series', x_n_2, n0)[1]
    x3 = inf_sequence(n, '1/k^2', 'series', x_n_3, n0)[1]
    t_2, x1_2 = inf_sequence(n, '1/k', 'integral', quad_n_1, n0) 
    x2_2 = inf_sequence(n, '1/klnk', 'integral', quad_n_2, n0)[1]
    x3_2 = inf_sequence(n, '1/k^2', 'integral', quad_n_3, n0)[1]
    plt.plot(t,x1,label = '1/k_series')
    plt.plot(t_2,x1_2,label = '1/k_integral')
    plt.plot(t,x2,label = '1/klnk_series')
    plt.plot(t_2, x2_2, label = '1/klnk_integral')
    plt.plot(t,x3,label = '1/k^2_series')
    plt.plot(t_2, x3_2, label = '1/k^2_integral')
    plt.legend(loc = 2)
    plt.show()

def inf_sequence(n, puzzle, solution, x_n, n0=0):
    print puzzle
    print solution
    x_n_v = np.vectorize(x_n)
    t = range(n0, n)
    x = x_n_v(t)
    print 'calculate x_n for n from %d to %d:' %(n0, n)
    print x
    return t,x

def demo_integral():
    puzzle1 = 1/(1+x**2)
    solutions1 = list()
    solutions1.append("solution:let x = tan(z), then dx = dtan(z) = 1/cos^2(z) * dz")
    solutions1.append(Integral(1/((tan(z)**2+1)*(cos(z)**2))))
    solutions1.append(Integral(1,z))
    integral(puzzle1, solutions1)
    n = symbols('n')
    puzzle2 = (x**n)*log(x)
    solution2 = list()
    solution2.append('solution: integrate by parts')
    solution2.append('for n != -1')
    solution2.append(1/(n+1) * x**(n+1) * log(x) - Integral(1/(n+1) * x**n, x))
    solution2.append(1/(n+1) * x**(n+1) * log(x) - 1/(n+1)**2 * x**(n+1))
    solution2.append('for n == -1')
    solution2.append(log(x)**2/2)
    solution2.append('this is an example that sympy got a different result')
    integral(puzzle2, solution2, x)
    I = IndexedBase('I')
    n = symbols('n', cls = Idx)
    puzzle3 = (log(x))**n
    solution3 = list()
    solution3.append('solution: integrate by parts + induction')
    solution3.append(log(x)**n * x - n * Integral(log(x)**(n-1), x))
    solution3.append('I[n]=')
    solution3.append(log(x)**n * x - n * I[n-1])
    solution3.append('I[0]=x')
    solution3.append('I[1]=')
    solution3.append(log(x) * x - I[0])
    solution3.append(x * (log(x) - 1))
    solution3.append('I[2]=')
    solution3.append(log(x)**2 * x - 2 * I[1])
    solution3.append(x * (log(x)**2 - 2*log(x) + 2))
    n, k = symbols('n, k')
    solution3.append('I[n]=')
    solution3.append(x * summation( (-1)**(n-k)*factorial(n)/factorial(k) *
        log(x)**k, (k, 0, n)))
    solution3.append('this is an example that sympy could not compute result')
    integral(puzzle3, solution3, x)

def integral(puzzle, solutions, v = x):
    init_printing(use_unicode=False, wrap_line=False, no_global=True)
    print '\npuzzle:'
    pprint(Integral(puzzle, v))
    print '\n'
    for solution in solutions:
        pprint(solution)
    I = integrate(puzzle, v)
    print '\nsympy calculation result:'
    pprint(I)
    return I


