# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
R Scott Collings
Math 347
9 Jan 2019
"""

import sympy as sy
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    #show the expression
    x,y = sy.symbols('x y')
    return sy.Rational(2,5) * sy.exp(x ** 2 - y) * sy.cosh(x + y) + sy.Rational(3,7) * sy.log(x * y + 1)


# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    #find the sum with sympy instead of for loops
    i,j,x = sy.symbols('i j x')
    return sy.simplify(sy.product(sy.summation(j * (sy.sin(x) + sy.cos(x)), (j,i,5)),(i,1,5)))


# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-3,3]. Plot e^(-y^2) over the same domain for comparison.
    """
    #make an expression for approximating e^x with maclaurin series
    x,y,i = sy.symbols('x y i')
    approx = sy.summation((x ** i) / sy.factorial(i),(i,0,N))
    #substitute to approx e^-y**2
    function = sy.lambdify((y,i),approx.subs(x,-y**2),'numpy')
    #plot approximation for various values of y compared with true value
    t = np.linspace(-3,3,1000)
    f = function(t,N)
    samples = np.exp(-t**2)
    plt.plot(t,f,label='approximation')
    plt.plot(t,samples,label='function')
    plt.ylim((0,1.5))
    plt.xlim((-2,2))
    plt.legend()
    plt.show()


# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    #make an expression
    x,y,r,theta = sy.symbols('x y r theta')
    expr = 1 - ((x**2+y**2)**sy.Rational(7,2) + 18*x**5*y - 60*x**3*y**3 + 18*x*y**5) / (x**2+y**2)**3
    #simplify the expression
    newExpr = expr.subs({x:r*sy.cos(theta),y:r*sy.sin(theta)}).simplify()
    #solve for r and plot rose curve for various theta with one of the r
    r = sy.solve(newExpr,r)
    f = sy.lambdify(theta,r[0],'numpy')
    t = np.linspace(0,2*np.pi,1000)
    samples = f(t)
    plt.plot(samples*np.cos(t),samples*np.sin(t))
    plt.show()


# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    x,y,l = sy.symbols('x y l')
    A = sy.Matrix([[x-y,x,0],[x,x-y,x],[0,x,x-y]])
    I = sy.Matrix([[1,0,0],[0,1,0],[0,0,1]])
    expr = (A-l*I).det()
    eigvals = sy.solve(expr,l)
    eigvecs = [(A-val*I).nullspace() for val in eigvals]
    return {a:b for a,b in zip(eigvals,eigvecs)}



# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points. Determine which points are
    maxima and which are minima.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    #initialize and take two derivatives of the polynomial
    x = sy.symbols('x')
    P = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x - 100
    evalP = sy.lambdify(x,P,'numpy')
    Pprime = sy.Derivative(P,x).doit()
    PdoublePrime = sy.lambdify(x,sy.Derivative(Pprime,x).doit(),'numpy')
    #solve for critical points and find concavity at those points
    criticalPoints = np.array(sy.solve(Pprime,x))
    concavity = PdoublePrime(criticalPoints)
    t = np.linspace(-5,5,1000)
    f = evalP(t)
    #determine whether critical points are maxes or mins
    maxes = np.where(concavity < 0)
    mins = np.where(concavity > 0)
    undetermined = np.where(concavity == 0)
    #plot critical points along with polynomial
    plt.plot(t,f)
    plt.plot(criticalPoints[maxes],evalP(criticalPoints[maxes]),'yo',label='local maxima')
    plt.plot(criticalPoints[mins],evalP(criticalPoints[mins]),'go',label='local minima')
    plt.plot(criticalPoints[undetermined],evalP(criticalPoints[undetermined]),'mo',label='inconlcusive')
    plt.xlim((-5,5))
    plt.legend()
    plt.show()
    return set(criticalPoints[mins]),set(criticalPoints[maxes])

# Problem 7
def prob7():
    """Calculate the integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    x,y,z,r,theta,phi,t = sy.symbols('x y z r theta phi t')
    #create function and Jacobian with respect to r theta phi
    matrix = sy.Matrix([x,y,z])
    f = (x**2 + y**2 + z**2)**2
    matrix = matrix.subs({x:r*sy.sin(phi)*sy.cos(theta),
                y:r*sy.sin(phi)*sy.sin(theta),
                z:r*sy.cos(phi)})
    J = matrix.jacobian([r,phi,theta])
    f = f.subs({x:r*sy.sin(phi)*sy.cos(theta),
                y:r*sy.sin(phi)*sy.sin(theta),
                z:r*sy.cos(phi)})
    #simplify function and jacobian
    f = f.simplify()
    expr = f*sy.simplify(J.det())
    #take the triple integral leaving the last step (with respect to r) as a lambda
    step1 = sy.integrate(expr,(theta,0,sy.pi*2))
    step2 = sy.integrate(step1,(phi,0,sy.pi))
    step3 = sy.lambdify(t,sy.integrate(step2,(r,0,t)),'numpy')
    samples = np.linspace(0,3,1000)
    #plot integral value with various radius sizes of integration
    plt.plot(samples,step3(samples))
    plt.show()
    return step3(2)

