#gaussian_quadrature.py
"""Volume 2: Gaussian Quadrature.
R Scott Collings
Math 323 Sec 002
29 Jan 2017
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as spars
import scipy.linalg as spla
import scipy.stats as stat
import scipy.integrate as intgt

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        # check to see if polynomial type is supported
        if polytype != "legendre" and polytype != "chebyshev":
            raise ValueError('supported basis are Legendre and Chebyshev')
        #set values depending on polynomial type
        self.polytype = polytype
        self.n = n
        if polytype == "legendre":
            self.w = lambda x: 1
        else:
            self.w = lambda x: np.sqrt(1-x**2)
        #calculate interpolation points and weights
        self.eigs,self.weights = self.points_weights(n)

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        #make weight function and b_k for each type of polynomial
        f = lambda k: .5 if k == 1 else .25
        w = np.pi
        if self.polytype == "legendre":
            f = lambda k: k**2/(4*k**2-1)
            w = 2
        #create Jacobi matrix and get eigenvalues and eigenvectors
        J = spars.diags([[np.sqrt(f(k)) for k in range(1,n)],[np.sqrt(f(k)) for k in range(1,n)]],[-1,1]).toarray()
        eigs,vecs = np.linalg.eig(J)
        #calculate interpolation weights and return along with interpolation points
        weights = np.array([w*vec[0]**2 for vec in vecs.T])
        return eigs,weights

    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        #compute integral with interpolation sum
        return np.sum([f(x)*self.w(x)*y for x,y in zip(self.eigs,self.weights)])

    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        #scale function to interval -1,1 and compute approximation
        h = lambda x: (b-a)*f((b-a)*x/2+(a+b)/2)/2
        return self.basic(h)

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        #scale function to interval [-1,1]x[-1,1] and return double interpolation sum
        g = lambda x,y: (b1-a1)*(b2-a2)*f((b1-a1)*x/2+(a1+b1)/2,(b2-a2)*x/2+(a2+b2)/2)/4
        return np.sum([np.sum([
                g(x2,y2)*self.w(x2)*self.w(y2)*x1*y1 for x2,x1 in zip(self.eigs,self.weights)])
            for y2,y1 in zip(self.eigs,self.weights)])


# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    #get true integral value and define function
    actual = stat.norm.cdf(2) - stat.norm.cdf(-3)
    f = lambda x: (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)
    #get scipy approximation
    quad = intgt.quad(f,-2,3)[0]
    legApprox = []
    chebyApprox = []
    #for several amounts of interpolation points get the Legendre and Chebyshev approximation
    t = range(5,55,5)
    for n in t:
        A = GaussianQuadrature(n)
        legApprox.append(A.integrate(f,-3,2))
        B = GaussianQuadrature(n,'chebyshev')
        chebyApprox.append(B.integrate(f,-3,2))
    #plot error for all 3 different approximations
    plt.semilogy(t,np.abs(legApprox-actual),label='Legendre Approximation')
    plt.semilogy(t,np.abs(chebyApprox-actual),label='Chebyshev Approximation')
    plt.plot(t,[quad for x in t],label='Scipy Approximation')
    plt.legend()
    plt.title('Approximation error for normal c.d.f. from -3 to 2')
    plt.ylabel('Error')
    plt.xlabel('Interpolation Points')
    plt.show()