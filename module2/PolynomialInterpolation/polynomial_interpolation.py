# solutions.py
"""Volume 2: Polynomial Interpolation.
R Scott Collings
Math 323 Sec 002
16 Jan 2017
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import BarycentricInterpolator

# Problem 1
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    #create lamda for sum of lagrange polynomials
    f = lambda t: np.sum(
        [y*np.prod(
                [(t - x_k)/(x - x_k) if x != x_k else 1 for x_k in xint]
            )
            for x,y in zip(xint,yint)
        ])
    #evaluate lambda at each point and return evaluation
    return np.array([f(p) for p in points])


# Problems 2 and 3
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        #set attributes
        self.x = xint
        self.y = yint
        self.n = len(xint)
        #calculate barycentric weights
        w = np.ones(self.n)
        C = (np.max(xint) - np.min(xint)) / 4
        shuffle = np.random.permutation(self.n-1) #mix order to reduce memory overflow likelihood
        for j in range(self.n):
            temp = (xint[j] - np.delete(xint,j)) / C
            temp = temp[shuffle]
            w[j] /= np.prod(temp)
        self.w = w

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        #create lambda for barycentric lagrange formula
        f = lambda t: self.y[np.where(self.x == t)] if t in self.x else np.sum(
            [self.y[j]*self.w[j]/(t-self.x[j]) for j in range(len(self.x))]
            )/np.sum(
            [self.w[j]/np.prod(t-self.x[j]) for j in range(len(self.x))])
        #evaluate lambda and return results
        return np.array([f(t) for t in points])

    # Problem 3
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        #update class attributes
        self.y = np.append(self.y,yint)
        self.x = np.append(self.x,xint)
        self.n = len(self.x)
        #recompute barycentric weights
        w = np.ones(self.n)
        C = (np.max(self.x) - np.min(self.x)) / 4
        shuffle = np.random.permutation(self.n-1)
        for j in range(self.n):
            temp = (self.x[j] - np.delete(self.x,j)) / C
            temp = temp[shuffle]
            w[j] /= np.prod(temp)
        self.w = w


# Problem 4
def prob4():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    #initialize runges function and domain
    runges = lambda t: 1/(1+25*t**2)
    t = np.linspace(-1,1,400)
    f = np.array([runges(p) for p in t])
    error1 = []
    error2 = []
    nVals = []
    #find error by L-inf norm for several numbers of interpolating points
    for j in range(2,9):
        n = 2**j
        nVals.append(n)
        #find interpolation polynomial with barycentric lagrange formula for equally spaced points
        #and chebyshev extremizers then get error for each interpolation
        equallySpaced = np.linspace(-1,1,n)
        chebyshevPoints = np.array(np.cos((np.pi/n)*np.arange(n+1)))
        b1 = BarycentricInterpolator(equallySpaced,np.array([runges(p) for p in equallySpaced]))
        b2 = BarycentricInterpolator(chebyshevPoints,np.array([runges(p) for p in chebyshevPoints]))
        error1.append(np.linalg.norm(f-b1(t),ord=np.inf))
        error2.append(np.linalg.norm(f-b2(t),ord=np.inf))
    #plot error for two methods of interpolation
    plt.loglog(nVals,error1,label='equally spaced interpolation error')
    plt.loglog(nVals,error2,label='chevyshev zeros interpolation error')
    plt.legend()
    plt.show()


# Problem 5
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    #get each interpolation point and funtion values at those points
    y = np.cos((np.pi * np.arange(2*n))/n)
    samples = np.array([f(t) for t in y])
    #take the fft of the function values at sample points and scale returned coefficients
    coeffs = np.real(np.fft.fft(samples))[:n+1] / n
    coeffs[0] = coeffs[0]/2
    coeffs[n] = coeffs[n]/2
    #get rid of negligable coefficients by rounding
    return np.round(coeffs,10)
    


# Problem 6
def prob6(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    data = np.load('airdata.npy')
    #find closest points to each chebyshev extremizer
    fx = lambda a,b,n: .5*(a+b+(b-a)*np.cos(np.arange(n+1)*np.pi/n))
    a,b = 0,360-1/24
    domain = np.linspace(0,b,8784)
    points = fx(a,b,n)
    temp = np.abs(points-domain.reshape(8784,1))
    temp2 = np.argmin(temp,axis=0)
    #get the interpolating polynomial using these points
    poly = Barycentric(domain[temp2],data[temp2])
    #plot against actual data
    plt.plot(domain,data,label='function')
    plt.plot(domain,poly(domain),label='interpolation')
    plt.legend()
    plt.show()
