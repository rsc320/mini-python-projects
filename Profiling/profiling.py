# profiling.py
"""Python Essentials: Profiling.
R Scott Collings
Math 347 Sec 002
8 Jan 2019
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
import math
from numba import jit
from matplotlib import pyplot as plt
from time import time

# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
        n = len(data)
    #start at second to last row
    k = n-2
    #iterate through rows until top row
    for i in range(n):
        #iterate through elements of row
        for j in range(k+1):
            #find max value for sum of current entries and lower entries and replace value of current
            data[k][j] = max([data[k][j]+data[k+1][j], data[k][j]+data[k+1][j+1]])
        k -= 1
    return int(data[0][0])



# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    primes_list = [2,3]
    current = 5
    while len(primes_list) < N:
        isPrime = True
        #find maximum divisor
        k = int(math.ceil(math.sqrt(current)))
        #only check if previous primes divide number
        for i in primes_list:
            #if greater than max divisor
            if i > k:
                break
            #if found a divisor
            if current % i == 0:
                isPrime = False
                break
        if isPrime:
            primes_list.append(current)
        current += 2
    return primes_list



# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    #find the index that has thi min norm
    return np.argmin(np.linalg.norm(A.T - x,axis=1))


# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    #initialize dictionary
    alphabet = {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]:(i+1) for i in range(26)}
    #array broadcasting to find name scores then sum to get total
    return sum([(j+1) * sum([alphabet[names[j][i]] for i in range(len(names[j]))]) for j in range(len(names))])


# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    #yield and initailize base cases
    yield 1
    yield 1
    x1 = 1
    x2 = 1
    while True:
        #yield additional values
        x = x1 + x2
        yield x
        x1 = x2
        x2 = x

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    #use index and value of generator
    for i, x in enumerate(fibonacci()):
        #stop getting values when length is N or longer and return index
        if len(str(x)) >= N:
            return i + 1



# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    #array of all numbers from 2 to N
    num = np.arange(2,N+1)
    #stop when one prime left in list
    while len(num) > 1:
        #remove all numbers that divide by the first entry
        n = num[0]
        num = num[np.where(num % n != 0)]
        yield n
    yield num[0]




# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

#use the same method as matrix_power but add jit to set data types
@jit
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        #find next power of each entry
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    #run matrix_power_numba once to compile and initialize plotting arrays
    matrix_power_numba(np.random.random((4,4)),n)
    time1 = []
    time2 = []
    time3 = []
    powers = []
    for m in range(2,8):
        #time each array power calculation using different methods
        powers.append(m**2)
        A = np.random.random((2**m,2**m))
        t = time()
        matrix_power(A,n)
        time1.append(time() - t)
        t = time()
        matrix_power_numba(A,n)
        time2.append(time() - t)
        t = time()
        np.linalg.matrix_power(A,n)
        time3.append(time() - t)
    #plot results
    plt.loglog(powers,time1,label='slow')
    plt.loglog(powers,time2,label='numba')
    plt.loglog(powers,time3,label='numpy')
    plt.gca().set_xlabel('matrix size (mxm)')
    plt.gca().set_ylabel('time (ms)')
    plt.gca().set_title('Matrix multiplication')
    plt.legend()
    plt.show()
