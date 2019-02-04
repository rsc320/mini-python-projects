# specs.py
"""Python Essentials: Unit Testing.
R Scott Collings
Math 345 Sec 002
04 Sept 2018
"""

from itertools import combinations

def add(a, b):
    """Add two numbers."""
    #add nums
    return a + b

def divide(a, b):
    """Divide two numbers, raising an error if the second number is zero."""
    #can't divide by 0
    if b == 0:
        raise ZeroDivisionError("second input cannot be zero")
    return a / b


# Problem 1
def smallest_factor(n):
    """Return the smallest prime factor of the positive integer n."""
    if n == 1: return 1
    #add 1 for prime squares because range doesn't include final number
    for i in range(2, int(n**.5) + 1):
        if n % i == 0: return i
    return n


# Problem 2
def month_length(month, leap_year=False):
    """Return the number of days in the given month."""
    if month in {"September", "April", "June", "November"}:
        return 30
    elif month in {"January", "March", "May", "July",
                        "August", "October", "December"}:
        return 31
    #feb has additional cases to check
    if month == "February":
        if not leap_year:
            return 28
        else:
            return 29
    else:
        return None


# Problem 3
def operate(a, b, oper):
    """Apply an arithmetic operation to a and b."""
    if type(oper) is not str:
        raise TypeError("oper must be a string")
    elif oper == '+':
        return a + b
    elif oper == '-':
        return a - b
    elif oper == '*':
        return a * b
    elif oper == '/':
        if b == 0:
            raise ZeroDivisionError("division by zero is undefined")
        return a / b
    #none of the specified operators
    raise ValueError("oper must be one of '+', '/', '-', or '*'")


# Problem 4
class Fraction(object):
    """Reduced fraction class with integer numerator and denominator."""
    def __init__(self, numerator, denominator):
        """Creates a new Fraction object reducing it as far as possible

        Parameters:
            numerator (int)
            denominator (int)
        """
        #don't work with undefined fractions
        if denominator == 0:
            raise ZeroDivisionError("denominator cannot be zero")
        elif type(numerator) is not int or type(denominator) is not int:
            raise TypeError("numerator and denominator must be integers")

        def gcd(a,b):
            """Gets the common factor to reduce the fraction

            Parameters:
                a (int): numerator
                b (int): denominator

            Returns:
                int: greatest common divisor of a and b
            """
            while b != 0:
                a, b = b, a % b
            return a
        #find gcd of numerator and denom to reduce fraction
        common_factor = gcd(numerator, denominator)
        self.numer = numerator // common_factor
        self.denom = denominator // common_factor

    def __str__(self):
        """Stringifies a Fraction object

        Returns:
            string: visual representation of Fraction
        """
        if self.denom != 1:
            return "{}/{}".format(self.numer, self.denom)
        else:
            #don't represent as fraction if denom is 1
            return str(self.numer)

    def __float__(self):
        """Computes the fraction in decimal form

        Returns:
            float: decimal representation of fraction
        """
        #show fraction as a float
        return self.numer / self.denom

    def __eq__(self, other):
        """Determines if a fraction is equal to this fraction

        Parameters:
            Fraction: fraction to compare against

        Returns:
            Fraction: result of equivalence
        """
        if type(other) is Fraction:
            return self.numer==other.numer and self.denom==other.denom
        else:
            #if they aren't the same representation see if they reduce the same
            return float(self) == other

    def __add__(self, other):
        """Addition operator on fractions

        Parameters:
            Fraction: fraction to compare against

        Returns:
            Fraction: result of adding fractions
        """
        #get common denom
        return Fraction(self.numer*other.denom + self.denom*other.numer,
            self.denom*other.denom)

    def __sub__(self, other):
        """Subtraction operator on fractions

        Parameters:
            Fraction: fraction to compare against

        Returns:
            Fraction: result of subtracting fractions
        """
        #get common denom
        return Fraction(self.numer*other.denom - self.denom*other.numer,
            self.denom*other.denom)

    def __mul__(self, other):
        """Multiplication operator on fractions

        Parameters:
            Fraction: fraction to compare against

        Returns:
            Fraction: result of multiplying fractions
        """
        #multiply numers and multiply denoms
        return Fraction(self.numer*other.numer, self.denom*other.denom)

    def __truediv__(self, other):
        """Division operator on fractions

        Parameters:
            Fraction: fraction to compare against

        Returns:
            Fraction: result of dividing fractions
        """
        if self.denom*other.numer == 0:
            raise ZeroDivisionError("cannot divide by zero")
        #flip and multiply
        return Fraction(self.numer*other.denom, self.denom*other.numer)


# Problem 6
def count_sets(cards):
    """Return the number of sets in the provided Set hand.

    Parameters:
        cards (list(str)) a list of twelve cards as 4-bit integers in
        base 3 as strings, such as ["1022", "1122", ..., "1020"].
    Returns:
        (int) The number of sets in the hand.
    Raises:
        ValueError: if the list does not contain a valid Set hand, meaning
            - there are not exactly 12 cards,
            - the cards are not all unique,
            - one or more cards does not have exactly 4 digits, or
            - one or more cards has a character other than 0, 1, or 2.
    """
    if type(cards) != list:
        raise TypeError("Cards must be a list")
    if any([type(x) != str for x in cards]):
        raise TypeError("Cards must be base 3 ints cast as string")
    if any([len(x) != 4 for x in cards]):
        raise ValueError("Cards must be defined as 4 digit base 3")
    for x in cards:
        if any([char not in ['0', '1', '2'] for char in x]):
            raise ValueError("Cards must be base 3")
    if len(cards) != 12:
        raise ValueError("A hand must consist of 12 cards")
    #cast as set to make sure cards are unique
    if len(set(cards)) != 12:
        raise ValueError("Cards must be unique")
    #find all groups of 3 cards
    possible_sets = list(combinations(cards, 3))
    true_set_count = 0
    for possible_set in possible_sets:
        #check validity of set
        if is_set(possible_set[0], possible_set[1], possible_set[2]):
            true_set_count += 1
    return true_set_count


def is_set(a, b, c):
    """Determine if the cards a, b, and c constitute a set.

    Parameters:
        a, b, c (str): string representations of 4-bit integers in base 3.
            For example, "1022", "1122", and "1020" (which is not a set).
    Returns:
        True if a, b, and c form a set, meaning the ith digit of a, b,
            and c are either the same or all different for i=1,2,3,4.
        False if a, b, and c do not form a set.
    """
    for i in range(4):
        attribute_sum = int(a[i]) + int(b[i]) + int(c[i]) 
        #to be a set attributes in each column must be divisible by 3
        if attribute_sum % 3 != 0:
            return False
    return True

