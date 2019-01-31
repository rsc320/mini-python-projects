# test_specs.py
"""Python Essentials: Unit Testing.
R Scott Collings
Math 321 Sec 002
19 Oct 2018
"""

import specs
import pytest


def test_add():
    """different types of addition"""
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8 #negative times positive

def test_divide():
    """test both integer and float divition"""
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0) #cant have 0 denominator
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    """test finding smallest prime factor of a number"""
    assert specs.smallest_factor(1)==1, "Failed on n=1"
    assert specs.smallest_factor(2)==2, "Failed on n=2"
    assert specs.smallest_factor(3)==3, "Failed on n=3"
    assert specs.smallest_factor(4)==2, "Failed on smallest square"
    assert specs.smallest_factor(10)==2, "Failed on even number"
    assert specs.smallest_factor(81)==3, "Failed on non-prime square" #tests square where root isn't prime
    assert specs.smallest_factor(49)==7, "Failed on prime square"
    assert specs.smallest_factor(23)==23, "Failed on large prime"
    assert specs.smallest_factor(63)==3, "Failed on odd"
    assert specs.smallest_factor(323)==17, "Failed on close primes"


# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    """Tests getting month length for various months and years"""
    assert specs.month_length('January', True) == 31, "Failed on leap-year Jan"
    assert specs.month_length('February', True) == 29, "Failed on leap-year Jan"
    assert specs.month_length('March', True) == 31, "Failed on leap-year Jan"
    assert specs.month_length('April', True) == 30, "Failed on leap-year Jan"
    assert specs.month_length('May', True) == 31, "Failed on leap-year Jan"
    assert specs.month_length('June', True) == 30, "Failed on leap-year Jan"
    assert specs.month_length('July', True) == 31, "Failed on leap-year Jan"
    assert specs.month_length('August', True) == 31, "Failed on leap-year Jan"
    assert specs.month_length('September', True) == 30, "Failed on leap-year Jan"
    assert specs.month_length('October', True) == 31, "Failed on leap-year Jan"
    assert specs.month_length('November', True) == 30, "Failed on leap-year Jan"
    assert specs.month_length('December', True) == 31, "Failed on leap-year Jan"
    assert specs.month_length('January') == 31, "Failed on non-leap-year Jan"
    assert specs.month_length('February') == 28, "Failed on non-leap-year Jan"
    assert specs.month_length('March') == 31, "Failed on non-leap-year Jan"
    assert specs.month_length('April') == 30, "Failed on non-leap-year Jan"
    assert specs.month_length('May') == 31, "Failed on non-leap-year Jan"
    assert specs.month_length('June') == 30, "Failed on non-leap-year Jan"
    assert specs.month_length('July') == 31, "Failed on non-leap-year Jan"
    assert specs.month_length('August') == 31, "Failed on non-leap-year Jan"
    assert specs.month_length('September') == 30, "Failed on non-leap-year Jan"
    assert specs.month_length('October') == 31, "Failed on non-leap-year Jan"
    assert specs.month_length('November') == 30, "Failed on non-leap-year Jan"
    assert specs.month_length('December') == 31, "Failed on non-leap-year Jan"
    assert specs.month_length('Not a month', True) == None, "Failed on non-month leap-year" #tests non-month
    assert specs.month_length('Not a month') == None, "Failed on non-month non-leap-year"


# Problem 3: write a unit test for specs.operate().
def test_operate():
    """tests operations on two numbers"""
    assert specs.operate(1,2,'+') == 3, "Failed on positive + positive"
    assert specs.operate(1,0,'+') == 1, "Failed on positive + zero"
    assert specs.operate(1,-2,'+') == -1, "Failed on positive + negative"
    assert specs.operate(0,0,'+') == 0, "Failed on zero + zero"
    assert specs.operate(1,2,'-') == -1, "Failed on positive + positive"
    assert specs.operate(1,0,'-') == 1, "Failed on positive + positive"
    assert specs.operate(1,-2,'-') == 3, "Failed on positive + positive"
    assert specs.operate(0,0,'-') == 0, "Failed on positive + positive"
    assert specs.operate(1,2,'*') == 2, "Failed on positive + positive"
    assert specs.operate(1,0,'*') == 0, "Failed on positive + positive"
    assert specs.operate(1,-2,'*') == -2, "Failed on positive + positive"
    assert specs.operate(0,0,'*') == 0, "Failed on positive + positive"
    assert specs.operate(1,2,'/') == 0.5, "Failed on positive + positive"
    assert specs.operate(1,-2,'/') == -0.5, "Failed on positive + positive"
    assert specs.operate(-1,2,'/') == -0.5, "Failed on positive + positive"
    assert specs.operate(0,2,'/') == 0, "Failed on positive + positive"
    pytest.raises(ZeroDivisionError, specs.operate, a=1, b=0, oper='/')
    pytest.raises(ZeroDivisionError, specs.operate, a=0, b=0, oper='/')
    pytest.raises(TypeError, specs.operate, a=1, b=2, oper=9) #tests non-operator
    pytest.raises(ValueError, specs.operate, a=1, b=2, oper='test')

# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    """creates some test fractions for other test functions"""
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3 #return several fraction to use

def test_fraction_init(set_up_fractions):
    """tests proper initialization of fractions objects"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42) # 30/42 reduces to 5/7.
    assert frac.numer == 5
    assert frac.denom == 7
    with pytest.raises(ZeroDivisionError) as err:
        specs.Fraction(1,0)
    assert err.value.args[0] == "denominator cannot be zero"
    with pytest.raises(TypeError) as err1:
        specs.Fraction(1.2, 1.3)
    assert err1.value.args[0] == "numerator and denominator must be integers"

def test_fraction_str(set_up_fractions):
    """tests fractions represented as strings"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    assert str(specs.Fraction(2,1)) == '2' #no denominator

def test_fraction_float(set_up_fractions):
    """tests fractions represented as floats"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3. #float division to compare with

def test_fraction_eq(set_up_fractions):
    """tests comparing fractions"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)
    assert frac_1_3 == float(1 / 3) #compare equivalence to floats as well

def test_fraction_add(set_up_fractions):
    """tests adding fractions"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    #compare to fraction that should be equal
    assert frac_1_2 + frac_1_3 == specs.Fraction(5, 6)
    assert frac_1_3 + frac_n2_3 == specs.Fraction(-1, 3)
    assert frac_n2_3 + frac_1_3 + frac_1_3 == specs.Fraction(0, 1)

def test_fraction_sub(set_up_fractions):
    """tests subtracting fractions"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    #compare to fraction that should be equal
    assert frac_1_2 - frac_1_3 == specs.Fraction(1, 6)
    assert frac_1_3 - frac_n2_3 == specs.Fraction(1, 1)
    assert frac_n2_3 - frac_n2_3 == specs.Fraction(0, 1)

def test_fraction_mul(set_up_fractions):
    """tests multiplying fractions"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    #compare to fraction that should be equal
    assert frac_1_2 * frac_1_3 == specs.Fraction(1, 6)
    assert frac_1_3 * frac_n2_3 == specs.Fraction(-2, 9)
    assert frac_n2_3 * specs.Fraction(0,1) == specs.Fraction(0, 1)

def test_fraction_truediv(set_up_fractions):
    """tests dividing fractions"""
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    #compare to fraction that should be equal
    assert frac_1_2 / frac_1_3 == specs.Fraction(3, 2)
    assert frac_1_3 / frac_n2_3 == specs.Fraction(-1, 2)
    assert specs.Fraction(0,1) / frac_n2_3 == specs.Fraction(0, 1)
    pytest.raises(ZeroDivisionError, specs.Fraction.__truediv__, frac_1_2, specs.Fraction(0,1))


# Problem 5: Write test cases for Set.
@pytest.fixture
def set_up_hand():
    """creates some test hands to test the cases for SET"""
    hand1 = ['1022', '1122', '0100', '2021', '0010', '2201', '2111', '0020', '1102', '0210', '2110', '1020']
    hand2 = ['1122', '0100', '2021', '0010', '2201', '2111', '0020', '1102', '0210', '2110', '1020']
    hand3 = ['102', '1122', '0100', '2021', '0010', '2201', '2111', '0020', '1102', '0210', '2110', '1020']
    hand4 = ['1022', '1022', '0100', '2021', '0010', '2201', '2111', '0020', '1102', '0210', '2110', '1020']
    hand5 = ['1022', '1422', 'a100', '2021', '0010', '2201', '2111', '0020', '1102', '0210', '2110', '1020']
    hand6 = ['1022', '1122', '0100', '2021', '0010', '2201', '2111', '0020', '1102', '0210', '2110', 1020]
    hand7 = '1022'
    #return all hands as tuple
    return hand1, hand2, hand3, hand4, hand5, hand6, hand7

def test_count_sets(set_up_hand):
    """test cases for counting possible sets in a hand"""
    hand1, hand2, hand3, hand4, hand5, hand6, hand7 = set_up_hand
    pytest.raises(ValueError, specs.count_sets, hand2) #not enough cards
    pytest.raises(ValueError, specs.count_sets, hand3) #one string not a card
    pytest.raises(ValueError, specs.count_sets, hand4) #two identitcal cards
    pytest.raises(ValueError, specs.count_sets, hand5) #card not represented as base 3
    pytest.raises(TypeError, specs.count_sets, hand6) #has a non-string card
    pytest.raises(TypeError, specs.count_sets, hand7) #not a list
    assert specs.count_sets(hand1) == 6

def test_is_set():
    """tests finding whether three cards is a set"""
    assert specs.is_set('2021', '2201', '2111') == True, "Failed on two different"
    #case with not a set
    assert specs.is_set('1022', '1122', '0100') == False, "Failed on false set"
    assert specs.is_set('1111', '2222', '0000') == True, "Failed on all different"
    assert specs.is_set('0120', '0121', '0122') == True, "Failed on one different"
    assert specs.is_set('0111', '0000', '0222') == True, "Failed on three different"