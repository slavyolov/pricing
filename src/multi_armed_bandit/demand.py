""""""

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import bernoulli
import sympy
from sympy.solvers import solve


def demand_curve(price: int, a: float = 0.5, b: float = 0.05):
    """
    Demand curve is modeled via logistic function that represents the probability of purchase rather than the demand
    directly.

    Args:
        price: price value
        a: the maximum achievable probability of purchase. The value of 'a' stays within the range of [0, 2]
            this typically stays at 2 so that when the price is equal to 0 we have a probaility of purchase equal to 1
        b: modulates the sensitivity of the demand curve against price changes.
            Value of 0 gives unfirorm probability of purchase equal to 1
            Value of 0.1 for the price levels [0, 20, 30, 40, 50] gives probability of
                purchase : [1.0, 0.9, 0.803, 0.709, 0.62, 0.538]
            Value of 0.1 for the price levels [0, 20, 30, 40, 50] gives probability of
                purchase : [1.0, 0.538, 0.238, 0.095, 0.036, 0.013]
            Value of 0.1 for the price levels [0, 20, 30, 40, 50] gives probability of
                purchase : [1.0, 0.238, 0.036, 0.005, 0.001, 0.0]
            Value of 1 for the price levels [0.49, 0.99, 1.49, 1.99, 2.49, 2.99] gives probability of
                purchase : [0.76, 0.542, 0.368, 0.241, 0.153, 0.096]
            Value of 0.1 for the price levels [0.49, 0.99, 1.49, 1.99, 2.49, 2.99] gives probability of
                purchase : [0.976, 0.951, 0.926, 0.901, 0.876, 0.852]
    Examples :
        >>> prices = [0, 20, 40, 60, 80, 100]
        # >>> prices = [0.49, 0.99, 1.49, 1.99, 2.49, 2.99]
        >>> a = 2
        >>> b = 0
        >>> purchase_probabilities = [demand_curve(price=price, a=a, b=b).round(3) for price in prices]
        >>> print(purchase_probabilities)

    Returns:
        Probability of purchase at a price point
    """
    return a / (1 + np.exp(b * price))


def revenue_derivative(x, a=0.5, b=0.05):
    # derivative of the function x*demand_curve
    return a / (np.exp(b * x) + 1) - (a * b * x * np.exp(b * x)) / (np.exp(b * x) + 1) ** 2


def get_optimal_price(a=0.5, b=0.05):
    """
    >>> from scipy.optimize import fsolve
    >>> a = 2
    >>> b = 0.042
    >>> print(fsolve(revenue_derivative, 0, args=(a, b))[0])
    Args:
        a:
        b:

    Returns:

    """
    # find the root of the revenue derivative
    return fsolve(revenue_derivative, 0, args=(a, b))[0]


def get_reward(price, a=0.5, b=0.05):
    """
    The reward is either 0 or 1 based on a Bernoulli distribution whose 'p' depends on the demand curve

    Args:
        price:
        a:
        b:

    Example :
        >>> from scipy.stats import bernoulli
        >>> probability = 0.1
        >>> reward = 0
        >>> for i in range(1,100):
        ...     reward += bernoulli.rvs(probability)
        >>> print(reward)

    Returns:
        Reward value. Either 0 (no purchase made at the current price) or 1 (purchase is made)
    """
    prob = demand_curve(price, a, b)
    return bernoulli.rvs(prob)


def expected_revenue(price, a, b):
    """

    Args:
        price:
        a:
        b:

    Examples:
        >>> prices = [0, 20, 30, 40, 50]
        >>> a = 2
        >>> b = 0.042
        >>> average_expected_revenue = [expected_revenue(price=price, a=a, b=b).round(3) for price in prices]
        >>> print(average_expected_revenue)

    Returns:

    """
    return (a * price) / (1 + np.exp(b * price))


def solve_demand_curve(price, prob=0.5):
    """
    Find the beta when alpha (2) and the probability of the event (pron) are given

    Args:
        price: current selling price
        prob: probability of event at the current selling price

    Returns:

    """
    b = sympy.Symbol('b')
    y = 2 / (1 + sympy.exp(b * price))  # This is the demand function equation # TODO: check how to put alpha value here
    solutions = solve(sympy.Eq(y, prob), b)

    # The possible solutions are :
    print(f"Solving the equation {y} where price = {price} are : \n {solutions}")

    return solutions[0]
