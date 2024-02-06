"""
Utility demand functions needed for the Bernoulli Multi-Armed bandit problem
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import bernoulli
from sympy import exp, Eq, Symbol
from sympy.solvers import solve


def demand_curve(price: int, a: float = 0.5, b: float = 0.05):
    """
    Demand curve is modeled via logistic function that represents the probability of purchase rather than the demand
    directly.

    Args:
        price: price value
        a: the maximum achievable probability of purchase. The value of 'a' stays within the range of [0, 2]
            this typically stays at a value close to 2 so that when the price is equal to 0 we have a probaility of
            purchase equal to 1.
        b: modulates the sensitivity of the demand curve against price changes. Beta value must be taken based on the
            data so that we have representative curve. Beta could be obtained if we have observed probability and solve
            the demand equation for beta. See method "multi_armed_bandit.demand.solve_demand_curve" for more details.

    Examples :
        >>> prices = [0, 20, 40, 60, 80, 100]
        >>> a = 2
        >>> b = 0.1
        >>> purchase_probabilities = [demand_curve(price=price, a=a, b=b).round(3) for price in prices]
        >>> print(purchase_probabilities)
        [1.0, 0.238, 0.036, 0.005, 0.001, 0.0]
        >>> prices = [0, 20, 40, 60, 80, 100]
        >>> a = 2
        >>> b = 0.042
        >>> purchase_probabilities = [demand_curve(price=price, a=a, b=b).round(3) for price in prices]
        >>> print(purchase_probabilities)
        [1.0, 0.603, 0.314, 0.149, 0.067, 0.03]
        >>> prices = [0.49, 0.99, 1.49, 1.99, 2.49, 2.99]
        >>> a = 2
        >>> b = 0.042
        >>> purchase_probabilities = [demand_curve(price=price, a=a, b=b).round(3) for price in prices]
        >>> print(purchase_probabilities)
        [0.99, 0.979, 0.969, 0.958, 0.948, 0.937]

    Returns:
        Probability of purchase at a price point
    """
    return a / (1 + np.exp(b * price))


def revenue_derivative(price, a=0.5, b=0.05):
    """
    Use combination of both the product and the chain rule to get the demand derivative

    Use the revenue derivative to find the exact price that maximizes the expected revenue curve. This formula
    determines the price that sets it to 0. This, in turn, is the price that maximizes the expected revenue.

    Args:
        price: price value
        a: the maximum achievable probability of purchase. The value of 'a' stays within the range of [0, 2]
            this typically stays at a value close to 2 so that when the price is equal to 0 we have a probaility of
            purchase equal to 1.
        b: modulates the sensitivity of the demand curve against price changes. Beta value must be taken based on the
            data so that we have representative curve. Beta could be obtained if we have observed probability and solve
            the demand equation for beta. See method "multi_armed_bandit.demand.solve_demand_curve" for more details.

    Returns:

    Examples:
        In the given examples the price that maximizes our expected revenue is 32 euro (the derivative is closer to 0)
        >>> revenue_derivative(price=32, a=2, b=0.04)
        -0.0006681897704714501
        >>> revenue_derivative(price=10, a=2, b=0.04)
        0.6104160831818727
    """
    # derivative of the function x*demand_curve
    return a / (np.exp(b * price) + 1) - (a * b * price * np.exp(b * price)) / (np.exp(b * price) + 1) ** 2


def get_optimal_price(a=0.5, b=0.05):
    """
    Apply solver to get the price that maximizes the expected revenue

    Args:
        a: the maximum achievable probability of purchase. The value of 'a' stays within the range of [0, 2]
            this typically stays at a value close to 2 so that when the price is equal to 0 we have a probaility of
            purchase equal to 1.
        b: modulates the sensitivity of the demand curve against price changes. Beta value must be taken based on the
            data so that we have representative curve. Beta could be obtained if we have observed probability and solve
            the demand equation for beta. See method "multi_armed_bandit.demand.solve_demand_curve" for more details.

    Returns:
        The optimal price at given alpha and beta parameters

    Examples:
        >>> get_optimal_price(a=2, b=0.042)
        30.439631970501754
    """
    # find the root of the revenue derivative
    return fsolve(revenue_derivative, 0, args=(a, b))[0]


def get_reward(price, a=0.5, b=0.05):
    """
    The reward is either 0 or 1 based on a Bernoulli distribution whose 'p' depends on the demand curve

    Args:
        price: price value
        a: the maximum achievable probability of purchase. The value of 'a' stays within the range of [0, 2]
            this typically stays at a value close to 2 so that when the price is equal to 0 we have a probaility of
            purchase equal to 1.
        b: modulates the sensitivity of the demand curve against price changes. Beta value must be taken based on the
            data so that we have representative curve. Beta could be obtained if we have observed probability and solve
            the demand equation for beta. See method "multi_armed_bandit.demand.solve_demand_curve" for more details.

    Returns:
        Reward value. Either 0 (no purchase made at the current price) or 1 (purchase is made)
    """
    prob = demand_curve(price, a, b)
    return bernoulli.rvs(prob)


def expected_revenue(price, a, b):
    """
    Get the average expected revenue at a given price based on the alpha and beta parameters

    Args:
        price:
        a:
        b:

    Returns:
        The average expected revenue

    Examples:
        >>> x = expected_revenue(price=32.08, a=2, b=0.04)
        13.923105289214105
    """
    return (a * price) / (1 + np.exp(b * price))


def solve_demand_curve(price, alpha=2, prob=0.5, display=False) -> float:
    """
    Find the beta when alpha (2) and the probability of the event (pron) are given

    Args:
        price: current selling price
        alpha: the maximum achievable probability of purchase
        prob: probability of event at the current selling price

    Returns:
        Get the first solution of the equation (this is the beta)

    Examples:
        >>> solve_demand_curve(price=10, alpha=2, prob=0.289707)
        0.17755499124375104
    """
    b = Symbol('b')
    y = alpha / (1 + exp(b * price))  # This is the demand function equation
    solutions = solve(Eq(y, prob), b)

    # The possible solutions are :
    if display:
        print(f"Solving the equation {y} where price = {price} are : \n {solutions}")

    return float(solutions[0])
