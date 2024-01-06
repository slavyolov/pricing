""""""

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import bernoulli


def demand_curve(price, a=0.5, b=0.05):
    # demand curve modeled via logisti function
    return a / (1 + np.exp(b * price))

def revenue_derivative(x, a=0.5, b=0.05):
    # derivative of the function x*demand_curve
    return a / (np.exp(b * x) + 1) - (a * b * x * np.exp(b * x)) / (np.exp(b * x) + 1) ** 2

def get_optimal_price(a=0.5, b=0.05):
    # find the root of the revenue derivative
    return fsolve(revenue_derivative, 0, args=(a, b))[0]

def get_reward(price, a=0.5, b=0.05):
    """
    If the probability is higher the changes of having a reward (1) increases and vice versa

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

    """
    # reward is 1 or 0 based on a Bernoulli distribution whose 'p' depends on the demand vurve
    prob = demand_curve(price, a, b)
    return bernoulli.rvs(prob)

def expected_revenue(price, a, b):
    return (a * price) / (1 + np.exp(b * price))
