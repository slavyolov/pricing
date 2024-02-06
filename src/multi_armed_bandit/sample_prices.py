"""
Pick at random prices from a set of prices
"""

import numpy as np


def sample_from_initial(low, high):
    """
    Sample prices at random within given range

    Args:
        low:
        high:

    Returns:
       Array with 3 sampled prices

    """
    # get random values at random between a price range
    random_prices = np.random.uniform(low=low, high=high, size=10).round(2)
    elem_differences = [np.abs(random_prices[i] - random_prices[i - 1]) for i in range(1, len(random_prices))]

    # Filter the initial prices array with elements that differ at least 0.5 cents from the original prices
    filter_arr = filter_clause(elem_differences, base_diff=0.5)

    random_prices = random_prices[:-1]
    random_prices = random_prices[filter_arr]

    # get unique prices
    random_prices = np.unique(random_prices)

    # Select 3 prices fom the random sample(min, median, max)
    return np.array([random_prices.min(), np.median(random_prices), random_prices.max()])


def filter_clause(input_arr, base_diff=0.5):
    """
    Apply array filter to select prices at random where the i'th element is different from the i'th - 1 element

    Args:
        input_arr:
        base_diff:

    Returns:

    """
    filter_arr = []
    for diff in input_arr:
        if diff >= base_diff:
            filter_arr.append(True)
        else:
            filter_arr.append(False)
    return filter_arr
