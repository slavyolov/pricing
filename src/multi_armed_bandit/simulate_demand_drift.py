import numpy as np

from multi_armed_bandit.demand import *

prices = [0, 1.99, 2.49, 2.99, 3.99, 4.99, 5.99]
a = 2
b = 0.2
initial_purchase_probabilities = [demand_curve(p, a, b).round(3) for p in prices]

#  Adjusted probability
best_price = 2.99
competitor_price = 2.99


def sigmoid_decay(x, k, x0):
    """

    Args:
        x: Input variable (price)
        k: Controls the steepness of the curve
        x0: this is the x-value of the sigmoid midpoint

    Returns:

    """
    return 1 / (1 + np.exp(k * (x - x0)))


# Parameters
k = 1 # TODO: search values between 0.1 and 0.5 (maybe we can select them at random
x0 = competitor_price  # TODO: to be selected based on the competitors price

# New probabilities
decay_values = sigmoid_decay(np.array(prices), 1, x0)

print("INITIAL proba : ", initial_purchase_probabilities)
print("UPDATED proba : ", decay_values)

# Can we calculate the optimal reward having these new probabilities ?

# AVERAGE :
expect_revenue_initial = prices * np.array(initial_purchase_probabilities)
expected_revenue_ = prices * decay_values

optimal_price = fsolve(func=revenue_derivative, x0=0, args=(a, b))[0]
# optimal_price = 6.39

np.linspace(start=0, stop=10, num=11000)

# NO
# x = 1.99
def sigmoid_derivative(x, k, x0):
    exp_term = np.exp(k * (x - x0))
    denominator = (exp_term + 1) ** 2
    # numerator = k * np.exp(k * (x-x0))
    # denominator = (1 + np.exp(k * (x-x0))) ** 2
    result = -k * exp_term / denominator
    return result

sigmoid_derivative(2, 1, 2.99)


equation = lambda x: sigmoid_decay(x, k, x0)

optimal_price = fsolve(func=equation, x0=x0)[0]



(1 + np.exp(k * (100 - x0))) ** 2 #

# Expected revenue of known points :
a = 2
b = 0.042
prices = [0, 10, 20, 30, 40]
prices = [0, 2, 3, 4, 5]
initial_purchase_probabilities = [demand_curve(p, a, b).round(3) for p in prices]
average_expected_revenue = [expected_revenue(price=price, a=a, b=b).round(3) for price in prices]

print(initial_purchase_probabilities)
print(average_expected_revenue)
print(initial_purchase_probabilities * np.array(prices))

# Sigmoid expected revenue :
k = 0.1  # TODO: search values between 0.1 and 0.5 (maybe we can select them at random
x0 = 20
sigmoid_probabilities = sigmoid_decay(x=np.array(prices), k=k, x0=x0)
print(sigmoid_probabilities)
print(sigmoid_probabilities * np.array(prices))


def logit_curve(a, b, price):
    return a / (1 + np.exp(a + b * price))


logit_probabilities = [logit_curve(a, b, p).round(3) for p in prices]
print(logit_probabilities)



# TODO : testing
solve_demand_curve(30, prob=0.5)


def sig(p, C, alpha, p_0):
    e = np.random.normal(10, 5, p.shape[0])
    # return (C / (1 + np.exp(alpha * (p - p_0)))) + e
    return (C / (1 + np.exp(alpha * (p - p_0))))

# a = 2
# C
# alpha = 0.0366204096222703
# p = 30
# p_0 = 50
# print((1 + np.exp(alpha * (p - p_0))))

#
#
#     return
#
# import math
# import sympy
# import numpy as np
#
# price = 5
# x = sympy.Symbol('x')
# y = 2 / (1 + sympy.exp(x * price))
# yprime = y.diff(x) # get derivative
# print(yprime) # print derivative
# print(y.diff(x, 5)) # print 5th derivative
#
#
# y.diff(sympy.exp(x**2), x)
#
#
# import sympy
#
# x = sympy.Symbol('x')
# y = (sympy.exp(-.0482 * x) * 100)
# yprime = y.diff(x) # get derivative
# print(yprime) # print derivative
# print(y.diff(x, 5)) # print 5th derivative
#
#
#
# from sympy import *
# import numpy as np
# x = Symbol('x')
# y = x**2 + 1
# yprime = y.diff(x)
# yprime
#
#
# from sympy.solvers import solve
# from sympy import Symbol
# x = Symbol('x')
# print(solve(x**2 - 1, x))
#
#
#
#
# # Solve equation :
# y = 2 / (1 + sympy.exp(x * 5))
# print(solve(Eq(2 / (1 + sympy.exp(x * 5)), 0), x))
#
# y = 2 / (1 + np.exp(0.0604561743745867 * 5))
# print("probability", (y))


# SEARCH :

x = np.linspace(-10, 10, 1000)

def sigmoid(x):
    return 1 / (1 + np.exp(x))

def __sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


y1 = sigmoid(x)
y2 = __sigmoid_derivative(x)

from matplotlib import pyplot as plt
plt.plot(x, y1, label='sigmoid')
plt.plot(x, y2, label='derivative')
plt.legend(loc='upper left')
plt.show()

def sigmoid_decay(x, k=1, x0=5):
    """

    Args:
        x: Input variable (price)
        k: Controls the steepness of the curve
        x0: this is the x-value of the sigmoid midpoint

    Returns:

    """
    return 1 / (1 + np.exp(k * (x - x0)))

def __sigmoid_derivative(x, k=1, x0=5):
    return sigmoid_decay(x, k, x0) * (1 - sigmoid_decay(x, k, x0))


x = np.linspace(0, 10, 10)

y1 = sigmoid_decay(x)
y2 = __sigmoid_derivative(x)

plt.plot(x, y1, label='sigmoid')
plt.plot(x, y2, label='derivative')
plt.legend(loc='upper left')
plt.show()