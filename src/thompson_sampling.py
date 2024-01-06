import numpy as np
from tabulate import tabulate
from scipy.optimize import linprog
import scipy.stats as stats
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation

np.set_printoptions(precision=2)

def tabprint(msg, A):
    print(msg)
    print(tabulate(A, tablefmt="fancy_grid"))

plt.rcParams.update({'font.family':'Candara', 'font.serif':['Candara']})
plt.rcParams.update({'pdf.fonttype': 'truetype', 'font.size': 18})


class ThompsonSampling:
    def __init__(self):
        # Hidden (true) demand parameters - a linear demand function is assumed
        self.demand_a = 50
        self.demand_b = 7
        self.epochs = 50

    def run(self):
        prices = [1.99, 2.49, 2.99, 3.49, 3.99, 4.49]

        # prior distribution for each price - gamma(α, β)
        theta = []
        for p in prices:
            theta.append({'price': p, 'alpha': 30.00, 'beta': 1.00, 'mean': 30.00})

        history = []
        for t in range(0, self.epochs):  # simulation loop
            demands = self.sample_demands_from_model(theta)

            # Display in tabular form the [price, alpha, beta, mean]
            print(tabulate(np.array(theta), tablefmt="fancy_grid"))

            print("demands = ", np.array(demands))

            price_probs = self.optimal_price_probabilities(prices, demands, 60)

            # select one best price
            price_index_t = np.random.choice(len(prices), 1, p=price_probs)[0]
            price_t = prices[price_index_t]

            # sell at the selected price and observe demand
            demand_t = self.sample_demand(price_t)
            print('selected price %.2f => demand %.2f, revenue %.2f' % (price_t, demand_t, demand_t * price_t))

            theta_trace = []
            for v in theta:
                theta_trace.append(v.copy())
            history.append([price_t, demand_t, demand_t * price_t, theta_trace])

            # update model parameters
            v = theta[price_index_t]
            v['alpha'] = v['alpha'] + demand_t
            v['beta'] = v['beta'] + 1
            v['mean'] = v['alpha'] / v['beta']

            print("")

    @staticmethod
    def gamma(alpha, beta):
        """
        
        Args:
            alpha:
            beta:

        Returns:

        """
        shape = alpha
        scale = 1 / beta
        return np.random.gamma(shape, scale)

    def sample_demand(self, price):
        demand = self.demand_a - self.demand_b * price
        return np.random.poisson(demand, 1)[0]

    def sample_demands_from_model(self, theta):
        return list(map(lambda v: self.gamma(v['alpha'], v['beta']), theta))

    # Find the optimal distribution of prices (price probabilities) given fixed price levels,
    # corresponding demand levels, and availbale product inventory.
    #
    # Inputs:
    #   prices, demands, and revenues are vectors (i-th element corresponds to i-th price level)
    #   inventory is a scalar (number of availbale units)
    def optimal_price_probabilities(self, prices, demands, inventory):
        revenues = np.multiply(prices, demands)

        L = len(prices)
        M = np.full([1, L], 1)
        B = [[1]]
        Df = [demands]

        res = linprog(-np.array(revenues).flatten(),
                      A_eq=M,
                      b_eq=B,
                      A_ub=Df,
                      b_ub=np.array([inventory]),
                      bounds=(0, None))

        price_prob = np.array(res.x).reshape(1, L).flatten()
        return price_prob
