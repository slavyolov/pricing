import numpy as np
from multi_armed_bandit.demand import get_reward, demand_curve, revenue_derivative
from scipy.optimize import fsolve


class MultiArmedBandit:
    def __init__(self, prices, alpha, beta, n_steps: int = 10000, n_epochs: int = 50, purchase_proba: list = None):
        # Input parameters
        self.prices = prices  # Set of prices (arms)
        self.n_steps = n_steps  # Number of episodes in a single epoch
        self.n_epochs = n_epochs  # number of epochs to run the simulation
        self.purchase_proba = purchase_proba  # probabilities of sales for any of the
        self.alpha = alpha
        self.beta = beta

        # Price parameters
        self.optimal_price = fsolve(func=revenue_derivative, x0=0, args=(self.alpha, self.beta))[0]
        self.optimal_probability = demand_curve(price=self.optimal_price, a=self.alpha, b=self.beta)

        # Bandit parameters
        self.reactivity = self.n_steps  # worst case scenario initialization
        self.react_counter = 10  # number of steps needed to confirm that the reactivity threshold has been hit
        self.cum_regret = np.zeros((self.n_step,))
        self.avg_reward = 0
        self.arm_counter = np.zeros_like(prices, dtype=float)
        self.arm_avg_reward = np.zeros_like(prices, dtype=float)
        self.cum_reward = 0

    def run(self, strategy):
        if strategy == "thompson":
            successes = np.zeros_like(self.prices, dtype=int)
            failures = np.zeros_like(self.prices, dtype=int)

        for iteration in range(self.n_steps):
            if strategy == "greedy":
                arm = self.greedy(arm_avg_reward=self.arm_avg_reward)
            elif strategy == "eps_greedy":
                arm = self.epsilon_greedy(arm_avg_reward=self.arm_avg_reward, epsilon=0.1)
            elif strategy == "thompson":
                arm = self.thompson_sampling(arm_prices=self.prices, successes=successes, failures=failures,
                                             prices=self.prices)

            # Reward is either 0 or 1 based on the binomial distribution
            reward = get_reward(self.prices[arm], a=self.alpha, b=self.beta)

            # compute cumulative regret using the known optimal_price
            # Regret(t) = Optimal Reward (t) - Actual Reward (t)
            # Where :
            #   Optimal reward is : optimal_price * optimal_probability
            #   Actual reward is : selected price by the MAB * reward
            self.cum_regret[iteration] = self.cum_regret[iteration - 1] + (
                        self.optimal_price * self.optimal_probability - self.prices[arm] * reward)

            if strategy == "thompson":
                if reward > 0:
                    successes[arm] += 1
                else:
                    failures[arm] += 1

            # update the value for the chosen arm using a running average
            self.arm_counter[arm] += 1
            reward *= self.prices[arm]
            self.arm_avg_reward[arm] = ((self.arm_counter[arm] - 1) * self.arm_avg_reward[arm] + reward) / self.arm_counter[arm]
            self.avg_reward = (iteration * self.avg_reward + reward) / (iteration + 1)
            self.cum_reward += reward

            # verify if the reactivity threshold has been hit
            if (iteration > 100 and self.react_counter != 0 and self.avg_reward >= 0.95 *
                    self.optimal_price * self.optimal_probability):
                self.react_counter -= 1
                if self.react_counter == 0:
                    reactivity = iteration + 1

        return self.cum_regret, reactivity, self.arm_counter, self.cum_reward

    def greedy(self, arm_avg_reward):
        if np.all(arm_avg_reward == 0):
            # if we have not gotten any reward, go random
            arm = np.random.choice(len(arm_avg_reward))
            # print(f"Arm selected at random : {arm}")
        else:
            # else choose the arm with the highest average reward
            arm = np.argmax(arm_avg_reward)
            # print(f"Selected arm with the highest average reward : {arm}")

        return arm

    def epsilon_greedy(self, arm_avg_reward, epsilon=0.1):
        if np.random.rand() < epsilon:
            # with probability epsilon choose a random arm
            arm = np.random.choice(len(arm_avg_reward))
        elif np.all(arm_avg_reward == 0):
            # if we have not gotten any reward, go random
            arm = np.random.choice(len(arm_avg_reward))
        else:
            # else choose the arm with the highest average reward
            arm = np.argmax(arm_avg_reward)
        return arm

    def thompson_sampling(self, arm_prices, successes, failures, prices):
        samples = [np.random.beta(successes[i] + 1, failures[i] + 1) for i in range(len(prices))]
        samples = [s * arm_prices[i] for i, s in enumerate(samples)]
        return np.argmax(samples)
