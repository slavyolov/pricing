import numpy as np
from multi_armed_bandit.demand import get_reward, demand_curve, revenue_derivative, solve_demand_curve
from scipy.optimize import fsolve
import random


class MultiArmedBandit:
    def __init__(self, prices, alpha, beta, n_steps: int = 10000, n_epochs: int = 50, purchase_proba: list = None,
                 drift=True, change_at_step=10000):
        # Input parameters
        self.prices = prices  # Set of prices (arms)
        self.n_steps = n_steps  # Number of episodes in a single epoch
        self.n_epochs = n_epochs  # number of epochs to run the simulation
        self.purchase_proba = purchase_proba  # probabilities of sales for any of the
        self.alpha = alpha
        self.beta = beta

        # Add non-stationary
        self.drift = drift
        self.competitor_prices = [6.5]  # Putting once price for transparency, in reality it could be a set of prices
        self.updated_probabilities = self._competitor_discount()
        self.at_step = change_at_step

    def run_simulation(self, strategy):
        """
        Args:
            strategy:

        Returns:

        """
        # Optimal prices
        optimal_price = fsolve(func=revenue_derivative, x0=0, args=(self.alpha, self.beta))[0]
        optimal_probability = demand_curve(price=optimal_price, a=self.alpha, b=self.beta)

        # Define the things we want to return :
        reactivity = self.n_steps  # worst case scenario initialization
        react_counter = 10  # number of steps needed to confirm that the reactivity threshold has been hit
        cum_regret = np.zeros((self.n_steps,))
        avg_reward = 0
        arm_counter = np.zeros_like(self.prices, dtype=float)
        arm_avg_reward = np.zeros_like(self.prices, dtype=float)
        cum_reward = 0
        arm_counter_drift = np.zeros_like(self.prices, dtype=float)
        arm_counter_static = np.zeros_like(self.prices, dtype=float)

        print(f"Running strategy : {strategy}")
        print(f"Initial Purchase Probabilities : {self.purchase_proba}")
        print(F"The optimal price is : {optimal_price}")

        if strategy == "thompson":
            successes = np.zeros_like(self.prices, dtype=int)
            failures = np.zeros_like(self.prices, dtype=int)

        for iteration in range(self.n_steps):
            # print(f"Episode : {iteration}")
            if strategy == "greedy":
                arm = self.greedy(arm_avg_reward=arm_avg_reward)
            elif strategy.startswith("eps_greedy"):
                _, epsilon = strategy.split("-")
                arm = self.epsilon_greedy(arm_avg_reward=arm_avg_reward, epsilon=float(epsilon))
            elif strategy == "thompson":
                arm = self.thompson_sampling(arm_prices=self.prices, successes=successes, failures=failures,
                                             prices=self.prices)
            elif strategy.startswith("ucb"):
                try:
                    if strategy.endswith("-norm"):
                        normalize = True
                        _, C, _ = strategy.split("-")
                    else:
                        normalize = False
                        _, C = strategy.split("-")
                    C = float(C)
                except:
                    C = 1
                    normalize = False
                arm = self.UCB1(arm_avg_reward, arm_counter, iteration, C=C, normalize=normalize)

            # Introduce non-stationary in the probabilities (Currently only once at the half of the episode)
            if iteration >= self.at_step and self.drift:
                probabilities = self.updated_probabilities
            else:
                probabilities = self.purchase_proba

            if iteration == self.at_step and self.drift:
                # Take the price
                # We are not solving this by beta as we touched significantly the probabilities by our decay assumption
                # (check the '_competitor_discount' method)
                # Instead we take as the optimal price the one that is the lowest between the competitors. For example,
                # If the competitors price == 6.5 and our prices (arms) are [..., 5.5, 6, 7] - the optimal price is
                # going to be 6
                ix_ = np.where(np.array(self.prices) <= self.competitor_prices[0])[0]
                optimal_price = self.prices[ix_.max()]
                optimal_probability = demand_curve(price=optimal_price, a=self.alpha, b=self.beta)
                print(f"Purchase Probabilities were adjusted at episode {iteration} to : {probabilities}")
                print(F"The new optimal price is : {optimal_price}")

            # Reward is either 0 or 1 based on the binomial distribution
            reward = get_reward(probabilities[arm])

            # compute cumulative regret using the known optimal_price
            # Regret(t) = Optimal Reward (t) - Actual Reward (t)
            # Where :
            #   Optimal reward is : optimal_price * optimal_probability
            #   Actual reward is : selected price by the MAB * reward
            # TODO : CHECK THE self.optimal_price * self.optimal_probability (class param or not ? Maybe probs will update this)
            cum_regret[iteration] = cum_regret[iteration - 1] + (optimal_price * optimal_probability -
                                                                 self.prices[arm] * reward)

            if strategy == "thompson":
                if reward > 0:
                    successes[arm] += 1
                else:
                    failures[arm] += 1

            # update the value for the chosen arm using a running average
            arm_counter[arm] += 1
            if iteration >= self.at_step and self.drift:
                arm_counter_drift[arm] += 1

            if iteration < self.at_step and self.drift:
                arm_counter_static[arm] += 1

            reward *= self.prices[arm]
            arm_avg_reward[arm] = ((arm_counter[arm] - 1) * arm_avg_reward[arm] + reward) / arm_counter[arm]
            avg_reward = (iteration * avg_reward + reward) / (iteration + 1)
            cum_reward += reward

            # verify if the reactivity threshold has been hit
            if (iteration > 100 and react_counter != 0 and avg_reward >= 0.95 *
                    optimal_price * optimal_probability):
                react_counter -= 1
                if react_counter == 0:
                    reactivity = iteration + 1

        return cum_regret, reactivity, arm_counter, cum_reward, arm_counter_static, arm_counter_drift

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

    def _competitor_discount(self):
        # Define competitors price
        selling_price_competitor_x = random.choice(self.competitor_prices)  # TODO: Currently hardcoded but could be adjusted

        # Get the prices (arms) that will be affected
        filtered_prices = [price for price in self.prices if price >= selling_price_competitor_x - 0.5]

        # Adjust the probabilities
        ix = len(filtered_prices)
        affected_probabilities = self.purchase_proba[-ix:]

        # Define decreasing factor for the probabilities that decays with each next price from the minimal
        initial_factor = random.uniform(1.1, 1.15)

        # Update the probabilities
        for i in range(len(affected_probabilities)):
            if i > 0:
                initial_factor = initial_factor * 1.2
            print(f"denominator {initial_factor} probability was {affected_probabilities[i]} now : ",
                  affected_probabilities[i] / initial_factor)
            affected_probabilities[i] = affected_probabilities[i] / initial_factor

        affected_probabilities = [round(x, 4) for x in affected_probabilities]

        return self.purchase_proba[:-ix] + affected_probabilities

    @staticmethod
    def UCB1(arm_avg_reward, arm_counter, iteration, C=1, normalize=False):
        """
        Suitable for Bernoulli bandit problems

        Args:
            arm_avg_reward:
            arm_counter:
            iteration:
            C:
            normalize:

        Returns:

        """
        if np.all(arm_avg_reward == 0):
            # if we have not gotten any reward, go random
            arm = np.random.choice(len(arm_avg_reward))
            return arm
        if 0 in arm_counter:
            # if there's an arm that hasn't been pulled yet, pull it.
            arm = np.argmin(arm_counter)
            return arm
        # Total number of times any arm has been played
        total_plays = iteration + 1  # since iteration starts from 0

        if normalize:
            max_reward = arm_avg_reward.max()
            arm_norm_reward = arm_avg_reward / max_reward
            # Calculate upper bounds for all arms
            ucb_values = arm_norm_reward + C * np.sqrt(2 * np.log(total_plays) / arm_counter)
            ucb_values *= max_reward
        else:
            # calculate upper bounds for all arms
            ucb_values = arm_avg_reward + C * np.sqrt(2 * np.log(total_plays) / arm_counter)

        # Return the arm which has the maximum upper bound
        # print("ucb_values:", ucb_values)
        # print(f"iteration {iteration}, arm = {np.argmax(ucb_values)} ")
        return np.argmax(ucb_values)
