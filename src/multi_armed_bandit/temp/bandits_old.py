import numpy as np
from multi_armed_bandit.demand import get_reward


def greedy(arm_avg_reward):
    if np.all(arm_avg_reward == 0):
        # if we have not gotten any reward, go random
        arm = np.random.choice(len(arm_avg_reward))
        # print(f"Arm selected at random : {arm}")
    else:
        # else choose the arm with the highest average reward
        arm = np.argmax(arm_avg_reward)
        # print(f"Selected arm with the highest average reward : {arm}")

    return arm


def epsilon_greedy(arm_avg_reward, epsilon=0.1):
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


def UCB1(arm_avg_reward, arm_counter, iteration, C=1, normalize=False):
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
    return np.argmax(ucb_values)


def thompson_sampling(arm_prices, successes, failures, prices):
    # print(successes/(successes+failures))
    samples = [np.random.beta(successes[i] + 1, failures[i] + 1) for i in range(len(prices))]
    samples = [s * arm_prices[i] for i, s in enumerate(samples)]
    return np.argmax(samples)


def run_simulation(prices, optimal_price, optimal_probability, a, b, nstep, strategy="epsgreedy",
                   detailed_display=False):
    reactivity = nstep  # worst case scenario initialization
    react_counter = 10  # number of steps needed to confirm that the reactivity threshold has been hit
    cum_regret = np.zeros((nstep,))
    avg_reward = 0
    arm_counter = np.zeros_like(prices, dtype=float)
    arm_avg_reward = np.zeros_like(prices, dtype=float)
    cum_reward = 0

    if strategy == "thompson":
        successes = np.zeros_like(prices, dtype=int)
        failures = np.zeros_like(prices, dtype=int)

    for iteration in range(nstep):
        if strategy == "greedy":
            arm = greedy(arm_avg_reward)
        elif strategy == "epsgreedy":
            arm = epsilon_greedy(arm_avg_reward, epsilon=0.1)
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
            arm = UCB1(arm_avg_reward, arm_counter, iteration, C=C, normalize=normalize)
        elif strategy == "thompson":
            arm = thompson_sampling(prices, successes, failures, prices)

        # Halfway through the episode, change the probabilities
        # TODO: we can also change the demand function by adding some noise or something
        # TODO: this will also change the reward signal
        drift = False
        if drift and iteration == int(nstep / 2):
            print("Probabilities were adjusted")
            a = 2
            b = 0.032

            # if b==0 the customer is not sensitive to the price - wil always buy
            # if b==1 will never buy (this depends on ot the prices ofc)

            from multi_armed_bandit.demand import demand_curve
            purchase_probabilities = [demand_curve(p, a, b).round(3) for p in prices]
            print(f"The NEW associated purchase probabilities for the price candidates : {prices} are {purchase_probabilities}")

        # Reward is either 0 or 1 based on the binomial distribution
        reward = get_reward(prices[arm], a, b)

        # compute cumulative regret using the known optimal_price
        # Regret(t) = Optimal Reward (t) - Actual Reward (t)
        # Where :
        #   Optimal reward is : optimal_price * optimal_probability
        #   Actual reward is : selected price by the MAB * reward
        cum_regret[iteration] = cum_regret[iteration - 1] + (optimal_price * optimal_probability - prices[arm] * reward)

        if detailed_display:
            print(f"Iteration {iteration} out of {nstep - 1} - reward {prices[arm] * reward} - cumulative regret {cum_regret[iteration]}")

        if strategy == "thompson":
            if reward > 0:
                successes[arm] += 1
            else:
                failures[arm] += 1

                # update the value for the chosen arm using a running average
        arm_counter[arm] += 1
        reward *= prices[arm]
        arm_avg_reward[arm] = ((arm_counter[arm] - 1) * arm_avg_reward[arm] + reward) / arm_counter[arm]
        avg_reward = ((iteration) * avg_reward + reward) / (iteration + 1)
        cum_reward += reward

        # verify if the reactivity threshold has been hit
        if iteration > 200 and react_counter != 0 and avg_reward >= 0.95 * optimal_price * optimal_probability:
            react_counter -= 1
            if react_counter == 0:
                reactivity = iteration + 1

    return cum_regret, reactivity, arm_counter, cum_reward


#
# change the reward function

# RL : to construct experiments
# Can be used to validate the curve
    # Partially observed process ?
    # To build different curves ?