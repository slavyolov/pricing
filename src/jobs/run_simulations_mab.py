import pandas as pd
from src.multi_armed_bandit.demand import solve_demand_curve, demand_curve
from pathlib import Path
from multi_armed_bandit.bandits import MultiArmedBandit
import numpy as np
from src.multi_armed_bandit.sample_prices import sample_from_initial
from matplotlib import pyplot as plt


check_probabilities = True


def get_prices(alpha):
    # TODO: maybe put this into a function get prices or class
    # Read data
    path = str(Path(Path(__file__).parents[2], "data/input/sales.csv"))
    df = pd.read_csv(path)

    # Get the average conversion rate per price (this is the probability of selling on average)
    avg_proba_observed = df.groupby("price")["conversion_rate"].agg("mean").values

    # product cost (this defines the minimum price for our problem. Below this value we are generating loses)
    product_costs = 5.0

    # get the prices from the observed data
    prices = df["price"].sort_values().unique()

    # Solve the demand equation for beta taking the midpoint of the probabilities
    midpoint_proba_observed = avg_proba_observed[1]
    beta = solve_demand_curve(price=np.median(prices), alpha=alpha, prob=midpoint_proba_observed)
    print("Beta coefficient:", beta)

    if check_probabilities:
        purchase_probabilities = [demand_curve(price=price, a=alpha, b=beta).round(3) for price in prices]
        print(f"Observed purchase probabilities : ", list(avg_proba_observed.round(3)))
        print(f"Observed data standard deviation : ",
              list(df.groupby("price")["conversion_rate"].agg("std").values.round(3)))
        print(f"Estimated purchase probabilities using beta = {beta} : ", purchase_probabilities)

    # Expand the price set to enable the MAB to test for other prices when searching for the best one
    # Get 10 prices (including the original 3)
    break_even_price = np.array([product_costs])
    wave_1 = sample_from_initial(low=5.25, high=7.25)  # At least .25 cents diff from min_possible and min_observed
    wave_2 = sample_from_initial(low=7.75, high=9.75)  # At least .25 cents diff between 7.5 and 10

    new_prices = np.concatenate((break_even_price, wave_1, wave_2, prices))
    new_prices.sort()
    new_prices = list(new_prices)

    # Sanity check (prices and probabilities)
    if check_probabilities:
        purchase_probabilities = [demand_curve(price=price, a=alpha, b=beta).round(3) for price in prices]
        print(f"Estimated purchase probabilities using beta = {beta} on new set of prices : ", purchase_probabilities)

    return new_prices, beta


if __name__ == "__main__":
    # Set the max probability at 95% to address the chance of missing the offer even if the price is equal to 0
    alpha = 1.9
    prices, beta = get_prices(alpha=alpha)

    # Run the simulation :
    regret_curves = {}
    strategies = ["eps_greedy"]
    n_steps = 100
    n_epochs = 1
    detailed_display = True
    bandit = MultiArmedBandit(prices=prices, alpha=alpha, beta=beta, n_steps=n_steps, n_epochs=n_epochs)

    for strategy in strategies:
        regret_curves[strategy] = np.zeros((n_steps,))
        regrets = []
        reactivities = []
        arm_counters = np.zeros((len(prices),))
        for ep in range(n_epochs):
            if detailed_display:
                print("=" * 100)
                print(f"Running epoch {ep} out of {n_epochs - 1}")
                print("=" * 100)
            regret, reactivity, arm_counter, cum_reward = bandit.run_simulation(strategy=strategy)
            regret_curves[strategy] += regret
            regrets.append(regret[-1])
            reactivities.append(reactivity)
            arm_counters += arm_counter / n_steps
            print(f"Epoch {ep} out of {n_epochs - 1} completed")

        regret_curves[strategy] /= n_epochs
        arm_allocation = 100 * arm_counters / n_epochs
        print("-------------\nStrategy: %s" % strategy)
        print("Regret -> mean: %.2f, median: %.2f, std: %.2f" % (np.mean(regrets), np.median(regrets), np.std(regrets)))
        print("Reactivity -> mean: %.2f, median: %.2f, std: %.2f" % (
            np.mean(reactivities), np.median(reactivities), np.std(reactivities)))
        print("Arm allocation -> %s" % (arm_allocation))
        print("Cumulative reward is : ", cum_reward)


    plt.figure(figsize=(12, 6))
    for label in regret_curves:
        plt.plot(regret_curves[label], label=label)
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Simulation done!")
