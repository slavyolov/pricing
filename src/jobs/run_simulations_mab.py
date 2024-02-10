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


def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data


if __name__ == "__main__":
    # Set the max probability at 95% to address the chance of missing the offer even if the price is equal to 0
    alpha = 1.9
    prices, beta = get_prices(alpha=alpha)

    # Get the probabilities
    purchase_probabilities = [demand_curve(p, a=alpha, b=beta).round(4) for p in prices]

    # Test the solution
    test = True
    if test == True:
        alpha = 2
        beta = 0.042
        prices = [20, 30, 40, 50, 60]
        purchase_probabilities = [demand_curve(p, a=alpha, b=beta).round(4) for p in prices]

    # Run the simulation :
    regret_curves = {}
    strategies = [
        "greedy",
        "eps_greedy-0.1",
        "eps_greedy-0.2",
        "thompson",
        "ucb1-0.7-norm",
        "ucb1-1-norm",
        "ucb1-1"
    ]
    n_steps = 100
    n_epochs = 3
    detailed_display = True
    change_at_step = n_steps / 2  # in case we want to add drift
    bandit = MultiArmedBandit(prices=prices, alpha=alpha, beta=beta, n_steps=n_steps, n_epochs=n_epochs,
                              purchase_proba=purchase_probabilities,
                              drift=False,
                              change_at_step=change_at_step
                              )

    statistics = {}
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
            regret, reactivity, arm_counter, cum_reward, arm_counter_static, arm_counter_drift \
                = bandit.run_simulation(strategy=strategy)
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
        print("Arm allocation Static -> %s" % (100 * arm_counter_static / change_at_step))
        print("Arm allocation Drift -> %s" % (100 * arm_counter_drift / change_at_step))
        print("Cumulative reward is : ", cum_reward)
        # TODO: arm allocation until the change and after the change

        statistics[strategy] = {
            "arm_allocation": arm_allocation,
            "arm_counter_static": arm_counter_static,
            "arm_counter_drift": arm_counter_drift,
            "mean_regret": np.mean(regrets),
            "median_regret": np.median(regrets),
            "std_regret": np.std(regrets),
            "mean_reactivities": np.mean(reactivities),
            "median_reactivities": np.median(reactivities),
            "std_reactivities": np.std(reactivities),
            "cum_reward": cum_reward
        }

    # Lineplot
    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap('plasma')
    for ix, label in enumerate(regret_curves):
        plt.plot(regret_curves[label], label=label, color=cmap(ix / (len(regret_curves) - 1)))
    plt.axvline(x=change_at_step, color='r', linestyle='--', label="Probabilities change")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create bar plot
    plt.figure(figsize=(15, 6))
    values = []
    labels = []
    for label in regret_curves:
        labels.append(label)
        values.append(regret_curves[label][n_steps - 1])
        # plt.bar(label, regret_curves[label][n_steps - 1])
    sorted_values, sorted_labels = zip(*sorted(zip(values, labels), reverse=True))
    cmap = plt.get_cmap('plasma')
    plt.bar(sorted_labels, sorted_values, color=cmap(np.linspace(0,1, len(sorted_values))))
    plt.xlabel("Algorithm")
    plt.ylabel(f"Cumulative Regret")
    plt.title(f"Cumulative Regret Comparison at step t = {n_steps}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Simulation done!")

    # TODO: store the arm_allocation to file (txt file or smth) and call later for plotting in the jobs section
    # # Second Round
    # arm_allocations = [
    #     [34.39607, 24.89851, 20.40018, 12.10207, 8.20317],  # greedy
    #     [12.1732, 64.81001, 17.89838, 2.89369, 2.22472],    # epsgreedy
    #     [4.85504, 69.37921, 19.35197, 4.37928, 2.0345],     # thompson
    #     [8.26354, 76.07635, 13.34003, 1.75242, 0.56766]     # ucb1-0.7-norm
    # ]
    #
    #
    # # Plotting
    # plot_arm_allocations(arm_allocations)

    # Get the arm allocations
    for key, value in statistics.items():
        print(value["arm_allocation"])
