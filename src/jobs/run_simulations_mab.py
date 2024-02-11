from src.multi_armed_bandit.demand import demand_curve
from multi_armed_bandit.bandits import MultiArmedBandit
import numpy as np
from matplotlib import pyplot as plt
import pprint

def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data


if __name__ == "__main__":
    # Input parameters :
    # Set the max probability at 95% to address the chance of missing the offer even if the price is equal to 0
    alpha = 1.9
    prices = [5.0, 5.31, 6.17, 6.86, 7.5, 7.81, 8.71, 9.34, 10.0, 11.0]
    beta = 0.171411
    n_steps = 20000
    n_epochs = 100

    drift = False  # For the non-stationary run (set this to True)
    change_at_step = n_steps / 2  # in case we want to add drift

    detailed_display = True

    # Get the probabilities
    purchase_probabilities = [demand_curve(p, a=alpha, b=beta).round(4) for p in prices]

    # Run the simulation :
    regret_curves = {}
    strategies = [
        "greedy",
        "eps_greedy-0.1",
        "eps_greedy-0.2",
        "thompson",
        "ucb1-1-norm",
        "ucb1-1"
    ]
    bandit = MultiArmedBandit(prices=prices, alpha=alpha, beta=beta, n_steps=n_steps, n_epochs=n_epochs,
                              purchase_proba=purchase_probabilities,
                              drift=drift,
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

        statistics[strategy] = {
            "arm_allocation": arm_allocation.tolist(),
            "arm_allocation_static": (100 * arm_counter_static / change_at_step).tolist(),
            "arm_allocation_drift": (100 * arm_counter_drift / change_at_step).tolist(),
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
    if drift:
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

    print("Run statistics")
    print(pprint.pprint(statistics, indent=4, width=120, compact=True))
    print("Simulation done!")

    # Get the arm allocations
    # for key, value in statistics.items():
    #     print(value["arm_allocation"])
