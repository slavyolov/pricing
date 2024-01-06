"""Run the multi armed bandit problem"""
import numpy as np
from scipy.optimize import fsolve
from multi_armed_bandit.demand import demand_curve, revenue_derivative
from multi_armed_bandit.bandits import run_simulation
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # PARAMETERS
    detailed_display = False

    a = 2  # the maximum achievable probability of purchase. The value of a stays within the range of [0, 2]. The demand function used estimates the probability of purchase
    b = 0.042  # modulates the sensitivity of the demand curve against price changes. For the current demand model reasonable values are between [0, 0.1]

    # price candidates
    prices = [20, 30, 40, 50, 60]

    nstep = 10000  # Number of time steps for every simulation
    nepoch = 50  # Number of simulation executions
    regret_curves = {}
    # strategies = ["greedy", "epsgreedy", "thompson", "ucb1-0.7-norm"]
    # strategies = ["epsgreedy", "thompson"]
    strategies = ["thompson"]

    optimal_price = fsolve(func=revenue_derivative, x0=0, args=(a, b))[0]
    optimal_probability = demand_curve(price=optimal_price, a=a, b=b)
    print("The optimal prices for the simulation is : ", optimal_price.round(3))
    print("The optimal probability for the simulation is : ", optimal_probability.round(3))

    best_price_index = index = (np.abs(np.array(prices) - optimal_price)).argmin()

    purchase_probabilities = [demand_curve(p, a, b).round(3) for p in prices]
    print(f"Associated purchase probabilities for the price candidates : {prices} are {purchase_probabilities}")

    # Run the simulation
    for strategy in strategies:
        regret_curves[strategy] = np.zeros((nstep,))
        regrets = []
        reactivities = []
        arm_counters = np.zeros((len(prices),))
        for ep in range(nepoch):
            if detailed_display:
                print("=" * 100)
                print(f"Running epoch {ep} out of {nepoch - 1}")
                print("=" * 100)
            regret, reactivity, arm_counter = run_simulation(prices, optimal_price, optimal_probability, a, b, nstep,
                                                             strategy=strategy)
            regret_curves[strategy] += regret
            regrets.append(regret[-1])
            reactivities.append(reactivity)
            arm_counters += arm_counter / nstep
            print(f"Epoch {ep} out of {nepoch - 1} completed")

        regret_curves[strategy] /= nepoch
        arm_allocation = 100 * arm_counters / nepoch
        print("-------------\nStrategy: %s" % strategy)
        print("Regret -> mean: %.2f, median: %.2f, std: %.2f" % (np.mean(regrets), np.median(regrets), np.std(regrets)))
        print("Reactivity -> mean: %.2f, median: %.2f, std: %.2f" % (
            np.mean(reactivities), np.median(reactivities), np.std(reactivities)))
        print("Arm allocation -> %s" % (arm_allocation))

    plt.figure(figsize=(12, 6))
    for label in regret_curves:
        plt.plot(regret_curves[label], label=label)
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
