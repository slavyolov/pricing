from multi_armed_bandit.plots import plot_arm_allocations, bar_plots
import ast
import pprint
import matplotlib.pyplot as plt


if __name__ == "__main__":
    drift = True

    # reading the data from the file
    # file = "/Users/syol07091/PycharmProjects/pricing/data/output/statistics_stationary_run.txt"
    file = "/Users/syol07091/PycharmProjects/pricing/data/output/statistics_non_stationary_run.txt"

    with open(file) as f:
        data = f.read()

    # print("Data type before reconstruction : ", type(data))

    # reconstructing the data as a dictionary
    statistics = ast.literal_eval(data)

    # print("Data type after reconstruction : ", type(statistics))
    # print(pprint.pprint(statistics, width=120, compact=True))

    plot_arm_allocations(statistics, arm_allocation_type="arm_allocation")
    if drift:
        plot_arm_allocations(statistics, arm_allocation_type="arm_allocation_static")
        plot_arm_allocations(statistics, arm_allocation_type="arm_allocation_drift")

    # fetch the lists
    algorithms = []
    means_regret = []
    std_regret = []
    medians_regret = []
    mean_reactivities = []
    std_reactivities = []
    medians_reactivity = []
    for key, value in statistics.items():
        algorithms.append(key)
        means_regret.append(value["mean_regret"])
        std_regret.append(value["std_regret"])
        medians_regret.append(value["median_regret"])

        mean_reactivities.append(value["mean_reactivities"])
        std_reactivities.append(value["std_reactivities"])
        medians_reactivity.append(value["median_reactivities"])

    bar_plots(algorithms, means_regret, medians_regret, std_regret, medians_reactivity)

    # Store stats to pandas
    import pandas as pd

    regret_df = pd.DataFrame(
        {'algorithm': algorithms,
         'mean_regret': means_regret,
         'std_regret': std_regret
        })

    print("Done")