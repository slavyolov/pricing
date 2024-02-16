import matplotlib.pyplot as plt
import numpy as np


def plot_arm_allocations(statistics, arm_allocation_type="arm_allocation"):
    # Get the arm allocations
    algorithms = []
    arm_allocations = []

    # Color codes for 10 prices
    colors = [
        '#e6194b', '#3cb44b',
        '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#bcf60c',
        '#fabebe', '#008080',
    ]

    # get prices with labels
    # prices = [5.0, 5.31, 6.17, 6.86, 7.5, 7.81, 8.71, 9.34, 10.0, 11.0]
    # "', ".join("'price : " + str(price) for price in prices) + "'"
    prices_w_labels = ['price : 5.0', 'price : 5.31', 'price : 6.17', 'price : 6.86', 'price : 7.5', 'price : 7.81',
                       'price : 8.71', 'price : 9.34', 'price : 10.0', 'price : 11.0']

    # fetch the arm_allocation
    for key, value in statistics.items():
        algorithms.append(key)
        arm_allocations.append(value[arm_allocation_type])

    # Pie Chart for each strategy
    if arm_allocation_type == 'arm_allocation':
        label = "stationary"
    elif arm_allocation_type == 'arm_allocation_static':
        label = "non_stationary_first_half"
    elif arm_allocation_type == 'arm_allocation_drift':
        label = "non_stationary_second_half"
    else:
        raise ValueError("allocation type not supported!")

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle(f"Arm Allocation {label}", fontsize=18, y=0.99)
    axes = axes.ravel()

    for i, strategy in enumerate(algorithms):
        print(strategy)
        ax = axes[i]
        wedges, texts, autotexts = ax.pie(
            arm_allocations[i][::-1],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors[::-1],
            pctdistance=0.85,
        )

        # Draw a center circle for 'donut' style
        # centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        # ax.add_artist(centre_circle)

        # Increase the size and weight of the percentage labels
        for text in autotexts:
            text.set(size=10)

        ax.set_title(strategy, fontsize=13, y=0.92)

    fig.legend(wedges[::-1], prices_w_labels, title="Arms", loc="upper right", fontsize='large')

    plt.tight_layout(pad=0.01)
    plt.show()


def bar_plots(strategies, means_regret, medians_regret, std_regret, medians_reactivity):
    # Bar plot setup
    barWidth = 0.3
    r1 = np.arange(len(means_regret))
    r2 = [x + barWidth for x in r1]

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))

    # Regret plot
    ax[0].bar(r1, means_regret, width=barWidth, color='#AED6F1', edgecolor='grey', yerr=std_regret, capsize=7, label='Mean Regret')
    ax[0].bar(r2, medians_regret, width=barWidth, color='#FAD7A0', edgecolor='grey', label='Median Regret')
    ax[0].set_xlabel('Algorithm', fontweight='bold')
    ax[0].set_ylabel('Regret Value', fontweight='bold')
    ax[0].set_title('Regret (Mean and Median) by Algorithm', fontweight='bold')
    ax[0].set_xticks([r + barWidth for r in range(len(means_regret))])
    ax[0].set_xticklabels(strategies, rotation=25)
    ax[0].legend()

    # Reactivity plot
    ax[1].bar(strategies, medians_reactivity, color='#D2B4DE', edgecolor='grey')
    ax[1].set_xlabel('Algorithm', fontweight='bold')
    ax[1].set_ylabel('Reactivity Value', fontweight='bold')
    ax[1].set_title('Reactivity (Median) by Algorithm', fontweight='bold')
    ax[1].set_xticklabels(strategies, rotation=25)

    plt.tight_layout()
    plt.show()
