import matplotlib.pyplot as plt
from demand import demand_curve, expected_revenue


prices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# prices = [0.49, 0.99, 1.49, 1.99, 2.49, 2.99]
a = [2]
b = [0.05]
# a = [1, 1, 2, 2]
# b = [0.06, 0.04, 0.06, 0.04]
# b = [0.1, 0.5, 1, 2]


def plot_curves(a, b, plot_coordinates=False):
    """
    Generate and plot the purchase probabilities for the price candidates given different a, b arguments

    Args:
        a:
        b:

    Returns:

    """
    # Generate and plot the purchase probabilities for the price candidates given different a, b arguments
    scenarios = {}

    for a, b in zip(a, b):
        print(a, b)
        label = f"a = {a}, b = {b}"
        purchase_probabilities = [demand_curve(price=price, a=a, b=b).round(3) for price in prices]
        scenarios[label] = purchase_probabilities

    # Plot the purchase probabilities for all scenarios
    plt.figure(figsize=(12, 6))
    for scenario in scenarios:
        plt.plot(prices, scenarios[scenario], label=scenario)
        if plot_coordinates:
            for i_x, i_y in zip(prices, scenarios[scenario]):
                plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
    plt.xlabel("Prices")
    plt.ylabel("Probability of purchase")
    plt.title("Demand curves for 'a / 1 + exp(b * price)'")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_average_expected_revenue(a, b, plot_coordinates=False):
    scenarios = {}

    for a, b in zip(a, b):
        print(a, b)
        label = f"a = {a}, b = {b}"
        average_expected_revenue = [expected_revenue(price=price, a=a, b=b).round(3) for price in prices]
        scenarios[label] = average_expected_revenue

    # Plot the purchase probabilities for all scenarios
    plt.figure(figsize=(12, 6))
    for scenario in scenarios:
        plt.plot(prices, scenarios[scenario], label=scenario)
        if plot_coordinates:
            for i_x, i_y in zip(prices, scenarios[scenario]):
                plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
    plt.xlabel("Prices")
    plt.ylabel("Average Expected Revenue")
    plt.title("Expected revenue curves for '(a * price) / (1 + exp(b * price))'")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_curves(a, b)
    plot_average_expected_revenue(a, b)
