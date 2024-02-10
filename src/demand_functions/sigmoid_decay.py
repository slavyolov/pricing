import matplotlib.pyplot as plt
import numpy as np


def sigmoid_decay(x, k, x0):
    """

    Args:
        x: Input variable (price)
        k: Controls the steepness of the curve
        x0: this is the x-value of the sigmoid midpoint

    Returns:

    """
    return 1 / (1 + np.exp(k * (x - x0)))


def sigmoid_derivative(x, k, x0):
    return sigmoid_decay(x, k, x0) * (1 - sigmoid_decay(x, k, x0))


def plot_decay_and_derivative():
    x = np.linspace(0, 10, 10)
    k = 1
    x0 = 5  # Midpoint of x

    y1 = sigmoid_decay(x, k, x0)
    y2 = sigmoid_derivative(x, k, x0)

    plt.plot(x, y1, label='sigmoid')
    plt.plot(x, y2, label='derivative')
    plt.legend(loc='upper left')
    plt.show()


def run():
    # Prices
    # prices = np.linspace(0, 100, 15)
    prices = np.array([0, 10, 12, 14, 16, 20])
    competitor_price = 14

    # Parameters
    k = 0.1  # TODO: search values between 0.1 and 0.5 (maybe we can select them at random
    x0 = competitor_price  # TODO: to be selected based on the competitors price

    # Calculate sigmoid decay for each price
    # high values of k control the decrease of probability
    for k in [0.1, 0.3, 0.5, 0.7, 1]:
        decay_values = sigmoid_decay(prices, k, x0)

        # Add random noise :
        # noise = np.random.uniform(-0.05, 0.05, len(prices))

        # Noisy sigmoid :
        # decay_values = decay_values + noise

        # print(sigmoid_decay(0, k, x0))
        # print(sigmoid_decay(50, k, x0))
        # print(sigmoid_decay(100, k, x0))

        # Plot
        plt.suptitle(f"Decaying with {k}")
        plt.plot(prices, decay_values, label='Sigmoid Decay')
        plt.scatter(prices, decay_values, label='Probabilities', color="orange")

        for i, txt in enumerate(decay_values):
            plt.annotate(f'{txt:.2f}', (prices[i], decay_values[i]), textcoords='offset points',
                         xytext=(20, 0), ha='center')

        plt.xlabel('Price')
        plt.ylabel('Sigmoid Decay Value')
        plt.title('Sigmoid Decay Function for Different Prices')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    run()
    plot_decay_and_derivative()
