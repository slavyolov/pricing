import pandas as pd
from src.multi_armed_bandit.demand import solve_demand_curve, demand_curve
from pathlib import Path
from multi_armed_bandit.bandits_rf import MultiArmedBandit
import numpy as np
from src.multi_armed_bandit.sample_prices import sample_from_initial


check_probabilities = True


if __name__ == "__main__":
    # Read data
    path = str(Path(Path(__file__).parents[2], "data/input/sales.csv"))
    df = pd.read_csv(path)

    # Get the average conversion rate per price (this is the probability of selling on average)
    avg_proba_observed = df.groupby("price")["conversion_rate"].agg("mean").values

    # product cost (this defines the minimum price for our problem. Below this value we are generating loses)
    product_costs = 5.0

    # get the prices from the observed data
    prices = df["price"].sort_values().unique()

    # Get alpha and beta parameters
    alpha = 1.9  # Set the max probability at 95% to address the chance of missing the offer even if the price is equal to 0

    # Solve the demand equation for beta taking the midpoint of the probabilities
    midpoint_proba_observed = avg_proba_observed[1]
    beta = solve_demand_curve(price=10, alpha=alpha, prob=midpoint_proba_observed)
    print("Beta coefficient:", beta)

    if check_probabilities:
        purchase_probabilities = [demand_curve(price=price, a=alpha, b=beta).round(3) for price in prices]
        print(f"Observed purchase probabilities : ", list(avg_proba_observed.round(3)))
        print(f"Observed data standard deviation : ",
              list(df.groupby("price")["conversion_rate"].agg("std").values.round(3)))
        print(f"Estimated purchase probabilities using beta = {beta} : ", purchase_probabilities)

    # Expand the price set to enable the MAB to test for other prices when searching for the best one
    mean_price = np.mean(prices)
    std_dev_price = np.std(prices)

    # Get 10 prices (including the original 3)
    break_even_price = np.array([product_costs])
    wave_1 = sample_from_initial(low=5.25, high=7.25)  # At least .25 cents diff from min_possible and min_observed
    wave_2 = sample_from_initial(low=7.75, high=9.75)  # At least .25 cents diff between 7.5 and 10

    new_prices = np.concatenate((break_even_price, wave_1, wave_2, prices))
    new_prices.sort()
    new_prices = list(new_prices)

    # Sanity check (prices and probabilities)
    if check_probabilities:
        purchase_probabilities = [demand_curve(price=price, a=alpha, b=beta).round(3) for price in new_prices]
        print(f"Estimated purchase probabilities using beta = {beta} on new set of prices : ", purchase_probabilities)

    MultiArmedBandit(prices=prices, alpha=alpha, beta=beta)
