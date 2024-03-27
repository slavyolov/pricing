# Target :

- To maximize the revenue.

# Question marks :
- if we get the optimal average reward by running the expected revenue, why we need the MAB algorithms ?
  - Potential answer : a and b parameters in practice are never predetermined. We could run different simulations
    - using the same price set [0, 20, 30, 40, 50] but different values of b.
    - The result can be averaged and better approximation of the price can be made

- How to find the value of b (or set of probable values ?)
  - Optimization algorithm can be executed until the stage we have a close to sigmoid curve 
  - Predefined values of b can be identified for prices >= X and <= y could be done. For example :
    - prices between 0-50 EURO (values of b between 0.01 and 0.1)
    - prices between 0-3 EURO (values of b between 1 and 2)

- But if the reward for every potential action is known why we need RL (MAB specifically) ?
  - If the reward for every potential action is known, then why do you need multi-armed bandits? The optimal policy becomes trivial, just pick the best action.
  - https://stats.stackexchange.com/a/108786

# TODOs
- maybe adding some random noise to the demand signal so that we get different probabilities with every iteration ?
- Also can we combine the probability of purchase with some gradients ?


# TODO: add initial idea of the probabilities based on the demand curve
# TODO: update the probabilities (few times) - maybe after some epochs (see which algorithm is adapting faster)