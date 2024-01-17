import numpy as np


class GradientOptimzer:

    def __init__(self, num_items, num_factors, beta=0.99):
        """Initialize
        Args:
        num_items:  Total number of items in the data
        num_factors: Number of factors
        beta_2: hyper-parameter
        """
        self.num_items = num_items
        self.num_factors = num_factors
        self.beta = beta

        self.v = np.zeros((self.num_items, self.num_factors))
        self.gradients = np.zeros((self.num_items, self.num_factors))

        self.t = 0

    # get past historical gradients estimates
    def historical_gradient_estimates(self, present_gradients, item_index):
        self.gradients[item_index, :] = present_gradients
        self.t += 1

        self.v = (self.beta * self.v) + (1.0 - self.beta) * np.square(self.gradients)  # Equation 14

        # Calculates the bias-corrected estimates
        v_hat = self.v / (1.0 - (self.beta ** self.t))  # Equation 16

        return v_hat