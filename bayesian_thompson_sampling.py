import numpy as np


class Item:
    """ the base item class """

    def __init__(self):
        self.initialize()  # reset the item

    def initialize(self):
        self.Q = 0  # the estimate of this item's reward value
        self.n = 0  # the number of times this item has been tried

    def update(self, R):
        """ update this item after it has returned reward value 'R' """

        # increment the number of times this item has been tested
        self.n += 1

        # the new estimate of the mean is calculated from the old estimate
        self.Q = (1 - 1.0 / self.n) * self.Q + (1.0 / self.n) * R


class GaussianThompsonItem(Item):
    def __init__(self):
        self.τ_0 = 1 #10000 #0.0001 #10000 #10000 0.01 #0.01 #0.0001 #10000  # 0.0001 #1000# the posterior precision (try 0.001, 0.0001)
        self.μ_0 = 0  # 1 - the posterior mean

        # initialize the base Item
        super().__init__()

    def sample(self):
        """ return a value from the the posterior normal distribution """
        return (np.random.randn() / np.sqrt(self.τ_0)) + self.μ_0

    def update(self, R):
        """ update this item after it has returned reward value 'R' """

        # do a standard update of the estimated mean
        super().update(R)

        # update the mean and precision of the posterior
        self.μ_0 = ((self.τ_0 * self.μ_0) + (self.n * self.Q)) / (self.τ_0 + self.n)
        self.τ_0 += 1