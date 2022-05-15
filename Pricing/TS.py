import numpy as np
from Learner import *


class TS(Learner):

    def __init__(self, n_arms, prices):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.means = np.zeros(self.prices.shape)
        self.widths = np.ones(self.prices.shape) * np.inf
        self.beta_parameters = np.ones(n_arms, 2)
        self.t = 0

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        super(TS, self).update(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward
