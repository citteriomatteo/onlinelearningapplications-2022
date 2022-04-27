import numpy as np
from Learner import *
from Pricing.pricingEnvironment import PricingEnvironment
from Social_Influence.Graph import Graph


class Ucb(Learner):
    def __init__(self, n_arms, prices):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.means = np.zeros(prices.shape)
        self.widths = np.ones(prices.shape) * np.inf

    def reset(self):
        self.__init__(self.n_arms, self.prices)

    def act(self):
        idxs = np.argmax((self.widths + self.means) * self.prices, axis=1)
        return idxs

    def update(self, arm_pulled, reward):
        super().update(arm_pulled, reward)
        for i in range(len(arm_pulled)):
            self.means[i][arm_pulled[i]] = np.mean(self.rewards_per_arm[i][arm_pulled[i]])  ## self.means[arm_pulled]? -> OK
        for prod in range(self.n_products):
            for idx in range(self.n_arms):
                n = len(self.rewards_per_arm[prod][idx])
                if n > 0:
                    self.widths[prod][idx] = np.sqrt((2 * np.max(np.log(self.t)) / (n)))  # (t-1)? self.prices)
                else:
                    self.widths[prod][idx] = np.inf


graph = Graph(mode="full", weights=True)
env = PricingEnvironment(4, graph, 1)
learner = Ucb(4, env.prices[0])

for i in range(100):
    pulled_arms = learner.act()
    rewards = env.round(pulled_arms)
    learner.update(pulled_arms, rewards)

    print(pulled_arms)
