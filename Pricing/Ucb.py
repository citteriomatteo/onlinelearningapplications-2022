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
        """

        :return: for each product return the arm to pull
        :rtype: int
        """
        idx = np.argmax((self.widths + self.means) * self.prices, axis=1)
        return idx

    def update(self, arm_pulled, reward):
        """
        update mean and widths
        :param arm_pulled: arm pulled for every product
        :type arm_pulled: list
        :param reward: reward obtained for every product
        :type reward: list
        :return: none
        :rtype: none
        """
        super().update(arm_pulled, reward)
        num_products = len(arm_pulled)
        '''update mean for every arm pulled for every product'''
        for prod in range(num_products):
            self.means[prod][arm_pulled[prod]] = np.mean(self.rewards_per_arm[prod][arm_pulled[prod]])
        '''update widths for every arm pulled for every product'''
        for prod in range(num_products):
            n = len(self.rewards_per_arm[prod][arm_pulled[prod]])
            if n > 0:
                self.widths[prod][arm_pulled[prod]] = np.sqrt((2 * np.max(np.log(self.t)) / n))
            else:
                self.widths[prod][arm_pulled[prod]] = np.inf


graph = Graph(mode="full", weights=True)
env = PricingEnvironment(4, graph, 1)
learner = Ucb(4, env.prices[0])

for i in range(10000):
    pulled_arms = learner.act()
    rewards = env.round(pulled_arms)
    learner.update(pulled_arms, rewards)
    print(pulled_arms)
