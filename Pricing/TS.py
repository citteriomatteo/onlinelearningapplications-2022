import numpy as np
from Learner import *
from Pricing.pricingEnvironment import PricingEnvironment
from Social_Influence.Graph import Graph


class TS(Learner):

    def __init__(self, n_arms, prices, graph):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.beta_parameters = np.ones((self.n_products, n_arms, 2))
        self.graph = graph

    def pull_arm(self):
        """

        :return: for every product choose the arm to pull
        :rtype: list
        """
        idx = [0 for _ in range(self.n_products)]
        for prod in range(self.n_products):
            # generate beta for every price of the current product
            beta = np.random.beta(self.beta_parameters[prod, :, 0], self.beta_parameters[prod, :, 1])
            # arm of the current product with highest expected reward
            idx[prod] = np.argmax(beta*self.prices[prod])
        return idx

    def update(self, pulled_arm, reward):
        """
        update alpha and beta parameters
        :param pulled_arm: arm pulled for every product
        :type pulled_arm: list
        :param reward: reward obtained for every product using the specified arm
        :type reward: list
        :return: none
        :rtype: none
        """
        super(TS, self).update(pulled_arm, reward)
        for prod in range(self.n_products):
            self.beta_parameters[prod, pulled_arm[prod], 0] = self.beta_parameters[prod, pulled_arm[prod], 0] + reward[
                prod]
            self.beta_parameters[prod, pulled_arm[prod], 1] = self.beta_parameters[prod, pulled_arm[prod], 1] + 1.0 - \
                                                              reward[prod]


graph = Graph(mode="full", weights=True)
env = PricingEnvironment(4, graph, 1)
learner = TS(4, env.prices[0], graph)

for i in range(10000):
    pulled_arms = learner.pull_arm()
    rewards = env.round(pulled_arms)
    learner.update(pulled_arms, rewards)
    print(pulled_arms)
