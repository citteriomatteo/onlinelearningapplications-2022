import numpy as np
from Learner import *
from Pricing.pricingEnvironment import PricingEnvironment
from Social_Influence.Graph import Graph


class TS(Learner):

    def __init__(self, n_arms, prices):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.beta_parameters = np.ones((self.n_products, n_arms, 2))

    def pull_arm(self):
        #idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))

        idx = [0 for i in range(self.n_products)]
        for prod in range(self.n_products):
            idx[prod] = np.argmax(np.random.beta(self.beta_parameters[prod,:, 0], self.beta_parameters[prod,:, 1]))
        return idx

    def update(self, pulled_arm, reward):
        super(TS, self).update(pulled_arm, reward)
        for prod in range(self.n_products):
            self.beta_parameters[prod, pulled_arm[prod], 0] = self.beta_parameters[prod, pulled_arm[prod], 0] + reward[prod]
            self.beta_parameters[prod, pulled_arm[prod], 1] = self.beta_parameters[prod, pulled_arm[prod], 1] + 1.0 - reward[prod]


graph = Graph(mode="full", weights=True)
env = PricingEnvironment(4, graph, 1)
learner = TS(4, env.prices[0])

for i in range(10000):
    pulled_arms = learner.pull_arm()
    rewards = env.round(pulled_arms)
    learner.update(pulled_arms, rewards)
    print(pulled_arms)