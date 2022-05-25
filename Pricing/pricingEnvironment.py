import numpy as np

import Settings
from Social_Influence.Graph import Graph
from user_data_generator import StandardDataGenerator


class PricingEnvironment:
    def __init__(self, n_arms, graph, probabilities, resources="../json/dataUsers.json"):
        self.n_arms = n_arms
        self.probabilities = probabilities

        self.user_data = StandardDataGenerator(resources)
        self.alpha_ratios = self.user_data.get_alpha_ratios()
        self.num_daily_users = self.user_data.get_num_daily_users()
        self.num_product_sold = self.user_data.get_num_product_sold()
        self.features = self.user_data.get_features()
        self.classes = self.user_data.get_classes()
        self.conversion_rates = self.user_data.get_conversion_rates()
        self.prices = self.user_data.get_prices()
        self.secondaries = self.user_data.get_secondaries()

        self.graph = graph
        self.theta = np.random.dirichlet(np.ones(len(self.graph.nodes)), size=1)
        self.arms_features = np.random.binomial(1, 0.5, size=(len(self.graph.edges), len(self.graph.nodes)))
        self.lam = Settings.LAMBDA
        for i in range(0, len(self.graph.edges)):
            self.graph.edges[i].probability = np.dot(self.theta, self.arms_features[i])

    def round(self, pulled_arm):
        """

        :param pulled_arm: arm pulled for each product
        :type pulled_arm: list
        :return: reward (0 or 1) for every product given the arm
        :rtype: list
        """
        num_product = len(pulled_arm)
        distributions = [0 for i in range(num_product)]
        for prod in range(num_product):
            distributions[prod] = np.random.binomial(1, self.conversion_rates[0][prod][pulled_arm[prod]])
        return distributions


graph = Graph(mode="full", weights=True)
env = PricingEnvironment(4, graph, 1)
print(env.prices[0].shape)
