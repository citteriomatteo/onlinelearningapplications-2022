import numpy as np

import Settings
from user_data_generator import StandardDataGenerator


class Environment:
    def __init__(self, n_arms, graph, probabilities, resources="../json/dataUsers.json"):
        self.n_arms = n_arms
        self.probabilities = probabilities

        self.user_data = StandardDataGenerator(resources)
        self.alpha_ratios = self.user_data.get_alpha_ratios()
        self.num_daily_users = self.user_data.get_num_daily_users()
        self.num_product_sold = self.user_data.get_num_product_sold()
        self.features = self.user_data.get_features()
        self.classes = self.user_data.get_classes()

        #TODO: discuss about this part
        self.graph = graph
        self.theta = np.random.dirichlet(np.ones(len(self.graph.nodes)), size=1)
        self.arms_features = np.random.binomial(1, 0.5, size=(len(self.graph.edges), len(self.graph.nodes)))
        self.lam = Settings.lam
        for i in range(0, len(self.graph.edges)):
            self.graph.edges[i].probability = np.dot(self.theta, self.arms_features[i])

    def round(self, pulled_arm):
        return np.random.binomial(1, self.probabilities[pulled_arm])
