import numpy as np

from Project_Code import Settings
from Project_Code.Social_Influence.Simulator import Simulator
from Project_Code.Pricing.user_data_generator import StandardDataGenerator


class EnvironmentPricing:
    def __init__(self, n_arms, graph, probabilities, resources="../Project_Code/json/dataUsers.json"):
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
        self.simulator = Simulator(self.graph, self.alpha_ratios, self.num_product_sold, self.secondaries,
                                   self.conversion_rates)
        self.theta = np.random.dirichlet(np.ones(len(self.graph.nodes)), size=1)
        self.arms_features = np.random.binomial(1, 0.5, size=(len(self.graph.edges), len(self.graph.nodes)))
        self.lam = Settings.LAMBDA

    def round(self, pulled_arm, customer=None):
        """
        :param customer: optional parameter needed for step7 (customer context must be known before the simulation!)
        :param pulled_arm: arm pulled for each product
        :type pulled_arm: list
        :return: reward (0 or 1) for every product given the arm
        :rtype: list
        """
        visited_products, num_bought_products, num_primary = self.simulator.simulate(selected_prices=pulled_arm,
                                                                                     customer=customer)
        return visited_products, num_bought_products, num_primary

