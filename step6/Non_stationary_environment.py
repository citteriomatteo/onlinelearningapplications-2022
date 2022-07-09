from Pricing.pricing_environment import EnvironmentPricing
from Social_Influence.Graph import Graph
from Pricing.user_data_generator import *
from Social_Influence.Simulator import *
import Settings

import numpy as np

class Non_stationary_environment:
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

    def setNewConvRates(self,new_conversion_rate):
        self.simulator.set_conversion_rates(new_conversion_rate)
        self.conversion_rates= new_conversion_rate



graph = Graph(mode="full", weights=True)
env = Non_stationary_environment(4, graph, 1)
new_conv_rates=[
    [
      [0.85, 0.47, 0.45, 0.2],
      [0.45, 0.4, 0.9, 0.25],
      [0.55, 0.8, 0.5, 0.4],
      [0.8, 0.35, 0.32, 0.25],
      [0.6, 0.55, 0.93, 0.52] ],

    [ [0.9, 0.45, 0.4, 0.35],
      [0.4, 0.8, 0.3, 0.25],
      [0.5, 0.45, 0.9, 0.35],
      [0.4, 0.35, 0.8, 0.3],
      [0.5, 0.45, 0.4, 0.9] ],
    [
      [0.25, 0.79, 0.4, 0.3],
      [0.45, 0.4, 0.35, 0.95],
      [0.55, 0.85, 0.5, 0.45],
      [0.4, 0.82, 0.32, 0.25],
      [0.4, 0.95, 0.35, 0.3]  ]
  ]


for i in range(1000):
    env.round([0,1,2,2,3])
    if(i==500):
        env.setNewConvRates(new_conv_rates)
        print(env.conversion_rates)





