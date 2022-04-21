import numpy as np
from user_data_generator import StandardDataGenerator


class Environment:
    def __init__(self, n_arms, probabilities, resources="../json/dataUsers.json"):
        self.n_arms = n_arms
        self.probabilities = probabilities

        self.user_data = StandardDataGenerator(resources)
        self.alpha_ratios = self.user_data.get_alpha_ratios()
        self.num_daily_users = self.user_data.get_num_daily_users()
        self.num_product_sold = self.user_data.get_num_product_sold()
        self.features = self.user_data.get_features()
        self.classes = self.user_data.get_classes()

    def round(self, pulled_arm):
        return np.random.binomial(1, self.probabilities[pulled_arm])
