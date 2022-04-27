import numpy as np


class Learner():

    def __init__(self, n_arms, n_products):
        self.n_arms = n_arms
        self.n_products = n_products

        # self.collected_rewards = np.array([])

        self.max_idxs = [0 for i in range(n_products)]
        self.max_revenue = 0


    def update_observations(self, pulled_arm, reward):
        """
        :param pulled_arm: arm pulled by the learner
        :param reward: reward returned by the environment
        :return: update the observation list
        """
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)

