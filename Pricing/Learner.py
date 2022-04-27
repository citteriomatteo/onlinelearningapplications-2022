import numpy as np


class Learner():

    def __init__(self, n_arms, n_products):
        self.n_arms = n_arms
        self.n_products = n_products

        self.max_idxs = [0 for i in range(n_products)]
        self.max_revenue = 0

        ####
        self.t = 0
        self.rewards = []
        self.rewards_per_arm = [[[] for i in range(n_arms)] for j in range(n_products)]
        self.pulled = []

    def reset(self):
        self.__init__(self.n_arms, self.t)

    def update(self, pulled_arms, reward):
        """
        :param pulled_arms: arms pulled by the learner for every product
        :param reward: reward returned by the environment
        :return: update the observation list
        """
        self.t += 1
        self.rewards.append(reward)
        for i in range(len(pulled_arms)):
            self.rewards_per_arm[i][pulled_arms[i]].append(reward)
        #self.rewards_per_arm[pulled_arms].append(reward)
        self.pulled.append(pulled_arms)
