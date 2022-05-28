import numpy as np


class Learner:

    def __init__(self, n_arms, n_products):
        """

        :param n_arms: number of arms of the learner
        :type n_arms: int
        :param n_products: number of products
        :type n_products: int
        """
        self.n_arms = n_arms
        self.n_products = n_products



        ####
        self.t = 1
        # self.rewards = []
        # matrix n_products X n_arms X list of previous rewards
        self.rewards_per_arm = [[[] for i in range(n_arms)] for j in range(n_products)]
        # list of pulled arm
        self.pulled = []

    def reset(self):
        self.__init__(self.n_arms, self.t)

    def update(self, pulled_arms, visited_products, num_bought_products):
        """
        update the observations list once the reward is returned by the environment
        :param pulled_arms: arms pulled by the learner for every product
        :type pulled_arms: list
        :param reward: rewards returned by the environment for every product
        :type reward: list
        :return: update the observation list
        :rtype: None
        """
        self.t += 1
        # self.rewards.append(reward)
        num_product = len(pulled_arms)
        #TODO: fix the way the append works
        for prod in range(num_product):
            if visited_products[prod] == 1:
                if num_bought_products[prod] == 0:
                    self.rewards_per_arm[prod][pulled_arms[prod]].append(0)
                else:
                    self.rewards_per_arm[prod][pulled_arms[prod]].append(1)
        self.pulled.append(pulled_arms)
