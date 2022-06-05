import numpy as np
from Pricing.Learner import *
from Pricing.pricingEnvironment import PricingEnvironment
from Social_Influence.Graph import Graph


class Ucb(Learner):
    def __init__(self, n_arms, prices, secondaries):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.pricesMeanPerProduct = np.mean(self.prices, 1)
        self.means = np.zeros(prices.shape)
        self.num_product_sold_estimation = np.ones(prices.shape)
        self.nearbyReward = np.zeros(prices.shape)
        self.widths = np.ones(prices.shape) * np.inf
        self.secondaries = secondaries
        self.currentBestArms = np.zeros(len(prices))
        self.visit_probability_estimation = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_visited_from_starting_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_visited_as_first_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.n = np.zeros((self.n_products, self.n_arms))

    def reset(self):
        self.__init__(self.n_arms, self.prices)

    def act(self):
        """
        :return: for each product returns the arm to pull based on which one gives the highest reward
        :rtype: int
        """
        idx = np.argmax((self.widths + self.means) * ((self.prices*self.num_product_sold_estimation) + self.nearbyReward), axis=1)
        return idx

    def totalNearbyRewardEstimation(self):
        """
        :return: a matrix containing the nearby rewards for all products and all prices
        """
        # contains the conversion rate of the current best price for each product
        conversion_of_current_best = [i[j] for i,j in zip(self.means, self.currentBestArms)]
        price_of_current_best = np.array([i[j] for i, j in zip(self.prices, self.currentBestArms)])
        num_product_sold_of_current_best = np.array([i[j] for i, j in zip(self.num_product_sold_estimation, self.currentBestArms)])
        nearbyRewardsTable = np.zeros(self.prices.shape)
        # it is created a list containing all the nodes/products that must be visited (initially all the products)
        nodesToVisit = [i for i in range(len(self.prices))]
        for node in nodesToVisit:
            # for each product and each price calculates its nearby reward
            for price in range(len(self.prices[0])):
                nearbyRewardsTable[node][price] = sum(self.visit_probability_estimation[node][price]
                                                      * conversion_of_current_best * price_of_current_best
                                                      * num_product_sold_of_current_best)
        return nearbyRewardsTable


    def updateHistory(self, arm_pulled, visited_products, num_bought_products, num_primary):
        super().update(arm_pulled, visited_products, num_bought_products)
        self.times_visited_as_first_node[num_primary][arm_pulled[num_primary]] += 1
        for i in range(len(visited_products)):
            if (visited_products[i] == 1) and i != num_primary:
                self.times_visited_from_starting_node[num_primary][arm_pulled[num_primary]][i] += 1

    def update(self, arm_pulled):
        """
        update mean and widths
        :param arm_pulled: arm pulled for every product
        :type arm_pulled: list
        :return: none
        :rtype: none
        """

        #DA CAMBIARE UCB
        self.currentBestArms = arm_pulled
        num_products = len(arm_pulled)
        '''update mean for every arm pulled for every product'''
        for prod in range(num_products):
            self.means[prod][arm_pulled[prod]] = np.mean(self.rewards_per_arm[prod][arm_pulled[prod]])
            self.num_product_sold_estimation[prod][arm_pulled[prod]] = np.mean(self.boughts_per_arm[prod][arm_pulled[prod]])
            self.visit_probability_estimation[prod] = self.times_visited_from_starting_node[prod] / self.times_visited_as_first_node[prod]
        '''update widths for every arm pulled for every product'''
        for prod in range(num_products):
            for arm in range(self.n_arms):
                self.n[prod,arm] = len(self.rewards_per_arm[prod][arm])
                if self.n[prod,arm] > 0:
                    self.widths[prod][arm] = np.sqrt((2 * np.max(np.log(self.t)) / n))
                else:
                    self.widths[prod][arm] = np.inf
        self.nearbyReward = self.totalNearbyRewardEstimation()
        aaa = 1


graph = Graph(mode="full", weights=True)
env = PricingEnvironment(4, graph, 1)
learner = Ucb(4, env.prices[0], env.secondaries)

for i in range(10000):
    if i == 9990:
        aaa = 1
    pulled_arms = learner.act()
    visited_products, num_bought_products, num_primary = env.round(pulled_arms)
    learner.updateHistory(pulled_arms, visited_products, num_bought_products, num_primary)
    # TODO  non hardcodare
    if (i % 10 == 0) and (i != 0):
        learner.update(pulled_arms)
    print(pulled_arms)
print(learner.means)
print(learner.widths)
aaa = [i[j] for i,j in zip(learner.nearbyReward, learner.currentBestArms)]
print(aaa)
print(sum(aaa))