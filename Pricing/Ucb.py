import numpy as np
from Learner import *
from Pricing.pricingEnvironment import PricingEnvironment
from Social_Influence.Graph import Graph


class Ucb(Learner):
    def __init__(self, n_arms, prices, secondaries, num_product_sold, graph):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.pricesMeanPerProduct = np.mean(self.prices, 1)
        self.means = np.zeros(prices.shape)
        self.nearbyReward = np.zeros(prices.shape)
        self.widths = np.ones(prices.shape) * np.inf
        self.graph = graph
        self.secondaries = secondaries
        self.currentBestArms = np.zeros(len(prices))
        self.num_product_sold = num_product_sold

    def reset(self):
        self.__init__(self.n_arms, self.prices, self.graph)

    def act(self):
        """
        :return: for each product return the arm to pull
        :rtype: int
        """
        idx = np.argmax((self.widths + self.means) * ((self.prices*self.num_product_sold) + self.nearbyReward), axis=1)
        return idx

    def totalNearbyRewardEstimation(self):
        conversion_of_current_best = [i[j] for i,j in zip(self.means, self.currentBestArms)]
        nearbyRewardsTable = np.zeros(self.prices.shape)
        nodesToVisit = [i for i in range(len(self.prices))]
        for node in nodesToVisit:
            copyList = nodesToVisit.copy()
            copyList.remove(node)
            for price in range(len(self.prices[0])):
                nearbyRewardsTable[node][price] += self.singleNearbyRewardEstimation(copyList, conversion_of_current_best,
                                                                                     node, self.means[node][price])
        return nearbyRewardsTable

    def singleNearbyRewardEstimation(self, nodesToVisit, conversion_estimation_for_best_arms, product, probabilityToEnter):
        valueToReturn = 0
        for j in (list(set(nodesToVisit).intersection(self.secondaries[product]))):
            probabilityToVisitSecondary = graph.search_edge_by_nodes(graph.search_product_by_number(product), graph.search_product_by_number(j)).probability
            probToBuyASecondary = conversion_estimation_for_best_arms[j] * probabilityToVisitSecondary * probabilityToEnter
            valueToReturn += self.prices[j][self.currentBestArms[j]] * probToBuyASecondary * self.num_product_sold[j][self.currentBestArms[j]]
            if(probToBuyASecondary>(1e-6)):
                copyList = nodesToVisit.copy()
                copyList.remove(j)
                valueToReturn += self.singleNearbyRewardEstimation(copyList, conversion_estimation_for_best_arms, j, probToBuyASecondary)
        return valueToReturn

    def updateHistory(self, arm_pulled, reward):
        super().update(arm_pulled, reward)

    def update(self, arm_pulled):
        """
        update mean and widths
        :param arm_pulled: arm pulled for every product
        :type arm_pulled: list
        :return: none
        :rtype: none
        """
        self.currentBestArms = arm_pulled
        self.nearbyReward = self.totalNearbyRewardEstimation()
        num_products = len(arm_pulled)
        '''update mean for every arm pulled for every product'''
        for prod in range(num_products):
            self.means[prod][arm_pulled[prod]] = np.mean(self.rewards_per_arm[prod][arm_pulled[prod]])
        '''update widths for every arm pulled for every product'''
        for prod in range(num_products):
            n = len(self.rewards_per_arm[prod][arm_pulled[prod]])
            if n > 0:
                self.widths[prod][arm_pulled[prod]] = np.sqrt((2 * np.max(np.log(self.t)) / n))
            else:
                self.widths[prod][arm_pulled[prod]] = np.inf


graph = Graph(mode="full", weights=True)
env = PricingEnvironment(4, graph, 1)
learner = Ucb(4, env.prices[0], env.secondaries, env.num_product_sold[0], graph)

for i in range(10000):
    if i == 50:
        aaa = 1
    pulled_arms = learner.act()
    rewards = env.round(pulled_arms)
    learner.updateHistory(pulled_arms,rewards)
    # TODO  non hardcodare
    if (i % 10 == 0) and (i!=0):
        learner.update(pulled_arms)
    print(pulled_arms)
print(learner.means)
print(learner.widths)
