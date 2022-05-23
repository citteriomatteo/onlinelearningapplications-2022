import numpy as np
from Learner import *
from Pricing.pricingEnvironment import PricingEnvironment
from Social_Influence.Graph import Graph


class Ucb(Learner):
    def __init__(self, n_arms, prices, graph):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.pricesMeanPerProduct = np.mean(self.prices,1)
        self.means = np.zeros(prices.shape)
        self.widths = np.ones(prices.shape) * np.inf
        self.graph = graph

    def reset(self):
        self.__init__(self.n_arms, self.prices, self.graph)

    def act(self):
        """

        :return: for each product return the arm to pull
        :rtype: int
        """
        aaa = self.totalNearbyRewardEstimation()
        #print(aaa)
        idx = np.argmax((self.widths + self.means) * self.prices, axis=1)
        return idx

    def totalNearbyRewardEstimation(self):
        conversionMeanPerProduct = np.mean(self.means, 1)
        nearbyRewardsTable = np.zeros(self.prices.shape)
        nodesToVisit = [i for i in range(len(self.prices))]
        for node in nodesToVisit:
            copyList = nodesToVisit.copy()
            copyList.remove(node)
            for price in range(len(self.prices[0])):
                nearbyRewardsTable[node][price] += self.singleNearbyRewardEstimationFirstLayer(copyList,conversionMeanPerProduct,
                                                                                               node, self.means[node][price])
        return  nearbyRewardsTable

    def singleNearbyRewardEstimationFirstLayer(self, nodesToVisit, conversionMeanPerProduct, product, probabilityToEnter):
        valueToReturn = 0
        for j in nodesToVisit:
            probabilityToVisitSecondary = graph.search_edge_by_nodes(graph.search_product_by_number(product),graph.search_product_by_number(j)).probability
            probToBuyASecondary = conversionMeanPerProduct[j] * probabilityToVisitSecondary * probabilityToEnter
            valueToReturn += self.pricesMeanPerProduct[j] * probToBuyASecondary
            if(probToBuyASecondary>(1e-6)):
                copyList = nodesToVisit.copy()
                copyList.remove(j)
                valueToReturn += self.singleNearbyRewardEstimationDeeperLayers(copyList, conversionMeanPerProduct, j, probToBuyASecondary)
        return valueToReturn

    def singleNearbyRewardEstimationDeeperLayers(self, nodesToVisit, conversionMeanPerProduct, product, probabilityToEnter):
        valueToReturn = 0
        for j in nodesToVisit:
            probabilityToVisitSecondary = graph.search_edge_by_nodes(graph.search_product_by_number(product),graph.search_product_by_number(j)).probability
            probToBuyASecondary = conversionMeanPerProduct[j] * probabilityToVisitSecondary * probabilityToEnter
            valueToReturn += self.pricesMeanPerProduct[j] * probToBuyASecondary
            if (probToBuyASecondary > (1e-6)):
                copyList = nodesToVisit.copy()
                copyList.remove(j)
                valueToReturn += self.singleNearbyRewardEstimationDeeperLayers(copyList, conversionMeanPerProduct, j,
                                                                               probToBuyASecondary)
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
learner = Ucb(4, env.prices[0], graph)

for i in range(10000):
    pulled_arms = learner.act()
    rewards = env.round(pulled_arms)
    learner.updateHistory(pulled_arms,rewards)
    # TODO  non hardcodare
    if i % 10 == 0:
        learner.update(pulled_arms)
    print(pulled_arms)
print(learner.means)
print(learner.widths)
