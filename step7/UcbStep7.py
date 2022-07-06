import Settings
from Pricing.Learner import *


class Ucb(Learner):
    def __init__(self, n_arms, prices, secondaries, graph):
        super().__init__(n_arms, prices.shape[0])
        self.prices = prices
        self.pricesMeanPerProduct = np.mean(self.prices, 1)
        self.means = np.zeros(prices.shape)
        self.num_product_sold_estimation = np.ones(prices.shape)
        self.nearbyReward = np.zeros(prices.shape)
        self.widths = np.ones(prices.shape) * np.inf
        self.graph = graph
        self.secondaries = secondaries
        self.currentBestArms = np.zeros(len(prices))
        self.n = np.zeros((self.n_products, self.n_arms))

    def reset(self):
        self.__init__(self.n_arms, self.prices, self.graph)

    def act(self):
        """
        :return: for each product returns the arm to pull based on which one gives the highest reward
        :rtype: int
        """
        idx = np.argmax(
            (self.widths + self.means) * ((self.prices * self.num_product_sold_estimation) + self.nearbyReward), axis=1)
        return idx

    def get_opt_arm_value(self):
        """
        :return: returns the value associated with the optimal arm
        :rtype: float
        """
        return np.max(
            (self.widths + self.means) * (self.prices * self.num_product_sold_estimation) + self.nearbyReward, axis=1)

    def totalNearbyRewardEstimation(self):
        """
        :return: a matrix containing the nearby rewards for all products and all prices
        """
        # contains the conversion rate of the current best price for each product
        conversion_of_current_best = [i[j] for i, j in zip(self.means, self.currentBestArms)]
        nearbyRewardsTable = np.zeros(self.prices.shape)
        # it is created a list containing all the nodes/products that must be visited (initially all the products)
        nodesToVisit = [i for i in range(len(self.prices))]
        for node in nodesToVisit:
            copyList = nodesToVisit.copy()
            copyList.remove(node)
            # for each product and each price calculates its nearby reward
            for price in range(len(self.prices[0])):
                nearbyRewardsTable[node][price] += self.singleNearbyRewardEstimation(copyList,
                                                                                     conversion_of_current_best,
                                                                                     node, self.means[node][price])
        return nearbyRewardsTable

    def singleNearbyRewardEstimation(self, nodesToVisit, conversion_estimation_for_best_arms, product,
                                     probabilityToEnter):
        """
        :return: nearby reward of a single price of a single product
        :rtype: float
        """
        valueToReturn = 0
        isSecondary = True
        # for each node that is possible to visit from the starting one, calculates its nearby reward
        for j in (list(set(nodesToVisit).intersection(self.secondaries[product]))):
            # the probability from a node to visit another one is given by the edge of the graph connecting the two
            # nodes/products
            probabilityToVisitSecondary = self.graph.search_edge_by_nodes(self.graph.search_product_by_number(product),
                                                                          self.graph.search_product_by_number(
                                                                              j)).probability
            if not isSecondary:
                probabilityToVisitSecondary = probabilityToVisitSecondary * Settings.LAMBDA
            isSecondary = False
            # the chance of buying a secondary product is given by the probability of visiting it, the probability
            # to buy the primary and the probability to buy the secondary once its page is reached (its
            # conversion rate)
            probToBuyASecondary = conversion_estimation_for_best_arms[
                                      j] * probabilityToVisitSecondary * probabilityToEnter
            valueToReturn += self.prices[j][self.currentBestArms[j]] * probToBuyASecondary \
                             * self.num_product_sold_estimation[j][self.currentBestArms[j]]
            # the tree must be ran across deeper, but it is useless to visit it if the chance of reaching a deeper node
            # is almost zero, so it is checked how much probable it is to going deeper before
            # doing the other calculations
            if (probToBuyASecondary > (1e-6)):
                copyList = nodesToVisit.copy()
                copyList.remove(j)
                valueToReturn += self.singleNearbyRewardEstimation(copyList, conversion_estimation_for_best_arms, j,
                                                                   probToBuyASecondary)

        return valueToReturn

    def updateHistory(self, arm_pulled, visited_products, num_bought_products):
        super().update(arm_pulled, visited_products, num_bought_products)
        current_prices = [i[j] for i, j in zip(self.prices, arm_pulled)]
        current_reward = sum(num_bought_products * current_prices)
        self.current_reward.append(current_reward)

    def update(self, arm_pulled):
        """
        update mean and widths
        :param arm_pulled: arm pulled for every product
        :type arm_pulled: list
        :return: none
        :rtype: none
        """
        self.currentBestArms = arm_pulled
        for prod in range(self.n_products):
            new_mean = np.mean(self.rewards_per_arm[prod][arm_pulled[prod]])
            if not np.isnan(new_mean):
                self.means[prod][arm_pulled[prod]] = new_mean
            sold_estimation = np.mean(self.boughts_per_arm[prod][arm_pulled[prod]])
            if not np.isnan(sold_estimation):  # to avoid Nan values in the matrix
                self.num_product_sold_estimation[prod][arm_pulled[prod]] = sold_estimation
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                self.n[prod, arm] = len(self.rewards_per_arm[prod][arm])
                if (self.n[prod, arm]) > 0:
                    self.widths[prod][arm] = np.sqrt((2 * np.max(np.log(self.t)) / self.n[prod, arm]))
                else:
                    self.widths[prod][arm] = np.inf
        self.nearbyReward = self.totalNearbyRewardEstimation()
