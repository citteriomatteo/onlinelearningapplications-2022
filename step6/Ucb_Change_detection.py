from step6.Non_stationary_environment import Non_stationary_environment
import numpy as np
from Pricing.Learner import *
from Pricing.pricing_environment import EnvironmentPricing
from Social_Influence.Graph import Graph
from Pricing.Clairvoyant import Clairvoyant
import Settings
from matplotlib import pyplot as plt
from step6.Cumulative_sum import CUSUM


class Ucb_Change_detection(Learner):
    def __init__(self,n_arms, prices, secondaries, M=100,eps=0.05,h=15, alpha=0.01):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.means = np.zeros(prices.shape)
        self.num_product_sold_estimation = np.ones(prices.shape)
        self.nearbyReward = np.zeros(prices.shape)
        self.widths = np.ones(prices.shape) * np.inf
        self.secondaries = secondaries
        self.currentBestArms = np.zeros(len(prices))
        self.visit_probability_estimation = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_visited_from_starting_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_visited_as_first_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_bought_as_first_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.n = np.zeros((self.n_products, self.n_arms))
        self.alpha_ratios = np.zeros(self.n_products)
        self.times_product_visited_as_first_node = np.zeros(self.n_products)

        self.change_detection = [[CUSUM(M,eps,h) for i in range(n_arms)] for j in range(self.n_products)] #deve essere nella stessa dimensione di self.prices
        self.valid_reward_per_arms=[[[0] for i in range(n_arms)] for j in range(self.n_products)]
        self.detections = [[[] for i in range(n_arms)] for j in range(self.n_products)] #stesso
        self.alpha = alpha

    def act(self):
        """
        :return: for each product returns the arm to pull based on which one gives the highest reward
        :rtype: int
        """
        if np.random.binomial(1, 1 - self.alpha):
            idx = np.argmax((self.widths + self.means) * ((self.prices*self.num_product_sold_estimation) + self.nearbyReward), axis=1)
            return idx
        else:
            random_arms = (np.random.randint(0,4,size=5))
            idx = random_arms  #np.argmax((cost_random),axis=1)
            return idx

    def update(self, arm_pulled):
        """
        update mean and widths
        :param arm_pulled: arm pulled for every product
        :type arm_pulled: list
        :return: none
        :rtype: none
        """
        self.currentBestArms = arm_pulled
        self.average_reward.append(np.mean(self.current_reward[-Settings.DAILY_INTERACTIONS:]))
        num_products = len(arm_pulled)
        '''update mean for every arm pulled for every product'''
        for prod in range(num_products):
            self.alpha_ratios[prod] = self.times_product_visited_as_first_node[prod] / np.sum(
                self.times_product_visited_as_first_node)
            if len(self.valid_reward_per_arms[prod][arm_pulled[prod]]) > 0:
                self.means[prod][arm_pulled[prod]] = np.mean(self.valid_reward_per_arms[prod][arm_pulled[prod]])
            if len(self.boughts_per_arm[prod][arm_pulled[prod]]) > 0:
                self.num_product_sold_estimation[prod][arm_pulled[prod]] = np.mean(
                    self.boughts_per_arm[prod][arm_pulled[prod]])
            for t1 in range(self.n_arms):
                for t2 in range(num_products):
                    if self.times_bought_as_first_node[prod][t1][t2] > 0:
                        self.visit_probability_estimation[prod][t1][t2] = \
                        self.times_visited_from_starting_node[prod][t1][t2] / self.times_bought_as_first_node[prod][t1][
                            t2]
                    else:
                        self.visit_probability_estimation[prod][t1][t2] = 0
        self.visit_probability_estimation[np.isnan(self.visit_probability_estimation)] = 0

        total_valid_samples=0
        for prod in range(num_products):
            for arm in range(self.n_arms):
                total_valid_samples += len(self.valid_reward_per_arms[prod][arm])

        '''update widths for every arm pulled for every product'''
        for prod in range(num_products):
            for arm in range(self.n_arms):
                self.n[prod, arm] = len(self.rewards_per_arm[prod][arm])
                if self.n[prod, arm] > 0:
                    self.widths[prod][arm] = np.sqrt((2 * np.log(total_valid_samples)) / self.n[prod, arm])
                else:
                    self.widths[prod][arm] = np.inf
        self.nearbyReward = self.totalNearbyRewardEstimation()
        self.nearbyReward[np.isnan(self.nearbyReward)] = 0


    def revenue_given_arms(self, arms):
        means = [i[j] for i, j in zip(self.means, arms)]
        prices = [i[j] for i, j in zip(self.prices, arms)]
        num_product_sold = [i[j] for i, j in zip(self.num_product_sold_estimation, arms)]
        nearby_reward = [i[j] for i, j in zip(self.nearbyReward, arms)]
        return np.sum(np.multiply(self.alpha_ratios, np.multiply(means, np.multiply(prices, num_product_sold))+nearby_reward))


    def updateHistory(self, arm_pulled, visited_products, num_bought_products, num_primary):

        for prod in range(self.n_products):
            if visited_products[prod] == 1:
                if num_bought_products[prod] == 0:
                    self.valid_reward_per_arms[prod][arm_pulled[prod]].append(0)
                else:
                    self.valid_reward_per_arms[prod][arm_pulled[prod]].append(1)


        super().update(arm_pulled, visited_products, num_bought_products)
        self.times_product_visited_as_first_node[num_primary] += 1

        #controllo se l'arm è stato tirato e aggiorno la quantità nel cumulative sum (io controllo solo il conversion rate)
        for prod in range(self.n_products):
            if(visited_products[prod] == 1):
                if(num_bought_products[prod] > 0):
                    quantity=1
                else:
                    quantity=0
                if self.change_detection[prod][arm_pulled[prod]].update(quantity):
                    print("Prod: "+str(prod) +"Arm: " + str(arm_pulled[prod])+"changet at time t: " + str(self.t))
                    self.detections[prod][arm_pulled[prod]].append(self.t)
                    self.valid_reward_per_arms[prod][arm_pulled[prod]] = []
                    self.change_detection[prod][arm_pulled[prod]].reset()

        self.times_visited_as_first_node[num_primary][arm_pulled[num_primary]] += 1
        if num_bought_products[num_primary] > 0:
            self.times_bought_as_first_node[num_primary][arm_pulled[num_primary]] += 1
        for i in range(len(visited_products)):
            if (visited_products[i] == 1) and i != num_primary:
                self.times_visited_from_starting_node[num_primary][arm_pulled[num_primary]][i] += 1


        current_prices = [i[j] for i, j in zip(self.prices, arm_pulled)]
        current_reward = sum(num_bought_products * current_prices)
        self.current_reward.append(current_reward)



    def totalNearbyRewardEstimation(self):
        """
        :return: a matrix containing the nearby rewards for all products and all prices
        """
        # contains the conversion rate of the current best price for each product
        conversion_of_current_best = [i[j] for i, j in zip(self.means, self.currentBestArms)]
        price_of_current_best = np.array([i[j] for i, j in zip(self.prices, self.currentBestArms)])
        num_product_sold_of_current_best = np.array(
            [i[j] for i, j in zip(self.num_product_sold_estimation, self.currentBestArms)])
        nearbyRewardsTable = np.zeros(self.prices.shape)
        # it is created a list containing all the nodes/products that must be visited (initially all the products)
        nodesToVisit = [i for i in range(len(self.prices))]
        for node in nodesToVisit:
            # for each product and each price calculates its nearby reward
            for price in range(len(self.prices[0])):
                nearbyRewardsTable[node][price] = sum(self.visit_probability_estimation[node][price]
                                                      * conversion_of_current_best * price_of_current_best
                                                      * num_product_sold_of_current_best * self.means[node][price])
        return nearbyRewardsTable
