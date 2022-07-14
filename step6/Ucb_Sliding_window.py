import numpy as np
from Pricing import Learner
from Pricing.Learner import *
from Pricing.pricing_environment import EnvironmentPricing
from Non_stationary_environment import Non_stationary_environment
from Pricing.Learner import *
from Pricing.pricing_environment import EnvironmentPricing
from Social_Influence.Graph import Graph
from Pricing.Clairvoyant import Clairvoyant
import Settings
from matplotlib import pyplot as plt
from Cumulative_sum import CUSUM

class UCB_Sliding_Window(Learner):
    def __init__(self, n_arms, prices):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.means = np.zeros(prices.shape)
        self.widths = np.ones(prices.shape) * np.inf
        self.num_product_sold_estimation = np.ones(prices.shape)
        self.currentBestArms = np.zeros(len(prices))
        self.nearbyReward = np.zeros(prices.shape)

        self.visit_probability_estimation = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_visited_from_starting_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_visited_as_first_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_bought_as_first_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.n = np.zeros((self.n_products, self.n_arms))

        self.max_boughts = 0
        self.max_rewards = 0

    def reset(self):
        self.__init__(self.n_arms, self.prices, self.graph)

    def act(self):
        """
        :return: for each product returns the arm to pull based on which one gives the highest reward
        :rtype: int
        """
        idx = np.argmax((self.widths + self.means) * ((self.prices * self.num_product_sold_estimation) + self.nearbyReward), axis=1)
        return idx

    def updateHistory(self, arm_pulled, visited_products, num_bought_products, num_primary, day):
        super().update(arm_pulled, visited_products, num_bought_products)
        self.times_visited_as_first_node[num_primary][arm_pulled[num_primary]] += 1
        if num_bought_products[num_primary] > 0:
            self.times_bought_as_first_node[num_primary][arm_pulled[num_primary]] += 1
        for i in range(len(visited_products)):
            if (visited_products[i] == 1) and i != num_primary:
                self.times_visited_from_starting_node[num_primary][arm_pulled[num_primary]][i] += 1
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
        self.average_reward.append(np.mean(self.current_reward[-Settings.DAILY_INTERACTIONS:]))
        num_products = len(arm_pulled)
        '''update mean for every arm pulled for every product'''
        for prod in range(num_products):
            if len(self.rewards_per_arm[prod][arm_pulled[prod]]) > 0:
                self.means[prod][arm_pulled[prod]] = np.mean(self.rewards_per_arm[prod][arm_pulled[prod]][-Settings.WINDOW_SIZE:])
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
        '''update widths for every arm pulled for every product'''
        for prod in range(num_products):
            for arm in range(self.n_arms):
                self.n[prod, arm] = len(self.rewards_per_arm[prod][arm])
                if self.n[prod, arm] > 0:
                    self.widths[prod][arm] = np.sqrt((2 * np.max(np.log(self.t)) / self.n[prod, arm]))
                else:
                    self.widths[prod][arm] = np.inf
        self.nearbyReward = self.totalNearbyRewardEstimation()
        self.nearbyReward[np.isnan(self.nearbyReward)] = 0



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
                                                      * num_product_sold_of_current_best * self.means[node][price])
        return nearbyRewardsTable



graph = Graph(mode="full", weights=True)
env = Non_stationary_environment(4, graph, 1)
# ho tolto env.secondaries
learner = UCB_Sliding_Window(4, env.prices)

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
      [0.4, 0.95, 0.35, 0.3]]
  ]


clairvoyant = Clairvoyant(env.prices, env.conversion_rates, env.classes, env.secondaries, env.num_product_sold, graph, env.alpha_ratios)
best_revenue = clairvoyant.revenue_given_arms([0, 1, 2, 2, 3], 0)
print(best_revenue)

clairvoyant_after_change = Clairvoyant(env.prices, new_conv_rates, env.classes, env.secondaries, env.num_product_sold, graph, env.alpha_ratios)
best_revenue_after_change = clairvoyant_after_change.revenue_given_arms([0, 2, 1, 0, 2], 0)
print(best_revenue_after_change)

best_revenue_array = [best_revenue for i in range(500)] + [best_revenue_after_change for i in range(500)]



for i in range(Settings.NUM_OF_DAYS):
    pulled_arms = learner.act()
    if i%100==0: print(pulled_arms)
    for j in range(Settings.DAILY_INTERACTIONS):
        visited_products, num_bought_products, num_primary = env.round(pulled_arms)
        learner.updateHistory(pulled_arms, visited_products, num_bought_products,num_primary, i)
    learner.update(pulled_arms)
    if(i==500):
        print("Cambio conv rates"+str(i))
        env.setNewConvRates(new_conv_rates)


#print(learner.means)
#print(learner.widths)


fig, ax = plt.subplots(nrows=2,ncols=1, figsize=(12,8))
ax[0].plot(learner.average_reward, color='blue', label='UCB Sliding Window')
ax[0].axhline(y=best_revenue, xmin=0., xmax=0.5, color='red', linestyle='--', label='Clairvoyant')
ax[0].axhline(y=best_revenue_after_change, xmin=0.5, xmax=1., color='green', linestyle='--', label='Clairvoyant after abrupt change')
ax[0].set_title('Average reward')
ax[0].axvline(x=500+Settings.WINDOW_SIZE)
ax[0].axvline(x=500)
ax[1].plot(np.cumsum(learner.average_reward), color='blue', label='UCB Sliding Window')
ax[1].plot(np.cumsum(best_revenue_array)[:500], color='red', linestyle='--', label='Clairvoyant')
x = [i for i in range(501,1000)]
ax[1].plot(x, np.cumsum(best_revenue_array)[501:], color='green', linestyle='--', label='Clairvoyant after abrupt change')
ax[1].set_title('Cumulative reward')
ax[1].axvline(x=500)
ax[0].legend()
ax[1].legend()
plt.show()