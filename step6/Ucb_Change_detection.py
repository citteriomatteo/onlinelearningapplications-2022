from Non_stationary_environment import Non_stationary_environment
import numpy as np
from Pricing.Learner import *
from Pricing.pricing_environment import EnvironmentPricing
from Social_Influence.Graph import Graph
from Pricing.Clairvoyant import Clairvoyant
import Settings
from matplotlib import pyplot as plt
from Cumulative_sum import CUSUM


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



new_conv_rates=[
    [
      [0.7, 0.47, 0.45, 0.2],
      [0.35, 0.3, 0.8, 0.25],
      [0.45, 0.8, 0.4, 0.4],
      [0.7, 0.35, 0.32, 0.25],
      [0.5, 0.55, 0.85, 0.52] ],

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

final_reward= np.zeros((Settings.NUM_PLOT_ITERATION, Settings.NUM_OF_DAYS))
final_cumulative_regret = np.zeros((Settings.NUM_PLOT_ITERATION, Settings.NUM_OF_DAYS))
final_cumulative_reward = np.zeros((Settings.NUM_PLOT_ITERATION, Settings.NUM_OF_DAYS))

for k in range (Settings.NUM_PLOT_ITERATION):
    graph = Graph(mode="full", weights=True)
    env = Non_stationary_environment(4, graph, 1)
    learner = Ucb_Change_detection(4, env.prices,env.secondaries)

    clairvoyant = Clairvoyant(env.prices, env.conversion_rates, env.classes, env.secondaries, env.num_product_sold, graph, env.alpha_ratios)
    clairvoyant_after_change = Clairvoyant(env.prices, new_conv_rates, env.classes, env.secondaries, env.num_product_sold, graph, env.alpha_ratios)
    best_revenue = clairvoyant.revenue_given_arms([0, 1, 2, 2, 3], 0)
    best_revenue_after_change = clairvoyant_after_change.revenue_given_arms([0, 2, 1, 0, 2], 0)
    best_revenue_array = [best_revenue for i in range(Settings.DAY_OF_ABRUPT_CHANGE)] + [best_revenue_after_change for i in range(Settings.DAY_OF_ABRUPT_CHANGE)]
    opt_rew = []
    actual_rew = []
    best_rew = best_revenue
    for i in range(Settings.NUM_OF_DAYS):
        pulled_arms = learner.act()
        print(pulled_arms)
        if i==Settings.DAY_OF_ABRUPT_CHANGE:
            env.setNewConvRates(new_conv_rates)
            print("Cambio")
            best_rew=best_revenue_after_change
        for j in range(Settings.DAILY_INTERACTIONS):
            visited_products, num_bought_products, num_primary = env.round(pulled_arms)
            learner.updateHistory(pulled_arms, visited_products, num_bought_products, num_primary)

        learner.update(pulled_arms)
        actual_rew.append(learner.revenue_given_arms(arms=pulled_arms))
        print(actual_rew[-1])
        opt_rew.append(best_rew)

    final_cumulative_regret[k, :] = np.cumsum(opt_rew) - np.cumsum(actual_rew)
    final_cumulative_reward[k,:] = np.cumsum(actual_rew)
    final_reward[k:] = actual_rew


#REGRET
print("FINAL CUM REGRET: ")
print(final_cumulative_regret)

mean_cumulative_regret = np.mean(final_cumulative_regret, axis=0)
stdev_regret= np.std(final_cumulative_regret, axis=0) / np.sqrt(Settings.NUM_OF_DAYS)
print("MEAN: ")
print(mean_cumulative_regret)

#Cumulative REWARD
print("FINAL CUM REWARD: ")
print(final_cumulative_reward)

mean_cumulative_reward = np.mean(final_cumulative_reward, axis=0)
stdev_cumulative_reward= np.std(final_cumulative_reward, axis=0) / np.sqrt(Settings.NUM_OF_DAYS)
print("MEAN: ")
print(mean_cumulative_reward)

#AREWARD
print("FINAL REWARD: ")
print(final_reward)

mean_final_reward = np.mean(final_reward, axis=0)
stdev_reward= np.std(final_reward, axis=0) / np.sqrt(Settings.NUM_OF_DAYS)
print("MEAN: ")
print(mean_final_reward)

fig, ax = plt.subplots(nrows=3,ncols=1, figsize=(12,12))
ax[0].plot(mean_cumulative_regret, color='blue', label='UCB Change Detection')
ax[0].fill_between(range(Settings.NUM_OF_DAYS), mean_cumulative_regret - stdev_regret,mean_cumulative_regret + stdev_regret, alpha=0.4)
ax[0].set_title('Cumulative Regret')

ax[1].plot(mean_cumulative_reward, color='blue', label='UCB Change Detection')
ax[1].fill_between(range(Settings.NUM_OF_DAYS), mean_cumulative_reward - stdev_cumulative_reward, mean_cumulative_reward + stdev_cumulative_reward, alpha=0.4)
ax[1].plot(np.cumsum(best_revenue_array)[:Settings.DAY_OF_ABRUPT_CHANGE], color='red', linestyle='--', label='Clairvoyant')
xx = [i for i in range(Settings.DAY_OF_ABRUPT_CHANGE+1,Settings.NUM_OF_DAYS)]
ax[1].plot(xx, np.cumsum(best_revenue_array)[Settings.DAY_OF_ABRUPT_CHANGE+1:],
           color='green', linestyle='--', label='Clairvoyant after abrupt change')
ax[1].set_title('Cumulative reward')

ax[2].plot(mean_final_reward, color='blue', label='UCB Change Detection')
ax[2].fill_between(range(Settings.NUM_OF_DAYS), mean_final_reward - stdev_reward, mean_final_reward + stdev_reward, alpha=0.4)
ax[2].axhline(y=best_revenue, xmin=0., xmax=Settings.DAY_OF_ABRUPT_CHANGE/Settings.NUM_OF_DAYS,
              color='red', linestyle='--', label='Clairvoyant')
ax[2].axhline(y=best_revenue_after_change, xmin=Settings.DAY_OF_ABRUPT_CHANGE/Settings.NUM_OF_DAYS, xmax=1.,
              color='green', linestyle='--', label='Clairvoyant after abrupt change')
ax[2].set_title('Reward')

ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()





