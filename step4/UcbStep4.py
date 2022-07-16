import numpy as np
from matplotlib import pyplot as plt

import Settings
from Pricing.Clairvoyant import Clairvoyant
from Pricing.Learner import *
from Pricing.pricing_environment import EnvironmentPricing
from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from Social_Influence.Page import Page


class Ucb(Learner):
    def __init__(self, n_arms, prices, secondaries, graph):
        super().__init__(n_arms, len(prices))
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
        self.alpha_ratios = np.zeros(self.n_products)
        self.times_visited_as_first_node = np.zeros(self.n_products)

    def reset(self):
        self.__init__(self.n_arms, self.prices, self.graph)

    def act(self):
        """
        :return: for each product returns the arm to pull based on which one gives the highest reward
        :rtype: int
        """
        aaa = (self.widths + self.means)
        bbb = ((self.prices*self.num_product_sold_estimation) + self.nearbyReward)
        idx = np.argmax((self.widths + self.means) * ((self.prices*self.num_product_sold_estimation) + self.nearbyReward), axis=1)
        return idx

    def revenue_given_arms(self, arms):
        means = [i[j] for i, j in zip(self.means, arms)]
        prices = [i[j] for i, j in zip(self.prices, arms)]
        num_product_sold = [i[j] for i, j in zip(self.num_product_sold_estimation, arms)]
        nearby_reward = [i[j] for i, j in zip(self.nearbyReward, arms)]
        return np.sum(np.multiply(self.alpha_ratios, np.multiply(means, np.multiply(prices, num_product_sold))+nearby_reward))

    def simulateTotalNearby(self, selected_price):
        times_visited_from_starting_node = np.zeros((self.n_products, self.n_products))
        for prod in range(self.n_products):
            for iteration in range(Settings.NUM_MC_SIMULATIONS):
                visited_products_ = self.simulateSingleNearby(selected_price, prod)
                for j in range(len(visited_products_)):
                    if (visited_products_[j] == 1) and j != prod:
                        times_visited_from_starting_node[prod][j] += 1
        return times_visited_from_starting_node / Settings.NUM_MC_SIMULATIONS

    def simulateSingleNearby(self, selected_prices, starting_node):
        customer = Customer(reservation_price=100, num_products=len(self.graph.nodes), graph=self.graph)
        num_prod = starting_node
        t = 0
        visited_products = np.zeros(len(selected_prices))
        num_bought_products = np.zeros(len(selected_prices))
        primary = self.graph.nodes[num_prod]
        second = self.graph.search_product_by_number(self.secondaries[primary.sequence_number][0])
        third = self.graph.search_product_by_number(self.secondaries[primary.sequence_number][1])
        page = Page(primary=primary, second=second, third=third)
        customer.click_on(page)
        visited_products[num_prod] = 1
        is_starting_node = True

        while len(customer.pages) > 0:
            # action = Action(user=customer)
            # -----------------------------------------------------------------------------------
            # 2: CUSTOMERS' CHOICE BETWEEN OPENING A NEW TAB AND USING AN ALREADY OPENED ONE
            # randomized choice: choice of page 0-to-(|pages|-1) or creating a new page
            chosen_index = np.random.randint(low=0, high=len(customer.pages))
            page = customer.pages[chosen_index]
            primary = page.primary
            second = page.second
            third = page.third
            p2 = self.graph.search_edge_by_nodes(primary, second).probability if (
                    visited_products[second.sequence_number] == 0) else 0
            p3 = self.graph.search_edge_by_nodes(primary, third).probability if (
                    visited_products[third.sequence_number] == 0) else 0
            # action.set_page(page)
            superare = self.means[primary.sequence_number][selected_prices[primary.sequence_number]]
            # -----------------------------------------------------------------------------------
            # 4: CUSTOMERS' CHOICE BETWEEN BUYING AND NOT BUYING THE PRIMARY PRODUCT
            if (is_starting_node) or (np.random.random() < superare):  # PRIMARY PRODUCT BOUGHT
                is_starting_node = False

                quantity = 0
                customer.add_product(product=primary, quantity=quantity)
                page.set_bought(True)
                # action.set_quantity_bought(quantity=quantity)
                num_bought_products[page.primary.sequence_number] += quantity

                # -----------------------------------------------------------------------------------
                # 5: CUSTOMERS' CLICK CHOICE BETWEEN: SECOND PRODUCT, THIRD PRODUCT OR CLOSE PAGE
                choice = [None] * 2

                if np.random.random() < p2:  # SECONDARY BOUGHT
                    choice[0] = 1
                    visited_products[second.sequence_number] += 1
                if np.random.random() < p3:  # TERTIARY BOUGHT
                    choice[1] = 1
                    visited_products[third.sequence_number] += 1

                if choice[0] == 1:  # SECONDARY PRODUCT CHOSEN
                    # CREATION OF THE NEW PAGE
                    new_primary = second
                    new_second = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][0])
                    new_third = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][1])

                    # --- page creation and insertion in the list of customer's pages ---
                    new_page = Page(new_primary, new_second, new_third)
                    customer.add_new_page(new_page)

                if choice[1] == 1:  # THIRD PRODUCT CHOSEN
                    # CREATION OF THE NEW PAGE
                    new_primary = third
                    new_second = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][0])
                    new_third = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][1])

                    # --- page creation and insertion in the list of customer's pages ---
                    new_page = Page(new_primary, new_second, new_third)
                    customer.add_new_page(new_page)

            customer.direct_close_page(page)
            # action.compute_for_social_influence(graph=self.graph)
            t += 1
        return visited_products

    def updateHistory(self, arm_pulled, visited_products, num_bought_products, num_primary):
        super().update(arm_pulled, visited_products, num_bought_products)
        self.times_visited_as_first_node[num_primary] += 1
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
        for prod in range(self.n_products):
            self.means[prod][arm_pulled[prod]] = np.mean(self.rewards_per_arm[prod][arm_pulled[prod]])
            self.num_product_sold_estimation[prod][arm_pulled[prod]] = np.mean(self.boughts_per_arm[prod][arm_pulled[prod]])
            self.alpha_ratios[prod] = self.times_visited_as_first_node[prod] / np.sum(self.times_visited_as_first_node)
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                self.n[prod,arm] = len(self.rewards_per_arm[prod][arm])
                if (self.n[prod,arm]) > 0:
                    self.widths[prod][arm] = np.sqrt((2 * np.max(np.log(self.t)) / self.n[prod,arm]))
                else:
                    self.widths[prod][arm] = np.inf
        self.nearbyReward = np.zeros((self.n_products, self.n_arms))
        self.visit_probability_estimation = self.simulateTotalNearby(arm_pulled)
        self.visit_probability_estimation[np.isnan(self.visit_probability_estimation)] = 0
        self.num_product_sold_estimation[np.isnan(self.num_product_sold_estimation)] = 1
        for prod in range(self.n_products):
            for price in range(self.n_arms):
                for temp in range(self.n_products):
                    self.nearbyReward[prod][price] += self.means[prod][price]*self.visit_probability_estimation[prod][temp]*self.means[temp][self.currentBestArms[temp]]*self.num_product_sold_estimation[temp][self.currentBestArms[temp]]*self.prices[temp][self.currentBestArms[temp]]

'''''''''
final_reward= np.zeros((Settings.NUM_PLOT_ITERATION, Settings.NUM_OF_DAYS))
final_cumulative_regret = np.zeros((Settings.NUM_PLOT_ITERATION, Settings.NUM_OF_DAYS))
final_cumulative_reward = np.zeros((Settings.NUM_PLOT_ITERATION, Settings.NUM_OF_DAYS))

for k in range (Settings.NUM_PLOT_ITERATION):
    graph = Graph(mode="full", weights=True)
    env = EnvironmentPricing(4, graph, 1)
    learner = Ucb(4, env.prices, env.secondaries, graph)
    clairvoyant = Clairvoyant(env.prices, env.conversion_rates, env.classes, env.secondaries, env.num_product_sold,
                              graph, env.alpha_ratios)
    best_revenue = clairvoyant.revenue_given_arms([0, 1, 2, 2, 3], 0)
    opt_rew = []
    actual_rew = []
    for i in range(Settings.NUM_OF_DAYS):
        pulled_arms = learner.act()
        print(pulled_arms)
        for j in range(Settings.DAILY_INTERACTIONS):
            visited_products, num_bought_products, num_primary = env.round(pulled_arms)
            learner.updateHistory(pulled_arms, visited_products, num_bought_products, num_primary)

        learner.update(pulled_arms)
        actual_rew.append(learner.revenue_given_arms(pulled_arms))
        opt_rew.append(best_revenue)

    a = learner.revenue_given_arms(pulled_arms)
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



best_revenue_array = [best_revenue for i in range(Settings.NUM_OF_DAYS)]


fig, ax = fig, ax = plt.subplots(nrows=3,ncols=1, figsize=(12,12))
ax[0].plot(mean_cumulative_regret, color='blue', label='UCB-4')
ax[0].fill_between(range(Settings.NUM_OF_DAYS), mean_cumulative_regret - stdev_regret,mean_cumulative_regret + stdev_regret, alpha=0.4)
ax[0].set_title('Cumulative Regret')

ax[1].plot(mean_cumulative_reward, color='blue', label='UCB-4')
ax[1].fill_between(range(Settings.NUM_OF_DAYS), mean_cumulative_reward - stdev_cumulative_reward, mean_cumulative_reward + stdev_cumulative_reward, alpha=0.4)
ax[1].plot(np.cumsum(best_revenue_array), color='red', linestyle='--', label='Clairvoyant')
ax[1].set_title('Cumulative reward')

ax[2].plot(mean_final_reward, color='blue', label='UCB-4')
ax[2].fill_between(range(Settings.NUM_OF_DAYS), mean_final_reward - stdev_reward, mean_final_reward + stdev_reward, alpha=0.4)
ax[2].axhline(y=best_revenue, color='red', linestyle='--', label='Clairvoyant')
ax[2].set_title('Reward')


ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()

'''''

