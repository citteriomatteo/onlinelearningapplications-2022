from random import random

import numpy as np
import random

import Settings
from Action import Action
from Pricing.Learner import *
from Pricing.pricing_environment import EnvironmentPricing
from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from Social_Influence.Page import Page


class Ucb(Learner):
    def __init__(self, n_arms, prices, secondaries, num_product_sold, graph):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.pricesMeanPerProduct = np.mean(self.prices, 1)
        self.means = np.zeros(prices.shape)
        self.nearbyReward = np.zeros((self.n_products, self.n_arms))
        self.widths = np.ones(prices.shape) * np.inf
        self.graph = graph
        self.secondaries = secondaries
        self.currentBestArms = np.zeros(len(prices))
        self.num_product_sold = num_product_sold
        self.n = np.zeros((self.n_products, self.n_arms))
        self.visit_probability_estimation = np.zeros((self.n_products, self.n_products))

    def reset(self):
        self.__init__(self.n_arms, self.prices, self.graph)

    def act(self):
        """
        :return: for each product returns the arm to pull based on which one gives the highest reward
        :rtype: int
        """
        idx = np.argmax((self.widths + self.means) * ((self.prices*self.num_product_sold) + self.nearbyReward), axis=1)
        return idx

    def updateHistory(self, arm_pulled, visited_products, num_bought_products):
        super().update(arm_pulled, visited_products, num_bought_products)

    def update(self, arm_pulled):
        """
        update mean and widths
        :param arm_pulled: arm pulled for every product
        :type arm_pulled: list
        :return: none
        :rtype: none
        """

        self.currentBestArms = arm_pulled
        self.nearbyReward = np.zeros((self.n_products, self.n_arms))
        self.visit_probability_estimation = self.simulateTotalNearby(arm_pulled)
        for prod in range(self.n_products):
            for price in range(self.n_arms):
                for temp in range(self.n_products):
                    self.nearbyReward[prod][price] += self.means[prod][price]*self.visit_probability_estimation[prod][temp]*self.means[temp][self.currentBestArms[temp]]*self.num_product_sold[temp][self.currentBestArms[temp]]*self.prices[temp][self.currentBestArms[temp]]

        for prod in range(self.n_products):
            self.means[prod][arm_pulled[prod]] = np.mean(self.rewards_per_arm[prod][arm_pulled[prod]])
        for prod in range(self.n_products):
            for arm in range(self.n_arms):
                self.n[prod,arm] = len(self.rewards_per_arm[prod][arm])
                if (self.n[prod,arm]) > 0:
                    self.widths[prod][arm] = np.sqrt((2 * np.max(np.log(self.t)) / self.n[prod,arm]))
                else:
                    self.widths[prod][arm] = np.inf

    def simulateTotalNearby(self, selected_price):
        times_visited_from_starting_node = np.zeros((self.n_products, self.n_products))
        for prod in range(self.n_products):
            for iteration in range(364):
                visited_products_ = self.simulateSingleNearby(selected_price, prod)
                for j in range(len(visited_products_)):
                    if (visited_products_[j] == 1) and j != prod:
                        times_visited_from_starting_node[prod][j] += 1
        return times_visited_from_starting_node / 364

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

graph = Graph(mode="full", weights=True)
env = EnvironmentPricing(4, graph, 1)
learner = Ucb(4, env.prices, env.secondaries, env.num_product_sold[0], graph)

for i in range(10000):
    if i == 9900:
        aaa = 1
    pulled_arms = learner.act()
    print(pulled_arms)

    visited_products, num_bought_products, a = env.round(pulled_arms)
    #print("Vidited products: ", visited_products)
    #print("Number of product sold: ", num_bought_products)

    learner.updateHistory(pulled_arms, visited_products, num_bought_products)

    # TODO  non hardcodare
    if (i % 10 == 0) and (i != 0):
        learner.update(pulled_arms)

    #print("Number of rewards per product:", learner.n)
    #print("T:",learner.t)
    #print("Means: ",learner.means)
    #print("Widths:",learner.widths)

print(learner.means)
print(learner.widths)
print((learner.widths + learner.means) * ((learner.prices*learner.num_product_sold) + learner.nearbyReward))

