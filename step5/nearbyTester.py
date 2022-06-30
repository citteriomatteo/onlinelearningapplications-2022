import random

import numpy as np

from Pricing.pricing_environment import EnvironmentPricing
from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from Social_Influence.Page import Page

class nearbySimulator():
    def __init__(self, graph, num_product_sold, secondaries, conversion_rates, alpha_ratios):
        self.times_bought_as_first_node = np.zeros((5, 5))
        self.times_visited_as_first_node = np.zeros((5, 5))
        self.times_visited_from_starting_node = np.zeros((5, 5))
        self.graph = graph
        self.num_product_sold = num_product_sold
        self.secondaries = secondaries
        self.conversion_rates = conversion_rates
        self.n_products = 5
        self.alpha_ratios = alpha_ratios


    @staticmethod
    def generateRandomQuantity(mean):
        deviation = mean - 1
        # print("mean: "+str(mean) + " deviation: "+str(deviation))
        return random.randint(round(mean - deviation), round(mean + deviation))

    def simulateTotalNearby(self, selected_price):
        times_visited_from_starting_node = np.zeros((self.n_products, 5))
        #for prod in range(self.n_products):
        for iteration in range(10000):
            visited_products_, num_bought_products, primary = self.simulateSingleNearby(selected_price)
            self.times_visited_as_first_node[primary] += 1
            if num_bought_products[primary] > 0:
                self.times_bought_as_first_node[primary] += 1
            for i in range(len(visited_products_)):
                if (visited_products_[i] == 1) and i != primary:
                    self.times_visited_from_starting_node[primary][i] += 1
        return self.times_visited_from_starting_node / self.times_visited_as_first_node, self.times_visited_from_starting_node / self.times_bought_as_first_node

    def simulateSingleNearby(self, selected_prices, customer=None):

        if customer is None:
            customer = Customer(reservation_price=100, num_products=len(self.graph.nodes), graph=self.graph)

        num_prod = random.choices([0, 1, 2, 3, 4], self.alpha_ratios[1:], k=1)[0]

        t = 0
        visited_products = np.zeros(5)
        num_bought_products = np.zeros(5)

        primary = self.graph.nodes[num_prod]
        second = self.graph.search_product_by_number(self.secondaries[primary.sequence_number][0])
        third = self.graph.search_product_by_number(self.secondaries[primary.sequence_number][1])
        page = Page(primary=primary, second=second, third=third)
        customer.click_on(page)
        visited_products[num_prod] = 1

        while len(customer.pages) > 0:
            chosen_index = np.random.randint(low=0, high=len(customer.pages))

            page = customer.pages[chosen_index]
            primary = page.primary
            second = page.second
            third = page.third
            p2 = self.graph.search_edge_by_nodes(primary, second).probability if (
                    visited_products[second.sequence_number] == 0) else 0
            p3 = self.graph.search_edge_by_nodes(primary, third).probability if (
                    visited_products[third.sequence_number] == 0) else 0

            # customer.print_all_pages()
            # print("· The customer chose the page " + str(chosen_index + 1) + ".")

            # -----------------------------------------------------------------------------------
            # 4: CUSTOMERS' CHOICE BETWEEN BUYING AND NOT BUYING THE PRIMARY PRODUCT
            if np.random.random() < self.conversion_rates[primary.sequence_number][
                selected_prices[primary.sequence_number]]:  # PRIMARY PRODUCT BOUGHT

                quantity = self.generateRandomQuantity(
                    self.num_product_sold[primary.sequence_number][selected_prices[primary.sequence_number]])
                # customer.add_product(product=primary, quantity=quantity)
                page.set_bought(True)
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
                    # customer.click_on(new_page)

                if choice[1] == 1:  # THIRD PRODUCT CHOSEN
                    # CREATION OF THE NEW PAGE
                    new_primary = third
                    new_second = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][0])
                    new_third = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][1])

                    # --- page creation and insertion in the list of customer's pages ---
                    new_page = Page(new_primary, new_second, new_third)
                    customer.add_new_page(new_page)
                    # customer.click_on(new_page)

            customer.direct_close_page(page)

            t += 1

        # print(num_bought_products)
        return visited_products, num_bought_products, num_prod

graph = Graph(mode="full", weights=True)
env = EnvironmentPricing(4, graph, 1)
learner = nearbySimulator(graph, env.num_product_sold[0], env.secondaries, env.conversion_rates[0], env.alpha_ratios[0])
aaa, bbb = learner.simulateTotalNearby([0,1,2,2,3])
print(aaa)
print(bbb)