import random

import numpy as np

from Pricing.pricing_environment import EnvironmentPricing
from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from Social_Influence.Page import Page

class nearbySimulator():
    def __init__(self, graph, num_product_sold, secondaries, conversion_rates):
        self.graph = graph
        self.num_product_sold = num_product_sold
        self.secondaries = secondaries
        self.conversion_rates = conversion_rates
        self.n_products = 5

    @staticmethod
    def generateRandomQuantity(mean):
        deviation = mean - 1
        # print("mean: "+str(mean) + " deviation: "+str(deviation))
        return random.randint(round(mean - deviation), round(mean + deviation))

    def simulateTotalNearby(self, selected_price):
        times_visited_from_starting_node = np.zeros((self.n_products, 5))
        #for prod in range(self.n_products):
        for iteration in range(10000):
            visited_products_, num_bought_ = self.simulateSingleNearby(selected_price,1)
            for j in range(len(visited_products_)):
                times_visited_from_starting_node[1][j] += visited_products_[j]
        return times_visited_from_starting_node / 10000

    def simulateSingleNearby(self, selected_prices, numm, customer=None):

        if customer is None:
            customer = Customer(reservation_price=100, num_products=len(self.graph.nodes), graph=self.graph)

        #num_prod = random.choices([0, 1, 2, 3, 4], self.alpha_ratios[1:], k=1)[0]
        num_prod=numm

        t = 0
        visited_products = np.zeros(5)
        num_bought_products = np.zeros(5)

        primary = self.graph.nodes[num_prod]
        second = self.graph.search_product_by_number(self.secondaries[primary.sequence_number][0])
        third = self.graph.search_product_by_number(self.secondaries[primary.sequence_number][1])
        page = Page(primary=primary, second=second, third=third)
        customer.click_on(page)
        is_starting_node = True
        visited_products[num_prod] = 1

        while len(customer.pages) > 0:

            # print("\n\n ----- ITERATION: " + str(t) + " -----")

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
                    #customer.click_on(new_page)

                if choice[1] == 1:  # THIRD PRODUCT CHOSEN
                    # CREATION OF THE NEW PAGE
                    new_primary = third
                    new_second = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][0])
                    new_third = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][1])

                    # --- page creation and insertion in the list of customer's pages ---
                    new_page = Page(new_primary, new_second, new_third)
                    customer.add_new_page(new_page)
                    #customer.click_on(new_page)

            customer.direct_close_page(page)

            t += 1

        # print(num_bought_products)
        return visited_products, num_bought_products

graph = Graph(mode="full", weights=True)
env = EnvironmentPricing(4, graph, 1)
learner = nearbySimulator(graph, env.num_product_sold[0], env.secondaries, env.conversion_rates[0])
aaa = learner.simulateTotalNearby([0,1,2,2,3])
print(aaa)