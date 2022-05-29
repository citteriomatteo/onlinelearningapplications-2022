import numpy as np
import random

from Action import Action

from Settings import LAMBDA, CONVERSION_RATE
from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from Social_Influence.Page import Page


class Simulator:
    def __init__(self, graph, alpha_ratios, num_product_sold, secondaries, conversion_rates):
        self.graph = graph
        self.alpha_ratios = alpha_ratios
        self.num_product_sold = num_product_sold
        self.secondaries = secondaries
        self.conversion_rates = conversion_rates

    @staticmethod
    def generateRandomQuantity(mean):
        deviation = mean - 1
        #print("mean: "+str(mean) + " deviation: "+str(deviation))
        return random.randint(round(mean - deviation), round(mean + deviation))

    def simulate(self, selected_prices):

        customer = Customer(reservation_price=100, num_products=len(self.graph.nodes), graph=self.graph)
        num_prod = random.choices([0, 1, 2, 3, 4], self.alpha_ratios[1:], k=1)[0]

        t = 0
        visited_products = np.zeros(len(self.alpha_ratios)-1)
        num_bought_products = np.zeros(len(self.alpha_ratios)-1)

        primary = self.graph.nodes[num_prod]
        second = self.graph.search_product_by_number(self.secondaries[primary.sequence_number][0])
        third = self.graph.search_product_by_number(self.secondaries[primary.sequence_number][1])
        page = Page(primary=primary, second=second, third=third)
        customer.click_on(page)

        visited_products[num_prod] = 1

        while len(customer.pages) > 0:

            action = Action(user=customer)

            #print("\n\n ----- ITERATION: " + str(t) + " -----")

            # -----------------------------------------------------------------------------------
            # 2: CUSTOMERS' CHOICE BETWEEN OPENING A NEW TAB AND USING AN ALREADY OPENED ONE

            # randomized choice: choice of page 0-to-(|pages|-1) or creating a new page

            chosen_index = np.random.randint(low=0, high=len(customer.pages))

            page = customer.pages[chosen_index]
            primary = page.primary
            second = page.second
            third = page.third
            p2 = self.graph.search_edge_by_nodes(primary, second).probability
            p3 = self.graph.search_edge_by_nodes(primary, third).probability

            #customer.print_all_pages()
            #print("· The customer chose the page " + str(chosen_index + 1) + ".")

            action.set_page(page)

            # -----------------------------------------------------------------------------------
            # 4: CUSTOMERS' CHOICE BETWEEN BUYING AND NOT BUYING THE PRIMARY PRODUCT

            if np.random.random() < self.conversion_rates[primary.sequence_number][selected_prices[primary.sequence_number]]:  # PRIMARY PRODUCT BOUGHT
                if not page.bought:
                    quantity = self.generateRandomQuantity(self.num_product_sold[primary.sequence_number][selected_prices[primary.sequence_number]])
                    #print("· The customer buys the primary product in quantity: " + str(quantity) + "!")
                    customer.add_product(product=primary, quantity=quantity)
                    page.set_bought(True)
                    action.set_quantity_bought(quantity=quantity)
                    num_bought_products[page.primary.sequence_number] += quantity

                # -----------------------------------------------------------------------------------
                # 5: CUSTOMERS' CLICK CHOICE BETWEEN: SECOND PRODUCT, THIRD PRODUCT OR CLOSE PAGE
                probabilities = [p2, p3 * LAMBDA, 1 - p2 - p3 * LAMBDA]
                choice = random.choices([0, 1, 2], probabilities, k=1)[0]

                if choice == 0:  # SECONDARY PRODUCT CHOSEN

                    # CREATION OF THE NEW PAGE
                    new_primary = second
                    new_second = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][0])
                    new_third = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][1])

                    # --- page creation and insertion in the list of customer's pages ---
                    new_page = Page(new_primary, new_second, new_third)
                    action.set_click_second(customer.click_on(new_page=new_page))

                    visited_products[new_primary.sequence_number] = 1
                    #print("· The customer clicks on: ", new_primary.name)

                else:
                    if choice == 1:  # THIRD PRODUCT CHOSEN

                        # CREATION OF THE NEW PAGE
                        new_primary = third
                        new_second = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][0])
                        new_third = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][1])

                        # --- page creation and insertion in the list of customer's pages ---
                        new_page = Page(new_primary, new_second, new_third)
                        action.set_click_third(customer.click_on(new_page=new_page))

                        visited_products[new_primary.sequence_number] = 1
                        #print("· The customer clicks on: ", new_primary.name)

                    else:  # CHOSEN "CLOSE PAGE" OPERATION
                        action.set_page_close(customer.close_page(page))
                        #print("· The customer closes the page.")

            else:  # PAGE CLOSED (PRIMARY PRODUCT NOT BOUGHT)
                #print("· The customer closes the page without buying.")
                customer.close_page(page)

            action.compute_for_social_influence(graph=self.graph)

            t += 1

        #print(num_bought_products)
        return visited_products, num_bought_products, num_prod


