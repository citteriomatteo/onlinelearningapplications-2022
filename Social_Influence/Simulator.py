import numpy as np
import random

from Action import Action
from Settings import LAMBDA, CONVERSION_RATE
from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from Social_Influence.Page import Page


class Simulator:
    def __init__(self, graph):
        self.graph = graph

    def simulate(self, customer, num_prod):

        t = 0
        visited_products = np.zeros(len(self.graph.nodes))
        num_bought_products = np.zeros(len(self.graph.nodes))

        primary = self.graph.nodes[num_prod]
        # TODO change the method for taking second and third
        second, third, p2, p3 = self.graph.pull_arms(node1=primary, products_state=customer.products_state)
        page = Page(primary=primary, second=second, third=third)
        customer.click_on(page)

        visited_products[num_prod] = 1

        while len(customer.pages) > 0:

            action = Action(user=customer)

            print("\n\n ----- ITERATION: " + str(t) + " -----")

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

            customer.print_all_pages()
            print("· The customer chose the page " + str(chosen_index + 1) + ".")

            action.set_page(page)

            # -----------------------------------------------------------------------------------
            # 4: CUSTOMERS' CHOICE BETWEEN BUYING AND NOT BUYING THE PRIMARY PRODUCT

            if np.random.random() < CONVERSION_RATE:  # PRIMARY PRODUCT BOUGHT

                quantity = np.random.randint(1, 5)
                print("· The customer buys the primary product in quantity: " + str(quantity) + "!")
                customer.add_product(product=primary, quantity=quantity)
                action.set_quantity_bought(quantity=quantity)
                num_bought_products[page.primary.sequence_number] += quantity

                # -----------------------------------------------------------------------------------
                # 5: CUSTOMERS' CLICK CHOICE BETWEEN: SECOND PRODUCT, THIRD PRODUCT OR CLOSE PAGE
                probabilities = [p2, p3 * LAMBDA, 1 - p2 - p3 * LAMBDA]
                choice = random.choices([0, 1, 2], probabilities, k=1)[0]

                if choice == 0:  # SECONDARY PRODUCT CHOSEN

                    # CREATION OF THE NEW PAGE
                    new_primary = second
                    new_second, new_third, new_p2, new_p3 = self.graph.pull_arms(node1=new_primary,
                                                                                 products_state=customer.products_state)

                    # --- page creation and insertion in the list of customer's pages ---
                    new_page = Page(new_primary, new_second, new_third)
                    action.set_click_second(customer.click_on(new_page=new_page))

                    visited_products[new_primary.sequence_number] = 1
                    print("· The customer clicks on: ", new_primary.name)

                else:
                    if choice == 1:  # THIRD PRODUCT CHOSEN

                        # CREATION OF THE NEW PAGE
                        new_primary = third
                        new_second, new_third, new_p2, new_p3 = self.graph.pull_arms(node1=new_primary,
                                                                                     products_state=customer.
                                                                                     products_state)

                        # --- page creation and insertion in the list of customer's pages ---
                        new_page = Page(new_primary, new_second, new_third)
                        action.set_click_third(customer.click_on(new_page=new_page))

                        visited_products[new_primary.sequence_number] = 1
                        print("· The customer clicks on: ", new_primary.name)

                    else:  # CHOSEN "CLOSE PAGE" OPERATION
                        action.set_page_close(customer.close_page(page))
                        print("· The customer closes the page.")

            else:  # PAGE CLOSED (PRIMARY PRODUCT NOT BOUGHT)
                print("· The customer closes the page without buying.")
                customer.close_page(page)

            action.compute_for_social_influence(graph=self.graph)

            t += 1

        return visited_products, num_bought_products



