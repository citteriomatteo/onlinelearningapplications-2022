import numpy as np
import random


class Customer:
    """
    states: -1 -> inactive
             0 -> susceptible
             1 -> active
    """

    def __init__(self, reservation_price, num_products, graph):
        self.reservation_price = reservation_price
        self.products_state = np.zeros(shape=num_products)
        self.graph = graph
        self.pages = []
        self.cart = []
        self.features = self.get_randomized_features()


    def set_susceptible(self, prod_number):
        self.products_state[prod_number] = 0

    def set_active(self, prod_number):
        if self.products_state[prod_number] == 0:
            self.products_state[prod_number] = 1
            return True

        return False

    def set_inactive(self, prod_number):
        if self.products_state[prod_number] != 0:
            self.products_state[prod_number] = -1
            return True

        return False

    def add_product(self, product, quantity):
        if quantity > 0:
            for i in range(quantity):
                self.cart.append(product)

    def click_on(self, new_page):
        if self.set_active(prod_number=new_page.primary.sequence_number):
            self.add_new_page(new_page=new_page)
            return True

        return False

    def add_new_page(self, new_page):
        self.pages.append(new_page)

    def direct_close_page(self, page):
        self.pages.remove(page)

    def close_page(self, page):
        if self.set_inactive(prod_number=page.primary.sequence_number):
            self.pages.remove(page)
            return True

        return False

    def direct_close_page(self, page):
        self.pages.remove(page)

    def get_randomized_features(self):
        """
        randomizes the features for the customer
        """
        features = []
        for i in range(2):
            features.append(random.choice([True, False]))

        return features

    def print_all_pages(self):
        first_prods = ""
        second_prods = ""
        third_prods = ""

        print("\nPAGES:")
        if len(self.pages) == 0:
            print("No pages.\n")
        else:
            header = ""
            for i in range(len(self.pages)):
                header += "  PAGE " + str(i + 1) + "             "
                first_prods += self.pages[i].get_formatted_primary()
                second_prods += self.pages[i].get_formatted_second()
                third_prods += self.pages[i].get_formatted_third()

            print(header)
            print(first_prods)
            print(second_prods)
            print(third_prods)
            print()
