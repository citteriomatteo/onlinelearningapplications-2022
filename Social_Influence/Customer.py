import numpy as np
import random


class Customer:
    """
    states: -1 -> inactive
             0 -> susceptible
             1 -> active
    """

    def __init__(self, reservation_price, num_products, graph, env=None):
        self.reservation_price = reservation_price
        self.products_state = np.zeros(shape=num_products)
        self.graph = graph
        self.pages = []
        self.cart = []
        if env is not None:
            self.features_class, self.features = self.get_right_user_class(env.classes)

            # setting of the simulator with the characteristics related to this specific customer's class
            env.simulator.set_alpha_ratios(env.alpha_ratios[self.features_class])
            env.simulator.set_num_product_sold(env.num_product_sold[self.features_class])
            env.simulator.set_conversion_rates(env.conversion_rates[self.features_class])

        else:       # if features are not useful (step 1-6) -> force the Customer to the class 1
            self.features_class = 0
            self.features = [True, True]


    def get_right_user_class(self, classes=None):
        """
        Gets the class of the Customer considering the fractions of probabilities on the .json file.
        """
        class_idx = 0
        features = None

        probs = [classes[c]['fraction'] for c in classes]
        class_names = list(classes.keys())

        class_idx = random.choices([0, 1, 2], probs, k=1)[0]
        features = classes[class_names[class_idx]]['features']

        return class_idx, features

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
