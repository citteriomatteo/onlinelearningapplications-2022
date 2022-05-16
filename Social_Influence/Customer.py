import numpy as np


class Customer:
    """
    states: -1 -> inactive
             0 -> susceptible
             1 -> active
    """

    products_state = []
    cart = []

    def __init__(self, reservation_price, num_products, graph):
        self.reservation_price = reservation_price
        self.products_state = np.zeros(shape=num_products)
        self.graph = graph

    def set_susceptible(self, prod_number):
        self.products_state[prod_number] = 0

    def set_active(self, prod_number):
        if self.products_state[prod_number] == 0:
            self.products_state[prod_number] = 1
            return True

        return False

    def set_inactive(self, prod_number):
        if self.products_state[prod_number] == 1:
            self.products_state[prod_number] = -1
            return True

        return False

    def add_product(self, product, quantity):
        if quantity > 0:
            for i in range(quantity):
                self.cart.append(product)

    def click_on(self, first, second):
        if self.set_active(prod_number=second.sequence_number):
            self.graph.update_estimation(node=second, reward=1)

        """
        ASSUMPTION: if the same product is clicked (to be set as primary) on multiple parallel pages, nothing happens
        """

    def close_page(self, first, second, third):
        self.set_inactive(prod_number=first.sequence_number)
        """We put reward = 0 for the nodes that have been visualized but not clicked on. 
        (still susceptible on close) """
        if self.products_state[second.sequence_number] == 0:
            self.graph.update_estimation(node=second, reward=0)
        if self.products_state[third.sequence_number] == 0:
            self.graph.update_estimation(node=third, reward=0)
