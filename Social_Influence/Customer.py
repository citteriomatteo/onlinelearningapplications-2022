import numpy as np


class Customer:
    """
    states: -1 -> inactive
             0 -> susceptible
             1 -> active
    """

    cart = []
    products_state = []
    pages = []

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

    def click_on(self, node):
        if self.set_active(prod_number=node.sequence_number):
            self.graph.update_estimation(node=node, reward=1)

        """
        ASSUMPTION: if the same product is clicked (to be set as primary) on multiple parallel pages, nothing happens
        """

    def add_new_page(self, new_page):
        duplicated = False
        for page in self.pages:
            if page.is_identical(new_page):
                duplicated = True
        if not duplicated:  # if it doesn't already exist an identical page, insert it
            self.pages.append(new_page)

    def close_page(self, page):
        self.set_inactive(prod_number=page.primary.sequence_number)
        """We put reward = 0 for the nodes that have been visualized but not clicked on. 
        (still susceptible on close) """
        if page.second is not None:
            if self.products_state[page.second.sequence_number] == 0:
                self.graph.update_estimation(node=page.second, reward=0)
        if page.third is not None:
            if self.products_state[page.third.sequence_number] == 0:
                self.graph.update_estimation(node=page.third, reward=0)

        self.pages.remove(page)
