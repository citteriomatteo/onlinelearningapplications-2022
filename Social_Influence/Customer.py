import numpy as np


class Customer:
    """
    states: -1 -> inactive
             0 -> susceptible
             1 -> active
    """

    products_state = []
    cart = []
    pages = []

    def __init__(self, reservation_price, num_products, firstPage):
        self.reservation_price = reservation_price
        self.products_state = np.zeros(shape=num_products)
        self.pages.append(firstPage)

    def set_susceptible(self, prod_number):
        self.products_state[prod_number] = 0

    def set_active(self, prod_number):
        if self.products_state[prod_number] == 0:
            self.products_state[prod_number] = 1

    def set_inactive(self, prod_number):
        if self.products_state[prod_number] == 1:
            self.products_state[prod_number] = -1

    def add_product(self, product, quantity):
        if quantity > 0:
            for i in range(quantity):
                self.cart.append(product)

    def click_on(self, new_page):
        self.set_active(prod_number=new_page.first.sequence_number)

        # ------- here we should update weights -------

        """
        ASSUMPTION: if the same product is clicked (to be set as primary) on multiple parallel pages, the same 
        result will be displayed -> no new page is created
        """

        if self.products_state[new_page.first.sequence_number] == 0:
            self.pages.append(new_page)

    def close_page(self, product):
        self.set_inactive(prod_number=product.sequence_number)
