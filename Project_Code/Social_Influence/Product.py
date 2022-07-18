

class Product:

    def __init__(self, name, price, sequence_number):
        """

        :param name: the name
        :type name: string
        :param price: the price
        :type price: int
        :param sequence_number: a number that identifies the product
        :type sequence_number: int
        :param x: Vector that indicates the connected products (parameters)
        :type x: vector of 0 or 1
        """
        self.name = name
        self.price = price
        self.sequence_number = sequence_number
        self.x = []

    def set_price(self, price):
        self.price = price

    def set_x(self, x):
        self.x = x
