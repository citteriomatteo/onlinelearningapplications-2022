class Page:

    def __init__(self, primary, second, third):
        self.primary = primary
        self.second = second
        self.third = third
        self.bought = False

    def set_bought(self, value):
        self.bought = value
    '''
    Used for getting the formatted texts of the products (for a pretty CLI printing of the pages)
    '''
    def get_formatted_primary(self):
        if self.primary is not None:
            return "| first: " + self.primary.name + "     |    "
        return "| first: EMPTY  |    "

    def get_formatted_second(self):
        if self.second is not None:
            return "| second: " + self.second.name + "    |    "
        return "| second: EMPTY |    "

    def get_formatted_third(self):
        if self.third is not None:
            return "| third: " + self.third.name + "     |    "
        return "| third: EMPTY  |    "

    def print(self):
        str = ""
        if self.primary is not None:
            str += "| first: " + self.primary.name + " |"
        else:
            str += "| first: EMPTY |"
        if self.second is not None:
            str += "| second: " + self.second.name + " |"
        else:
            str += "| second : EMPTY |"
        if self.third is not None:
            str += "| third: " + self.third.name + " |"
        else:
            str += "| third: EMPTY |"
        print(str)
