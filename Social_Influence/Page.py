class Page:

    def __init__(self, primary, second, third):
        self.primary = primary
        self.second = second
        self.third = third

    '''
    Used to check whether the page has already been opened before (with the same primary)
    :param page: the page to confront
    :type page: Page 
    '''

    def is_identical(self, page):
        if self.primary == page.primary and self.second == page.second and self.third == page.third:
            return True
        return False

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
