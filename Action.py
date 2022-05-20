

class Action:
    def __init__(self, user):
        self.user = user
        self.page = None
        self.quantity_bought = 0
        self.click_second = False
        self.click_third = False
        self.page_close = False

    def set_page(self, page):
        self.page = page

    def set_quantity_bought(self, quantity):
        self.quantity_bought = quantity

    def set_click_second(self, value):
        self.click_second = value

    def set_click_third(self, value):
        self.click_third = value

    def set_page_close(self, value):
        self.page_close = value

    def compute_for_pricing(self, graph):
        return True

    def compute_for_social_influence(self, graph):
        if self.click_second:
            graph.update_estimation(node=self.page.second, reward=1)

        if self.click_third:
            graph.update_estimation(node=self.page.third, reward=1)

        if self.page_close:
            """We put reward = 0 for the nodes that have been visualized but not clicked on. 
                    (still susceptible on close) """
            if self.page.second is not None:
                if self.user.products_state[self.page.second.sequence_number] == 0:
                    graph.update_estimation(node=self.page.second, reward=0)
            if self.page.third is not None:
                if self.user.products_state[self.page.third.sequence_number] == 0:
                    graph.update_estimation(node=self.page.third, reward=0)

    def update_history(self, history):
        if self.click_second:
            history[0] += 1
        if self.click_third:
            history[1] += 1
        if self.page_close:
            history[2] += 1

    def show(self):
        print(self.click_second)
        print(self.click_third)
        print(self.page_close)