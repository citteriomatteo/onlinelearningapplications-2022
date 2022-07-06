import random

from Pricing.Learner import *
from Pricing.pricing_environment import EnvironmentPricing
from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from Social_Influence.Page import Page


class Clairvoyant(Learner):

    def __init__(self, prices, conversion_rates, classes, secondaries, num_product_sold, graph, alpha_ratios):
        """

        :param prices: list of products and each product is a list of prices
        :type prices: list of list
        :param conversion_rates: matrix n_products X arms
        :type conversion_rates: list of list
        """

        # num of arms (prices for each product)
        self.prices = prices
        self.conversion_rates = conversion_rates
        self.n_arms = prices.shape[1]
        self.n_products = prices.shape[0]
        self.classes = classes
        self.secondaries = secondaries
        self.num_product_sold = num_product_sold
        self.graph = graph
        self.visit_probability_estimation = np.zeros((self.n_products, self.n_products))
        self.alpha_ratios = alpha_ratios
        super().__init__(self.n_arms, self.n_products)

    def revenue_given_arms(self, arms, chosen_class):
        """
        Returns the revenue of a given combination of arms of a given user class
        :param arms: list of arms
        :type arms: list
        :param chosen_class: the user class given by the external of the method
        """
        nearby_reward = []
        self.visit_probability_estimation = self.simulateTotalNearby(arms, chosen_class)
        for prod in range(self.n_products):
            nearby_reward_temporary = 0
            for temp in range(self.n_products):
                nearby_reward_temporary += self.conversion_rates[chosen_class][prod][arms[prod]] * self.visit_probability_estimation[prod][
                    temp] * self.conversion_rates[chosen_class][temp][arms[temp]] * self.num_product_sold[chosen_class][temp][
                                                      arms[temp]] * self.prices[temp][
                                                      arms[temp]]
            nearby_reward.append(nearby_reward_temporary)

        revenue = []
        for i in range(self.n_products):
            revenue.append(self.prices[i][arms[i]] * self.conversion_rates[chosen_class][i][arms[i]] * self.num_product_sold[chosen_class][i][arms[i]])

        average_total = 0
        for i in range(5):
            average_total += (revenue[i]+nearby_reward[i])*self.alpha_ratios[chosen_class][i+1]

        return average_total

    def disaggr_revenue_given_arms(self, arms, env):
        """
        Returns the revenue of a given combination of arms by weighting wrt all the classes (disaggregated average)
        :param arms: list of arms
        :type arms: list
        :param env: environment, to access to all the classes
        """
        disaggr_average_total = 0
        for c in range(len(env.classes)):
            disaggr_average_total += env.classes["C" + str(c + 1)]["fraction"] \
                                     * self.revenue_given_arms(arms=arms, chosen_class=c)

        return disaggr_average_total


    def simulateTotalNearby(self, selected_price, chosen_class):
        times_visited_from_starting_node = np.zeros((self.n_products, self.n_products))
        for prod in range(self.n_products):
            for iteration in range(364):
                visited_products_ = self.simulateSingleNearby(selected_price, prod, chosen_class)
                for j in range(len(visited_products_)):
                    if (visited_products_[j] == 1) and j != prod:
                        times_visited_from_starting_node[prod][j] += 1
        return times_visited_from_starting_node / 364

    def simulateSingleNearby(self, selected_prices, starting_node, chosen_class):
        customer = Customer(reservation_price=100, num_products=len(self.graph.nodes), graph=self.graph)
        num_prod = starting_node
        t = 0
        visited_products = np.zeros(len(selected_prices))
        num_bought_products = np.zeros(len(selected_prices))
        primary = self.graph.nodes[num_prod]
        second = self.graph.search_product_by_number(self.secondaries[primary.sequence_number][0])
        third = self.graph.search_product_by_number(self.secondaries[primary.sequence_number][1])
        page = Page(primary=primary, second=second, third=third)
        customer.click_on(page)
        visited_products[num_prod] = 1
        is_starting_node = True

        while len(customer.pages) > 0:
            # action = Action(user=customer)
            # -----------------------------------------------------------------------------------
            # 2: CUSTOMERS' CHOICE BETWEEN OPENING A NEW TAB AND USING AN ALREADY OPENED ONE
            # randomized choice: choice of page 0-to-(|pages|-1) or creating a new page
            chosen_index = np.random.randint(low=0, high=len(customer.pages))
            page = customer.pages[chosen_index]
            primary = page.primary
            second = page.second
            third = page.third
            p2 = self.graph.search_edge_by_nodes(primary, second).probability if (
                    visited_products[second.sequence_number] == 0) else 0
            p3 = self.graph.search_edge_by_nodes(primary, third).probability if (
                    visited_products[third.sequence_number] == 0) else 0
            # action.set_page(page)
            superare = self.conversion_rates[chosen_class][primary.sequence_number][selected_prices[primary.sequence_number]]
            # -----------------------------------------------------------------------------------
            # 4: CUSTOMERS' CHOICE BETWEEN BUYING AND NOT BUYING THE PRIMARY PRODUCT
            if (is_starting_node) or (np.random.random() < superare):  # PRIMARY PRODUCT BOUGHT
                is_starting_node = False

                quantity = 0
                customer.add_product(product=primary, quantity=quantity)
                page.set_bought(True)
                # action.set_quantity_bought(quantity=quantity)
                num_bought_products[page.primary.sequence_number] += quantity

                # -----------------------------------------------------------------------------------
                # 5: CUSTOMERS' CLICK CHOICE BETWEEN: SECOND PRODUCT, THIRD PRODUCT OR CLOSE PAGE
                choice = [None] * 2

                if np.random.random() < p2:  # SECONDARY BOUGHT
                    choice[0] = 1
                    visited_products[second.sequence_number] += 1
                if np.random.random() < p3:  # TERTIARY BOUGHT
                    choice[1] = 1
                    visited_products[third.sequence_number] += 1

                if choice[0] == 1:  # SECONDARY PRODUCT CHOSEN
                    # CREATION OF THE NEW PAGE
                    new_primary = second
                    new_second = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][0])
                    new_third = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][1])

                    # --- page creation and insertion in the list of customer's pages ---
                    new_page = Page(new_primary, new_second, new_third)
                    customer.add_new_page(new_page)

                if choice[1] == 1:  # THIRD PRODUCT CHOSEN
                    # CREATION OF THE NEW PAGE
                    new_primary = third
                    new_second = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][0])
                    new_third = self.graph.search_product_by_number(self.secondaries[new_primary.sequence_number][1])

                    # --- page creation and insertion in the list of customer's pages ---
                    new_page = Page(new_primary, new_second, new_third)
                    customer.add_new_page(new_page)

            customer.direct_close_page(page)
            # action.compute_for_social_influence(graph=self.graph)
            t += 1
        return visited_products