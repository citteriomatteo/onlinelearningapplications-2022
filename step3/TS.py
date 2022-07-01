import numpy as np
from Pricing.Learner import *
from Pricing.pricing_environment import EnvironmentPricing
from Settings import LAMBDA
from Social_Influence.Graph import Graph
from Social_Influence.Customer import Customer
from Social_Influence.Page import Page


class TS(Learner):

    def __init__(self, n_arms, prices, secondaries, num_product_sold, graph):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.beta_parameters = np.ones((self.n_products, n_arms, 2))
        self.graph = graph
        self.success_per_arm_batch = np.zeros((self.n_products, self.n_arms))
        self.pulled_per_arm_batch = np.zeros((self.n_products, self.n_arms))
        self.secondaries = secondaries # da togliere?
        self.num_product_sold = num_product_sold
        self.nearbyReward = np.zeros(prices.shape)
        self.currentBestArms = np.zeros(len(prices))
        self.visit_probability_estimation = np.zeros((self.n_products, self.n_products))

    def pull_arm(self):
        """

        :return: for every product choose the arm to pull
        :rtype: list
        """
        idx = [0 for _ in range(self.n_products)]
        for prod in range(self.n_products):
            # generate beta for every price of the current product
            beta = np.random.beta(self.beta_parameters[prod, :, 0], self.beta_parameters[prod, :, 1])
            # arm of the current product with highest expected reward
            idx[prod] = np.argmax(beta * ((self.prices[prod] * self.num_product_sold[prod]) + self.nearbyReward[prod]))
            # print("rewards prod %d: %s" % (prod, beta * self.prices[prod]))
            # print("NEARBY REWARDS - old - %d: %s" % (prod, self.expected_nearby_reward(prod)[prod]))
            # print("NEARBY REWARDS -check- %d: %s" % (prod, self.reward_of_node_without_nearby(prod)[prod]))
            # print("NEARBY REWARDS - new - %d: %s" % (prod, self.expected_reward(prod)))
        # print("arm pulled", idx)
        return idx

    def updateHistory(self, arm_pulled, visited_products, num_bought_products):
        super().update(arm_pulled, visited_products, num_bought_products)

    def update(self, pulled_arm, visited_products, num_bought_products):
        """
        update alpha and beta parameters
        :param pulled_arm: arm pulled for every product
        :type pulled_arm: list
        :param reward: reward obtained for every product using the specified arm
        :type reward: list
        :return: none
        :rtype: none
        """

        self.currentBestArms = pulled_arm
        self.nearbyReward = np.zeros((self.n_products, self.n_arms))
        self.visit_probability_estimation = self.simulateTotalNearby(pulled_arm)
        for prod in range(self.n_products):
            for price in range(self.n_arms):
                alpha_actual = self.beta_parameters[prod][price][0]
                beta_actual = self.beta_parameters[prod][price][1]
                for temp in range(self.n_products):
                    alpha_near = self.beta_parameters[temp][self.currentBestArms[temp]][0]
                    beta_near = self.beta_parameters[temp][self.currentBestArms[temp]][1]

                    self.nearbyReward[prod][price] += (alpha_actual/(alpha_actual+beta_actual)) * self.visit_probability_estimation[prod][
                        temp] * (alpha_near/(alpha_near+beta_near)) * self.num_product_sold[temp][
                                                          self.currentBestArms[temp]] * self.prices[temp][
                                                          self.currentBestArms[temp]]


        # TODO vedere qua sotto che non ricordo
        self.nearbyReward = [self.nearby_reward(node) for node in range(self.n_products)]

        # super(TS, self).update(pulled_arm, visited_products, num_bought_products)
        for prod in range(self.n_products):
            if visited_products[prod] == 1:
                if num_bought_products[prod] > 0:
                    self.success_per_arm_batch[prod, pulled_arm[prod]] += 1
                self.pulled_per_arm_batch[prod, pulled_arm[prod]] += 1

        # for prod in range(self.n_products):
        # self.beta_parameters[prod, pulled_arm[prod], 0] = self.beta_parameters[prod, pulled_arm[prod], 0] + reward[
        # prod]
        # self.beta_parameters[prod, pulled_arm[prod], 1] = self.beta_parameters[prod, pulled_arm[prod], 1] + 1.0 - \
        # reward[prod]

    def simulateTotalNearby(self, selected_price):
        times_visited_from_starting_node = np.zeros((self.n_products, self.n_products))
        for prod in range(self.n_products):
            for iteration in range(364):
                visited_products_ = self.simulateSingleNearby(selected_price, prod)
                for j in range(len(visited_products_)):
                    if (visited_products_[j] == 1) and j != prod:
                        times_visited_from_starting_node[prod][j] += 1
        return times_visited_from_starting_node / 364

    def simulateSingleNearby(self, selected_prices, starting_node):
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
            alpha = self.beta_parameters[primary.sequence_number][selected_prices[primary.sequence_number]][0]
            beta = self.beta_parameters[primary.sequence_number][selected_prices[primary.sequence_number]][1]
            superare = alpha / (alpha+beta)
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

    # TODO dove la uso? non ricordo
    def update_beta_distributions(self):

        self.beta_parameters[:, :, 0] = self.beta_parameters[:, :, 0] + self.success_per_arm_batch[:, :]
        self.beta_parameters[:, :, 1] = self.beta_parameters[:, :,
                                        1] + self.pulled_per_arm_batch - self.success_per_arm_batch

        self.pulled_per_arm_batch = np.zeros((self.n_products, self.n_arms))
        self.success_per_arm_batch = np.zeros((self.n_products, self.n_arms))


graph = Graph(mode="full", weights=True)
env = EnvironmentPricing(4, graph, 1)
learner = TS(4, env.prices[0], env.secondaries, env.num_product_sold[0], graph)

for i in range(1000):
    pulled_arms = learner.pull_arm()
    visited_products, num_bought_products, a = env.round(pulled_arms)
    learner.update(pulled_arms, visited_products, num_bought_products)
    if (i % 10 == 0) and (i != 0):
        learner.update_beta_distributions()
    print(pulled_arms)
