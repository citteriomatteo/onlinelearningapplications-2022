import random

from Pricing.Learner import *
from Pricing.pricing_environment import EnvironmentPricing
from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from Social_Influence.Page import Page


class Greedy_Learner(Learner):

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

        # for each class save the list of arm to pull for each product (3x5)
        self.max_idxs = [[0 for i in range(self.n_products)] for _ in range(len(classes))]
        # for each class (3 types of classes) save the max_revenue (3x1)
        self.max_revenue = [self.revenue_given_arms(self.max_idxs[i], i) for i in range(len(classes))]

        self.classes_probability = []
        for c in self.classes:
            self.classes_probability.append(self.classes[c]['fraction'])

    def pull_arm(self):
        """

        :return: arms and total revenue of the best arms near the previous arms
        :rtype: int, float
        """
        # for each product save the total revenue increasing by 1 the arm of that product
        revenues = [0 for i in range(self.n_products)]
        # Discuss whether to use the method chooseClass here or in line 51
        classChoice = self.chooseClass()
        for i in range(self.n_products):
            new_arms = self.max_idxs[classChoice].copy()
            # calculate the revenue of each nearby arm (if the best combination is 00000, it tries
            # 00001, 00010, 00100, 01000, 10000
            if new_arms[i] < self.n_arms - 1:
                new_arms[i] += 1
                revenues[i] = self.revenue_given_arms(new_arms, classChoice)

        # index of the best product arm to increase
        price_index_increased = revenues.index(max(revenues))

        return_arms = self.max_idxs[classChoice].copy()
        return_arms[price_index_increased] += 1
        return return_arms, revenues[price_index_increased], classChoice

    def chooseClass(self):
        return random.choices([0, 1, 2], self.classes_probability, k=1)[0]

    def revenue_given_arms(self, arms, chosen_class):
        """
        Returns the revenue of a given combination of arms of a given user class
        :param arms: list of arms
        :type arms: list
        :param chosen_class: the user class given by the external of the method
        :return:
        :rtype:
        """
        nearby_reward = []
        self.visit_probability_estimation = self.simulateTotalNearby(arms, chosen_class)
        for prod in range(self.n_products):
            nearby_reward_temporary = 0
            for temp in range(self.n_products):
                nearby_reward_temporary += self.conversion_rates[chosen_class][prod][arms[prod]] * \
                                           self.visit_probability_estimation[prod][
                                               temp] * self.conversion_rates[chosen_class][temp][arms[temp]] * \
                                           self.num_product_sold[chosen_class][temp][
                                               arms[temp]] * self.prices[temp][
                                               arms[temp]]
            nearby_reward.append(nearby_reward_temporary)

        revenue = []
        for i in range(self.n_products):
            revenue.append(self.prices[i][arms[i]] * self.conversion_rates[chosen_class][i][arms[i]] *
                           self.num_product_sold[chosen_class][i][arms[i]])

        average_total = 0
        for i in range(5):
            average_total += (revenue[i] + nearby_reward[i]) * self.alpha_ratios[chosen_class][i + 1]
        return average_total

    @staticmethod
    def calculateNodesToVisit(index):
        list_to_return = [0, 1, 2, 3, 4]
        list_to_return.remove(index)
        return list_to_return

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

    def update(self):
        """
        :return: update the max_idx and max_revenues
        :rtype: none
        """
        local_max_found = [False for i in range(len(self.classes))]
        # iterate until the local maximum has been found for each class
        while not local_max_found == [True, True, True]:
            print(learner.max_idxs)
            print(learner.max_revenue)

            new_arms, new_revenue, chosen_class = self.pull_arm()
            if new_revenue > self.max_revenue[chosen_class]:
                self.max_revenue[chosen_class] = new_revenue
                self.max_idxs[chosen_class] = new_arms
                #self.current_reward.append(new_revenue)
            else:
                local_max_found[chosen_class] = True
                # when a local minimum for a certain class is found, we make sure this class will
                # be no more pulled by putting its probability equal to 0
                self.classes_probability[chosen_class] = 0


graph_sample = Graph(mode="full", weights=True)
env = EnvironmentPricing(4, graph_sample, 1)
learner = Greedy_Learner(env.prices, env.conversion_rates, env.classes, env.secondaries, env.num_product_sold,
                         graph_sample, env.alpha_ratios)
learner.update()
print('\nFINAL')
print('Greedy algorithm chosen arms: ',learner.max_idxs)
print('Clearvoyant best arms: [[0, 1, 2, 2, 3], [0, 2, 1, 0, 2], [1, 3, 1, 1, 1]]')
print('Average reward with greedy algorithm choices: ', learner.revenue_given_arms(learner.max_idxs[0],0),
      learner.revenue_given_arms(learner.max_idxs[1],1), learner.revenue_given_arms(learner.max_idxs[2],2))
print('Average reward with best arms: ', learner.revenue_given_arms([0, 1, 2, 2, 3],0),
      learner.revenue_given_arms([0, 2, 1, 0, 2],1), learner.revenue_given_arms([1, 3, 1, 1, 1],2))
print('Average regret per iteration: ', learner.revenue_given_arms([0, 1, 2, 2, 3],0) - learner.revenue_given_arms(learner.max_idxs[0],0),
      learner.revenue_given_arms([0, 2, 1, 0, 2], 1) - learner.revenue_given_arms(learner.max_idxs[1],1),
      learner.revenue_given_arms([1, 3, 1, 1, 1], 2) - learner.revenue_given_arms(learner.max_idxs[2],2))
