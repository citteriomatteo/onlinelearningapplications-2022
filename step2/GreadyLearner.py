import random

from Pricing.Learner import *
from Pricing.pricingEnvironment import PricingEnvironment
from Social_Influence.Graph import Graph


class Greedy_Learner(Learner):

    def __init__(self, prices, conversion_rates, classes, secondaries, num_product_sold):
        """

        :param prices: list of products and each product is a list of prices
        :type prices: list of list
        :param conversion_rates: matrix n_products X arms
        :type conversion_rates: list of list
        """

        # num of arms (prices for each product)
        self.prices = prices
        self.conversion_rates = conversion_rates
        self.n_arms = len(prices[0][0])
        self.n_products = len(prices[0])
        self.classes = classes
        self.secondaries = secondaries
        self.num_product_sold = num_product_sold
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
        revenue = 0
        for i in range(self.n_products):
            conversion_for_best_arms = [i[j] for i, j in zip(self.conversion_rates[chosen_class],
                                                             self.max_idxs[chosen_class])]
            revenue += (self.prices[chosen_class][i][arms[i]] * self.conversion_rates[chosen_class][i][arms[i]]
                        * self.num_product_sold[chosen_class][i][arms[i]]) + self.singleNearbyRewardEstimation \
                           (self.calculateNodesToVisit(i), conversion_for_best_arms, i,
                            self.conversion_rates[chosen_class][i][arms[i]], chosen_class)
        return revenue

    @staticmethod
    def calculateNodesToVisit(index):
        list_to_return = [0, 1, 2, 3, 4]
        list_to_return.remove(index)
        return list_to_return

    def singleNearbyRewardEstimation(self, nodes_to_visit, conversion_estimation_for_best_arms, product,
                                     probability_to_enter, chosen_class):
        """
        :return: nearby reward of a single price of a single product
        :rtype: float
        """
        value_to_return = 0
        # for each node that is possible to visit from the starting one, calculates its nearby reward
        for j in (list(set(nodes_to_visit).intersection(self.secondaries[product]))):
            # the probability from a node to visit another one is given by the edge of the graph connecting the two
            # nodes/products
            probability_to_visit_secondary = graph.search_edge_by_nodes(graph.search_product_by_number(product),
                                                                        graph.search_product_by_number(j)).probability
            # the chance of buying a secondary product is given by the probability of visiting it, the probability
            # to buy the primary and the probability to buy the secondary once its page is reached (its
            # conversion rate)
            prob_to_buy_a_secondary = conversion_estimation_for_best_arms[
                                          j] * probability_to_visit_secondary * probability_to_enter
            value_to_return += self.prices[chosen_class][j][self.max_idxs[chosen_class][j]] * prob_to_buy_a_secondary \
                               * self.num_product_sold[chosen_class][j][self.max_idxs[chosen_class][j]]
            # the tree must be ran across deeper, but it is useless to visit it if the chance of reaching a deeper node
            # is almost zero, so it is checked how much probable it is to going deeper before
            # doing the other calculations
            if prob_to_buy_a_secondary > 1e-6:
                copy_list = nodes_to_visit.copy()
                copy_list.remove(j)
                value_to_return += self.singleNearbyRewardEstimation(copy_list, conversion_estimation_for_best_arms,
                                                                     j, prob_to_buy_a_secondary, chosen_class)
        return value_to_return

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
            else:
                local_max_found[chosen_class] = True
                # when a local minimum for a certain class is found, we make sure this class will
                # be no more pulled by putting its probability equal to 0
                self.classes_probability[chosen_class] = 0


graph = Graph(mode="full", weights=True)
env = PricingEnvironment(4, graph, 1)
learner = Greedy_Learner(env.prices, env.conversion_rates, env.classes, env.secondaries, env.num_product_sold)
learner.update()
print('\nFINAL')
print(learner.max_idxs)
print(learner.max_revenue)