import numpy as np
from Learner import *
from Pricing.pricingEnvironment import PricingEnvironment
from Settings import LAMBDA
from Social_Influence.Graph import Graph


class TS(Learner):

    def __init__(self, n_arms, prices, secondaries, num_product_sold, graph):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.beta_parameters = np.ones((self.n_products, n_arms, 2))
        self.graph = graph
        self.success_per_arm_batch = np.zeros((self.n_products, self.n_arms))
        self.pulled_per_arm_batch = np.zeros((self.n_products, self.n_arms))
        self.secondaries = secondaries
        self.num_product_sold = num_product_sold

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
            # TODO: add expected reward
            idx[prod] = np.argmax(beta * ((self.prices[prod] * self.num_product_sold[prod])))
            # print("rewards prod %d: %s" % (prod, beta * self.prices[prod]))
            # print("NEARBY REWARDS - old - %d: %s" % (prod, self.expected_nearby_reward(prod)[prod]))
            # print("NEARBY REWARDS -check- %d: %s" % (prod, self.reward_of_node_without_nearby(prod)[prod]))
            # print("NEARBY REWARDS - new - %d: %s" % (prod, self.expected_reward(prod)))
        # print("arm pulled", idx)
        return idx

    def update(self, pulled_arm, reward):
        """
        update alpha and beta parameters
        :param pulled_arm: arm pulled for every product
        :type pulled_arm: list
        :param reward: reward obtained for every product using the specified arm
        :type reward: list
        :return: none
        :rtype: none
        """
        super(TS, self).update(pulled_arm, reward)
        for prod in range(self.n_products):
            self.success_per_arm_batch[prod, pulled_arm[prod]] += reward[prod]
            self.pulled_per_arm_batch[prod, pulled_arm[prod]] += 1

        # for prod in range(self.n_products):
        # self.beta_parameters[prod, pulled_arm[prod], 0] = self.beta_parameters[prod, pulled_arm[prod], 0] + reward[
        # prod]
        # self.beta_parameters[prod, pulled_arm[prod], 1] = self.beta_parameters[prod, pulled_arm[prod], 1] + 1.0 - \
        # reward[prod]

    def update_beta_distributions(self):

        self.beta_parameters[:, :, 0] = self.beta_parameters[:, :, 0] + self.success_per_arm_batch[:, :]
        self.beta_parameters[:, :, 1] = self.beta_parameters[:, :,
                                        1] + self.pulled_per_arm_batch - self.success_per_arm_batch

        self.pulled_per_arm_batch = np.zeros((self.n_products, self.n_arms))
        self.success_per_arm_batch = np.zeros((self.n_products, self.n_arms))

    def nearby_reward(self, actual_node):
        node_to_visit = [i for i in range(self.n_products)]
        node_to_visit.remove(actual_node)

        expected_reward_actual_node = np.zeros(self.n_arms)

        probability_to_observe = 1
        for node in (list(set(node_to_visit).intersection(self.secondaries[actual_node]))):
            # delete the actual_node from the node to visit
            new_node_to_visit = node_to_visit.copy()
            new_node_to_visit.remove(node)

            # probability to click node (probability_to_observe will be 1 for the first node and LAMBDA for the second)
            prob_to_clik_node = probability_to_observe * graph.search_edge_by_nodes(
                graph.search_product_by_number(actual_node),
                graph.search_product_by_number(node)).probability[0]  # senza [0] è un array con un elemento

            expected_reward_actual_node += prob_to_clik_node * self.expected_reward(node, new_node_to_visit)

            probability_to_observe = LAMBDA

        for arm in range(self.n_arms):
            alpha = self.beta_parameters[actual_node][arm][0]
            beta = self.beta_parameters[actual_node][arm][1]
            # for each arm calculate the expected reward of the actual_node
            expected_reward_actual_node[arm] = (alpha / (alpha + beta)) * expected_reward_actual_node[arm]

        return expected_reward_actual_node

    def expected_reward(self, actual_node, node_to_visit):
        """
        the first time i call expected_reward i pass only the actual_node. the parameter node_to_visit is managed during the recursion
        :param actual_node: node from which calculate the expected nearby reward
        :type actual_node: int
        :param node_to_visit: list of node to be visited
        :type node_to_visit: list
        :return: for each product and price return the expected reward obtained by the potential revenue of other product
        :rtype: matrix 5x4
        """
        # array containing for the actual_node the expected reward to be calculated
        expected_reward_actual_node = np.zeros(self.n_arms)
        # calculate expected_reward of actual node. I put this is else
        for arm in range(self.n_arms):
            alpha = self.beta_parameters[actual_node][arm][0]
            beta = self.beta_parameters[actual_node][arm][1]
            # for each arm calculate the expected reward of the actual_node
            expected_reward_actual_node[arm] = (alpha / (alpha + beta)) * self.prices[actual_node][arm] * \
                                               self.num_product_sold[actual_node][arm]

        # adds the expected rewards of the 2 secondary products
        probability_to_observe = 1
        for node in (list(set(node_to_visit).intersection(self.secondaries[actual_node]))):
            # delete the actual_node from the node to visit
            new_node_to_visit = node_to_visit.copy()
            new_node_to_visit.remove(node)

            # probability to click node (probability_to_observe will be 1 for the first node and LAMBDA for the second)
            prob_to_clik_node = probability_to_observe * graph.search_edge_by_nodes(
                graph.search_product_by_number(actual_node),
                graph.search_product_by_number(node)).probability[0]  # senza [0] è un array con un elemento

            expected_reward_actual_node += prob_to_clik_node * self.expected_reward(node, new_node_to_visit)

            probability_to_observe = LAMBDA

        return np.mean(expected_reward_actual_node)


graph = Graph(mode="full", weights=True)
env = PricingEnvironment(4, graph, 1)
learner = TS(4, env.prices[0], env.secondaries, env.num_product_sold[0], graph)

for i in range(10000):
    pulled_arms = learner.pull_arm()
    rewards = env.round(pulled_arms)
    learner.update(pulled_arms, rewards)
    if (i % 10 == 0):
        learner.update_beta_distributions()
    print(pulled_arms)
