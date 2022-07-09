import numpy as np
from Pricing.Learner import *
from Pricing.pricing_environment import EnvironmentPricing
from Settings import LAMBDA
from Social_Influence.Graph import Graph
from Pricing.Clairvoyant import Clairvoyant
import Settings
from matplotlib import pyplot as plt


class TS(Learner):

    def __init__(self, n_arms, prices, secondaries):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.beta_parameters = np.ones((self.n_products, n_arms, 2))
        self.num_product_sold_estimation = np.ones(prices.shape) * np.inf
        self.nearbyReward = np.zeros(prices.shape)
        self.secondaries = secondaries
        self.currentBestArms = np.zeros(len(prices))
        self.visit_probability_estimation = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_visited_from_starting_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_visited_as_first_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_bought_as_first_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.success_per_arm_batch = np.zeros((self.n_products, self.n_arms))
        self.pulled_per_arm_batch = np.zeros((self.n_products, self.n_arms))



    def act(self):
        """
        :return: for every product choose the arm to pull
        :rtype: list
        """
        idx = [0 for _ in range(self.n_products)]
        for prod in range(self.n_products):
            # generate beta for every price of the current product
            beta = np.random.beta(self.beta_parameters[prod, :, 0], self.beta_parameters[prod, :, 1])
            # arm of the current product with highest expected reward
            idx[prod] = np.argmax(beta * ((self.prices[prod] * self.num_product_sold_estimation[prod]) + self.nearbyReward[prod]))
        return idx

    def updateHistory(self, pulled_arm, visited_products, num_bought_products, num_primary):
        """
        update alpha and beta parameters
        :param pulled_arm: arm pulled for every product
        :type pulled_arm: list
        :param reward: reward obtained for every product using the specified arm
        :type reward: list
        :return: none
        :rtype: none
        """
        super().update(pulled_arm, visited_products, num_bought_products)
        self.times_visited_as_first_node[num_primary][pulled_arm[num_primary]] += 1
        if num_bought_products[num_primary] > 0:
            self.times_bought_as_first_node[num_primary][pulled_arm[num_primary]] += 1
        for i in range(len(visited_products)):
            if (visited_products[i] == 1) and i != num_primary:
                self.times_visited_from_starting_node[num_primary][pulled_arm[num_primary]][i] += 1

        #update of the batch related to the success
        for prod in range(self.n_products):
            if visited_products[prod] == 1:
                if num_bought_products[prod] > 0:
                    self.success_per_arm_batch[prod, pulled_arm[prod]] += 1
                self.pulled_per_arm_batch[prod, pulled_arm[prod]] += 1

        #saving rewards for thr graphical representation
        current_prices = [i[j] for i, j in zip(self.prices, pulled_arm)]
        current_reward = sum(num_bought_products * current_prices)
        self.current_reward.append(current_reward)

    def totalNearbyRewardEstimation(self):
        """
        :return: a matrix containing the nearby rewards for all products and all prices
        """
        # contains the conversion rate of the current best price for each product
        conversion_of_current_best = [i[j] for i, j in zip(self.means, self.currentBestArms)]
        price_of_current_best = np.array([i[j] for i, j in zip(self.prices, self.currentBestArms)])
        num_product_sold_of_current_best = np.array(
            [i[j] for i, j in zip(self.num_product_sold_estimation, self.currentBestArms)])
        nearbyRewardsTable = np.zeros(self.prices.shape)
        # it is created a list containing all the nodes/products that must be visited (initially all the products)
        nodesToVisit = [i for i in range(len(self.prices))]
        for node in nodesToVisit:
            # for each product and each price calculates its nearby reward
            for price in range(len(self.prices[0])):
                nearbyRewardsTable[node][price] = sum(self.visit_probability_estimation[node][price]
                                                      * conversion_of_current_best * price_of_current_best
                                                      * num_product_sold_of_current_best * self.means[node][price])
        return nearbyRewardsTable


        '''''''''
        self.nearbyReward = [self.nearby_reward(node) for node in range(self.n_products)]

        super(TS, self).update(pulled_arm, visited_products, num_bought_products)
        self.times_visited_as_first_node[num_primary][pulled_arm[num_primary]] += 1
        for i in range(len(visited_products)):
            if (visited_products[i] == 1) and i != num_primary:
                self.times_visited_from_starting_node[num_primary][pulled_arm[num_primary]][i] += 1
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
        '''''

    def update(self,pulled_arm):
        self.beta_parameters[:, :, 0] = self.beta_parameters[:, :, 0] + self.success_per_arm_batch[:, :]
        self.beta_parameters[:, :, 1] = self.beta_parameters[:, :, 1] \
                                        + self.pulled_per_arm_batch - self.success_per_arm_batch
        num_products = len(pulled_arm)
        for prod in range(num_products):
            if len(self.boughts_per_arm[prod][pulled_arm[prod]])!=0:
                self.num_product_sold_estimation[prod][pulled_arm[prod]] = np.mean(self.boughts_per_arm[prod][pulled_arm[prod]])
                if (self.num_product_sold_estimation[prod][pulled_arm[prod]] == 0):
                    self.num_product_sold_estimation[prod][pulled_arm[prod]] = np.inf
            for t1 in range(self.n_arms):
                for t2 in range(num_products):
                    if self.times_bought_as_first_node[prod][t1][t2] > 0:
                        self.visit_probability_estimation[prod][t1][t2] = self.times_visited_from_starting_node[prod][t1][t2] / self.times_bought_as_first_node[prod][t1][t2]
                    else:
                        self.visit_probability_estimation[prod][t1][t2] = 0
        self.visit_probability_estimation[np.isnan(self.visit_probability_estimation)] = 0


        self.pulled_per_arm_batch = np.zeros((self.n_products, self.n_arms))
        self.success_per_arm_batch = np.zeros((self.n_products, self.n_arms))

        self.nearbyReward = self.totalNearbyRewardEstimation()
        self.nearbyReward[np.isnan(self.nearbyReward)] = 0


'''''''''

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

            expected_reward_actual_node[arm] = sum(self.visit_probability_estimation[actual_node] * (alpha / (alpha + beta)) * self.prices[actual_node][arm])

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
                                               self.num_product_sold_estimation[actual_node][arm]

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
'''''


graph = Graph(mode="full", weights=True)
env = EnvironmentPricing(4, graph, 1)
learner = TS(4, env.prices, env.secondaries)

clairvoyant = Clairvoyant(env.prices, env.conversion_rates, env.classes, env.secondaries, env.num_product_sold, graph, env.alpha_ratios)
best_revenue = clairvoyant.revenue_given_arms([0, 1, 2, 2, 3], 0)
best_revenue_array = [best_revenue for i in range(Settings.NUM_OF_DAYS)]


for i in range(Settings.NUM_OF_DAYS):
    pulled_arms = learner.act()
    print(pulled_arms)
    for j in range(Settings.DAILY_INTERACTIONS):
        visited_products, num_bought_products, num_primary = env.round(pulled_arms)
        learner.updateHistory(pulled_arms, visited_products, num_bought_products,num_primary)
    learner.update(pulled_arms)

print(learner.num_product_sold_estimation)

fig, ax = plt.subplots(nrows=1,ncols=2)
ax[0].plot(learner.average_reward, color='blue', label='UCB-5')
ax[0].axhline(y=best_revenue, color='red', linestyle='--', label='Clairvoyant')
ax[0].set_title('Average reward')
ax[1].plot(np.cumsum(learner.average_reward), color='blue', label='UCB-5')
ax[1].plot(np.cumsum(best_revenue_array), color='red', linestyle='--', label='Clairvoyant')
ax[1].set_title('Cumulative reward')
ax[0].legend()
ax[1].legend()
plt.show()

