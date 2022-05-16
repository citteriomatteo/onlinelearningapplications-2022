from Learner import *
from Pricing.pricingEnvironment import PricingEnvironment
from Social_Influence.Graph import Graph


class Greedy_Learner(Learner):

    def __init__(self, prices, conversion_rates):
        """

        :param prices: list of products and each product is a list of prices
        :type prices: list of list
        :param conversion_rates: matrix n_products X arms
        :type conversion_rates: list of list
        """

        # num of arms (prices for each product)
        self.prices = prices
        self.conversion_rates = conversion_rates
        self.n_arms = len(prices[0])
        self.n_products = len(prices)

        self.max_idxs = [0 for i in range(self.n_products)]
        self.max_revenue = 0

        super().__init__(self.n_arms, self.n_products)

        self.max_revenue = self.revenue_given_arms(self.max_idxs)

    def pull_arm(self):
        """

        :return: arms and total revenue of the best arms near the previous arms
        :rtype: int, float
        """
        # for each product save the total revenue increasing by 1 the arm of that product
        revenues = [0 for i in range(self.n_products)]
        for i in range(self.n_products):
            new_arms = self.max_idxs.copy()
            if new_arms[i] < self.n_arms - 1:
                new_arms[i] += 1
                revenues[i] = self.revenue_given_arms(new_arms)

        # index of the best product arm to increase
        price_index_increased = revenues.index(max(revenues))
        return_arms = self.max_idxs.copy()
        return_arms[price_index_increased] += 1
        return return_arms, revenues[price_index_increased]

    def revenue_given_arms(self, arms):
        """

        :param arms: list of arms
        :type arms: list
        :return:
        :rtype:
        """
        revenue = 0
        for i in range(self.n_products):
            revenue += self.prices[i][arms[i]] * self.conversion_rates[i][arms[i]]
        return revenue

    def update(self):
        """

        :return: update the max_idx and max_revenues
        :rtype: none
        """
        local_max_found = False
        while not local_max_found:
            print(learner.max_idxs)
            print(learner.max_revenue)

            new_arms, new_revenue = self.pull_arm()
            if new_revenue > self.max_revenue:
                self.max_revenue = new_revenue
                self.max_idxs = new_arms
            else:
                local_max_found = True


graph = Graph(mode="full", weights=True)
env = PricingEnvironment(4, graph, 1)
learner = Greedy_Learner(env.prices[0], env.conversion_rates[0])
learner.update()
print('\nFINAL')
print(learner.max_idxs)
print(learner.max_revenue)
