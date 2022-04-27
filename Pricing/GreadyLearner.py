from Learner import *
from Pricing.pricingEnvironment import PricingEnvironment
from Social_Influence.Graph import Graph


class Greedy_Learner(Learner):

    def __init__(self, prices, demand_curve):
        """
            prices: list of products, each product is a list of prices
        """
        # num of arms (prices for each product)
        self.prices = prices
        self.demand_curve = demand_curve
        self.n_arms = len(prices[0])
        self.n_products = len(prices)

        super().__init__(self.n_arms, self.n_products)

        self.max_revenue = self.revenue_given_arms(self.max_idxs)

    def pull_arm(self):
        revenues = [0 for i in range(self.n_products)]
        for i in range(self.n_products):
            new_arms = self.max_idxs.copy()
            if new_arms[i] < self.n_arms-1:
                new_arms[i] += 1
                revenues[i] = self.revenue_given_arms(new_arms)

        #print('revenues ',revenues)
        price_index_increased = revenues.index(max(revenues))
        #print('index ', price_index_increased)
        return_arms = self.max_idxs.copy()
        return_arms[price_index_increased] += 1
        return return_arms, revenues[price_index_increased]

    def revenue_given_arms(self, arms):
        revenue = 0
        for i in range(self.n_products):
            revenue += self.prices[i][arms[i]] * self.demand_curve[i][arms[i]]
        return revenue

    def update(self):
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
learner = Greedy_Learner(env.prices[0], env.demand_curve[0])
learner.update()
print('\nFINAL')
print(learner.max_idxs)
print(learner.max_revenue)



