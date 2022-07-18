import numpy as np

from Project_Code.Pricing.Clairvoyant import Clairvoyant
from Project_Code.Pricing.pricing_environment import EnvironmentPricing
from Project_Code.Social_Influence.Graph import Graph

graph = Graph(mode="full", weights=True)
env = EnvironmentPricing(4, graph, 1)
clairvoyant = Clairvoyant(env.prices, env.conversion_rates, env.classes, env.secondaries, env.num_product_sold, graph, env.alpha_ratios)
best_arms = [0, 1, 2, 2, 3]
optimal_revenue = clairvoyant.revenue_given_arms(best_arms, 0)
bound = 0

for prod in range(5):
    for price in range(4):
        if price != best_arms[prod]:
            arms = best_arms.copy()
            arms[prod] = price
            arm_revenue = clairvoyant.revenue_given_arms(arms, 0)
            print("arm n." + str(arms))
            delta = optimal_revenue - arm_revenue
            bound += (8 * delta) + (np.log(50000) / delta)
print(bound)