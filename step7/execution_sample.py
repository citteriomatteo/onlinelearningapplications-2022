import numpy as np

import Settings
from Pricing.Clairvoyant import Clairvoyant
from Pricing.pricing_environment import EnvironmentPricing
from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from step7.ContextGenerator import ContextGenerator
from step7.ContextNode import ContextNode
from step7.ContextualLearner import ContextualLearner
from step7.UcbStep7 import Ucb
from step7.TSstep7 import TS
from matplotlib import pyplot as plt

mode = 'TS'
color = None

graph = Graph(mode="full", weights=True)
env = EnvironmentPricing(4, graph, 1, mode='multi_class')
context_learner = ContextualLearner(features=env.features, n_arms=env.n_arms, n_products=len(env.graph.nodes))

clairvoyant = Clairvoyant(env.prices, env.conversion_rates, env.classes, env.secondaries, env.num_product_sold, graph, env.alpha_ratios)
best_revenue = clairvoyant.revenue_given_arms(arms=[0, 1, 2, 2, 3], chosen_class=0)
best_revenue_array = [best_revenue for _ in range(Settings.NUM_OF_DAYS)]

# optimal arm for C1: [0, 1, 2, 2, 3]
# optimal arm for C2: [0, 2, 1, 0, 2]
# optimal arm for C3: [1, 3, 1, 1, 1]
best_arms_per_class = [[0, 1, 2, 2, 3], [0, 2, 1, 0, 2], [1, 3, 1, 1, 1]]
best_disaggr_revenue = clairvoyant.disaggr_revenue_given_arms(arms=best_arms_per_class, env=env)
best_disaggr_revenue_array = [best_disaggr_revenue for _ in range(Settings.NUM_OF_DAYS)]

# choice of the algorithm to execute
root_learner = None
if mode == "TS":
    color = 'green'
    root_learner = TS(4, env.prices, env.secondaries, graph)
if mode == "Ucb":
    color = 'blue'
    root_learner = Ucb(4, env.prices, env.secondaries, graph)

root_node = ContextNode(features=env.features, base_learner=root_learner)
context_learner.update_context_tree(root_node)

# confidence used for lower bounds is hardcoded to 0.1!
context_generator = ContextGenerator(features=env.features, contextual_learner=context_learner, confidence=0.1)

for i in range(Settings.NUM_OF_DAYS):

    if i % 14 == 0 and i != 0:
        context_generator.context_generation()

    customer = Customer(reservation_price=100, num_products=len(graph.nodes), graph=graph, env=env)

    learner = context_learner.get_learner_by_context(current_features=customer.features)

    pulled_arms = learner.act()
    print("Context "+ str(customer.features) +" :" + str(pulled_arms))

    for j in range(Settings.DAILY_INTERACTIONS):
        visited_products, num_bought_products, a = env.round(pulled_arms, customer)
        learner.updateHistory(pulled_arms, visited_products, num_bought_products)
        context_generator.collect_daily_data(pulled_arms=pulled_arms,
                                             visited_products=visited_products,
                                             num_bought_products=num_bought_products,
                                             features=customer.features)
        customer.set_as_new()

    context_generator.update_average_rewards(current_features=customer.features)
    learner.update(pulled_arms)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
ax[0].plot(context_generator.average_rewards, color=color, label=mode)
ax[0].axhline(y=best_revenue, color='grey', linestyle='--', label='Clairvoyant (Aggregate)')
ax[0].axhline(y=best_disaggr_revenue, color='black', linestyle='--', label='Clairvoyant (Disaggregate)')
ax[0].set_title('Average reward')
ax[1].plot(np.cumsum(context_generator.average_rewards), color=color, label=mode)
ax[1].plot(np.cumsum(best_revenue_array), color='grey', linestyle='--', label='Clairvoyant (Aggregate)')
ax[1].plot(np.cumsum(best_disaggr_revenue_array), color='black', linestyle='--', label='Clairvoyant (Disaggregate)')
ax[1].set_title('Cumulative reward')
ax[0].legend()
ax[1].legend()
plt.show()





