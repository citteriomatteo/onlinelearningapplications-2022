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

'''
Values for a good looking result:
TS: 90 NUM_OF_DAYS, 225 DAILY_INTERACTIONS, 20 NUM_PLOT_ITERATION
UCB: 90 NUM_OF_DAYS, 100 DAILY_INTERACTIONS, 20 NUM_PLOT_ITERATION
'''

# CHOOSE BETWEEN 'Ucb' AND 'TS' AND MODIFY mode ACCORDINGLY TO MAKE EXECUTION WITH A SPECIFIC LEARNER'S TYPE
mode = 'Ucb'

color = None
if mode == "TS":
    color = 'green'
if mode == "Ucb":
    color = 'blue'

final_reward= np.zeros((Settings.NUM_PLOT_ITERATION, Settings.NUM_OF_DAYS))
final_cumulative_regret = np.zeros((Settings.NUM_PLOT_ITERATION, Settings.NUM_OF_DAYS))
final_cumulative_reward = np.zeros((Settings.NUM_PLOT_ITERATION, Settings.NUM_OF_DAYS))

for k in range (Settings.NUM_PLOT_ITERATION):

    graph = Graph(mode="full", weights=True)
    env = EnvironmentPricing(4, graph, 1)
    context_learner = ContextualLearner(features=env.features, n_arms=env.n_arms, n_products=len(env.graph.nodes))

    clairvoyant = Clairvoyant(env.prices, env.conversion_rates, env.classes, env.secondaries, env.num_product_sold, graph, env.alpha_ratios)
    best_revenue = clairvoyant.revenue_given_arms(arms=[0, 1, 2, 2, 3], chosen_class=0)
    best_revenue_array = [best_revenue for _ in range(Settings.NUM_OF_DAYS)]

    # optimal arm for C1: [0, 1, 2, 2, 3]
    # optimal arm for C2: [0, 2, 1, 0, 2]
    # optimal arm for C3: [1, 3, 1, 1, 1]
    best_arms_per_class = [[0, 1, 2, 2, 3], [0, 2, 1, 0, 2], [1, 3, 1, 1, 1]]
    best_disaggr_revenue = clairvoyant.disaggr_revenue_given_arms(arms=best_arms_per_class, env=env)

    # choice of the algorithm to execute
    root_learner = None
    if mode == "TS":
        root_learner = TS(4, env.prices, env.secondaries, graph)
    if mode == "Ucb":
        root_learner = Ucb(4, env.prices, env.secondaries, graph)

    root_node = ContextNode(features=env.features, base_learner=root_learner)
    context_learner.update_context_tree(root_node)

    # confidence used for lower bounds is hardcoded to 0.1!
    context_generator = ContextGenerator(features=env.features, contextual_learner=context_learner, confidence=0.1)

    best_revenue = clairvoyant.disaggr_revenue_given_arms(arms=best_arms_per_class, env=env)
    opt_rew = []
    actual_rew = []

    for i in range(Settings.NUM_OF_DAYS):

        print("DAY ", i)

        if i % 14 == 0 and i != 0:
            context_generator.context_generation()

        for j in range(Settings.DAILY_INTERACTIONS):

            customer = Customer(reservation_price=100, num_products=len(graph.nodes), graph=graph, env=env)

            learner = context_learner.get_learner_by_context(current_features=customer.features)

            pulled_arms = learner.act()

            visited_products, num_bought_products, num_primary = env.round(pulled_arms, customer)
            learner.updateHistory(pulled_arms, visited_products, num_bought_products, num_primary)
            context_generator.collect_daily_data(pulled_arms=pulled_arms,
                                                 visited_products=visited_products,
                                                 num_bought_products=num_bought_products,
                                                 num_primaries=num_primary,
                                                 features=customer.features)
            customer.set_as_new()

        learner.update(pulled_arms)
        context_generator.update_average_rewards(current_features=customer.features)

        if mode == 'Ucb':
            actual_rew.append(context_generator.average_rewards[-1])
        if mode == 'TS':
            actual_rew.append(learner.average_reward[-1])
        opt_rew.append(best_revenue)

    final_cumulative_regret[k, :] = np.cumsum(opt_rew) - np.cumsum(actual_rew)
    final_cumulative_reward[k, :] = np.cumsum(actual_rew)
    final_reward[k:] = actual_rew


#REGRET
print("FINAL CUMULATIVE REGRET: ")
print(final_cumulative_regret)

mean_cumulative_regret = np.mean(final_cumulative_regret, axis=0)
stdev_regret = np.std(final_cumulative_regret, axis=0) / np.sqrt(Settings.NUM_OF_DAYS*Settings.DAILY_INTERACTIONS)
print("MEAN: ")
print(mean_cumulative_regret)


#Cumulative REWARD
print("FINAL CUMULATIVE REWARD: ")
print(final_cumulative_reward)

mean_cumulative_reward = np.mean(final_cumulative_reward, axis=0)
stdev_cumulative_reward = np.std(final_cumulative_reward, axis=0) / np.sqrt(Settings.NUM_OF_DAYS)
print("MEAN: ")
print(mean_cumulative_reward)

#REWARD
print("FINAL REWARD: ")
print(final_reward)

mean_final_reward = np.mean(final_reward, axis=0)
stdev_reward = np.std(final_reward, axis=0) / np.sqrt(Settings.NUM_OF_DAYS)
print("MEAN: ")
print(mean_final_reward)

best_revenue_array = [best_revenue for i in range(Settings.NUM_OF_DAYS)]

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
ax[0].plot(mean_cumulative_regret, color=color, label=mode)
ax[0].fill_between(range(Settings.NUM_OF_DAYS), mean_cumulative_regret - stdev_regret, mean_cumulative_regret + stdev_regret, alpha=0.4)
ax[0].set_title('Cumulative Regret')

ax[1].plot(mean_cumulative_reward, color=color, label=mode)
ax[1].fill_between(range(Settings.NUM_OF_DAYS), mean_cumulative_reward - stdev_cumulative_reward, mean_cumulative_reward + stdev_cumulative_reward, alpha=0.4)
ax[1].plot(np.cumsum(best_revenue_array), color='grey', linestyle='--', label='Disaggregated Clairvoyant')
ax[1].set_title('Cumulative reward')

ax[2].plot(mean_final_reward, color=color, label=mode)
ax[2].fill_between(range(Settings.NUM_OF_DAYS), mean_final_reward - stdev_reward, mean_final_reward + stdev_reward, alpha=0.4)
ax[2].axhline(y=best_revenue, color='grey', linestyle='--', label='Disaggregated Clairvoyant')
ax[2].axvline(x=14, color='red', label="Split attempt")
ax[2].axvline(x=28, color='red')
ax[2].set_title('Reward')

ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()




