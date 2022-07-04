from Pricing.pricing_environment import EnvironmentPricing
from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from step7.ContextGenerator import ContextGenerator
from step7.ContextNode import ContextNode
from step7.ContextualLearner import ContextualLearner
from step7.UcbStep7 import Ucb

graph = Graph(mode="full", weights=True)
env = EnvironmentPricing(4, graph, 1, mode='multi_class')
context_learner = ContextualLearner(features=env.features, n_arms=env.n_arms, n_products=len(env.graph.nodes))
root_learner = Ucb(4, env.prices, env.secondaries, graph)
root_node = ContextNode(features=env.features, base_learner=root_learner)
context_learner.update_context_tree(root_node)

# confidence used for lower bounds is hardcoded to 0.1!
context_generator = ContextGenerator(features=env.features, contextual_learner=context_learner, confidence=0.1)

for i in range(1000):
    if i % 14 == 0 and i != 0:
        context_generator.context_generation()

    customer = Customer(reservation_price=100, num_products=len(graph.nodes), graph=graph, env=env)

    learner = context_learner.get_learner_by_context(current_features=customer.features)

    pulled_arms = learner.act()

    visited_products, num_bought_products, a = env.round(pulled_arms, customer)

    learner.updateHistory(pulled_arms, visited_products, num_bought_products)

    context_generator.collect_daily_data(pulled_arms=pulled_arms,
                                         visited_products=visited_products,
                                         num_bought_products=num_bought_products,
                                         features=customer.features)

    # TODO  non hardcodare
    if (i % 10 == 0) and (i != 0):
        learner.update_TS_History(pulled_arms)


context_learner.print_mean()




