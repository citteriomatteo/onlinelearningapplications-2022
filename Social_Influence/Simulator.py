import numpy as np

from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from Social_Influence.Learner import Learner

T = 100
n_experiments = 100
lin_ucb_rewards_per_experiment = []
customer = Customer(reservation_price=1000, num_products=5)

for e in range(0, n_experiments):
    graph = Graph(mode="full", weights=True)
    learner = Learner(graph)
    for t in range(0, T):
        alpha = np.random.dirichlet(np.ones(len(graph.nodes)), size=1)
        # HERE THERE IS ALL THE SIMULATION OF THE ACTIONS OF THE CUSTOMER, SELECTION OF THE SECONDARY ETC...
        """Selection of the primary in the main page"""
        rand = np.random.random_sample()
        scale = 1
        i = 0
        for i in range(len(alpha)):
            if scale > rand:
                scale -= alpha[i]

        primary = graph.search_product_by_number(number=i)
        # HERE BUY OR NOT BUY
        second, third = graph.get_secondary_products(primary=primary, products_state=customer.products_state)
        # HERE THE USER CLICKS OR CLOSES PAGE
        customer.click_on(first=primary, second=second)
        customer.close_page(first=primary, second=second, third=third)