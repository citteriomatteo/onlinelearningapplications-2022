import numpy as np

from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from Settings import lam

T = 10
n_experiments = 1
lin_ucb_rewards_per_experiment = []

for e in range(0, n_experiments):
    graph = Graph(mode="reduced", weights=True)
    # customers generation
    customer = Customer(reservation_price=1000, num_products=5, graph=graph)
    for t in range(0, T):
        print("ITERATION: ", t)
        # alpha ratios generation
        alpha = np.random.dirichlet(np.ones(len(graph.nodes)), size=1)
        alpha = alpha * (1 - np.absolute(customer.products_state))
        # HERE THERE IS ALL THE SIMULATION OF THE ACTIONS OF THE CUSTOMER, SELECTION OF THE SECONDARY ETC...
        """Selection of the primary in the main page"""
        i = np.argmax(alpha)

        primary = graph.search_product_by_number(number=i)
        customer.set_active(prod_number=primary.sequence_number)
        second, third, p2, p3 = graph.pull_arms(node1=primary, products_state=customer.products_state)
        print("first: ", primary.name)
        print("second: ", second.name)
        print("third: ", third.name)
        print("waiting for the choice...")
        # HERE BUY OR NOT BUY
        if np.random.random() > 0:
            print("the customer buys the product.")
            customer.add_product(product=primary, quantity=1)

            # HERE THE USER CLICKS OR CLOSES PAGE
            print("the customer sees the products...")
            probabilities_scale = [p2, p2 + p3*lam, 1]
            rand = np.random.random()
            scale = 1
            i = 0
            while rand > probabilities_scale[i]:
                i += 1

            print("ucbs before: ", graph.compute_ucbs(primary=primary, products_state=customer.products_state))
            if i == 0:
                customer.click_on(first=primary, second=second)
                print("the customer clicks on: ", second.name)
            else:
                if i == 1:
                    customer.click_on(first=primary, second=third) # change primary
                    print("the customer clicks on: ", third.name)
                else:
                    customer.close_page(first=primary, second=second, third=third)
                    print("the customer closes the page: ")

            print("ucbs after: ", graph.compute_ucbs(primary=primary, products_state=customer.products_state))
        else:
            print("the customer closes the page without buy.")
            customer.close_page(first=primary, second=second, third=third)

