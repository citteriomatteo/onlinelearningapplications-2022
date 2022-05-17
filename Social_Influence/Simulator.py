import numpy as np

from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from Settings import LAMBDA, CONVERSION_RATE
from Social_Influence.Page import Page

T = 10
n_experiments = 1
lin_ucb_rewards_per_experiment = []

graph = Graph(mode="reduced", weights=True)

for e in range(0, n_experiments):

    # -----------------------------------------------------------------------------------
    # 1: CUSTOMERS GENERATION

    customer = Customer(reservation_price=1000, num_products=5, graph=graph)

    for t in range(0, T):
        print(" ----- ITERATION: "+str(t)+" -----")
        # alpha ratios generation
        alpha = np.random.dirichlet(np.ones(len(graph.nodes)), size=1)

        # -----------------------------------------------------------------------------------
        # 2: CUSTOMERS' CHOICE BETWEEN OPENING A NEW TAB AND USING AN ALREADY OPENED ONE

        page = None

        # randomized choice: choice of page 0-to-(|pages|-1) or creating a new page

        chosen_index = np.random.randint(low=0, high=len(customer.pages)+1)
        new_tab_choice = (chosen_index == len(customer.pages))

        if new_tab_choice:  # NEW TAB OPENING

            # Filter alpha ratios to keep only the ones of the susceptible nodes (a = a * (1-|states|))
            alpha = alpha * (1 - np.absolute(customer.products_state))
            # Selection of the primary in the main page. Used rule: the one with the highest alpha ratio
            i = np.argmax(alpha)
            primary = graph.search_product_by_number(number=i)
            if not customer.set_active(prod_number=primary.sequence_number):
                print("No more new tabs!")
                new_tab_choice = False
                # if all the products are inactive, no more tabs can be opened -> break the customer!
                try:
                    chosen_index = np.random.randint(low=0, high=len(customer.pages))
                except ValueError:
                    print("Every product has been visited. Customer eliminated.")
                    print(len(customer.pages))
                    break

            else:
                second, third, p2, p3 = graph.pull_arms(node1=primary, products_state=customer.products_state)

                # --- page creation and insertion in the list of customer's pages ---
                page = Page(primary=primary, second=second, third=third)
                customer.add_new_page(page)

                print("The customer opened a new tab.")
                page.print()
                print("Waiting for the choice...")

        if not new_tab_choice:  # OLD TAB USAGE

            page = customer.pages[chosen_index]
            primary = page.primary
            second = page.second
            third = page.third
            p2 = graph.search_edge_by_nodes(primary, second).probability
            p3 = graph.search_edge_by_nodes(primary, third).probability

            print("The customer chose the tab " + str(chosen_index) + ".")
            page.print()

        # -----------------------------------------------------------------------------------
        # 4: CUSTOMERS' CHOICE BETWEEN BUYING AND NOT BUYING THE PRIMARY PRODUCT

        if np.random.random() < CONVERSION_RATE:        # PRIMARY PRODUCT BOUGHT

            print("The customer buys the primary product!")
            customer.add_product(product=primary, quantity=1)

            # -----------------------------------------------------------------------------------
            # 5: CUSTOMERS' CLICK CHOICE BETWEEN: SECOND PRODUCT, THIRD PRODUCT OR CLOSE PAGE
            print("The customer sees the products...")
            probabilities_scale = [p2, p2 + p3*LAMBDA, 1]
            rand = np.random.random()
            scale = 1
            i = 0
            while rand > probabilities_scale[i]:
                i += 1

            print("ucbs before: ", graph.compute_ucbs_complete())
            if i == 0:  # SECONDARY PRODUCT CHOSEN

                customer.click_on(node=second)

                # CREATION OF THE NEW PAGE
                new_primary = second
                new_second, new_third, new_p2, new_p3 = graph.pull_arms(node1=new_primary,
                                                                        products_state=customer.products_state)

                # --- page creation and insertion in the list of customer's pages ---
                new_page = Page(new_primary, new_second, new_third)
                customer.add_new_page(new_page)
                print("the customer clicks on: ", new_primary.name)

            else:
                if i == 1:  # THIRD PRODUCT CHOSEN
                    customer.click_on(node=third)

                    # CREATION OF THE NEW PAGE
                    new_primary = third
                    new_second, new_third, new_p2, new_p3 = graph.pull_arms(node1=new_primary,
                                                                            products_state=customer.products_state)

                    # --- page creation and insertion in the list of customer's pages ---
                    new_page = Page(new_primary, new_second, new_third)
                    customer.add_new_page(new_page)
                    print("the customer clicks on: ", new_primary.name)

                else: # CHOSEN "CLOSE PAGE" OPERATION
                    customer.close_page(page)
                    print("the customer closes the page: ")

            print("ucbs after: ", graph.compute_ucbs_complete())

        else:           # PAGE CLOSED (PRIMARY PRODUCT NOT BOUGHT)
            print("the customer closes the page without buy.")
            customer.close_page(page)

