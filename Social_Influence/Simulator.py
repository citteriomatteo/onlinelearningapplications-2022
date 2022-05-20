import numpy as np
import matplotlib.pyplot as plt

from Action import Action
from Social_Influence.Customer import Customer
from Social_Influence.Graph import Graph
from Settings import LAMBDA, CONVERSION_RATE
from Social_Influence.Page import Page

T = 20
n_experiments = 1
lin_ucb_rewards_per_experiment = []
history_clicks = np.array([0, 0, 0])

graph = Graph(mode="reduced", weights=True)

for e in range(0, n_experiments):

    # -----------------------------------------------------------------------------------
    # 1: CUSTOMERS GENERATION

    customer = Customer(reservation_price=1000, num_products=5, graph=graph)

    for t in range(0, T):
        action = Action(user=customer)
        print("\n\n ----- ITERATION: " + str(t) + " -----")
        # alpha ratios generation
        alpha = np.random.dirichlet(np.ones(len(graph.nodes)), size=1)

        # -----------------------------------------------------------------------------------
        # 2: CUSTOMERS' CHOICE BETWEEN OPENING A NEW TAB AND USING AN ALREADY OPENED ONE

        page = None

        # randomized choice: choice of page 0-to-(|pages|-1) or creating a new page

        chosen_index = np.random.randint(low=0, high=len(customer.pages) + 1)
        new_tab_choice = (chosen_index == len(customer.pages))

        if new_tab_choice:  # NEW TAB OPENING

            # Filter alpha ratios to keep only the ones of the susceptible nodes (a = a * (1-|states|))
            alpha = alpha * (1 - np.absolute(customer.products_state))
            # Selection of the primary in the main page. Used rule: the one with the highest alpha ratio
            i = np.argmax(alpha)
            primary = graph.search_product_by_number(number=i)
            if not customer.set_active(prod_number=primary.sequence_number):
                print("· The customer can open no more new tabs!")
                new_tab_choice = False
                # if all the products are inactive, no more tabs can be opened -> break the customer!
                try:
                    chosen_index = np.random.randint(low=0, high=len(customer.pages))
                except ValueError:
                    customer.print_all_pages()
                    print("Every product has been visited. Customer finished the run.")
                    break

            else:
                second, third, p2, p3 = graph.pull_arms(node1=primary, products_state=customer.products_state)

                # --- page creation and insertion in the list of customer's pages ---
                page = Page(primary=primary, second=second, third=third)
                customer.add_new_page(page)

                print("· The customer opened a new tab: the product " + page.primary.name + " is displayed as primary.")
                customer.print_all_pages()

        if not new_tab_choice:  # OLD TAB USAGE

            page = customer.pages[chosen_index]
            primary = page.primary
            second = page.second
            third = page.third
            p2 = graph.search_edge_by_nodes(primary, second).probability
            p3 = graph.search_edge_by_nodes(primary, third).probability

            customer.print_all_pages()
            print("· The customer chose the page " + str(chosen_index+1) + ".")


        action.set_page(page)

        # -----------------------------------------------------------------------------------
        # 4: CUSTOMERS' CHOICE BETWEEN BUYING AND NOT BUYING THE PRIMARY PRODUCT

        if np.random.random() < CONVERSION_RATE:  # PRIMARY PRODUCT BOUGHT

            quantity = 1
            print("· The customer buys the primary product in quantity: " + str(quantity) + "!")
            customer.add_product(product=primary, quantity=quantity)
            action.set_quantity_bought(quantity=quantity)

            # -----------------------------------------------------------------------------------
            # 5: CUSTOMERS' CLICK CHOICE BETWEEN: SECOND PRODUCT, THIRD PRODUCT OR CLOSE PAGE
            probabilities_scale = [p2, p2 + p3 * LAMBDA, 1]
            rand = np.random.random()
            scale = 1
            i = 0
            while rand > probabilities_scale[i]:
                i += 1

            if i == 0:  # SECONDARY PRODUCT CHOSEN

                # CREATION OF THE NEW PAGE
                new_primary = second
                new_second, new_third, new_p2, new_p3 = graph.pull_arms(node1=new_primary,
                                                                        products_state=customer.products_state)

                # --- page creation and insertion in the list of customer's pages ---
                new_page = Page(new_primary, new_second, new_third)
                action.set_click_second(customer.click_on(new_page=new_page))

                print("· The customer clicks on: ", new_primary.name)

            else:
                if i == 1:  # THIRD PRODUCT CHOSEN

                    # CREATION OF THE NEW PAGE
                    new_primary = third
                    new_second, new_third, new_p2, new_p3 = graph.pull_arms(node1=new_primary,
                                                                            products_state=customer.products_state)

                    # --- page creation and insertion in the list of customer's pages ---
                    new_page = Page(new_primary, new_second, new_third)
                    action.set_click_third(customer.click_on(new_page=new_page))
                    print("· The customer clicks on: ", new_primary.name)

                else:  # CHOSEN "CLOSE PAGE" OPERATION
                    action.set_page_close(customer.close_page(page))
                    print("· The customer closes the page.")

        else:  # PAGE CLOSED (PRIMARY PRODUCT NOT BOUGHT)
            print("· The customer closes the page without buying.")
            customer.close_page(page)

        # print("ucbs before: ", graph.compute_ucbs_complete())

        action.compute_for_social_influence(graph=graph)
        action.update_history(history_clicks)

        action.compute_for_pricing(graph=graph)

        # print("ucbs after: ", graph.compute_ucbs_complete())

"""
print(history_clicks)

plt.hist(history_clicks, bins=[])
plt.show()
"""
