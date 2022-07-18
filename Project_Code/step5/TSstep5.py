from Project_Code.Pricing.Learner import *
from Project_Code import Settings


class TS(Learner):

    def __init__(self, n_arms, prices, secondaries):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.beta_parameters = np.ones((self.n_products, n_arms, 2))
        self.num_product_sold_estimation = np.ones(prices.shape) * np.inf
        self.nearbyReward = np.zeros(prices.shape)
        self.secondaries = secondaries
        self.currentBestArms = np.zeros(len(prices))
        self.visit_probability_estimation = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_visited_from_starting_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_visited_as_first_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.times_bought_as_first_node = np.zeros((self.n_products, self.n_arms, self.n_products))
        self.success_per_arm_batch = np.zeros((self.n_products, self.n_arms))
        self.pulled_per_arm_batch = np.zeros((self.n_products, self.n_arms))
        self.alpha_ratios = np.zeros(self.n_products)



    def act(self):
        """
        :return: for every product choose the arm to pull
        :rtype: list
        """
        idx = [0 for _ in range(self.n_products)]
        for prod in range(self.n_products):
            # generate beta for every price of the current product
            beta = np.random.beta(self.beta_parameters[prod, :, 0], self.beta_parameters[prod, :, 1])
            # arm of the current product with highest expected reward
            idx[prod] = np.argmax(beta * ((self.prices[prod] * self.num_product_sold_estimation[prod]) + self.nearbyReward[prod]))
        return idx

    def updateHistory(self, pulled_arm, visited_products, num_bought_products, num_primary):
        """
        update alpha and beta parameters
        :param pulled_arm: arm pulled for every product
        :type pulled_arm: list
        :param reward: reward obtained for every product using the specified arm
        :type reward: list
        :return: none
        :rtype: none
        """
        super().update(pulled_arm, visited_products, num_bought_products)
        self.times_visited_as_first_node[num_primary][pulled_arm[num_primary]] += 1
        if num_bought_products[num_primary] > 0:
            self.times_bought_as_first_node[num_primary][pulled_arm[num_primary]] += 1
        for i in range(len(visited_products)):
            if (visited_products[i] == 1) and i != num_primary:
                self.times_visited_from_starting_node[num_primary][pulled_arm[num_primary]][i] += 1

        #update of the batch related to the success
        for prod in range(self.n_products):
            if visited_products[prod] == 1:
                if num_bought_products[prod] > 0:
                    self.success_per_arm_batch[prod, pulled_arm[prod]] += 1
                self.pulled_per_arm_batch[prod, pulled_arm[prod]] += 1

        #saving rewards for thr graphical representation
        current_prices = [i[j] for i, j in zip(self.prices, pulled_arm)]
        current_reward = sum(num_bought_products * current_prices)
        self.current_reward.append(current_reward)

    def totalNearbyRewardEstimation(self):
        """
        :return: a matrix containing the nearby rewards for all products and all prices
        """
        # contains the conversion rate of the current best price for each product
        conversion_of_current_best = np.zeros(self.n_products)
        conversion_of_current_best_alpha = [i[j][0] for i, j in zip(self.beta_parameters, self.currentBestArms)]
        conversion_of_current_best_beta = [i[j][1] for i, j in zip(self.beta_parameters, self.currentBestArms)]

        for prod in range(self.n_products):
            conversion_of_current_best[prod] = conversion_of_current_best_alpha[prod]/(conversion_of_current_best_beta[prod]+conversion_of_current_best_alpha[prod])

        price_of_current_best = np.array([i[j] for i, j in zip(self.prices, self.currentBestArms)])
        num_product_sold_of_current_best = np.array(
            [i[j] for i, j in zip(self.num_product_sold_estimation, self.currentBestArms)])
        nearbyRewardsTable = np.zeros(self.prices.shape)
        # it is created a list containing all the nodes/products that must be visited (initially all the products)
        nodesToVisit = [i for i in range(len(self.prices))]
        for node in nodesToVisit:
            # for each product and each price calculates its nearby reward
            for price in range(len(self.prices[0])):
                alpha_near= self.beta_parameters[node][price][0]
                beat_near = self.beta_parameters[node][price][1]
                nearbyRewardsTable[node][price] = sum(self.visit_probability_estimation[node][price]
                                                      * conversion_of_current_best * price_of_current_best
                                                      * num_product_sold_of_current_best * (alpha_near/(alpha_near+beat_near)))
        return nearbyRewardsTable

    def revenue_given_arms(self, arms):
        conversion_of_current = np.zeros(self.n_products)
        conversion_of_current_alpha = [i[j][0] for i, j in zip(self.beta_parameters, arms)]
        conversion_of_current_beta = [i[j][1] for i, j in zip(self.beta_parameters, arms)]
        nearby_reward = [i[j] for i, j in zip(self.nearbyReward, arms)]

        for prod in range(self.n_products):
            conversion_of_current[prod] = conversion_of_current_alpha[prod] / (
                    conversion_of_current_beta[prod] + conversion_of_current_alpha[prod])

        price_of_current = np.array([i[j] for i, j in zip(self.prices, arms)])
        num_product_sold = [i[j] for i, j in zip(self.num_product_sold_estimation, arms)]
        for prod in range(self.n_products):
            if(num_product_sold[prod]== np.inf):
                num_product_sold[prod]=0
        return np.sum(np.multiply(self.alpha_ratios, np.multiply(conversion_of_current, np.multiply(price_of_current,
                                                                                                    num_product_sold)) + nearby_reward))



    def update(self,pulled_arm):

        self.currentBestArms = pulled_arm
        self.average_reward.append(np.mean(self.current_reward[-Settings.DAILY_INTERACTIONS:]))

        self.beta_parameters[:, :, 0] = self.beta_parameters[:, :, 0] + self.success_per_arm_batch[:, :]
        self.beta_parameters[:, :, 1] = self.beta_parameters[:, :, 1] \
                                        + self.pulled_per_arm_batch - self.success_per_arm_batch
        num_products = len(pulled_arm)
        for prod in range(num_products):
            self.alpha_ratios[prod] = np.sum(self.times_visited_as_first_node[prod][:]) / np.sum(
                self.times_visited_as_first_node)
            if len(self.boughts_per_arm[prod][pulled_arm[prod]])!=0:
                self.num_product_sold_estimation[prod][pulled_arm[prod]] = np.mean(self.boughts_per_arm[prod][pulled_arm[prod]])
                if (self.num_product_sold_estimation[prod][pulled_arm[prod]] == 0):
                    self.num_product_sold_estimation[prod][pulled_arm[prod]] = np.inf

            for t1 in range(self.n_arms):
                for t2 in range(num_products):
                    if self.times_bought_as_first_node[prod][t1][t2] > 0:
                        self.visit_probability_estimation[prod][t1][t2] = self.times_visited_from_starting_node[prod][t1][t2] / self.times_bought_as_first_node[prod][t1][t2]
                    else:
                        self.visit_probability_estimation[prod][t1][t2] = 0
        self.visit_probability_estimation[np.isnan(self.visit_probability_estimation)] = 0


        self.pulled_per_arm_batch = np.zeros((self.n_products, self.n_arms))
        self.success_per_arm_batch = np.zeros((self.n_products, self.n_arms))

        self.nearbyReward = self.totalNearbyRewardEstimation()
        self.nearbyReward[np.isnan(self.nearbyReward)] = 0