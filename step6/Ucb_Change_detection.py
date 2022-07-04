from Non_stationary_environment import Non_Stationary_Environment
from Cumulative_sum import *
from Pricing import Learner

class Ucb_Change_detection(Learner):
    def __init__(self,n_arms, prices, secondaries, M=100,eps=0.05,h=20, alpha=0.01):
        super().__init__(n_arms, len(prices))
        self.prices = prices
        self.pricesMeanPerProduct = np.mean(self.prices, 1)
        self.means = np.zeros(prices.shape)
        self.num_product_sold_estimation = np.ones(prices.shape)
        self.nearbyReward = np.zeros(prices.shape)
        self.widths = np.ones(prices.shape) * np.inf
        self.secondaries = secondaries
        self.currentBestArms = np.zeros(len(prices))
        self.visit_probability_estimation = np.zeros((self.n_products, self.n_products))
        self.times_visited_from_starting_node = np.zeros((self.n_products, self.n_products))
        self.times_visited_as_first_node = np.zeros(self.n_products)

        self.change_detection = [[[CUSUM(M,eps,h)] for i in range(n_arms)] for j in range(self.n_products)] #deve essere nella stessa dimensione di self.prices
        self.valid_reward_per_arms=[[[] for i in range(n_arms)] for j in range(self.n_products)]
        self.detections = [[[] for i in range(n_arms)] for j in range(self.n_products)] #stesso
        self.alpha = alpha

    def act(self):
        """
        :return: for each product returns the arm to pull based on which one gives the highest reward
        :rtype: int
        """
        if np.random.binomial(1, 1 - self.alpha):
            idx = np.argmax((self.widths + self.means) * ((self.prices*self.num_product_sold_estimation) + self.nearbyReward), axis=1)
            return idx
        else:
            cost_random = np.random.randint(0, 10, size=self.prices)
            idx = np.argmax((cost_random),axis=1)
            return idx

    def update(self, arm_pulled):
        """
        update mean and widths
        :param arm_pulled: arm pulled for every product
        :type arm_pulled: list
        :return: none
        :rtype: none
        """
        self.currentBestArms = arm_pulled
        num_products = len(arm_pulled)
        total_valid_samples=0


        for prod in range(num_products):
            self.means[prod][arm_pulled[prod]] = np.mean(self.valid_reward_per_arms[prod][arm_pulled[prod]])
            self.num_product_sold_estimation[prod][arm_pulled[prod]] = np.mean(self.boughts_per_arm[prod][arm_pulled[prod]])
            self.visit_probability_estimation[prod] = self.times_visited_from_starting_node[prod] / self.times_visited_as_first_node[prod]

        for prod in range(num_products):
            for arm in range(self.n_arms):
                total_valid_samples += len(self.valid_reward_per_arms[prod][arm])

        for prod in range(num_products):
            for arm in range(self.n_arms):
                self.n[prod,arm] = len(self.valid_reward_per_arms[prod][arm])
                if self.n[prod,arm] > 0:
                    self.widths[prod][arm] = np.sqrt((2 * np.max(total_valid_samples) / self.n[prod,arm]))
                else:
                    self.widths[prod][arm] = np.inf
        self.nearbyReward = self.totalNearbyRewardEstimation()
        aaa = 1


    def updateHistory(self, arm_pulled, visited_products, num_bought_products, num_primary):
        #super().update(arm_pulled, visited_products, num_bought_products)
        self.t += 1
        # self.rewards.append(reward)
        for prod in range(self.num_products):
            if(visited_products[prod] == 1):
                if(num_bought_products[prod] > 0):
                    quantity=1
                else:
                    quantity=0
                if self.detections[prod][arm_pulled[prod]].update_TS_History(quantity):  # non ho reward nella funzione
                    self.detections[prod][arm_pulled[prod]].append(self.t)
                    self.valid_reward_per_arms[prod][arm_pulled[prod]] = []
                    self.change_detection[prod][arm_pulled[prod]].reset()

        num_product = len(arm_pulled)
        # TODO: fix the way the append works
        for prod in range(num_product):
            if visited_products[prod] == 1:
                if num_bought_products[prod] == 0:
                    self.rewards_per_arm[prod][arm_pulled[prod]].append(0)
                    self.valid_reward_per_arms[prod][arm_pulled].append(0)
                else:
                    self.rewards_per_arm[prod][arm_pulled[prod]].append(1)
                    self.boughts_per_arm[prod][arm_pulled[prod]].append(num_bought_products[prod])
                    self.valid_reward_per_arms[prod][arm_pulled].append(1)
        self.pulled.append(arm_pulled)

        self.times_visited_as_first_node[num_primary][arm_pulled[num_primary]] += 1
        for i in range(len(visited_products)):
            if (visited_products[i] == 1) and i != num_primary:
                self.times_visited_from_starting_node[num_primary][arm_pulled[num_primary]][i] += 1


    def totalNearbyRewardEstimation(self):
        """
        :return: a matrix containing the nearby rewards for all products and all prices
        """
        # contains the conversion rate of the current best price for each product
        conversion_of_current_best = [i[j] for i,j in zip(self.means, self.currentBestArms)]
        price_of_current_best = np.array([i[j] for i, j in zip(self.prices, self.currentBestArms)])
        num_product_sold_of_current_best = np.array([i[j] for i, j in zip(self.num_product_sold_estimation, self.currentBestArms)])
        nearbyRewardsTable = np.zeros(self.prices.shape)
        # it is created a list containing all the nodes/products that must be visited (initially all the products)
        nodesToVisit = [i for i in range(len(self.prices))]
        for node in nodesToVisit:
            # for each product and each price calculates its nearby reward
            for price in range(len(self.prices[0])):
                nearbyRewardsTable[node][price] = sum(self.visit_probability_estimation[node][price]
                                                      * conversion_of_current_best * price_of_current_best
                                                      * num_product_sold_of_current_best)
        return nearbyRewardsTable


p0 = np.array([
      [0.9, 0.45, 0.4, 0.35],
      [0.4, 0.8, 0.3, 0.25],
      [0.5, 0.45, 0.9, 0.35],
      [0.2, 0.18, 0.8, 0.1],
      [0.6, 0.45, 0.4, 0.9] ])
p1 = np.array([
      [0.5, 0.47, 0.45, 0.35],
      [0.45, 0.4, 0.35, 0.25],
      [0.55, 0.54, 0.5, 0.45],
      [0.4, 0.35, 0.32, 0.1],
      [0.6, 0.55, 0.53, 0.52]  ])
p2 = np.array([
      [0.25, 0.3, 0.4, 0.3],
      [0.45, 0.4, 0.35, 0.25],
      [0.55, 0.6, 0.5, 0.45],
      [0.4, 0.42, 0.32, 0.1],
      [0.4, 0.45, 0.35, 0.3] ])
P = [p0, p1, p2]
T = 5000
n_exp = 5
#regret_cusum = np.zeros((n_exp, T))
#regret_ucb = np.zeros((n_exp, T))
#detections = [[] for _ in range(n_exp)]
M = 100
eps = 0.1
h = np.log(T)*2
for j in range(n_exp):
    e_UCB = Non_Stationary_Environment(4, P, T)
    e_CD = Non_Stationary_Environment(p0.size, P, T)
    learner_CD = CUSUM_UCB_Matching(p0.size, *p0.shape, M, eps, h)
    learner_UCB = UCB_Matching(p0.size, *p0.shape)
    opt_rew = []
    rew_CD = []
    rew_UCB = []
    for t in range(T):
        p = P[int(t / e_UCB.phase_size)]
        opt = linear_sum_assignment(-p)
        opt_rew.append(p[opt].sum())

        pulled_arm = learner_CD.act()
        reward = e_CD.round(pulled_arm)
        learner_CD.update_TS_History(pulled_arm, reward)
        rew_CD.append(reward.sum())

        pulled_arm = learner_UCB.act()
        reward = e_UCB.round(pulled_arm)
        learner_UCB.update_TS_History(pulled_arm, reward)
        rew_UCB.append(reward.sum())

    regret_cusum[j, :] = np.cumsum(opt_rew)-np.cumsum(rew_CD)
    regret_ucb[j, :] = np.cumsum(opt_rew)-np.cumsum(rew_UCB)
mean_regret_cusum = np.mean(regret_cusum, axis=0)
mean_regret_ucb = np.mean(regret_ucb, axis=0)
std_cusum = np.std(regret_cusum, axis=0)/np.sqrt(n_exp)
std_ucb = np.std(regret_ucb, axis=0)/np.sqrt(n_exp)





