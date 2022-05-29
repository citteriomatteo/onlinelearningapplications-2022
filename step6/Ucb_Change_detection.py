
from Cumulative_sum import *
from Pricing import Learner

class Ucv_Change_detection(Learner):
    def __init__(self,n_arms, prices, secondaries, M=100,eps=0.05,h=20,alpha=0.01):
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

        self.change_detection = [CUMSUM(M,eps,h) for _ in range(n_arms)]
        self.valid_reward_per_arms=[[] for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha

'''''''''
    def pull_arm(self):
        if np.random.binomial(1,1-self.alpha):
            upper_conf= self.empirical_means + self.confidence
            upper_conf[np.isinf(upper_conf)]=1e3
            row_ind,col_ind = linear_sum_assignment(-upper_conf.reshape(self.n_rows,self.n_cols))
            return row_ind,col_ind
        else:
            cost_random=np.random.randint(0,10,size=(self.n_rows,self.n_cols))
            return linear_sum_assignment(cost_random.reshape)
 



    def update(self,pulled_arms,rewards):
        self.t += 1
        pulled_arm_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))
        for pulled_arm, reward in zip(pulled_arm_flat, rewards):
            if self.change_detection[pulled_arm].update(reward):
                self.detections[pulled_arm].append(self.t)
                self.valid_reward_per_arms[pulled_arm]=[]
                self.change_detection[pulled_arm].reset()
            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = np.mean(self.valid_reward_per_arms[pulled_arm])

        total_valid_samples = sum([len(x) for x in self.valid_reward_per_arms])
        for a in range(self.n_arms):
            n_samples = len(self.valid_rewards_per_arm[a])
            self.confidence[a] = (2 * np.log(total_valid_samples) / n_samples) ** 0.5 if n_samples > 0 else np.inf


    def update_observation(self,pulled_arm,reward):
        self.reward_per_arm[pulled_arm].append(reward)
        self.valid_reward_per_arms[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards,reward)
        
'''''''''