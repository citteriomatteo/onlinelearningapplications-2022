import numpy as np
from Learner import *

class Ucb(Learner):
    def __init__(self, n_arms,prices):
        super().__init__(n_arms)
        self.prices = prices
        self.means = np.zeros(n_arms)
        self.widths = np.array([np.inf for _ in range(n_arms)])


    def reset(self):
        self.__init__(self.n_arms,self.prices)

    def act(self):
        idx = np.argmax((self.widths + self.means) * self.prices)   #* self.prices) fallito
        return idx

    def update(self,arm_pulled,reward):
        super().update(arm_pulled,reward)
        self.means[arm_pulled]  = np.mean(self.rewards_per_arm[arm_pulled]) ## self.means[arm_pulled]? -> OK
        for idx in range(self.n_arms):
            n = len(self.rewards_per_arm[idx])
            if n>0:
                self.widths[idx] = np.sqrt((2 * np.max( np.log(self.t))/(n)) )    #(t-1)? self.prices)
            else:
                self.widths[idx] = np.inf
