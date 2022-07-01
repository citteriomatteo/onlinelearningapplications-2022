from Pricing.pricing_environment import EnvironmentPricing
import numpy as np

#Da fare bene
class Non_Stationary_Environment(EnvironmentPricing):
    def __init__(self, n_arms, graph, probabilities, horizon):
        super().__init__(n_arms, graph, probabilities, resources="../json/dataUsers.json")
        self.t = 1
        n_phases = len(self.probabilities)
        print(n_phases)
        self.phase_size = horizon/n_phases

    def round(self, pulled_arm):
        current_phase = int(self.t / self.phase_size)
        p = self.probabilities[current_phase][pulled_arm]
        self.t += 1
        reward = np.random.binomial(1, p)
        return reward