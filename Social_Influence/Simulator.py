import numpy as np

from Social_Influence.Graph import Graph
from Social_Influence.Learner import Learner

T = 100
n_experiments = 100
lin_ucb_rewards_per_experiment = []

for e in range(0, n_experiments):
    graph = Graph(mode="full", weights=True)
    learner = Learner(graph)
    for t in range(0, T):
        alpha = np.random.dirichlet(np.ones(len(graph.nodes)), size=1)
        # HERE THERE IS ALL THE SIMULATION OF THE ACTIONS OF THE CUSTOMER, SELECTION OF THE SECONDARY ETC...
        first_arm, second_arm = learner.pull_arm()
        reward = env.round(pulled_arm)
        lin_ucb_learner.update(pulled_arm, reward)

    lin_ucb_rewards_per_experiment.append(lin_ucb_learner.collected_rewards)