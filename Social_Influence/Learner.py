import numpy as np


class Learner:
    def __init__(self, graph):
        self.graph = graph
        self.arms = np.zeros(shape=(len(graph.edges), len(graph.edges)))
        for i in range(len(graph.edges)):
            self.arms[i, :] = graph.edges[i].getFeatures()
        self.dim = len(graph.edges)
        self.collected_rewards = []
        self.pulled_arms = []
        self.c = 2.0
        self.M = np.identity(self.dim)
        self.b = np.atleast_2d(np.zeros(self.dim)).T
        self.theta = np.dot(np.linalg.inv(self.M), self.b)

    def compute_ucbs(self):
        self.theta = np.dot(np.linalg.inv(self.M), self.b)
        ucbs = []
        for arm in self.arms:
            arm = np.atleast_2d(arm).T
            ucb = np.dot(self.theta.T, arm) + self.c * np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.M), arm)))
            ucbs.append(ucb[0][0])
        return ucbs

    def pull_arm(self):
        ucbs = self.compute_ucbs()
        first = np.argmax(ucbs)
        ucbs.remove(first)
        second = np.argmax(ucbs)
        return self.graph.nodes[first], self.graph.nodes[second]

    def update_estimation(self, first, reward):
        arm = np.atleast_2d(self.arms[first]).T
        self.M += np.dot(arm, arm.T)
        self.b += reward * arm





