import numpy as np
import Settings


class SIEnvironment:
    """
    def __init__(self, n_arms, dim):
        self.theta = np.random.dirichlet(np.ones(dim), size=1)
        self.arms_features = np.random.binomial(1, 0.5, size=(n_arms, dim))
        self.p = np.zeros(n_arms)
        self.lam = Settings.lam

        for i in range(0, n_arms):
            self.p[i] = np.dot(self.theta, self.arms_features[i])

    #def round(self, pulled_arm):
    #    return 1 if np.random.random() < self.p[pulled_arm] else 0

    def opt(self):
        return np.max(self.p)

    """

    def __init__(self, graph):
        self.graph = graph
        self.theta = np.random.dirichlet(np.ones(len(self.graph.nodes)), size=1)
        self.arms_features = np.random.binomial(1, 0.5, size=(len(self.graph.edges), len(self.graph.nodes)))
        self.lam = Settings.lam

        for i in range(0, len(self.graph.edges)):
            self.graph.edges[i].probability = np.dot(self.theta, self.arms_features[i])

    def round(self, pulled_arm):
        return 1 if np.random.random() < self.graph.edges[pulled_arm].probability else 0

    def opt(self):
        max = 0.0
        opt_edge = self.graph.edges[0]
        for e in self.graph.edges:
            if e.probability > max:
                max = e.probability
                opt_edge = e

        return opt_edge.probability

