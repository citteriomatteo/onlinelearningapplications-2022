

class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.probability = 0.0

    def set_probability(self, probability):
        self.probability = probability
