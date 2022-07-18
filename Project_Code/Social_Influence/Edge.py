

class Edge:

    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.probability = 0.0

    def __init__(self, node1, node2, probability):
        self.node1 = node1
        self.node2 = node2
        self.probability = float(probability)

    def set_probability(self, probability):
        self.probability = probability

    def getNode1(self):
        return self.node1

    def getNode2(self):
        return self.node2

    def getProbability(self):
        return self.probability