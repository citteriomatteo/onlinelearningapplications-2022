from Edge import Edge


class LearnerEdge(Edge):
    def __init__(self, node1, node2):
        super().__init__(node1, node2)
        self.features = None

    def __init__(self, node1, node2, probability):
        super().__init__(node1, node2, probability)
        self.features = None

    def setFeatures(self, features):
        self.features = features

    def getNode1(self):
        return super().getNode1()

    def getNode2(self):
        return super().getNode2()

    def getProbability(self):
        return super().getProbability()

    def getFeatures(self):
        return self.features
