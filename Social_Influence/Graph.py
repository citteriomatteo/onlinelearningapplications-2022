import json

from Social_Influence.Edge import Edge
from Social_Influence.Product import Product
from Social_Influence.SIEnvironment import SIEnvironment


class Graph:
    nodes = []
    edges = []

    def __init__(self, mode, weights):

        if mode == "reduced":
            path = "../json/graphs/reduced"
        else:
            path = "../json/graphs/full"
        if weights:
            path += "_weights.json"
        else:
            path += "_noweights.json"

        with open(path, 'r') as f:
            data = json.load(f)

        for name in data["nodes"]:
            self.nodes.append(Product(name=name["type"], price=0))  # price is fixed for now, waiting for pricing part

        for edge in data["edges"]:
            node1 = self.search_product(edge["node1"])
            node2 = self.search_product(edge["node2"])
            if not weights:
                self.edges.append(Edge(node1=node1, node2=node2))
            else:
                self.edges.append(Edge(node1=node1, node2=node2, probability=edge["probability"]))

    def search_product(self, name):
        for n in self.nodes:
            if n.name == name:
                return n

    # searching the first and second nodes connected to "primary" with the highest probability
    def get_secondary_products(self, primary, products_state):

        first_prob = 0
        second_prob = 0
        first = None
        second = None

        for e in self.edges:
            if e.node1 == primary and products_state[e.node2.sequence_number] == 0:
                if e.probability > first_prob:
                    # the old "first" becomes now "second"
                    second_prob = first_prob
                    second = first
                    # saving the new "first"
                    first_prob = e.probability
                    first = e.node2
                else:
                    if e.probability > second_prob:
                        second_prob = e.probability
                        second = e.node2

        return [[first, first_prob], [second, second_prob]]

    def print_all(self):
        for edge in self.edges:
            print(edge.node1.name)
            print(edge.node2.name)
            print(edge.probability)


"""

graph = Graph(mode="full", weights=True)
env = SIEnvironment(graph=graph, randomize=False)

print(env.opt())

graph.print_all()

"""
