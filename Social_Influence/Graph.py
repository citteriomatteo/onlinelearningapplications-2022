import json

from Social_Influence.Edge import Edge
from Social_Influence.Product import Product
from Social_Influence.SIEnvironment import SIEnvironment


class Graph:
    nodes = []
    edges = []

    def __init__(self, param):

        if param:
            path = "../json/full.json"
        else:
            path = "../json/reduced.json"

        with open(path, 'r') as f:
            data = json.load(f)

        for name in data["nodes"]:
            self.nodes.append(Product(name=name["type"]))

        for edge in data["edges"]:
            node1 = self.search_product(edge["node1"])
            node2 = self.search_product(edge["node2"])
            self.edges.append(Edge(node1, node2))

    def search_product(self, name):
        for n in self.nodes:
            if n.name == name:
                return n

    def print_all(self):
        for edge in self.edges:
            print(edge.node1.name)
            print(edge.node2.name)
            print(edge.probability)

"""
graph = Graph(True)
env = SIEnvironment(graph)

print(env.opt())

graph.print_all()
"""
