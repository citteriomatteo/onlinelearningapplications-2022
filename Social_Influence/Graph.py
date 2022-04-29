import json

import numpy as np

from Social_Influence.Edge import Edge
from Social_Influence.Learner import Learner
from Social_Influence.LearnerEdge import LearnerEdge
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

        i = 0
        for name in data["nodes"]:
            self.nodes.append(Product(name=name["type"], price=0, sequence_number=i))  # price is fixed for now, waiting for pricing part
            i += 1

        for edge in data["edges"]:
            node1 = self.search_product(edge["node1"])
            node2 = self.search_product(edge["node2"])
            if not weights:
                self.edges.append(LearnerEdge(node1=node1, node2=node2))
            else:
                self.edges.append(LearnerEdge(node1=node1, node2=node2, probability=edge["probability"]))

        for edge in self.edges:
            edge.setFeatures(np.random.binomial(1, 0.5, size=(len(self.edges))))

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
            if e.getNode1() == primary and products_state[e.getNode2().sequence_number] == 0:
                if e.getProbability() > first_prob:
                    # the old "first" becomes now "second"
                    second_prob = first_prob
                    second = first
                    # saving the new "first"
                    first_prob = e.getProbability()
                    first = e.getNode2()
                else:
                    if e.getProbability() > second_prob:
                        second_prob = e.getProbability()
                        second = e.getNode2()

        return [[first, first_prob], [second, second_prob]]

    def print_all(self):
        for edge in self.edges:
            print(edge.getNode1().name)
            print(edge.getNode2().name)
            print(edge.getProbability())




