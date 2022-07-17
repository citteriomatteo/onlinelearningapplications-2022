import json

import numpy as np

from Social_Influence.Edge import Edge
from Social_Influence.Product import Product


class Graph:
    nodes = []
    edges = []

    def __init__(self, mode, weights):
        """
        :param mode: The type of graph (reduced or full)
        :type mode: string
        :param weights: The choice of loading probabilities from the file or not
        :type weights: boolean
        :param nodes: The set of nodes in the graph
        :type nodes: Product
        :param edges: The set of edges
        :type edges: Edge
        :param dummy_edge: edge with probability=0
                            (that we use as the null value, since an edge with zero probability is never chosen)
        :type dummy_edge: Edge
        :param M: The matrix used to exploit the upper confidence bounds
        :type M: matrix of floats
        :param c: Exploration coefficient
        :type c: integer
        :param b: vector used to exploit the upper confidence bounds
        :type b: vector of floats
        """

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
            self.nodes.append(Product(name=name["type"], price=0,
                                      sequence_number=i))  # price is fixed for now, waiting for pricing part
            i += 1

        x = np.zeros(shape=(len(self.nodes), len(self.nodes)))

        for edge in data["edges"]:
            node1 = self.search_product_by_name(edge["node1"])
            node2 = self.search_product_by_name(edge["node2"])
            prob = edge['probability']
            x[node1.sequence_number, node2.sequence_number] = 1
            if not weights:
                self.edges.append(Edge(node1=node1, node2=node2))
            else:
                self.edges.append(Edge(node1=node1, node2=node2, probability=edge["probability"]))

        aaa = 1
        # DUMMY EDGE WITH PROBABILITY 0: USEFUL WHEN ALL NODES ARE ACTIVE/INACTIVE
        self.dummy_edge = Edge(node1=None, node2=None, probability=0.0)

        for node in self.nodes:
            node.set_x(x[node.sequence_number, :])

        self.c = 2.0
        self.M = np.identity(len(self.nodes))
        self.b = np.atleast_2d(np.zeros(len(self.nodes))).T

    def search_product_by_name(self, name):
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def search_product_by_number(self, number):
        for n in self.nodes:
            if n.sequence_number == number:
                return n
        return None

    def search_edge_by_nodes(self, node1, node2):
        for edge in self.edges:
            if edge.node1 == node1 and edge.node2 == node2:
                return edge
        return self.dummy_edge

    def print_all(self):
        for edge in self.edges:
            print(edge.getNode1().name)
            print(edge.getNode2().name)
            print(edge.getProbability())