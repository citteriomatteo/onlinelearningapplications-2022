import copy
import numpy as np


class ContextNode:
    def __init__(self, features, base_learner):
        self.left_child: ContextNode = None
        self.right_child: ContextNode = None
        self.features: dict = features
        self.base_learner = base_learner
        self.features_subspace: dict = {}

    def is_leaf(self) -> bool:
        """
        Chek if the the node is a leaf or not
        """
        return self.left_child is None and self.right_child is None

    def can_grow(self) -> bool:
        """
        Check if the node already covers all the available features.
        """
        return len(self.features_subspace.keys()) < len(self.features)

    def get_leaves(self):
        """ Recursive method that returns the leaves of the tree. """
        # base case of the recursion
        if self.is_leaf():
            return [self]
        # otherwise this is not a leaf node, so check the child
        left_leaves = self.left_child.get_leaves()
        right_leaves = self.right_child.get_leaves()
        # concatenation of the children' leaves
        return left_leaves + right_leaves

    def split(self, splitting_feature, left_learner, right_learner):
        """
        Method used to actually create a new context.
        :param splitting_feature: the feature according to which the split is performed
        :param left_learner: the learner of the left child
        :param right_learner: the learner of the right child
        """

        # use deepcopy to get a child object that does not interfere with the parent one
        self.left_child = ContextNode(self.features, left_learner)
        self.left_child.features_subspace = copy.deepcopy(self.features_subspace)
        self.left_child.features_subspace[splitting_feature] = False
        # right node --> feature = True
        self.right_child = ContextNode(self.features, right_learner)
        self.right_child.features_subspace = copy.deepcopy(self.features_subspace)
        self.right_child.features_subspace[splitting_feature] = True

    def print(self):
        if self.left_child:
            self.left_child.print()
        print("[ " + str(self.features_subspace) + " ]"),
        if self.right_child:
            self.right_child.print()

    def print_mean(self):
        if self.left_child:
            self.left_child.print_mean()
        sos = np.zeros((5,4))
        for i in range(5):
            for j in range(4):
                sos[i][j] = sum(self.base_learner.rewards_per_arm[i][j])
        sos1 = np.zeros((5, 4))
        for i in range(5):
            for j in range(4):
                sos1[i][j] = len(self.base_learner.rewards_per_arm[i][j])
        print("Features of the context: ", self.features_subspace)
        print("[ " + str(self.base_learner.means) + " ]")
        print("Widths of the context: ", self.features_subspace)
        print("[ " + str(self.base_learner.widths) + " ]")
        print("Pulled per arms: ")
        print(sos1)
        print("Successes per arms: ")
        print(sos)
        if self.right_child:
            self.right_child.print_mean()