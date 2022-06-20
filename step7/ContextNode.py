import copy

from Pricing import Learner


class ContextNode:
    def __init__(self, features, base_learner):
        self.left_child: ContextNode = None
        self.right_child: ContextNode = None
        self.features: dict = features
        self.base_learner: Learner = base_learner
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
        return len(self.feature_subspace.keys()) < len(self.all_features)

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
        self.left_child = ContextNode(self.all_features, left_learner)
        self.left_child.feature_subspace = copy.deepcopy(self.feature_subspace)
        self.left_child.feature_subspace[splitting_feature] = False
        # right node --> feature = True
        self.right_child = ContextNode(self.all_features, right_learner)
        self.right_child.feature_subspace = copy.deepcopy(self.feature_subspace)
        self.right_child.feature_subspace[splitting_feature] = True
