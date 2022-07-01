
from Pricing.Learner import Learner


class ContextualLearner(Learner):
    def __init__(self, features, n_arms, n_products):
        super().__init__(n_arms, n_products)
        self.features = features
        # definition of the datastructures that handle the contexts
        self.context_tree = None

    def update_context_tree(self, new_context_tree):
        self.context_tree = new_context_tree

    def get_root_learner(self):
        return self.context_tree.base_learner

    def get_learner_by_context(self, current_features):
        # navigation of the tree up to the leaf with the correct context
        cur_node = self.context_tree
        navigate = True
        while navigate:
            if cur_node.is_leaf():
                navigate = False
                break
            left_subspace = cur_node.left_child.feature_subspace
            go_left = True
            for feature in left_subspace:
                feature_idx = self.features.index(feature)
                if current_features[feature_idx] != left_subspace[feature]:
                    go_left = False
                    break
            if go_left:
                cur_node = cur_node.left_child
            else:
                # optional check: it should be the right child by construction
                right_subspace = cur_node.right_child.feature_subspace
                go_right = True
                for feature in left_subspace:
                    feature_idx = self.features.index(feature)
                    if current_features[feature_idx] != right_subspace[feature]:
                        go_right = False
                        break
                if go_right:
                    cur_node = cur_node.right_child
                else:
                    raise NotImplementedError("An error occurs: neither the left and the right child are compliant "
                                              "with the given features.")
        return cur_node.base_learner

