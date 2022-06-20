import numpy as np

from Pricing.Learner import Learner


class ContextualLearner(Learner):
    def __init__(self, features, n_arms, n_products):
        super().__init__(n_arms, n_products)
        self.features = features
        # definition of the datastructures that handle the contexts
        self.context_tree = None

    def update_context_tree(self, new_context_tree):
        self.context_tree = new_context_tree

    def get_learner_by_context(self, current_features):
        # TODO: probably a faster way can be achieved by using the get_leaves method
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

    def pull_arm(self, user_features):
        """ get a structure of arm to pull according to the context """

        return self.get_learner_by_context(user_features).act()

    def update(self, daily_reward, pulled_arms, user_features):
        # scan and divide according to the features
        leaves = self.context_tree.get_leaves()
        distributions = np.zeros(len(leaves))
        for i, obs in enumerate(daily_reward):
            # i -> index used to scan the data received by the environment
            update_done = False
            for idx, leaf in enumerate(leaves):
                leaf_subspace = leaf.feature_subspace
                good_leaf = True
                for feature in leaf_subspace:
                    feature_idx = self.features.index(feature)
                    if user_features[i][feature_idx] != leaf_subspace[feature]:
                        good_leaf = False
                        break
                if good_leaf:
                    leaf.base_learner.update(pulled_arms[i], obs[0], obs[1])
                    update_done = True
                    distributions[idx] += 1
                    break
            if not update_done:
                raise AttributeError
        return distributions.tolist()



