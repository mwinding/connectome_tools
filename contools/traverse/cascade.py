from anytree import Node
import numpy as np
from anytree import LevelOrderGroupIter, Node
from .traverse import BaseTraverse


def to_transmission_matrix(adj, p, method="uniform", in_weights=None):

    neg_inds = np.where(adj.sum(axis=1)<0)[0] # negative edges indicate inhibitory connections
    adj = adj.copy() 
    adj = np.abs(adj) # convert to positive values for probs calculation

    if method == "uniform":
        not_probs = (
            1 - p
        ) ** adj  # probability of none of the synapses causing postsynaptic
        probs = 1 - not_probs  # probability of ANY of the synapses firing onto next
        
        for i in neg_inds:
            probs[i] = -probs[i] # restore negative edges
            probs[probs==-0] = 0 # prevent -0 values

    elif method == "input_weighted":
        raise NotImplementedError()
        alpha = p
        flat = np.full(adj.shape, alpha)
        # deg = meta["dendrite_input"].values
        in_weights[in_weights == 0] = 1
        flat = flat / in_weights[None, :]
        not_probs = np.power((1 - flat), adj)
        probs = 1 - not_probs
    return probs


class Cascade(BaseTraverse):
    def _choose_next(self):
        node_transition_probs = self.transition_probs[self._active]

        # identify active inhibitory nodes and deduct transition probs from active excitatory nodes
        all_neg_inds = self.neg_inds

        neg_inds_active = np.intersect1d(self._active, all_neg_inds) # identify active inhibitory nodes
        if(len(neg_inds_active)>0):

            # sum all activate negative edges if multiple activate negative nodes
            if(len(np.shape(node_transition_probs[neg_inds]))>1):
                summed_neg = node_transition_probs[neg_inds_active].sum(axis=0) 
                node_transition_probs[self._active] = node_transition_probs[self._active] + summed_neg # reduce probability of activating positive edges by magnitude of sum of negative edges

            # if only one activate negative node
            else:
                node_transition_probs[self._active] = node_transition_probs[self._active] + node_transition_probs[neg_inds_active] # reduce probability of activating positive edges by magnitude of negative edges
        
        # where the probabilistic signal transmission occurs
        # probs must be positive, so all negative values are converted to zero
        node_transition_probs[node_transition_probs<0] = 0

        transmission_indicator = np.random.binomial(
            np.ones(node_transition_probs.shape, dtype=int), node_transition_probs
        )
        nxt = np.unique(np.nonzero(transmission_indicator)[1])
        if len(nxt) > 0:
            return nxt
        else:
            return None

    def start(self, start_node):
        if isinstance(start_node, int):
            start_node = np.array([start_node])
            super().start(start_node)
        else:
            start_node = np.array(start_node)
            super().start(start_node)

    def _check_visited(self):
        self._active = np.setdiff1d(self._active, self._visited)
        return len(self._active) > 0

    def _check_stop_nodes(self):
        self._active = np.setdiff1d(self._active, self.stop_nodes)
        return len(self._active) > 0


def generate_cascade_paths(
    start_ind, all_start_inds, probs, depth, stop_inds=[], visited=[], max_depth=10 # added all_start_inds
):
    visited = visited.copy()
    visited.append(start_ind)

    probs = probs.copy()
    neg_inds = np.where(probs.sum(axis=1)<0)[0] # identify nodes with negative edges

    if (depth < max_depth) and (start_ind not in stop_inds) and (start_ind not in neg_inds): # transmission not allowed through negative edges

        neg_inds_active = np.intersect1d(all_start_inds, neg_inds) # identify active inhibitory nodes
        if(len(neg_inds_active)>0):

            # sum all activate negative edges if multiple activate negative nodes
            if(len(np.shape(probs[neg_inds]))>1):
                summed_neg = probs[neg_inds_active].sum(axis=0) 
                probs[start_ind] = probs[start_ind] + summed_neg # reduce probability of activating positive edges by magnitude of sum of negative edges

            # if only one activate negative node
            else:
                probs[start_ind] = probs[start_ind] + probs[neg_inds_active] # reduce probability of activating positive edges by magnitude of negative edges
        
        # where the probabilistic signal transmission occurs
        # probs must be positive, so all negative values are converted to zero
        probs[probs<0] = 0

        transmission_indicator = np.random.binomial(
            np.ones(len(probs), dtype=int), probs[start_ind]
        )
        next_inds = np.where(transmission_indicator == 1)[0]
        paths = []
        for i in next_inds:
            if i not in visited:
                next_paths = generate_cascade_paths(
                    i,
                    next_inds,
                    probs,
                    depth + 1,
                    stop_inds=stop_inds,
                    visited=visited,
                    max_depth=max_depth,
                )
                for p in next_paths:
                    paths.append(p)
        return paths
    else:
        return [visited]


def generate_cascade_tree(
    node, probs, depth, stop_inds=[], visited=[], max_depth=10, loops=False
):
    start_ind = node.name
    if (depth < max_depth) and (start_ind not in stop_inds):
        transmission_indicator = np.random.binomial(
            np.ones(len(probs), dtype=int), probs[start_ind]
        )
        next_inds = np.where(transmission_indicator == 1)[0]
        for n in next_inds:
            if n not in visited:
                next_node = Node(n, parent=node)
                visited.append(n)
                generate_cascade_tree(
                    next_node,
                    probs,
                    depth + 1,
                    stop_inds=stop_inds,
                    visited=visited,
                    max_depth=max_depth,
                )
    return node


def cascades_from_node(
    start_ind,
    probs,
    stop_inds=[],
    max_depth=10,
    n_sims=1000,
    seed=None,
    n_bins=None,
    method="tree",
):
    if n_bins is None:
        n_bins = max_depth
    np.random.seed(seed)
    n_verts = len(probs)
    node_hist_mat = np.zeros((n_verts, n_bins), dtype=int)
    for n in range(n_sims):
        if method == "tree":
            _cascade_tree_helper(start_ind, probs, stop_inds, max_depth, node_hist_mat)
        elif method == "path":
            _cascade_path_helper(start_ind, probs, stop_inds, max_depth, node_hist_mat)
    return node_hist_mat


def _cascade_tree_helper(start_ind, probs, stop_inds, max_depth, node_hist_mat):
    root = Node(start_ind)
    root = generate_cascade_tree(
        root, probs, 1, stop_inds=stop_inds, visited=[], max_depth=max_depth
    )
    for level, children in enumerate(LevelOrderGroupIter(root)):
        for node in children:
            node_hist_mat[node.name, level] += 1


def _cascade_path_helper(start_ind, probs, stop_inds, max_depth, node_hist_mat):
    paths = generate_cascade_paths(
        start_ind, probs, 1, stop_inds=stop_inds, max_depth=max_depth
    )
    for path in paths:
        for i, node in enumerate(path):
            node_hist_mat[node, i] += 1
    return paths
