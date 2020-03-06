import numpy as np
import pdb

from .base import StructuredModel
from ..inference import inference_dispatch, get_installed
from ..inference import saddle_point_sum_product as spsp
from .utils import loss_augment_unaries


import numpy as np

from .graph_crf import GraphCRF


import numpy as np

from .crf import CRF
from ..utils import expand_sym, compress_sym


###############################################################################
# GeneralizedCRF
###############################################################################

class GeneralizedCRF(StructuredModel):
    """Abstract base class"""
    def __init__(self, n_states=None, n_features=None, inference_method=None,
                class_weight=None, Loss=None):
        self.n_states = n_states
        if inference_method is None:
            # get first in list that is installed
            inference_method = get_installed(['ad3', 'max-product', 'lp'])[0]
        self.inference_method = inference_method
        self.inference_calls = 0
        self.n_features = n_features
        self.class_weight = class_weight
        self._set_size_joint_feature()
        self._set_class_weight()
        self.Loss = Loss


    def initialize(self, X, Y):
        # Works for both GridCRF and GraphCRF, but not ChainCRF.
        # funny that ^^
        n_features = X[0][0].shape[1]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.n_features, n_features))

        n_states = len(np.unique(np.hstack([y.ravel() for y in Y])))
        if self.n_states is None:
            self.n_states = n_states
        elif self.n_states != n_states:
            raise ValueError("Expected %d states, got %d"
                             % (self.n_states, n_states))

        self._set_size_joint_feature()
        self._set_class_weight()

    def __repr__(self):
        return ("%s(n_states: %s, inference_method: %s)"
                % (type(self).__name__, self.n_states,
                   self.inference_method))

    def _check_size_x(self, x):
        features = self._get_features(x)
        if features.shape[1] != self.n_features:
            raise ValueError("Unary evidence should have %d feature per node,"
                             " got %s instead."
                             % (self.n_features, features.shape[1]))

    def loss_augmented_inference(self, x, mu, w, relaxed=False,
                                 return_energy=False):
        """Loss-augmented Inference for x relative to y using parameters w.

        Finds (approximately)
        armin_y_hat np.dot(w, joint_feature(x, y_hat)) + loss(y, y_hat)
        using self.inference_method.


        Parameters
        ----------
        x : tuple
            Instance of a graph with unary evidence.
            x=(unaries, edges)
            unaries are an nd-array of shape (n_nodes, n_features),
            edges are an nd-array of shape (n_edges, 2)

        y : ndarray, shape (n_nodes,)
            Ground truth labeling relative to which the loss
            will be measured.

        w : ndarray, shape=(size_joint_feature,)
            Parameters for the CRF energy function.

        relaxed : bool, default=False
            Whether relaxed inference should be performed.
            Only meaningful if inference method is 'lp' or 'ad3'.
            By default fractional solutions are rounded. If relaxed=True,
            fractional solutions are returned directly.

        return_energy : bool, default=False
            Whether to return the energy of the solution (x, y) that was found.

        Returns
        -------
        y_pred : ndarray or tuple
            By default an inter ndarray of shape=(n_nodes)
            of variable assignments for x is returned.
            If ``relaxed=True`` and inference_method is ``lp`` or ``ad3``,
            a tuple (unary_marginals, pairwise_marginals)
            containing the relaxed inference result is returned.
            unary marginals is an array of shape (n_nodes, n_states),
            pairwise_marginals is an array of
            shape (n_states, n_states) of accumulated pairwise marginals.

        """
        self.inference_calls += 1
        self._check_size_w(w)
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)

        #######################################################################
        # Initialization
        #######################################################################
        # length = y.shape[0]
        mu0 = self.extract_marginals(mu)
        mu_nodes, mu_edges, dual_gap = spsp(mu0, unary_potentials, pairwise_potentials, edges, self.Loss, tol=1e-5)
        mu = np.hstack([mu_nodes.ravel(), mu_edges.ravel()])
        return mu, dual_gap

    # def batch_loss_augmented_inference(X, Y, w, relaxed=True):

    # NO NEED TO IMPLEMENT THE BATCH VERSION

    def inference(self, x, w, relaxed=False, return_energy=False):
        """Inference for x using parameters w.

        Finds (approximately)
        armin_y np.dot(w, joint_feature(x, y))
        using self.inference_method.


        Parameters
        ----------
        x : tuple
            Instance of a graph with unary evidence.
            x=(unaries, edges)
            unaries are an nd-array of shape (n_nodes, n_states),
            edges are an nd-array of shape (n_edges, 2)

        w : ndarray, shape=(size_joint_feature,)
            Parameters for the CRF energy function.

        relaxed : bool, default=False
            Whether relaxed inference should be performed.
            Only meaningful if inference method is 'lp' or 'ad3'.
            By default fractional solutions are rounded. If relaxed=True,
            fractional solutions are returned directly.

        return_energy : bool, default=False
            Whether to return the energy of the solution (x, y) that was found.

        Returns
        -------
        y_pred : ndarray or tuple
            By default an inter ndarray of shape=(width, height)
            of variable assignments for x is returned.
            If ``relaxed=True`` and inference_method is ``lp`` or ``ad3``,
            a tuple (unary_marginals, pairwise_marginals)
            containing the relaxed inference result is returned.
            unary marginals is an array of shape (width, height, n_states),
            pairwise_marginals is an array of
            shape (n_states, n_states) of accumulated pairwise marginals.

        """
        self._check_size_w(w)
        self.inference_calls += 1
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)
        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)


###############################################################################
# GeneralizedGraphCRF
###############################################################################


class GeneralizedGraphCRF(GeneralizedCRF):
    """Pairwise CRF on a general graph.

    Pairwise potentials the same for all edges, are symmetric by default
    (``directed=False``).  This leads to n_classes parameters for unary
    potentials.

    If ``directed=True``, there are ``n_classes * n_classes`` parameters
    for pairwise potentials, if ``directed=False``, there are only
    ``n_classes * (n_classes + 1) / 2`` (for a symmetric matrix).

    Examples, i.e. X, are given as an iterable of n_examples.
    An example, x, is represented as a tuple (features, edges) where
    features is a numpy array of shape (n_nodes, n_attributes), and
    edges is is an array of shape (n_edges, 2), representing the graph.

    Labels, Y, are given as an iterable of n_examples. Each label, y, in Y
    is given by a numpy array of shape (n_nodes,).

    There are n_states * n_features parameters for unary
    potentials. For edge potential parameters, there are n_state *
    n_states permutations, i.e. ::

                state_1 state_2 state 3
        state_1       1       2       3
        state_2       4       5       6
        state_3       7       8       9

    The fitted parameters of this model will be returned as an array
    with the first n_states * n_features elements representing the
    unary potentials parameters, followed by the edge potential
    parameters.

    Say we have two state, A and B, and two features 1 and 2. The unary
    potential parameters will be returned as [A1, A2, B1, B2].

    If ``directed=True`` the edge potential parameters will return
    n_states * n_states parameters. The rows are senders and the
    columns are recievers, i.e. the edge potential state_2 -> state_1
    is [2,1]; 4 in the above matrix.

    The above edge potential parameters example would be returned as
    [1, 2, 3, 4, 5, 6, 7, 8, 9] (see numpy.ravel).

    If edges are undirected, the edge potential parameter matrix is
    assumed to be symmetric and only the lower triangle is returned, i.e.
    [1, 4, 5, 7, 8, 9].


    Parameters
    ----------
    n_states : int, default=None
        Number of states for all variables. Inferred from data if not provided.

    n_features : int, default=None
        Number of features per node. Inferred from data if not provided.

    inference_method : string or None, default=None
        Function to call do do inference and loss-augmented inference.
        Possible values are:

            - 'max-product' for max-product belief propagation.
                Recommended for chains an trees. Loopy belief propagation in
                case of a general graph.
            - 'lp' for Linear Programming relaxation using cvxopt.
            - 'ad3' for AD3 dual decomposition.
            - 'qpbo' for QPBO + alpha expansion.
            - 'ogm' for OpenGM inference algorithms.

        If None, ad3 is used if installed, otherwise lp.

    class_weight : None, or array-like
        Class weights. If an array-like is passed, it must have length
        n_classes. None means equal class weights.

    directed : boolean, default=False
        Whether to model directed or undirected connections.
        In undirected models, interaction terms are symmetric,
        so an edge ``a -> b`` has the same energy as ``b -> a``.

    """
    def __init__(self, n_states=None, n_features=None, inference_method=None,
                 class_weight=None, Loss=None, directed=False):
        self.directed = directed
        GeneralizedCRF.__init__(self, n_states, n_features, inference_method,
                     class_weight=class_weight, Loss=Loss)
        # n_states unary parameters, upper triangular for pairwise

    def _set_size_joint_feature(self):
        # try to set the size of joint_feature if possible
        if self.n_features is not None and self.n_states is not None:
            if self.directed:
                self.size_joint_feature = (self.n_states * self.n_features +
                                           self.n_states ** 2)
            else:
                self.size_joint_feature = (
                    self.n_states * self.n_features
                    + self.n_states * (self.n_states + 1) / 2)

    def _get_edges(self, x):
        return x[1]

    def _get_features(self, x):
        return x[0]

    def _get_pairwise_potentials(self, x, w):
        """Computes pairwise potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_joint_feature,)
            Weight vector for CRF instance.

        Returns
        -------
        pairwise : ndarray, shape=(n_states, n_states)
            Pairwise weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        pw = w[self.n_states * self.n_features:]
        if self.directed:
            return pw.reshape(self.n_states, self.n_states)
        return expand_sym(pw)

    def _get_unary_potentials(self, x, w):
        """Computes unary potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_joint_feature,)
            Weight vector for CRF instance.

        Returns
        -------
        unary : ndarray, shape=(n_states)
            Unary weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        features = self._get_features(x)
        unary_params = w[:self.n_states * self.n_features].reshape(
            self.n_states, self.n_features)
        return np.dot(features, unary_params.T)

    def joint_feature(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation joint_feature, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, joint_feature(x, y)).

        Parameters
        ----------
        x : tuple
            Unary evidence.

        y : ndarray or tuple
            Either y is an integral ndarray, giving
            a complete labeling for x.
            Or it is the result of a linear programming relaxation. In this
            case, ``y=(unary_marginals, pariwise_marginals)``.

        Returns
        -------
        p : ndarray, shape (size_joint_feature,)
            Feature vector associated with state (x, y).

        """
        self._check_size_x(x)
        features, edges = self._get_features(x), self._get_edges(x)
        n_nodes = features.shape[0]

        if isinstance(y, tuple):
            # y is result of relaxation, tuple of unary and pairwise marginals
            unary_marginals, pw = y
            unary_marginals = unary_marginals.reshape(n_nodes, self.n_states)
            # accumulate pairwise
            pw = pw.reshape(-1, self.n_states, self.n_states).sum(axis=0)
        else:
            y = y.reshape(n_nodes)
            gx = np.ogrid[:n_nodes]

            #make one hot encoding
            unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
            gx = np.ogrid[:n_nodes]
            unary_marginals[gx, y] = 1

            ##accumulated pairwise
            pw = np.dot(unary_marginals[edges[:, 0]].T,
                        unary_marginals[edges[:, 1]])
        unaries_acc = np.dot(unary_marginals.T, features)
        if self.directed:
            pw = pw.ravel()
        else:
            pw = compress_sym(pw)
        joint_feature_vector = np.hstack([unaries_acc.ravel(), pw])
        return joint_feature_vector

    def mean_joint_feature(self, x, mu):
        """
        mu_nodes: length * n_states
        mu_edges: (length - 1) * n_states * n_states  (pwpot same at all edges)
        edges: (length - 1) * 2
        L: n_states * n_states
        """
        mu_nodes, mu_edges = self.extract_marginals(mu)
        features, edges = self._get_features(x), self._get_edges(x)
        n_nodes = features.shape[0]
        unaries_acc = np.dot(mu_nodes.T, features)
        pw = mu_edges.sum(0) # from (length-1, n_classes**2) to (n_classes**2)
        mean_joint_feature_vector = np.hstack([unaries_acc.ravel(), pw.ravel()])
        return mean_joint_feature_vector

    def batch_mean_joint_feature(self, X, MU, Y_true=None):
        result = np.zeros(self.size_joint_feature)
        for i in range(X.shape[0]):
                result += self.mean_joint_feature(X[i], MU[i])
        return result

    def loss(self, y, y_hat):
        L = 0
        for m in range(y.shape[0]):
            L += self.Loss[y[m], y_hat[m]]
        return L / y.shape[0]

    def batch_loss(self, Y, Y_hat):
        L = []
        for i in range(Y.shape[0]):
            L.append(self.loss(Y[i], Y_hat[i]))
        return np.array(L)

    def extract_marginals(self, mu, nodes_only=False):
        length = (mu.shape[0] + self.n_states**2) / (self.n_states**2+self.n_states)
        mu_length_comp = (length - 1) * self.n_states ** 2 + length * self.n_states
        assert mu_length_comp  == mu.shape[0]
        mu_nodes = mu[:length * self.n_states].reshape(length, self.n_states)
        if nodes_only:
            return mu_nodes
        else:
            mu_edges = mu[length * self.n_states:].reshape(length-1, self.n_states, self.n_states)
        return mu_nodes, mu_edges

    def cond_loss(self, y, mu):
        """
        y has shape length
        mu_nodes has shape length * n_classes
        """
        mu_nodes = self.extract_marginals(mu, nodes_only=True)
        return np.dot(mu_nodes, self.Loss.T)[y]

    def batch_cond_loss(self, Y, MU):
        cond_losses = [self.cond_loss(Y[i], MU[i]) for i in range(Y.shape[0])]
        return np.array(cond_losses)

    def Bayes_risk(self, mu):
        mu_nodes = self.extract_marginals(mu, nodes_only=True)
        cond_loss = np.dot(mu_nodes, self.Loss.T) # shape length * n_classes
        return cond_loss.min(1).sum()

    def batch_Bayes_risk(self, MU):
        bayes_risks = [self.Bayes_risk(MU[i]) for i in range(len(MU))]
        bayes_risks = np.array(bayes_risks) 
        return bayes_risks

###############################################################################
# GeneralizedChainCRF
###############################################################################

def make_chain_edges(x):
    # this can be optimized sooooo much!
    inds = np.arange(x.shape[0])
    edges = np.concatenate([inds[:-1, np.newaxis], inds[1:, np.newaxis]],
                           axis=1)
    return edges

class GeneralizedChainCRF(GeneralizedGraphCRF):
    """Linear-chain CRF.

    Pairwise potentials are symmetric and the same for all edges.
    This leads to ``n_classes`` parameters for unary potentials.
    If ``directed=True``, there are ``n_classes * n_classes`` parameters
    for pairwise potentials, if ``directed=False``, there are only
    ``n_classes * (n_classes + 1) / 2`` (for a symmetric matrix).

    Unary evidence ``x`` is given as array of shape (n_nodes, n_features), and
    labels ``y`` are given as array of shape (n_nodes,). Chain lengths do not
    need to be constant over the dataset.

    Parameters
    ----------
    n_states : int, default=None
        Number of states for all variables.
        Inferred from data if not provided.

    inference_method : string or None, default=None
        Function to call do do inference and loss-augmented inference.
        Defaults to "max-product" for max-product belief propagation.
        As chains can be solved exactly and efficiently, other settings
        are not recommended.

    class_weight : None, or array-like
        Class weights. If an array-like is passed, it must have length
        n_classes. None means equal class weights.

    directed : boolean, default=False
        Whether to model directed or undirected connections.
        In undirected models, interaction terms are symmetric,
        so an edge ``a -> b`` has the same energy as ``b -> a``.
    """
    def __init__(self, n_states=None, n_features=None, inference_method=None,
                 class_weight=None, Loss=None, directed=True):
        if inference_method is None:
            inference_method = "max-product"
        GeneralizedGraphCRF.__init__(self, n_states=n_states, n_features=n_features,
                          inference_method=inference_method,
                          class_weight=class_weight, Loss=Loss, directed=directed)

    def output_embedding(self, X, Y):
        n_samples = Y.shape[0]
        MU = []
        for i in range(n_samples):
            length = Y[i].shape[0]
            # set unaries embeddings
            node_embeddings = np.zeros((length, self.n_states))
            node_embeddings[np.arange(length), Y[i]] = 1
            # set pairwise embeddings
            edge_embeddings = np.expand_dims(node_embeddings, 1)[:-1] * np.expand_dims(node_embeddings, 2)[1:]
            # flatten
            MU.append(np.hstack([node_embeddings.ravel(), edge_embeddings.ravel()]))
        return MU

    def _get_edges(self, x):
        return make_chain_edges(x)

    def _get_features(self, x):
        return x

    def initialize(self, X, Y):
        n_features = X[0].shape[1]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.n_features, n_features))

        n_states = len(np.unique(np.hstack([y for y in Y])))
        if self.n_states is None:
            self.n_states = n_states
        elif self.n_states != n_states:
            raise ValueError("Expected %d states, got %d"
                             % (self.n_states, n_states))

        self._set_size_joint_feature()
        self._set_class_weight()
