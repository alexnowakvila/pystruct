import numpy as np
from scipy import sparse
import scipy.special as sp
import pdb

from .common import _validate_params
from ..utils.graph_functions import is_forest
from maxprod import inference_max_product as mp


def edges_to_graph(edges, n_vertices=None):
    if n_vertices is None:
        n_vertices = np.max(edges) + 1
    graph = sparse.coo_matrix((np.ones(len(edges)), edges.T),
                              shape=(n_vertices, n_vertices)).tocsr()
    return graph


def is_chain(edges, n_vertices):
    """Check if edges specify a chain and are in order."""
    return (np.all(edges[:, 0] == np.arange(0, n_vertices - 1))
            and np.all(edges[:, 1] == np.arange(1, n_vertices)))

def saddle_point_sum_product_debug(mu0, unary_potentials, pairwise_potentials,
                                edges, L, max_iter=5000, check_dual_gap=True, tol=1e-5, warm_start=False):
    """
        INPUT 

        unary_potentials: length * n_states
        pairwise_potentials: n_states * n_states  (pwpot same at all edges)
        edges: (length - 1) * 2
        L: n_states * n_states

        OUPTUT

        node_marginals: length * n_states
        pairwise_marginals: (length - 1) * n_states * n_states
    """

    n_states = pairwise_potentials.shape[0]
    length = unary_potentials.shape[0]
    if warm_start:
        p = mu0[0]
        # nu = np.log(mu0[0] + 1e-5)
    else:
        # initialize optimization variables
        p = np.ones((length, n_states)) / n_states
        nu = np.ones((length, n_states)) / n_states
        # nu = np.log(nu_nodes + 1e-16)

    q = np.zeros((length, n_states)) 
    mu = np.zeros((length, n_states)) 
    
    # initialize averages
    q_avg = np.zeros((length, n_states)) 
    mu_avg = np.zeros((length, n_states)) 

    # repeated_potentials = np.tile(pairwise_potentials, length - 1)
    repeated_potentials = np.repeat(pairwise_potentials[np.newaxis, :, :], length - 1, axis=0)

    eta = 1.
    # pdb.set_trace()
    pairwise_potentials = 0 * pairwise_potentials

    for k in range(max_iter):
        # FIRST PROXIMAL MAPPING
        q = sp.softmax(-eta * np.dot(nu, L.T) + np.log(p +1e-16), axis=1)
        mu = sp.softmax(eta * np.dot(p, L.T) + eta * unary_potentials + np.log(nu + 1e-6))
        # prepare uscores
        # uscores = eta * np.dot(p, L) + eta * unary_potentials - nu_nodes
        # uscores[0] = uscores[0] + nu_nodes[0]
        # uscores[-1] = uscores[-1] + nu_nodes[-1]
        # bscores = eta * repeated_potentials + nu_edges
        # mu_nodes, mu_edges, _ = sequence_sum_product(uscores, bscores)
        
        # SECOND PROXIMAL MAPPING
        p = sp.softmax(-eta * np.dot(mu, L.T) + np.log(q +1e-16), axis=1)
        nu = sp.softmax(eta * np.dot(q, L.T) + eta * unary_potentials + np.log(mu + 1e-6))
        # prepare uscores
        # uscores = eta * np.dot(q, L) + eta * unary_potentials - mu_nodes
        # uscores[0] = uscores[0] + mu_nodes[0]
        # uscores[-1] = uscores[-1] + mu_nodes[-1]
        # bscores = eta * repeated_potentials + mu_edges
        # nu_nodes, nu_edges, _ = sequence_sum_product(uscores, bscores)

        # UPDATE AVERAGES
        q_avg = k * q_avg / (k+1) + q / (k+1) 
        mu_avg = k * mu_avg / (k+1) + mu / (k+1)

        # COMPUTE DUAL GAP
        if check_dual_gap and k % 1000 == 0:
            # # do vitervi
            # ymax = mp(np.dot(q_avg, L) + unary_potentials, pairwise_potentials, edges)
            # #make one hot encoding
            # node_embeddings = np.zeros((length, n_states), dtype=np.int)
            # gx = np.ogrid[:length]
            # node_embeddings[gx, ymax] = 1
            # ##accumulated pairwise
            # sum_edge_embeddings = np.dot(node_embeddings[edges[:, 0]].T,
            #             node_embeddings[edges[:, 1]])
            # # compute value of y_max
            # m1 = (np.dot(q_avg, L) + unary_potentials)[np.arange(length), ymax].sum()
            # m2 = (pairwise_potentials * sum_edge_embeddings).sum()
            # maxval = m1 + m2
            # en1 = (unary_potentials * mu_avg_nodes).sum()
            # en2 = (repeated_potentials * mu_avg_edges).sum()
            # minval = np.min(np.dot(mu_avg_nodes, L), axis=1).sum() + en1 + en2
            maxval = np.max(np.dot(q_avg, L.T) + unary_potentials, 1).sum()
            en1 = (unary_potentials * mu_avg).sum(1)
            minval = np.min(np.dot(mu_avg, L.T) + en1, 1).sum()
            dual_gap = maxval - minval

            print("Dual gap ai iteration %f : %f" % (k, dual_gap))
            print("Primal is %f" % (maxval))
            # do max
            # substact
    # pdb.set_trace()
    nu_edges = np.ones((length - 1, n_states, n_states)) / n_states
    return mu_avg, nu_edges

def saddle_point_sum_product(mu0, unary_potentials, pairwise_potentials,
                                edges, L, max_iter=10, check_dual_gap=True, tol=1e-5, warm_start=True, eta=1.5):
    """
        INPUT 

        unary_potentials: length * n_states
        pairwise_potentials: n_states * n_states  (pwpot same at all edges)
        edges: (length - 1) * 2
        L: n_states * n_states

        OUPTUT

        node_marginals: length * n_states
        pairwise_marginals: (length - 1) * n_states * n_states
    """

    n_states = pairwise_potentials.shape[0]
    length = unary_potentials.shape[0]
    if warm_start:
        p = mu0[0]
        nu_nodes, nu_edges = np.log(mu0[0] + 1e-5), np.log(mu0[1] + 1e-5)
    else:
        # initialize optimization variables
        p = np.ones((length, n_states)) / n_states
        nu_nodes = np.ones((length, n_states)) / n_states
        nu_nodes = np.log(nu_nodes + 1e-5)
        nu_edges = np.ones((length - 1, n_states, n_states)) / n_states
        nu_edges = np.log(nu_edges + 1e-5)

    q = np.zeros((length, n_states)) 
    mu_nodes = np.zeros((length, n_states)) 
    mu_edges = np.zeros((length - 1, n_states, n_states))
    
    # initialize averages
    q_avg = np.zeros((length, n_states)) 
    mu_avg_nodes = np.zeros((length, n_states)) 
    mu_avg_edges = np.zeros((length - 1, n_states, n_states))

    # repeated_potentials = np.tile(pairwise_potentials, length - 1)
    repeated_potentials = np.repeat(pairwise_potentials[np.newaxis, :, :], length - 1, axis=0)

    eta = eta / float(length)
    # pdb.set_trace()
    # pairwise_potentials = 0 * pairwise_potentials

    for k in range(max_iter * length):
        # pdb.set_trace()
        # FIRST PROXIMAL MAPPING
        if np.isnan(p.sum()):
            pdb.set_trace()
        q = sp.softmax(-eta * np.dot(np.exp(nu_nodes), L.T) + np.log(p + 1e-4), axis=1)
        # prepare uscores
        uscores = eta * np.dot(p, L) + eta * unary_potentials - nu_nodes
        uscores[0] = uscores[0] + nu_nodes[0]
        uscores[-1] = uscores[-1] + nu_nodes[-1]
        bscores = eta * repeated_potentials + nu_edges
        mu_nodes, mu_edges, _ = sequence_sum_product(uscores, bscores)
        
        # SECOND PROXIMAL MAPPING
        p = sp.softmax(-eta * np.dot(np.exp(mu_nodes), L.T) + np.log(q +1e-4), axis=1)
        # prepare uscores
        uscores = eta * np.dot(q, L) + eta * unary_potentials - mu_nodes
        uscores[0] = uscores[0] + mu_nodes[0]
        uscores[-1] = uscores[-1] + mu_nodes[-1]
        bscores = eta * repeated_potentials + mu_edges
        nu_nodes, nu_edges, _ = sequence_sum_product(uscores, bscores)

        # UPDATE AVERAGES
        q_avg = k * q_avg / (k+1) + q / (k+1) 
        mu_avg_nodes = k * mu_avg_nodes / (k+1) + np.exp(mu_nodes) / (k+1) 
        mu_avg_edges = k * mu_avg_edges / (k+1) + np.exp(mu_edges) / (k+1) 

        # COMPUTE DUAL GAP
        if k  == (max_iter * length -1):
        # if k % 20 == 0:
            # do vitervi
            ymax = mp(np.dot(q_avg, L) + unary_potentials, pairwise_potentials, edges)
            #make one hot encoding
            node_embeddings = np.zeros((length, n_states), dtype=np.int)
            gx = np.ogrid[:length]
            node_embeddings[gx, ymax] = 1
            ##accumulated pairwise
            sum_edge_embeddings = np.dot(node_embeddings[edges[:, 0]].T,
                        node_embeddings[edges[:, 1]])
            # compute value of y_max
            m1 = (np.dot(q_avg, L) + unary_potentials)[np.arange(length), ymax].sum()
            m2 = (pairwise_potentials * sum_edge_embeddings).sum()
            maxval = m1 + m2
            en1 = (unary_potentials * mu_avg_nodes).sum()
            en2 = (repeated_potentials * mu_avg_edges).sum()
            minval = np.min(np.dot(mu_avg_nodes, L), axis=1).sum() + en1 + en2
            dual_gap = maxval - minval
    # print("Dual gap is : %f" % (dual_gap))
    # pdb.set_trace()
    return mu_avg_nodes, mu_avg_edges, dual_gap

def grad_entropy(MU, edges):
    marginal_nodes, marginal_edges = MU
    grad_nodes = np.log(marginal_nodes + 1e-5) + 1
    grad_edges = -np.log(marginal_edges + 1e-5) - 1
    return grad_nodes, grad_edges

def logsumexp(arr, axis=None):
    themax = np.amax(arr)
    return themax + np.log(np.sum(np.exp(arr - themax) + 1e-6, axis=axis))

def sequence_sum_product(uscores, bscores):
    """Apply the sum-product algorithm on a chain
    :param uscores: array T*K, (unary) scores on individual nodes
    :param bscores: array (T-1)*K*K, (binary) scores on the edges
    :return: log-marginals on nodes, log-marginals on edges, log-partition
    """

    # I keep track of the islog messages instead of the messages
    # This is more stable numerically

    length, nb_class = uscores.shape

    if length == 1:
        log_partition = logsumexp(uscores[0])
        umargs = uscores - log_partition
        bmargs = np.zeros([length - 1, nb_class, nb_class])
        return umargs, bmargs, log_partition

    bm = np.zeros([length - 1, nb_class])  # backward_messages
    fm = np.zeros([length - 1, nb_class])  # forward_messages

    # backward pass
    bm[-1] = logsumexp(bscores[-1] + uscores[-1], axis=-1)
    for t in range(length - 3, -1, -1):
        bm[t] = logsumexp(bscores[t] + uscores[t + 1] + bm[t + 1], axis=-1)

    # we compute the log-partition and include it in the forward messages
    log_partition = logsumexp(bm[0] + uscores[0])

    # forward pass
    fm[0] = logsumexp(bscores[0].T + uscores[0] - log_partition, axis=-1)
    for t in range(1, length - 1):
        fm[t] = logsumexp(bscores[t].T + uscores[t] + fm[t - 1], axis=-1)

    # unary marginals
    umargs = np.empty([length, nb_class])
    umargs[0] = uscores[0] + bm[0] - log_partition
    umargs[-1] = fm[-1] + uscores[-1]
    for t in range(1, length - 1):
        umargs[t] = fm[t - 1] + uscores[t] + bm[t]

    # binary marginals
    bmargs = np.empty([length - 1, nb_class, nb_class])

    if length == 2:
        bmargs[0] = uscores[0, :, np.newaxis] + bscores[0] + uscores[1] - log_partition
    else:
        bmargs[0] = uscores[0, :, np.newaxis] + bscores[0] + uscores[1] + bm[1] - log_partition
        bmargs[-1] = fm[-2, :, np.newaxis] + uscores[-2, :, np.newaxis] + bscores[-1] + uscores[-1]
        for t in range(1, length - 2):
            bmargs[t] = fm[t - 1, :, np.newaxis] + uscores[t, :, np.newaxis] + bscores[t] + \
                        uscores[t + 1] + bm[t + 1]

    return umargs, bmargs, log_partition


def tree_max_product(unary_potentials, pairwise_potentials, edges):
    n_vertices, n_states = unary_potentials.shape
    parents = -np.ones(n_vertices, dtype=np.int)
    visited = np.zeros(n_vertices, dtype=np.bool)
    neighbors = [[] for i in range(n_vertices)]
    pairwise_weights = [[] for i in range(n_vertices)]
    for pw, edge in zip(pairwise_potentials, edges):
        neighbors[edge[0]].append(edge[1])
        pairwise_weights[edge[0]].append(pw)
        neighbors[edge[1]].append(edge[0])
        pairwise_weights[edge[1]].append(pw.T)

    messages_forward = np.zeros((n_vertices, n_states))
    messages_backward = np.zeros((n_vertices, n_states))
    pw_forward = np.zeros((n_vertices, n_states, n_states))
    # build a breadth first search of the tree
    traversal = []
    lonely = 0
    while lonely < n_vertices:
        for i in range(lonely, n_vertices):
            if not visited[i]:
                queue = [i]
                lonely = i + 1
                visited[i] = True
                break
            lonely = n_vertices

        while queue:
            node = queue.pop(0)
            traversal.append(node)
            for pw, neighbor in zip(pairwise_weights[node], neighbors[node]):
                if not visited[neighbor]:
                    parents[neighbor] = node
                    queue.append(neighbor)
                    visited[neighbor] = True
                    pw_forward[neighbor] = pw

                elif not parents[node] == neighbor:
                    raise ValueError("Graph not a tree")
    # messages from leaves to root
    for node in traversal[::-1]:
        parent = parents[node]
        if parent != -1:
            message = np.max(messages_backward[node] + unary_potentials[node] +
                             pw_forward[node], axis=1)
            message -= message.max()
            messages_backward[parent] += message
    # messages from root back to leaves
    for node in traversal:
        parent = parents[node]
        if parent != -1:
            message = messages_forward[parent] + unary_potentials[parent] + pw_forward[node].T
            # leaves to root messages from other children
            message += messages_backward[parent] - np.max(messages_backward[node]
                                                          + unary_potentials[node]
                                                          + pw_forward[node], axis=1)
            message = message.max(axis=1)
            message -= message.max()
            messages_forward[node] += message

    return np.argmax(unary_potentials + messages_forward + messages_backward, axis=1)


def iterative_max_product(unary_potentials, pairwise_potentials, edges,
                          max_iter=10, damping=.5, tol=1e-5):
    n_edges = len(edges)
    n_vertices, n_states = unary_potentials.shape
    messages = np.zeros((n_edges, 2, n_states))
    all_incoming = np.zeros((n_vertices, n_states))
    for i in range(max_iter):
        diff = 0
        for e, (edge, pairwise) in enumerate(zip(edges, pairwise_potentials)):
            # update message from edge[0] to edge[1]
            update = (all_incoming[edge[0]] + pairwise.T +
                      unary_potentials[edge[0]]
                      - messages[e, 1])
            old_message = messages[e, 0].copy()
            new_message = np.max(update, axis=1)
            new_message -= np.max(new_message)
            new_message = damping * old_message + (1 - damping) * new_message
            messages[e, 0] = new_message
            update = new_message - old_message
            all_incoming[edge[1]] += update
            diff += np.abs(update).sum()

            # update message from edge[1] to edge[0]
            update = (all_incoming[edge[1]] + pairwise +
                      unary_potentials[edge[1]]
                      - messages[e, 0])
            old_message = messages[e, 1].copy()
            new_message = np.max(update, axis=1)
            new_message -= np.max(messages[e, 1])
            new_message = damping * old_message + (1 - damping) * new_message
            messages[e, 1] = new_message
            update = new_message - old_message
            all_incoming[edge[0]] += update
            diff += np.abs(update).sum()
        if diff < tol:
            break
    return np.argmax(all_incoming + unary_potentials, axis=1)
