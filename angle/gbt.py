import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import rankdata

'''
Part1: generate similarity graph
'''


def get_graph(dist, k, graph_type='mstree'):
    """
    get similarity graph based on distance matrix

    Parameters
    -------------
    dist : n by n distance matrix

    k : the value of k in "k-MST" or "k-NNL" to construct the similarity graph.

    graph_type : Specify the type of the constructing graph.
    "mstree": minimal spanning tree.
    "nnlink": nearest neighbor link method.

    Returns:
        An edge matrix representing a similarity graph on the distinct values
        with the number of edges in the similarity graph being the number of
        rows and 2 columns. Each row records the subject indices of the two
        ends of an edge in the similarity grap
    """

    if graph_type not in ("mstree", "nnlink"):
        raise AttributeError(
            'graph_type must be mstree or nnlink.'
        )
    if graph_type == "mstree":
        graph = mstree(dist, k)
    else:
        graph = nnlink(dist, k)

    return graph


def mstree(dist, k=1):
    '''
    Parameters:
        dist: distance matrices, 2 dimensions
        k: the value of k in "k-MST" to construct the similarity graph,
        means to add n supplementary edges k-1 times.

    Returns:
        span_tree: csr matrix
        The n * n representation of the undirected k-mst over the input.
    '''
    n = len(dist)

    span_tree = np.zeros([n, n])
    dist_tri_upper = np.triu(dist)

    for i in range(k):
        span_tree_temp = minimum_spanning_tree(dist_tri_upper)
        span_tree += span_tree_temp
        dist_tri_upper[np.nonzero(span_tree_temp)] = 0.

    return np.array(span_tree)


def nnlink(dist, k=1):
    '''
    Para:
        dist: distance matrices, 2 dimensions
        k: the value of k in "k-NNL" to construct the similarity graph.

    Returns:
        neighbors: matrices, 2 dimensions
        The n * n representation of the undirected k-mst over the input.
    '''
    n = len(dist)
    neighbors = np.zeros([n, n])

    ranks = rankdata(a=dist, axis=1) - 1
    ranks[ranks > k] = 0.

    indices = np.nonzero(ranks)
    neighbors[indices] = dist[indices]

    neighbors_sym = np.triu(neighbors) + np.tril(neighbors).T
    neighbors_upper = np.triu(neighbors_sym)

    return neighbors_upper


'''
Part2: graph based two sample test
'''


def _get_r(edge, sample1_id):
    '''
    helper angle
    '''
    r1 = r2 = 0
    n = len(edge)
    for i in range(n):
        e1 = (edge[i, 0] in sample1_id)
        e2 = (edge[i, 1] in sample1_id)
        if not e1 and not e2:
            r1 = r1 + 1
        if e1 and e2:
            r2 = r2 + 1

    return([r1, r2])


def graph_based_test(graph, sample1_id, sample2_id):
    '''
    implement Graph-Based Two-Sample Tests based on
    Chen, H., & Friedman, J. H. (2017).
    A new graph-based two-sample test for multivariate and object data. JASA

    Para:
        graph: similarity graph, N*N, 2 dimensions
        sample1_id: the id of the nodes from sample 1.

    Returns:
        test statistics based on GBT methods.
    '''

    edge = np.transpose(np.nonzero(graph))
    ranks = _get_r(edge=edge, sample1_id=sample1_id)
    r1, r2 = ranks
    m, n = len(sample1_id), len(sample2_id)
    N = n + m

    sub_graph_counts = np.count_nonzero(graph+graph.T, axis=1)
    num_edge = np.sum(sub_graph_counts) / 2
    num_pair_sharing = sum(sub_graph_counts**2) / 2 - num_edge
    # pair of nodes sharing a node * 2
    mu1 = num_edge * n * (n-1) / N / (N-1)
    mu2 = num_edge * m * (m-1) / N / (N-1)

    const1_m = m * (m-1) / N / (N-1)
    const1_n = n * (n-1) / N / (N-1)
    const2_m = const1_m * (m - 2) / (N - 2)
    const2_n = const1_n * (n - 2) / (N - 2)
    const3_m = const2_m * (m - 3) / (N - 3)
    const3_n = const2_n * (n - 3) / (N - 3)
    const_mn = m * n * (m-1) * (n-1) / N / (N-1) / (N-2) / (N-3)
    sigma_1 = 2*num_pair_sharing * const2_n + \
        (num_edge * (num_edge - 1) - 2*num_pair_sharing) * const3_n + mu1 - mu1**2
    sigma_2 = 2*num_pair_sharing * const2_m + \
        (num_edge * (num_edge - 1) - 2*num_pair_sharing) * const3_m + mu2 - mu2**2
    sigma_12 = (num_edge * (num_edge - 1) - 2*num_pair_sharing) * const_mn - mu1 * mu2
    Sigma = [[sigma_1, sigma_12], [sigma_12, sigma_2]]

    r_centered = np.array([r1 - mu1, r2 - mu2])
    test_stat = r_centered.dot(np.linalg.inv(Sigma)).dot(r_centered.T)

    return test_stat
# In[ ]:
