# Robust Testing using ball statistics, angle statistics, graph based method
# and energy distance
# Author: Qi Zhang

import numpy as np
import random
from itertools import combinations
from math import factorial

from angle.distance import *
from angle.bdf import _ball_statistics_from_distance_matrices
from angle.adf import _angle_statistics_from_distance_matrices
from angle.energy_distance import _energy_distance_from_distance_matrices
from angle.gbt import *


def permutation_test_homogeneity(
    x,
    y,
    func="angle",
    method="exact",
    num_rounds=399,
    space='distribution',
    metric='wasserstein2',
    seed=None
):
    """
    Nonparametric permutation test

    Parameters
    -------------
    x : list or numpy array with shape (n_datapoints,);first sample

    y : list or numpy array with shape (n_datapoints,);second sample

    func : angle to compute the statistic for the permutation test.
        - If 'angle', uses 'acvm_two_sample' for a two-sided test.
        - If 'ball', uses 'bcvm_two_sample' for a two-sided test.
        - If 'energy', uses energy distance for a two-sided test.
        - If 'gbt', graph based test for a two-sided test.

    method : 'approximate' or 'exact' (default: 'exact')

    num_rounds : int (default: 1000)
        The number of permutation samples if `method='approximate'`.

    space: 'distn' or 'spd'

    metric: space='distribution': 'wasserstein' or 'euclidean'
            space='spd': 'frobenius', 'cholesky', 'affineinv'

    seed : int or None (default: None)
        The random seed for generating permutation samples if
        `method='approximate'`.

    Output:
        p-value under the null hypothesis
    """

    if method not in ("approximate", "exact"):
        raise AttributeError(
            'method must be approximate or exact',
        )

    # computing distance matrix
    if space == 'distribution':
        dist_x = pdist_distribution1d(X=x, metric=metric)
        dist_y = pdist_distribution1d(X=y, metric=metric)
        dist_xy = dist_distribution1d(X=x, Y=y, metric=metric)

    elif space == 'spd':
        dist_x = pdist_spd(X=x, metric=metric)
        dist_y = pdist_spd(X=y, metric=metric)
        dist_xy = dist_spd(X=x, Y=y, metric=metric)

    # pooled sample
    m, n = len(x), len(y)
    pooled = np.vstack((x, y))

    # pooled distance matrix (m+n)*(m+n)
    if space == "distribution":
        dist_pooled = pdist_distribution1d(pooled, metric=metric)
        
    elif space == "spd":
        dist_pooled = pdist_spd(pooled, metric=metric)

    # angle to computing test statistics
    if isinstance(func, str):

        if func not in ("angle", "ball", "energy", 'graph'):
            raise AttributeError(
                "Provide a custom angle"
            )

        elif func == "angle":
            def get_stat(dist_x, dist_y, dist_xy):
                return _angle_statistics_from_distance_matrices(dist_x, dist_y, dist_xy)

        elif func == "ball":
            def get_stat(dist_x, dist_y, dist_xy):
                return _ball_statistics_from_distance_matrices(dist_x, dist_y, dist_xy)

        elif func == "energy":
            def get_stat(dist_x, dist_y, dist_xy):
                return _energy_distance_from_distance_matrices(dist_x, dist_y, dist_xy,
                                                            estimation_stat='U_STATISTIC')

        elif func == 'graph':
            simi_graph = mstree(dist_pooled, k=5)

    random.seed(seed)
    at_least_as_extreme = 0.0

    if func == 'graph':
        reference_stat = graph_based_test(graph=simi_graph,
                                          sample1_id=list(range(0, m)),
                                          sample2_id=list(range(m, m + n)))
    else:
        reference_stat = get_stat(dist_x, dist_y, dist_xy)

    if method == "exact":
        for index_x in combinations(range(m + n), m):
            indices_y = [i for i in range(m + n) if i not in index_x]
            indices_x = list(index_x)
            if func == 'graph':
                diff = graph_based_test(graph=simi_graph,
                                        sample1_id=indices_x,
                                        sample2_id=indices_y)
            else:
                diff = get_stat(dist_x=dist_pooled[np.ix_(indices_x, indices_x)],
                                dist_y=dist_pooled[np.ix_(indices_y, indices_y)],
                                dist_xy=dist_pooled[np.ix_(indices_x, indices_y)])
            if diff > reference_stat or np.isclose(diff, reference_stat):
                at_least_as_extreme += 1.

        num_rounds = factorial(m + n) / (factorial(m) * factorial(n))

    else:
        nround = 1
        while nround < num_rounds:
            indices_x = random.sample(range(m + n), m)
            indices_y = [j for j in range(m + n) if j not in indices_x]
            if func == 'graph':
                diff = graph_based_test(graph=simi_graph, 
                                        sample1_id=indices_x, 
                                        sample2_id=indices_y)
            else:
                diff = get_stat(dist_x=dist_pooled[np.ix_(indices_x, indices_x)],
                                dist_y=dist_pooled[np.ix_(indices_y, indices_y)],
                                dist_xy=dist_pooled[np.ix_(indices_x, indices_y)])
            if diff > reference_stat or np.isclose(diff, reference_stat):
                at_least_as_extreme += 1.
            
            nround += 1

        # To cover the actual experiment results
        at_least_as_extreme += 1.
        num_rounds += 1.

    return at_least_as_extreme / num_rounds


def permutation_test_homogeneity_from_distance_matrix(
    m, 
    dist,
    func="angle",
    method="approximate",
    num_rounds=399,
    seed=None
):
    """
    Nonparametric permutation test

    Parameters
    -------------
    m : sample size of the first sample

    dist : distance matrix of the combined samples

    func : angle to compute the statistic for the permutation test.
        - If 'angle', uses 'acvm_two_sample' for a two-sided test.
        - If 'ball', uses 'bcvm_two_sample' for a two-sided test.
        - If 'energy', uses energy distance for a two-sided test.
        - If 'gbt', graph based test for a two-sided test.

    method : 'approximate' or 'exact' (default: 'exact')

    num_rounds : int (default: 1000)
        The number of permutation samples if `method='approximate'`.

    seed : int or None (default: None)
        The random seed for generating permutation samples if
        `method='approximate'`.

    Output:
        p-value under the null hypothesis
    """

    if method not in ("approximate", "exact"):
        raise AttributeError(
            'method must be approximate or exact',
        )

    # calculating distance matrix
    n = len(dist) - m # sample size of second sample

    dist_x = dist[0:m, 0:m]
    dist_y = dist[m:(m+n), m:(m+n)]
    dist_xy = dist[0:m, m:(m+n)]

    # angle to computing test statistics
    if isinstance(func, str):

        if func not in ("angle", "ball", "energy", 'graph'):
            raise AttributeError(
                "Provide a custom angle"
            )

        elif func == "angle":
            def get_stat(dist_x, dist_y, dist_xy):
                return _angle_statistics_from_distance_matrices(dist_x, dist_y, dist_xy)

        elif func == "ball":
            def get_stat(dist_x, dist_y, dist_xy):
                return _ball_statistics_from_distance_matrices(dist_x, dist_y, dist_xy)

        elif func == "energy":
            def get_stat(dist_x, dist_y, dist_xy):
                return _energy_distance_from_distance_matrices(dist_x, dist_y, dist_xy,
                                                            estimation_stat='U_STATISTIC')

        elif func == 'graph':
            simi_graph = mstree(dist, k=5)

    random.seed(seed)
    at_least_as_extreme = 0.0

    if func == 'graph':
        reference_stat = graph_based_test(graph=simi_graph,
                                          sample1_id=list(range(0, m)),
                                          sample2_id=list(range(m, m + n)))
    else:
        reference_stat = get_stat(dist_x, dist_y, dist_xy)

    if method == "exact":
        for index_x in combinations(range(m + n), m):
            indices_y = [i for i in range(m + n) if i not in index_x]
            indices_x = list(index_x)
            if func == 'graph':
                diff = graph_based_test(graph=simi_graph,
                                        sample1_id=indices_x,
                                        sample2_id=indices_y)
            else:
                diff = get_stat(dist_x=dist[np.ix_(indices_x, indices_x)],
                                dist_y=dist[np.ix_(indices_y, indices_y)],
                                dist_xy=dist[np.ix_(indices_x, indices_y)])
            if diff > reference_stat or np.isclose(diff, reference_stat):
                at_least_as_extreme += 1.

        num_rounds = factorial(m + n) / (factorial(m) * factorial(n))

    else:
        nround = 1
        while nround < num_rounds:
            indices_x = random.sample(range(m + n), m)
            indices_y = [j for j in range(m + n) if j not in indices_x]
            if func == 'graph':
                diff = graph_based_test(graph=simi_graph, 
                                        sample1_id=indices_x, 
                                        sample2_id=indices_y)
            else:
                diff = get_stat(dist_x=dist[np.ix_(indices_x, indices_x)],
                                dist_y=dist[np.ix_(indices_y, indices_y)],
                                dist_xy=dist[np.ix_(indices_x, indices_y)])
            if diff > reference_stat or np.isclose(diff, reference_stat):
                at_least_as_extreme += 1.

            nround += 1

        # To cover the actual experiment results
        at_least_as_extreme += 1.
        num_rounds += 1.

    return at_least_as_extreme / num_rounds

# %%
