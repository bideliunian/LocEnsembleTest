import random
from itertools import combinations
from math import factorial

import numpy as np
from angle.distance import *
from angle.local_angle_stats import *


def permutation_test_homogeneity_local_ensemble(
        x,
        y,
        reference="uniform",
        n_ref=200,
        metric='wasserstein2',
        method="approximate",
        num_rounds=399,
        seed=None
):
    """
    Nonparametric permutation test

    Parameters
    -------------
    dist_xx : distance matrix of between x and x

    reference : reference distribution of the angle.
        - If 'uniform', uniform distribution over 1d wasserstein space
        - If 'gaussian', gaussian distribution over 1d wasserstein space.
        - If 'average', uses (Px + Py) /2.
        - If 'barycenter', midpoint of geodesic interpolation between Px and Py.

    method : 'approximate' or 'exact' (default: 'approximate')

    num_rounds : int (default: 1000)
        The number of permutation samples if `method='approximate'`.

    seed : int or None (default: None)
        The random seed for generating permutation samples if
        `method='approximate'`.

    Returns:
        p-value under the null hypothesis
    """

    if method not in ("approximate", "exact"):
        raise AttributeError(
            'method must be approximate or exact',
        )

    m, d = x.shape
    n = y.shape[0]

    data_pooled = np.vstack(x, y)
    dist_xyxy = dist_euclidean(X=data_pooled, Y=data_pooled)
    dist_xx = dist_xyxy[0:m, 0:m]
    dist_yy = dist_xyxy[m:(m + n), m:(m + n)]
    dist_xy = dist_xyxy[0:m, m:(m + n)]

    random.seed(seed)
    at_least_as_extreme = 0.0

    if reference == 'uniform':
        z = np.random.uniform(0, 1, size=(n_ref, d))
    elif reference == "gaussian":
        z = np.random.normal(0, 1, size=(n_ref, d))
    elif reference == "average":
        size_x = int(n_ref * m / (m + n))
        random_indices_x = np.random.choice(m, size_x, replace=False)
        random_indices_y = np.random.choice(n, size_x, replace=False)
        z = np.vstack((x[random_indices_x], y[random_indices_y]))
    elif reference == "barycenter":
        z = np.random.uniform(0, 1, size=(n_ref, d))
    else:
        raise AttributeError(
            "unkown reference distribution"
        )

    dist_xyz = dist_euclidean(data_pooled, z)
    dist_xz = dist_xyz[0:m, :]
    dist_yz = dist_xyz[m:(m + n), :]
    # angle_dist_joint = local_angle_pdist(dist_xyz, dist_xyxy)
    reference_stat = angle_statistics_from_distance_matrices(dist_xx, dist_yy, dist_xy, dist_xz, dist_yz)

    if method == "exact":
        for index_x in combinations(range(m + n), m):
            indices_y = [i for i in range(m + n) if i not in index_x]
            indices_x = list(index_x)

            dist_xx = dist_xyxy[np.ix_(indices_x, indices_x)],
            dist_yy = dist_xyxy[np.ix_(indices_y, indices_y)],
            dist_xy = dist_xyxy[np.ix_(indices_x, indices_y)],
            dist_xz = dist_xyz[indices_x, :],
            dist_yz = dist_xyz[indices_y, :]

            angle_stats = angle_statistics_from_distance_matrices(dist_xx, dist_yy, dist_xy, dist_xz, dist_yz)

            if angle_stats > reference_stat or np.isclose(angle_stats, reference_stat):
                at_least_as_extreme += 1.

        num_rounds = factorial(m + n) / (factorial(m) * factorial(n))

    else:
        nround = 1
        while nround < num_rounds:
            indices_x = random.sample(range(m + n), m)
            indices_y = [j for j in range(m + n) if j not in indices_x]

            dist_xx = dist_xyxy[np.ix_(indices_x, indices_x)],
            dist_yy = dist_xyxy[np.ix_(indices_y, indices_y)],
            dist_xy = dist_xyxy[np.ix_(indices_x, indices_y)],
            dist_xz = dist_xyz[indices_x, :],
            dist_yz = dist_xyz[indices_y, :]

            angle_stats = angle_statistics_from_distance_matrices(dist_xx, dist_yy, dist_xy, dist_xz, dist_yz)

            if angle_stats > reference_stat or np.isclose(angle_stats, reference_stat):
                at_least_as_extreme += 1.

            nround += 1

    # To cover the actual experiment results
    at_least_as_extreme += 1.
    num_rounds += 1.

    return at_least_as_extreme / num_rounds

# %%
