import numpy as np


def _energy_distance_from_distance_matrices(
    distance_xx,
    distance_yy,
    distance_xy,
        estimation_stat='U_STATISTIC'):
    """
    Compute energy distance with precalculated distance matrices.

    Input:
        distance_xx: Pairwise distances of X.
        distance_yy: Pairwise distances of Y.
        distance_xy: Pairwise distances between X and Y.
        estimation_stat: If 'U_STATISTIC', calculate energy
            distance using Hoeffding's unbiased U-statistics. Otherwise, use
            von Mises's biased V-statistics.

    Output:
        energy distance
    """
    m, n = len(distance_xx), len(distance_yy)

    if estimation_stat == 'U_STATISTIC':
        # If using u-statistics, we exclude the central diagonal of 0s for the
        return (
            2 * np.mean(distance_xy) - np.sum(distance_xx) / (m * (m - 1)) \
            - np.sum(distance_yy) / (n * (n - 1))
        )
    else:
        return (
            2 * np.mean(distance_xy) - np.mean(distance_xx) - np.mean(distance_yy)
        )

# In[ ]:
