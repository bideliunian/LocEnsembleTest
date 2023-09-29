import numpy as np


def angle_statistics_from_distance_matrices(dist_xx, dist_yy, dist_xy, dist_xz, dist_yz):
    """
    computing angle statistics for two sample test based on distance matrices

    Parameters:
        dist_xx: distance matrix of samples from group 1
        dist_yy, dist_xy, dist_xz, dist_yz
    Returns:
        angle ensemble statistics
    """

    angle_dist_xx = local_angle_pdist(dist_xz, dist_xx)
    angle_dist_yy = local_angle_pdist(dist_yz, dist_yy)
    angle_dist_xy = local_angle_dist(dist_xz, dist_yz, dist_xy)

    angle_stat = np.mean(2 * angle_dist_xy) - np.mean(angle_dist_xx) - np.mean(angle_dist_yy)

    return angle_stat


def local_angle_pdist(dist_xz, dist_xx):
    """
    get angle distance matrix between two samples X and Y with respect to Z, anlge_z(x, y)
    sample size X: n; Y: m; Z: n_ref
    Parameters:
        dist_xz: m * n_ref distance matrix between x and z
        dist_xx: m * m distance matrix between y and z
    Returns:
        n * m distance matrix
    """
    m, n_ref = dist_xz.shape
    angle_matrix_cul = np.zeros([m, m])

    for idx in range(n_ref):
        angle_matrix = np.zeros([m, m])  # angle distance metric
        for i in range(m):
            for j in range(i):
                denominator = 2 * dist_xz[i, idx] * dist_xz[j, idx]
                if denominator == 0.:
                    num = 1.
                else:
                    num = (dist_xz[i, idx] ** 2 + dist_xz[j, idx] ** 2 - dist_xx[i, j] ** 2) / denominator
                angle = np.arccos(max(-1, min(num, 1)))
                angle_matrix[i, j] = angle / n_ref
                angle_matrix[j, i] = angle / n_ref

        angle_matrix_cul += angle_matrix

    return angle_matrix_cul


def local_angle_dist(dist_xz, dist_yz, dist_xy):
    """
    get angle distance matrix between two samples X and Y with respect to Z, anlge_z(x, y)
    sample size X: n; Y: m; Z: n_ref
    Parameters:
        dist_xz: m * n_ref distance matrix between x and z
        dist_yz: n * n_ref distance matrix between y and z
        dist_xy: m * n distance matrix between x and y
    Returns:
        n * m distance matrix
    """
    m, n_ref = dist_xz.shape
    n = dist_yz.shape[0]
    angle_matrix_cul = np.zeros([m, n])

    for idx in range(n_ref):
        angle_matrix = np.zeros([m, n])  # angle distance metric
        for i in range(m):
            for j in range(n):
                denominator = 2 * dist_xz[i, idx] * dist_yz[j, idx]
                if denominator == 0.:
                    num = 1.
                else:
                    num = (dist_xz[i, idx] ** 2 + dist_yz[j, idx] ** 2 - dist_xy[i, j] ** 2) / denominator
                angle = np.arccos(max(-1, min(num, 1)))
                angle_matrix[i, j] = angle / n_ref

        angle_matrix_cul += angle_matrix

    return angle_matrix_cul
