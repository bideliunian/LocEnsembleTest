#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import distance


def emerical_adf(x):
    '''
    evluating ADF at n*n data grids
    Input:
        x: n*n distance matrix,
    Output:
        n * n matrix
    '''
    n = len(x)
    A = np.zeros([n, n])

    for k in range(n):
        for l in range(k):
            acd_kl = 0.
            for r in range(n):
                if x[r, k] * x[r, l] == 0.:
                    num = 1.
                else:
                    num = (x[r, k]**2 + x[r, l]**2 - x[k, l]**2) \
                           / (2 * x[r, k] * x[r, l])

                if num > 1.:
                    num = 1.
                elif num < -1.:
                    num = -1.
                acd_kl = acd_kl + np.arccos(num) / n
            A[k, l] = A[l, k] = acd_kl / np.pi

    return A


def _adf_helper(dist1, dist2):
    '''
    cvm divergence of y||x
    Input:
        dist1: distance matrix of x
        dist2: distance matrix of dist(x,y)
    Output:
        cvm divergence
    '''
    (m, n) = dist2.shape
    div = 0.
    for i in range(m):
        for j in range(i):
            adf_1_ij = 0.
            adf_2_ij = 0.
            for idx_1 in range(m):
                denominator = 2 * dist1[idx_1, i] * dist1[idx_1, j]
                if denominator == 0.:
                    num = 1.
                else:
                    num = (dist1[idx_1, i]**2 + dist1[idx_1, j]**2 - dist1[i, j]**2) / denominator

                if num > 1.:
                    num = 1.
                elif num < -1.:
                    num = -1.

                adf_1_ij = adf_1_ij + np.arccos(num) / m

            for idx_2 in range(n):
                denominator = (2 * dist2[i, idx_2] * dist2[j, idx_2])
                if denominator == 0.:
                    num = 1.
                else:
                    num = (dist2[i, idx_2]**2 + dist2[j, idx_2]**2 - dist1[i, j]**2) / denominator

                if num > 1.:
                    num = 1.
                elif num < -1.:
                    num = -1.
                    
                adf_2_ij = adf_2_ij + np.arccos(num) / n

            div += (adf_1_ij - adf_2_ij)**2

    return div


def _angle_statistics_from_distance_matrices(dist_xx, dist_yy, dist_xy):
    '''
    computing angle statistics for two sample test based on distance matrices

    Input:
        dist_xx: distance matrix of samples from group 1
        dist_yy: distance matrix of samples from group 2
        dist_xy: cross distance matrix between x and y m*n
    Output:
        ACvM statistics
    '''
    m, n = len(dist_xx), len(dist_yy)

    sum1 = _adf_helper(dist1=dist_xx, dist2=dist_xy)
    sum2 = _adf_helper(dist1=dist_yy, dist2=dist_xy.T)

    const = 2*(m*n/(m**2+n**2))**2
    return const*2*(sum1+sum2)


def angle_statistics(x, y, space, metric):
    '''
    computing angle statistics for two sample test

    Input:
        x: samples from group 1
        y: samples from group 2
        space: 'distribution' or 'spd'
        metric: 'distribution': 'wasserstein' or 'euclidean'
                'spd': 'frobenius', 'cholesky', 'affineinv'
    Output:
        ACvM statistics
    '''

    if space == 'distribution':
        dist_xx = distance.pdist_distribution1d(X=x, metric=metric)
        dist_yy = distance.pdist_distribution1d(X=y, metric=metric)
        dist_xy = distance.dist_distribution1d(X=x, Y=y, metric=metric)
    elif space == 'spd':
        dist_xx = distance.pdist_spd(X=x, metric=metric)
        dist_yy = distance.pdist_spd(X=y, metric=metric)
        dist_xy = distance.dist_spd(X=x, Y=y, metric=metric)

    return _angle_statistics_from_distance_matrices(dist_xx, dist_yy, dist_xy)


# %%
