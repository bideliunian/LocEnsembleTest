#!/usr/bin/env python
# coding: utf-8

# In[7]:

import numpy as np
from scipy.stats import rankdata
import distance


def empirical_bdf(x):
    '''
    evluating ADF at n*n data grids

    Input:
        x: m*n distance matrix,
    Output:
        n * n matrix
    '''
    n = len(x)
    B = rankdata(a=x, axis=1) / n
    return B


def _bdf_helper(dist1, dist2):
    '''
    cvm divergence of y||x
    Input:
        dist1: distance matrix of x m*m
        dist2: distance matrix of dist(x,y) m*n
    Output:
        cvm divergence
    '''
    m, n = dist2.shape
    rank1 = rankdata(a=dist1, axis=1)

    dist12 = np.concatenate((dist1, dist2), axis=1)
    rank12 = rankdata(a=dist12, axis=1)

    mdf1 = rank1 / m
    mdf2 = (rank12[0:m, 0:m] - rank1) / n

    div = np.sum((mdf1 - mdf2)**2)

    return div


def _ball_statistics_from_distance_matrices(dist_xx, dist_yy, dist_xy):
    '''
    computing ball statistics for two sample test based on distance matrices

    Input:
        dist_xx: distance matrix of samples from group 1
        dist_yy: distance matrix of samples from group 2
        dist_xy: cross distance matrix between x and y m*n
    Output:
        ACvM statistics
    '''
    m, n = len(dist_xx), len(dist_yy)

    sum1 = _bdf_helper(dist1=dist_xx, dist2=dist_xy)
    sum2 = _bdf_helper(dist1=dist_yy, dist2=dist_xy.T)

    const = 2*(m*n/(m**2+n**2))**2
    return const*2*(sum1+sum2)


def ball_statistics(x, y, space, metric):
    '''
    computing ball statistics for two sample test

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

    return _ball_statistics_from_distance_matrices(dist_xx, dist_yy, dist_xy)
