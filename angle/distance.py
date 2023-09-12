import scipy.linalg as la
from angle.wasserstein_1d import *


def pdist_distribution1d(X, metric='wasserstein2'):
    """
    get distance_matrix using 1-d wasserstein distance

    Input:
        X: n*m matrix with each row m iid observations from mu_i
    Output:
        n * n distance matrix
    """
    n = len(X)
    d_matrix = np.zeros([n, n])  # distance metric

    if metric == 'wasserstein2':
        for i in range(n):
            for j in range(i):
                d_matrix[i, j] = d_matrix[j, i] = wasserstein_1d(X[i], X[j], p=2)
    if metric == 'euclidean':
        for i in range(n):
            for j in range(i):
                d_matrix[i, j] = d_matrix[j, i] = la.norm(X[i] - X[j])

    return d_matrix


def dist_distribution1d(X, Y, metric='wasserstein2'):
    """
    get distance_matrix using 1-d wasserstein distance

    Input:
        X: n1*m1 matrix with each row m iid observations from mu_i
        Y: n2*m2 matrix with each row m iid observations from mu_i
    Output:
        n1 * n2 distance matrix
    """
    m, n = len(X), len(Y)
    d_matrix = np.zeros([m, n])  # distance metric

    if metric == 'wasserstein2':
        for i in range(m):
            for j in range(n):
                d_matrix[i, j] = wasserstein_1d(X[i], Y[j])
    if metric == 'euclidean':
        for i in range(m):
            for j in range(n):
                d_matrix[i, j] = la.norm(X[i] - X[j])

    return d_matrix


def pdist_spd(X, metric='frobenius'):
    """
    get distance_matrix for spd matrices

    Input:
        X: n*d*d array with X[i,,] iid observations of random spd matrix
        metric: 'frobenius', 'cholesky', 'affineinv'
    Output:
        n * n distance matrix
    """
    n, d = X.shape[0:2]
    # distance metric
    d_matrix = np.zeros([n, n])

    if metric == 'frobenius':
        for i in range(n):
            for j in range(i):
                d_matrix[j, i] = d_matrix[i, j] = la.norm(X[i] - X[j])

    elif metric == 'cholesky':
        X_cholesky = np.zeros([n, d, d])
        for i in range(n):
            X_cholesky[i] = la.cholesky(X[i])
        for i in range(n):
            for j in range(i):
                d_matrix[j, i] = d_matrix[i, j] = la.norm(X_cholesky[i] - X_cholesky[j])

    elif metric == 'affineinv':
        X_halfinv = np.zeros([n, d, d])
        for i in range(n):
            X_halfinv[i] = la.fractional_matrix_power(X[i], -0.5)
        for i in range(n):
            for j in range(i):
                mat = np.matmul(np.matmul(X_halfinv[i], X[j]), X_halfinv[i])
                mat = la.logm(mat)
                d_matrix[j, i] = d_matrix[i, j] = la.norm(mat)

    return d_matrix


def dist_spd(X, Y, metric='frobenius'):
    """
    get distance_matrix for spd matrices between two samples

    Input:
        X: n1*d*d array with X[i, , ] iid observations of random spd matrix
        Y: n2*d*d array with X[i, , ] iid observations of random spd matrix
        metric: 'frobenius', 'cholesky', 'affineinv'
    Output:
        n1 * n2 distance matrix
    """
    m, n = len(X), len(Y)
    d = X.shape[1]
    # distance metric
    d_matrix = np.zeros([m, n])

    if metric == 'frobenius':
        for k in range(m):
            for l in range(n):
                d_matrix[k, l] = la.norm(X[k] - Y[l])

    elif metric == 'cholesky':
        X_cholesky = np.zeros([m, d, d])
        Y_cholesky = np.zeros([n, d, d])
        for i in range(m):
            X_cholesky[i] = la.cholesky(X[i])
        for j in range(n):
            Y_cholesky[j] = la.cholesky(Y[j])

        for k in range(m):
            for l in range(n):
                d_matrix[k, l] = la.norm(X_cholesky[k] - Y_cholesky[l])

    elif metric == 'affineinv':
        X_halfinv = np.zeros([m, d, d])
        Y_halfinv = np.zeros([n, d, d])
        for i in range(m):
            X_halfinv[i] = la.fractional_matrix_power(X[i], -0.5)
        for j in range(n):
            Y_halfinv[j] = la.fractional_matrix_power(Y[j], -0.5)
        for k in range(m):
            for l in range(n):
                mat = np.matmul(np.matmul(X_halfinv[k], Y[l]), X_halfinv[k])
                mat = la.logm(mat)
                d_matrix[k, l] = la.norm(mat)

    return d_matrix


def dist_euclidean(X, Y):
    x2 = np.sum(X ** 2, axis=1)  # shape of (m)
    y2 = np.sum(Y ** 2, axis=1)  # shape of (n)

    xy = np.matmul(X, Y.T)
    x2 = x2.reshape(-1, 1)
    dist = np.sqrt(x2 - 2 * xy + y2)

    return dist
