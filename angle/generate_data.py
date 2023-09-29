import numpy as np
from scipy.linalg import expm
from scipy.stats import multivariate_t


def generate_data_distribution_1d(n, m, model, delta=0):
    """
    angle generating distributional data
    x~N(mu, sigma)
    mu~tN(delta, r*0.5)
    sigma~Gamma(4, 0.2)
    Para:
        n: sample size
        m: number of observations for each distribution

    Return:
        n * m matrix
    """
    if model == 'model1':
        mu = np.random.standard_cauchy(n) + delta
        sigma = 1
        # sigma = np.random.gamma(shape=2, scale=0.5, size=n)
    
    elif model == 'model2':
        sigma_mu = 0.5
        mu = np.random.normal(loc=0.0, scale=delta*sigma_mu, size=n)
        sigma = 1
        # sigma = np.random.gamma(shape=2, scale=0.5, size=n)

    elif model == 'model3':
        mu = 0.0
        z = np.random.normal(loc=0.0, scale=1.0, size=n)
        sigma = np.exp(delta * z)

    data = np.zeros([n, m])
    for i in range(m):
        data[:, i] = np.random.normal(loc=mu, scale=sigma)

    return data


def generate_data_spd(n, d, model, delta):
    '''
    angle generating spd matrix data by cholesky decomposition
    A = LL^T
    P = (L+Wv)(L+Wv)^T
    Wv is a random sparse matrix
    A from a whishart distribution
    Para:
        n: sample size
        p: dimension, defalt 100
        m: sparsity of random matrix Wv

    return:
        n * d * d array
    '''

    data = np.zeros([n, d, d])

    if model == 'model1':

        for i in range(n):
            L = np.zeros([d, d])
            lower_tri_idx = np.tril_indices(d)
            p = int(d * (d + 1) / 2)
            U = multivariate_t.rvs(loc = np.full(p,0), shape = np.eye(p), df=1)
            U = delta * U
            
            L[lower_tri_idx] = U
            np.fill_diagonal(L, np.abs(L.diagonal()))
    
            W = np.matmul(L, L.T)
            data[i] = W

    elif model == 'model2':

        for i in range(n):
            L = np.zeros([d, d])
            lower_tri_idx = np.tril_indices(d)
            p = int(d * (d + 1) / 2)
            U = multivariate_t.rvs(loc = np.full(p,0), shape = np.eye(p), df=1)
            U = delta * U

            L[lower_tri_idx] = U
            np.fill_diagonal(L, np.abs(L.diagonal()))

            W = expm(L + L.T)
            data[i] = W

    return data


