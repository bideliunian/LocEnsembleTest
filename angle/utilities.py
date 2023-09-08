import numpy as np


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def double_centered(a):
    """
    Return a copy of the matrix : `a` which is double centered.
    For every element its row and column averages are subtracted, and the total average is added.

    Args:
        a: Original square matrix.

    Returns:
        Double centered matrix.
    """
    mu = np.mean(a)
    mu_cols = np.mean(a, axis=0, keepdims=True)
    mu_rows = np.mean(a, axis=1, keepdims=True)

    # Do one operation at a time, to improve broadcasting memory usage.
    a -= mu_rows
    a -= mu_cols
    a += mu

    return a


def u_centered(a):
    """
    Return a copy of the matrix `a` which is `U`-centered.
    If the element of the i-th row and j-th column of the original
    matrix :math:`a` is :math:`a_{i,j}`, then the new element will be
    .. math::
        \tilde{a}_{i, j} =
        \begin{cases}
        a_{i,j} - \frac{1}{n-2}\sum_{l=1}^n a_{il} -
        \frac{1}{n-2}\sum_{k=1}^n a_{kj} +
        \frac{1}{(n-1)(n-2)}\sum_{k=1}^n a_{kj},
        &\text{if } i \neq j, \\
        0,
        &\text{if } i = j.
        \end{cases}
    Args:
        a: Original square matrix.
    Returns:
        `U`-centered matrix.
    """
    dim = a.shape[0]

    u_mu = np.sum(a) / ((dim - 1) * (dim - 2))
    sum_cols = np.sum(a, axis=0, keepdims=True)
    sum_rows = np.sum(a, axis=1, keepdims=True)
    u_mu_cols = sum_cols / (dim - 2)
    u_mu_rows = sum_rows / (dim - 2)

    # Do one operation at a time, to improve broadcasting memory usage.
    a -= u_mu_rows
    a -= u_mu_cols
    a += u_mu

    # The diagonal is zero
    a[np.eye(dim, dtype=np.bool)] = 0

    return a


# In[ ]:




