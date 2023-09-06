import numpy as np


def quantile_function(qs, cws, xs):
    """ Computes the quantile function of an empirical distribution

    Parameters
    ----------
    qs: array-like, shape (n,)
        Quantiles at which the quantile function is evaluated
    cws: array-like, shape (m, ...)
        cumulative weights of the 1D empirical distribution, if batched, must be similar to xs
    xs: array-like, shape (n, ...)
        locations of the 1D empirical distribution, batched against the `xs.ndim - 1` first dimensions

    Returns
    -------
    q: array-like, shape (..., n)
        The quantiles of the distribution
    """
    n = xs.shape[0]
    cws = cws.T
    qs = qs.T
    idx = np.searchsorted(cws, qs).T
    return np.take_along_axis(xs, np.clip(idx, 0, n - 1), axis=0)


def wasserstein_1d(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    """
    Computes the 1 dimensional OT loss between two (batched) empirical
    distributions

    .. math:
        OT_{loss} = \int_0^1 |cdf_u^{-1}(q)  cdf_v^{-1}(q)|^p dq

    It is formally the p-Wasserstein distance raised to the power p.
    We do so in a vectorized way by first building the individual quantile functions then integrating them.

    This function should be preferred to `emd_1d` whenever the backend is
    different to numpy, and when gradients over
    either sample positions or weights are required.

    Parameters
    ----------
    u_values: array-like, shape (n, ...)
        locations of the first empirical distribution
    v_values: array-like, shape (m, ...)
        locations of the second empirical distribution
    u_weights: array-like, shape (n, ...), optional
        weights of the first empirical distribution, if None then uniform weights are used
    v_weights: array-like, shape (m, ...), optional
        weights of the second empirical distribution, if None then uniform weights are used
    p: int, optional
        order of the ground metric used, should be at least 1 (see [2, Chap. 2], default is 1
    require_sort: bool, optional
        sort the distributions atoms locations, if False we will consider they have been sorted prior to being passed to
        the function, default is True

    Returns
    -------
    cost: float/array-like, shape (...)
        the batched EMD

    References
    ----------
    .. [15] Peyré, G., & Cuturi, M. (2018). Computational Optimal Transport.

    """

    assert p >= 1, "The OT loss is only valid for p>=1, {p} was given".format(p=p)

    u_values = np.array(u_values)
    v_values = np.array(v_values)
    n = u_values.shape[0]
    m = v_values.shape[0]

    if u_weights is None:
        u_weights = np.full(u_values.shape, 1. / n)
    elif u_weights.ndim != u_values.ndim:
        u_weights = np.repeat(u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = np.full(v_values.shape, 1. / m)
    elif v_weights.ndim != v_values.ndim:
        v_weights = np.repeat(v_weights[..., None], v_values.shape[-1], -1)

    if require_sort:
        u_sorter = np.argsort(u_values, 0)
        u_values = np.take_along_axis(u_values, u_sorter, 0)

        v_sorter = np.argsort(v_values, 0)
        v_values = np.take_along_axis(v_values, v_sorter, 0)

        u_weights = np.take_along_axis(u_weights, u_sorter, 0)
        v_weights = np.take_along_axis(v_weights, v_sorter, 0)

    u_cumweights = np.cumsum(u_weights, 0)
    v_cumweights = np.cumsum(v_weights, 0)

    qs = np.sort(np.concatenate((u_cumweights, v_cumweights), 0), 0)
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)
    qs = np.pad(qs, pad_width=[(1, 0)] + (qs.ndim - 1) * [(0, 0)])
    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = np.abs(u_quantiles - v_quantiles)

    if p == 1:
        return np.sum(delta * np.abs(diff_quantiles), axis=0)
    return np.power(np.sum(delta * np.power(diff_quantiles, p), axis=0), 1/p)
