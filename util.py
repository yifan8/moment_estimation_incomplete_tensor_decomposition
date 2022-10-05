

import numpy as np
from scipy.stats import multivariate_normal as norm  
from numpy.random import choice, poisson, exponential, binomial, gamma
import scipy.linalg as la 
from scipy.optimize import linear_sum_assignment


def match(A, A0, *args):
        """Match columns (or elements) in A with columns (or elements) in A0, apply the same permutation to *args

        Parameters
        ----------
        A : ndarray
            Array to be rearranged
        A0 : ndarray
            Constant array as matching target 

        Returns
        -------
        ndarray, ...
            A with columns reordered to match A0, along with elements in *args after the same reordering
        """
        
        if A.shape != A0.shape:
            raise ValueError(f'The shape of A and A0, {A.shape} and {A0.shape}, must match.')
        if A.ndim < 2:
            A = A[None, :]
            A0 = A0[None, :]
        r = A.shape[1]
        s = A0.shape[1]
        costmat = np.zeros((s, r))
        for ro in range(s):
            for co in range(r):
                costmat[ro, co] = la.norm(A[:, co] - A0[:, ro])
        _, coidx = linear_sum_assignment(costmat)

        A = A[:, coidx]
        
        if args:
            ret = []
            for arg in args:
                shape = arg.shape
                newshape = list(shape)
                newshape[-1] = len(coidx)
                newshape = tuple(newshape)
                ret.append(arg.reshape(-1, shape[-1])[:, coidx].reshape(newshape))
            return A, *ret
        return A


def sample_poisson(M, ws, N = 2000, get_label = False):
    """Get a sample from GMM

    Parameters
    ----------
    M : np.array (n * r)
        Matrix of mean vectors (columns)
    ws : +np.array (r)
        Weights, not assumed to be normalized
    N : int, optional
        Sample size, by default 2000

    Returns
    -------
    np.array (n * N)
        Matrix of sampled data (columnwise)
    np.array (n)
        Number of data for each component
    np.array (n)
        Empirical weights of the sample 
    """

    n, r = M.shape
    true_M = M * 1.
    true_std = M * 1.
    ws = ws * 1.
    ws /= np.sum(ws)
    js = choice(r, size = N, p = ws)
    Ns = np.array([np.sum(js == i) for i in range(r)])
    V = np.zeros((n, N))
    start = 0
    
    for j in range(r):
        Nj = Ns[j]
        temp = poisson(lam = M[:, j], size = (Nj, n))
        true_mean = np.mean(temp, axis = 0)
        true_M[:, j] = true_mean
        js[start:start + Nj] = j
        true_std[:, j] = (1/(Nj-1) * np.sum((temp - true_mean)**2, axis = 0))**(1/2)
        V[:, start:start + Nj] = temp.T 
        start += Nj
    true_ws = Ns / N
    if not get_label:
        return V, Ns, true_ws, true_M, true_std
    return V, Ns, js, true_ws, true_M, true_std


def sample_gauss(A, lams, std, N = 2000, get_label = False):
    """Get a sample from GMM

    Parameters
    ----------
    A : np.array (n * r)
        Matrix of mean vectors (columns)
    lams : +np.array (r)
        Weights, not assumed to be normalized
    std : +float or or np.array (r) np.array (n * r)   
        Standard deviation of normal variables
        + float: same spherical for all mixture
        (r,) array: different spherical for mixutres
        (n, r) array: ellipses for mixtures
    N : int, optional
        Sample size, by default 2000

    Returns
    -------
    np.array (n * N)
        Matrix of sampled data (columnwise)
    """

    n, r = A.shape
    true_A = A.copy()
    true_std = std.copy()
    lams = lams * 1.
    lams /= np.sum(lams)

    #std = np.ones((n, r)) * std
    randvars = [norm(mean = A[:, j], cov = np.diag(std[:, j]**2)) for j in range(r)]

    js = choice(r, size = N, p = lams)
    Ns = np.array([np.sum(js == i) for i in range(r)])
    V = np.zeros((n, N))
    start = 0

    for j in range(r):
        Nj = Ns[j]
        temp = randvars[j].rvs(size = Nj)
        V[:, start:start + Nj] = temp.T 
        js[start:start + Nj] = j
        start += Nj
        true_mean = np.mean(temp, axis = 0)
        true_A[:, j] = true_mean
        true_std[:, j] = (1/(Nj-0) * np.sum((temp - true_mean)**2, axis = 0))**(1/2)
    true_lam = Ns / N
    if not get_label:
        return V, Ns, true_lam, true_A, true_std
    return V, Ns, js, true_lam, true_A, true_std


def sample_exp(scale, ws, loc = None, N = 2000):
    """_summary_

    Parameters
    ----------
    scale : _type_
        _description_
    ws : _type_
        _description_
    loc : _type_, optional
        _description_, by default None
    N : int, optional
        _description_, by default 2000

    Returns
    -------
    _type_
        _description_
    """
    n, r = scale.shape
    true_M = scale * 1.
    true_std = scale * 1.
    ws = ws * 1.
    ws /= np.sum(ws)
    js = choice(r, size = N, p = ws)
    Ns = np.array([np.sum(js == i) for i in range(r)])
    V = np.zeros((n, N))
    start = 0
    for j in range(r):
        Nj = Ns[j]
        temp = exponential(scale = scale[:, j], size = (Nj, n))
        if loc is not None:
            temp += loc[:, j] 
        true_mean = np.mean(temp, axis = 0)
        true_M[:, j] = true_mean
        true_std[:, j] = (1/(Nj-1) * np.sum((temp - true_mean)**2, axis = 0))**(1/2)
        V[:, start:start + Nj] = temp.T
        start += Nj
    true_ws = Ns / N
    return V, Ns, true_ws, true_M, true_std


def sample_gamma(scale, shape, ws, N = 2000):
    """_summary_

    Parameters
    ----------
    scale : _type_
        _description_
    shape : _type_
        _description_
    ws : _type_
        _description_
    N : int, optional
        _description_, by default 2000

    Returns
    -------
    _type_
        _description_
    """
    n, r = scale.shape
    true_M = np.zeros((n, r))
    true_std = np.zeros((n, r))
    ws = np.array(ws) * 1.
    ws /= np.sum(ws)
    js = choice(r, size = N, p = ws)
    Ns = np.array([np.sum(js == i) for i in range(r)])
    V = np.zeros((n, N))
    start = 0
    for j in range(r):
        Nj = Ns[j]
        temp = gamma(scale = scale[:, j], shape = shape[:, j], size = (Nj, n))
        true_mean = np.mean(temp, axis = 0)
        true_M[:, j] = true_mean
        true_std[:, j] = np.std(temp.T, axis = 1, ddof = 1)
        V[:, start:start + Nj] = temp.T
        start += Nj
    true_ws = Ns / N
    return V, Ns, true_ws, true_M, true_std


def sample_bernoulli(P, ws, N = 2000):
    n, r = P.shape
    true_M = P * 1.
    true_std = np.sqrt(P * (1 - P))
    ws = ws * 1.
    ws /= np.sum(ws)
    js = choice(r, size = N, p = ws)
    Ns = np.array([np.sum(js == i) for i in range(r)])
    V = np.zeros((n, N))
    start = 0
    for j in range(r):
        Nj = Ns[j]
        temp = binomial(n = np.ones(n, dtype = np.int8), p = P[:, j], size = (Nj, n))
        true_mean = np.mean(temp, axis = 0)
        true_M[:, j] = true_mean
        true_std[:, j] = (1/(Nj-1) * np.sum((temp - true_mean)**2, axis = 0))**(1/2)
        V[:, start:start + Nj] = temp.T
        start += Nj
    true_ws = Ns / N
    return V, Ns, true_ws, true_M, true_std