# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def sample_simplex(d):
    '''Return one sample uniformly on the d-dimensional simplex.'''
    Exp = -np.log(np.random.uniform(size=d))
    return Exp/np.sum(Exp)

def bures_wasserstein(mean_1, mean_2, cov_1, cov_2):
    '''Return the OT distance between two Gaussian distributions N(mean_1,cov_1) and N(mean_1,cov_2).'''
    assert mean_1.shape == mean_2.shape
    assert cov_1.shape == cov_2.shape
    d = mean_1.shape[0]
    assert cov_2.shape == (d,d)
    
    e,v = np.linalg.eigh(cov_1)
    e[e<0]=0.
    sqrt_1 = v.dot(np.diag(np.sqrt(e))).dot(v.T)
    
    cross = sqrt_1.dot(cov_2).dot(sqrt_1)
    e,v = np.linalg.eigh(cross)
    e[e<0]=0.
    cross = v.dot(np.diag(np.sqrt(e))).dot(v.T)
    
    return np.linalg.norm(mean_1-mean_2)**2 + np.trace(cov_1 + cov_2 - 2*cross)

def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex."""
    # Adrien Gaidon - INRIA - 2011
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

def projection_Omega(matrix, k, max_iter_Dykstra=10):
    '''Project the matrix onto {0 <= Omega <= I with Trace(Omega)=k}, using Dykstra's projection algorithm.'''
    d = matrix.shape[0]
    matrix = 0.5*(matrix + matrix.T)
    e,v = np.linalg.eigh(matrix)
    x = e
    p = np.zeros(d)
    q = np.zeros(d)
    for _ in range(max_iter_Dykstra):
        y = x + p
        y[y>1] = 1.
        p = x + p - y
        x = euclidean_proj_simplex(y+q, s=k)
        q = y + q - x
    x[x<0.] = 0.
    return v.dot(np.diag(x)).dot(v.T)