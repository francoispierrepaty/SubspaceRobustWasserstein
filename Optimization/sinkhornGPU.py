#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This code is adapted from the package POT (https://pot.readthedocs.io/en/stable/_modules/ot/gpu/bregman.html)

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Leo Gautheron <https://github.com/aje>
#
# License: MIT License


import cupy as np  # np used for matrix computation
import cupy as cp  # cp used for cupy specific operations


def sinkhorn_knopp_gpu(a, b, M, reg, numItermax=1000, stopThr=1e-9, warm_u=None, warm_v=None, verbose=False, log=True, to_numpy=False):
    """Solve the entropic regularization optimal transport problem and return the OT matrix."""

    if len(a) == 0:
        a = np.ones((M.shape[0],)) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],)) / M.shape[1]

    # init data
    Nini = len(a)
    Nfin = len(b)

    if log:
        log = {'err': []}
    
    if warm_u is None:
        u = np.ones(Nini) / Nini
    else:
        u = warm_u
    if warm_v is None:
        v = np.ones(Nfin) / Nfin
    else:
        v = warm_v

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)
    
    tmp2 = np.empty(b.shape, dtype=M.dtype)

    Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v

        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)

        if (np.any(KtransposeU == 0) or
                np.any(np.isnan(u)) or np.any(np.isnan(v)) or
                np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        if cpt % 5 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
            tmp2 = np.sum(u[:, None] * K * v[None, :], 0)
            #tmp2=np.einsum('i,ij,j->j', u, K, v)
            err = np.linalg.norm(tmp2 - b)**2  # violation of marginal
            #if log:
            #    log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
    
    if log:
        log['u'] = u
        log['v'] = v
        log['nb_iterations'] = cpt

    res = u.reshape((-1, 1)) * K * v.reshape((1, -1))
    if to_numpy:
        res = cp.asnumpy(res)
    if log:
        return res, log
    else:
        return res