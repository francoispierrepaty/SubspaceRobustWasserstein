#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This code is adapted from the package POT (https://pot.readthedocs.io/en/stable/_modules/ot/bregman.html)

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np

def sinkhorn_knopp(a, b, M, reg, numItermax=1000, stopThr=1e-9, warm_u=None, warm_v=None, verbose=False, log=False):
    """Solve the entropic regularization optimal transport problem and return the OT matrix."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

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

    # print(np.min(K))
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
            np.einsum('i,ij,j->j', u, K, v, out=tmp2)
            err = np.linalg.norm(tmp2 - b)**2  # violation of marginal
            if log:
                log['err'].append(err)

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

    if log:
        return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
    else:
        return u.reshape((-1, 1)) * K * v.reshape((1, -1))