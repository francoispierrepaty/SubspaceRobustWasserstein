# -*- coding: utf-8 -*-

####################################################################
# Experiment for Figure 12 in https://arxiv.org/pdf/1901.08949.pdf #
####################################################################

import numpy as np
import matplotlib.pyplot as plt
import time
import cupy as cp

from SRW import SubspaceRobustWasserstein
from Optimization.frankwolfe import FrankWolfe
from sinkhornGPU import sinkhorn_knopp_gpu


def T(x,d,dim=2):
    assert dim <= d
    assert dim >= 1
    assert dim == int(dim)
    return x + 2*cp.sign(x)*cp.array(dim*[1]+(d-dim)*[0])

def fragmented_hypercube(n,d,dim):
    assert dim <= d
    assert dim >= 1
    assert dim == int(dim)
    
    a = (1./n) * cp.ones(n)
    b = (1./n) * cp.ones(n)

    # First measure : uniform on the hypercube
    X = cp.random.uniform(-1, 1, size=(n,d))

    # Second measure : fragmentation
    Y = T(cp.random.uniform(-1, 1, size=(n,d)), d, dim)
    
    return a,b,X,Y



ds = [10, 25, 50, 100, 250, 500, 1000] # Dimensions for which to compute the SRW computation time
nb_ds = len(ds)
n = 100 # Number of points in the measures
k = 2 # Dimension parameter
reg = 0.2 # Entropic regularization strengh
max_iter = 1000 # Maximum number of iterations
max_iter_sinkhorn = 30 # Maximum number of iterations in Sinkhorn
threshold = 0.05 # Stopping threshold
threshold_sinkhorn = 1e-3 # Stopping threshold in Sinkhorn
nb_exp = 100 # Number of experiments


times_SRW = cp.zeros((nb_exp, nb_ds))
times_W = cp.zeros((nb_exp, nb_ds))


tic = time.time()
tac = time.time()
for t in range(nb_exp):
    print(t)
    
    for ind_d in range(nb_ds):
        d = ds[ind_d]
        print(d)
        
        a,b,X,Y = fragmented_hypercube(n,d,dim=2)
        
        if d>=250:
            reg=0.5
        if d>=1000:
            reg=1.
        
        print('SRW')
        algo = FrankWolfe(reg=reg, step_size_0=None, max_iter=max_iter, max_iter_sinkhorn=max_iter_sinkhorn, threshold=threshold, threshold_sinkhorn=threshold_sinkhorn, use_gpu=True)
        SRW = SubspaceRobustWasserstein(X, Y, a, b, algo, k)
        tic = time.time()
        SRW.run()
        tac = time.time()
        times_SRW[t,ind_d] = tac-tic
        print(SRW.get_value(), len(SRW.get_minmax_values()), cp.abs(min(SRW.get_minmax_values())-SRW.get_value())/SRW.get_value())
        
        print('W')
        tic = time.time()
        ones = cp.ones((n,n))
        C = cp.diag(cp.diag(X.dot(X.T))).dot(ones) + ones.dot(cp.diag(cp.diag(Y.dot(Y.T)))) - 2*X.dot(Y.T)
        OT_plan = sinkhorn_knopp_gpu(a, b, C, reg, numItermax=max_iter_sinkhorn, stopThr=threshold_sinkhorn)
        tac = time.time()
        times_W[t,ind_d] = tac-tic

times_SRW = cp.asnumpy(times_SRW)
times_W = cp.asnumpy(times_W)

times_SRW_mean = np.mean(times_SRW, axis=0)
times_SRW_min = np.min(times_SRW, axis=0)
times_SRW_10 = np.percentile(times_SRW, 10, axis=0)
times_SRW_25 = np.percentile(times_SRW, 25, axis=0)
times_SRW_75 = np.percentile(times_SRW, 75, axis=0)
times_SRW_90 = np.percentile(times_SRW, 90, axis=0)
times_SRW_max = np.max(times_SRW, axis=0)

times_W_mean = np.mean(times_W, axis=0)
times_W_min = np.min(times_W, axis=0)
times_W_10 = np.percentile(times_W, 10, axis=0)
times_W_25 = np.percentile(times_W, 25, axis=0)
times_W_75 = np.percentile(times_W, 75, axis=0)
times_W_90 = np.percentile(times_W, 90, axis=0)
times_W_max = np.max(times_W, axis=0)


import matplotlib.ticker as ticker
plt.figure(figsize=(17,6))

mean, = plt.loglog(ds[1:], times_W_mean[1:], 'o-', lw=8, ms=20, label='Wasserstein')
col = mean.get_color()
plt.fill_between(ds[1:], times_W_25[1:], times_W_75[1:], facecolor=col, alpha=0.3)
plt.fill_between(ds[1:], times_W_10[1:], times_W_90[1:], facecolor=col, alpha=0.2)
plt.fill_between(ds[1:], times_W_min[1:], times_W_max[1:], facecolor=col, alpha=0.15)

mean, = plt.loglog(ds[1:], times_SRW_mean[1:], 'o-', lw=8, ms=20, label='Subspace Robust Wasserstein')
col = mean.get_color()
plt.fill_between(ds[1:], times_SRW_25[1:], times_SRW_75[1:], facecolor=col, alpha=0.3)
plt.fill_between(ds[1:], times_SRW_10[1:], times_SRW_90[1:], facecolor=col, alpha=0.2)
plt.fill_between(ds[1:], times_SRW_min[1:], times_SRW_max[1:], facecolor=col, alpha=0.15)
  
    
plt.xlabel('Dimension', fontsize=25)
plt.ylabel('Execution time in seconds', fontsize=25)
plt.xticks(ds[1:], fontsize=20)
plt.yticks(fontsize=20)
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
plt.grid(ls=':')
plt.legend(loc='best', fontsize=25)
plt.show()