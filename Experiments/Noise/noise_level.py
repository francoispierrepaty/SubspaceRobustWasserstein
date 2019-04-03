#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################################################
# Experiment for Figure 11 in https://arxiv.org/pdf/1901.08949.pdf #
####################################################################

import numpy as np
import matplotlib.pyplot as plt

from SRW import SubspaceRobustWasserstein
from Optimization.projectedascent import ProjectedGradientAscent


d = 20 # Total dimension
n = 100 # Number of points in each measure
k = 5 # Dimension of the Wishart (i.e. of support of the measures)
nb_exp = 100 # Number of experiments to run
reg = 0. # No regularization
max_iter = 1000 # Maximum number of iterations (the bigger the more precise)
thr = 1e-5 # Stopping threshold (not attained here since we are in unregularized SRW)

a = (1./n) * np.ones(n)
b = (1./n) * np.ones(n)

mean_1 = np.zeros(d)
mean_2 = np.zeros(d)

# Noise levels to test
ind = [0., 0.01, 0.1, 1, 2, 4, 7, 10]

SRW = np.zeros((nb_exp, len(ind)))
W = np.zeros((nb_exp, len(ind)))


for t in range(nb_exp):
    print(t)
    cov_1 = np.random.randn(d,k)
    cov_1 = cov_1.dot(cov_1.T)
    cov_2 = np.random.randn(d,k)
    cov_2 = cov_2.dot(cov_2.T)
    
    # Draw the measures
    X = np.random.multivariate_normal(mean_1, cov_1, size=n)
    Y = np.random.multivariate_normal(mean_2, cov_2, size=n)

    lst_rsw = []
    lst_w = []
    for epsilon in ind:
        # Add noise of level epsilon
        noiseX = np.random.randn(n,d)
        noiseY = np.random.randn(n,d)
        Xe = X + epsilon*noiseX
        Ye = Y + epsilon*noiseY
        
        # Choice of step size
        ones = np.ones((n,n))
        C = np.diag(np.diag(Xe.dot(Xe.T))).dot(ones) + ones.dot(np.diag(np.diag(Ye.dot(Ye.T)))) - 2*Xe.dot(Ye.T)
        step_size_0 = 1./np.max(C)
        
        # Compute SRW
        algo = ProjectedGradientAscent(reg=reg, step_size_0=step_size_0, max_iter=max_iter, max_iter_sinkhorn=50, threshold=thr, threshold_sinkhorn=1e-04, use_gpu=False)
        SRW_ = SubspaceRobustWasserstein(Xe, Ye, a, b, algo, k=k)
        SRW_.run()
        
        # Compute Wasserstein
        algo = ProjectedGradientAscent(reg=reg, step_size_0=step_size_0, max_iter=1, max_iter_sinkhorn=50, threshold=0.05, threshold_sinkhorn=1e-04, use_gpu=False)
        W_ = SubspaceRobustWasserstein(Xe, Ye, a, b, algo, k=d)
        W_.run()
        
        lst_rsw.append(SRW_.get_value())
        lst_w.append(W_.get_value())
    
    SRW[t,:] = np.array(lst_rsw)
    W[t,:] = np.array(lst_w)

# Relative change
SRW_percent = np.abs(SRW-np.array([SRW[:,0],]*len(ind)).transpose())/np.array([SRW[:,0],]*len(ind)).transpose()
W_percent = np.abs(W-np.array([W[:,0],]*len(ind)).transpose())/np.array([W[:,0],]*len(ind)).transpose()

SRW_percent = SRW_percent[:,1:]
W_percent = W_percent[:,1:]

SRW_mean = np.mean(SRW_percent, axis=0)
SRW_min = np.min(SRW_percent, axis=0)
SRW_10 = np.percentile(SRW_percent, 10, axis=0)
SRW_25 = np.percentile(SRW_percent, 25, axis=0)
SRW_75 = np.percentile(SRW_percent, 75, axis=0)
SRW_90 = np.percentile(SRW_percent, 90, axis=0)
SRW_max = np.max(SRW_percent, axis=0)

W_mean = np.mean(W_percent, axis=0)
W_min = np.min(W_percent, axis=0)
W_10 = np.percentile(W_percent, 10, axis=0)
W_25 = np.percentile(W_percent, 25, axis=0)
W_75 = np.percentile(W_percent, 75, axis=0)
W_90 = np.percentile(W_percent, 90, axis=0)
W_max = np.max(W_percent, axis=0)


# PLOT
import matplotlib.ticker as ticker
plt.figure(figsize=(17,6))

plotW, = plt.loglog(ind[1:], W_mean, 'o-', label='Wasserstein', lw=8, ms=20)
col_W = plotW.get_color()
plt.fill_between(ind[1:], W_25, W_75, facecolor=col_W, alpha=0.3)
plt.fill_between(ind[1:], W_10, W_90, facecolor=col_W, alpha=0.2)

plotSRW, = plt.loglog(ind[1:], SRW_mean, 'o-', label='Subspace Robust Wasserstein', lw=8, ms=20)
col_SRW = plotSRW.get_color()
plt.fill_between(ind[1:], SRW_25, SRW_75, facecolor=col_SRW, alpha=0.3)
plt.fill_between(ind[1:], SRW_10, SRW_90, facecolor=col_SRW, alpha=0.2)

plt.xlabel('Noise level (log scale)', fontsize=25)
plt.ylabel('Relative error (log scale)', fontsize=25)


plt.yticks(fontsize=20)
plt.xticks(ind[1:], fontsize=20)
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2g'))

plt.legend(loc=2, fontsize=25)
plt.grid(ls=':')
plt.show()