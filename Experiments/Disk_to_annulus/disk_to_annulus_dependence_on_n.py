#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################
# Experiment for Figures 7 and 8 in https://arxiv.org/pdf/1901.08949.pdf #
##########################################################################

import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

from SRW import SubspaceRobustWasserstein
from Optimization.frankwolfe import FrankWolfe



REAL_VALUE = (14./5) + (8./(5*np.sqrt(5)))*np.log((3+np.sqrt(5))/2)

def disk_annulus(n,d,dim=2):

    a = (1./n) * np.ones(n)
    b = (1./n) * np.ones(n)

    # First measure : uniform on the unit disk
    angles = np.random.randn(n,dim)
    angles = (angles.T/np.linalg.norm(angles, axis=1)).T
    radii = np.random.uniform(0,1,size=n)**(1./dim)
    X = np.diag(radii).dot(angles)
    X = np.append(X, np.random.uniform(size=(n,d-dim)), axis=1)

    # Second measure : uniform on the annulus 2≤r≤3
    angles = np.random.randn(n,dim)
    angles = (angles.T/np.linalg.norm(angles, axis=1)).T
    
    radii = ((3**dim - 2**dim)*np.random.uniform(0, 1, size=n) + 2**dim)**(1./dim)
    Y = np.diag(radii).dot(angles)
    Y = np.append(Y, np.random.uniform(size=(n,d-dim)), axis=1)
    
    return a,b,X,Y


d = 30 # Total dimension
k = 2 # k* = 2 and compute SRW with k = 2
nb_exp = 500 # Do 500 experiments
ns = [25, 50, 100, 250, 500, 1000] # Compute SRW between measures with 'n' points for 'n' in 'ns'

values = np.zeros((len(ns), nb_exp))
values_subspace = np.zeros((len(ns), nb_exp))

proj = cp.zeros((d,d)) # Real optimal subspace
proj[0,0] = 1
proj[1,1] = 1
for indn in range(len(ns)):
    n = ns[indn]
    # Sample nb_exp times
    for t in range(nb_exp):
        FW = FrankWolfe(reg=0.2, step_size_0=None, max_iter=15, threshold=0.01, max_iter_sinkhorn=30, threshold_sinkhorn=10e-04, use_gpu=False)
        a,b,X,Y = disk_annulus(n,d,dim=2)
        SRW_FW = SubspaceRobustWasserstein(X, Y, a, b, FW, k)
        SRW_FW.run()
        values[indn, t] = np.abs(REAL_VALUE-SRW_FW.get_value())
        values_subspace[indn, t] = cp.linalg.norm(SRW_FW.get_Omega() - proj)
    
    print('n =',n,'/', np.mean(values[indn,:]))


values_mean = np.mean(values, axis=1)
values_min = np.min(values, axis=1)
values_10 = np.percentile(values, 10, axis=1)
values_25 = np.percentile(values, 25, axis=1)
values_75 = np.percentile(values, 75, axis=1)
values_90 = np.percentile(values, 90, axis=1)
values_max = np.max(values, axis=1)


import matplotlib.ticker as ticker
plt.figure(figsize=(17,6))
mean, = plt.loglog(ns, values_mean, 'o-', lw=4, ms=11)
col = mean.get_color()
plt.fill_between(ns, values_25, values_75, facecolor=col, alpha=0.3)
plt.fill_between(ns, values_10, values_90, facecolor=col, alpha=0.2)
    
plt.xlabel('Number of points', fontsize=25)
plt.ylabel('$|W^2(\mu,\\nu) - S^2(\hat\mu, \hat\\nu)|$', fontsize=25)
plt.xticks(ns, fontsize=20)
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1], fontsize=20)
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
plt.grid(ls=':')
plt.show()



values_subspace_mean = np.mean(values_subspace, axis=1)
values_subspace_min = np.min(values_subspace, axis=1)
values_subspace_10 = np.percentile(values_subspace, 10, axis=1)
values_subspace_25 = np.percentile(values_subspace, 25, axis=1)
values_subspace_75 = np.percentile(values_subspace, 75, axis=1)
values_subspace_90 = np.percentile(values_subspace, 90, axis=1)
values_subspace_max = np.max(values_subspace, axis=1)

plt.figure(figsize=(17,6))
mean, = plt.loglog(ns, values_subspace_mean, 'o-', lw=4, ms=11)
col = mean.get_color()
plt.fill_between(ns, values_subspace_25, values_subspace_75, facecolor=col, alpha=0.3)
plt.fill_between(ns, values_subspace_10, values_subspace_90, facecolor=col, alpha=0.2)
plt.fill_between(ns, values_subspace_min, values_subspace_max, facecolor=col, alpha=0.15)
    
plt.xlabel('Number of points', fontsize=25)
plt.ylabel('$||\Omega^* - \widehat\Omega||_F$', fontsize=25)
plt.xticks(ns, fontsize=20)
plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], fontsize=20)
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
plt.grid(ls=':')
plt.show()