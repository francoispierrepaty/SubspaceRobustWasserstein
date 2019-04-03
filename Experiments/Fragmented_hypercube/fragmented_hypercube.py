# -*- coding: utf-8 -*-

###################################################################
# Experiment for Figure 2 in https://arxiv.org/pdf/1901.08949.pdf #
###################################################################

import numpy as np
import matplotlib.pyplot as plt

from SRW import SubspaceRobustWasserstein
from Optimization.frankwolfe import FrankWolfe


def T(x,d,dim=2):
    assert dim <= d
    assert dim >= 1
    assert dim == int(dim)
    return x + 2*np.sign(x)*np.array(dim*[1]+(d-dim)*[0])

def fragmented_hypercube(n,d,dim):
    assert dim <= d
    assert dim >= 1
    assert dim == int(dim)
    
    a = (1./n) * np.ones(n)
    b = (1./n) * np.ones(n)

    # First measure : uniform on the hypercube
    X = np.random.uniform(-1, 1, size=(n,d))

    # Second measure : fragmentation
    Y = T(np.random.uniform(-1, 1, size=(n,d)), d, dim)
    
    return a,b,X,Y


n = 100 # Number of points for each measure
d = 30 # Total dimension
k = list(range(1,d+1)) # Compute SRW for all parameters 'k'
nb_exp = 100 # Do 100 experiments
dims = [2, 4, 7, 10] # Plot for 'true' dimension k* = 2, 4, 7, 10

values = np.zeros((4, nb_exp, d))
for dim_index in range(4):
    dim = dims[dim_index]
    for t in range(nb_exp):
        a,b,X,Y = fragmented_hypercube(n,d,dim)
        FW = FrankWolfe(reg=0.2, step_size_0=None, max_iter=15, threshold=0.01, max_iter_sinkhorn=50, threshold_sinkhorn=10e-4, use_gpu=True)
        SRW = SubspaceRobustWasserstein(X, Y, a, b, FW, k)
        SRW.run()
        values[dim_index, t,:] = np.sort(list(SRW.get_value().values()))


values_mean = np.mean(values, axis=1)
values_min = np.min(values, axis=1)
values_10 = np.percentile(values, 10, axis=1)
values_25 = np.percentile(values, 25, axis=1)
values_75 = np.percentile(values, 75, axis=1)
values_90 = np.percentile(values, 90, axis=1)
values_max = np.max(values, axis=1)


plt.figure(figsize=(17,6))
col = []

for dim_index in range(4):
    mean, = plt.plot(range(1,d+1), values_mean[dim_index], label='Dimension $'+str(dims[dim_index])+'$', lw=4)
    col.append(mean.get_color())
    plt.fill_between(range(1,d+1), values_25[dim_index], values_75[dim_index], facecolor=col[-1], alpha=0.3)
    plt.fill_between(range(1,d+1), values_10[dim_index], values_90[dim_index], facecolor=col[-1], alpha=0.2)
    plt.fill_between(range(1,d+1), values_min[dim_index], values_max[dim_index], facecolor=col[-1], alpha=0.15)
    plt.xlabel('Dimension', fontsize=25)
plt.ylabel('SRW value', fontsize=25)
plt.xticks(range(1,d+1), fontsize=20)
ymax = plt.gca().get_ylim()[1]
for dim_index in range(4):
    plt.gca().get_xticklabels()[dims[dim_index]-1].set_color(col[dim_index])
    plt.axvline(x=dims[dim_index], ymax=(values_mean[dim_index,dims[dim_index]]/ymax)-0.025, c=col[dim_index], ls='--')
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(ls=':')
plt.show()