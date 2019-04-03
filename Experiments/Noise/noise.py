# -*- coding: utf-8 -*-

####################################################################
# Experiment for Figure 10 in https://arxiv.org/pdf/1901.08949.pdf #
####################################################################

import numpy as np
import matplotlib.pyplot as plt

from SRW import SubspaceRobustWasserstein
from Optimization.projectedascent import ProjectedGradientAscent

noise_level = 1
d = 20 # Total dimension
n = 100 # Number of points for each measure
l = 5 # Dimension of Wishart
nb_exp = 100 # Number of experiments
k = list(range(1,d+1)) # Compute SRW for all dimension parameter k

# Save the values
no_noise = np.zeros((nb_exp,d))
noise = np.zeros((nb_exp,d))


for t in range(nb_exp): # Fore each experiment
    print(t)
    
    a = (1./n) * np.ones(n)
    b = (1./n) * np.ones(n)
    
    mean_1 = 0.*np.random.randn(d)
    mean_2 = 0.*np.random.randn(d)

    cov_1 = np.random.randn(d,l)
    cov_1 = cov_1.dot(cov_1.T)
    cov_2 = np.random.randn(d,l)
    cov_2 = cov_2.dot(cov_2.T)
    
    # Draw measures
    X = np.random.multivariate_normal(mean_1, cov_1, size=n)
    Y = np.random.multivariate_normal(mean_2, cov_2, size=n)
    
    # Add noise
    Xe = X + noise_level*np.random.randn(n,d)
    Ye = Y + noise_level*np.random.randn(n,d)
    
    # Compute SRW begtween X and Y
    algo = ProjectedGradientAscent(reg=0., step_size_0=0.01, max_iter=30, threshold=0.01, max_iter_sinkhorn=30, threshold_sinkhorn=10e-04, use_gpu=False)
    SRW = SubspaceRobustWasserstein(X, Y, a, b, algo, k)
    SRW.run()
    no_noise[t,:] = np.sort(list(SRW.get_value().values()))
    
    # Compute SRW begtween Xe and Ye
    algo = ProjectedGradientAscent(reg=0., step_size_0=0.01, max_iter=30, threshold=0.01, max_iter_sinkhorn=30, threshold_sinkhorn=10e-04, use_gpu=False)
    SRWe = SubspaceRobustWasserstein(Xe, Ye, a, b, algo, k)
    SRWe.run()
    noise[t,:] = np.sort(list(SRWe.get_value().values()))
    
    no_noise[t,:] /= no_noise[t,(d-1)]
    noise[t,:] /= noise[t,(d-1)]


no_noise_mean = np.mean(no_noise, axis=0)
no_noise_min = np.min(no_noise, axis=0)
no_noise_10 = np.percentile(no_noise, 10, axis=0)
no_noise_25 = np.percentile(no_noise, 25, axis=0)
no_noise_75 = np.percentile(no_noise, 75, axis=0)
no_noise_90 = np.percentile(no_noise, 90, axis=0)
no_noise_max = np.max(no_noise, axis=0)

noise_mean = np.mean(noise, axis=0)
noise_min = np.min(noise, axis=0)
noise_10 = np.percentile(noise, 10, axis=0)
noise_25 = np.percentile(noise, 25, axis=0)
noise_75 = np.percentile(noise, 75, axis=0)
noise_90 = np.percentile(noise, 90, axis=0)
noise_max = np.max(noise, axis=0)


# PLOT
plt.figure(figsize=(17,6))

plotnonoise, = plt.plot(range(d), no_noise_mean, label='Without Noise', lw=8)
col_nonoise = plotnonoise.get_color()
plt.fill_between(range(d), no_noise_25, no_noise_75, facecolor=col_nonoise, alpha=0.3)
plt.fill_between(range(d), no_noise_10, no_noise_90, facecolor=col_nonoise, alpha=0.2)
plt.fill_between(range(d), no_noise_min, no_noise_max, facecolor=col_nonoise, alpha=0.15)

plotnoise, = plt.plot(range(d), noise_mean, label='With Noise', lw=8)
col_noise = plotnoise.get_color()
plt.fill_between(range(d), noise_25, noise_75, facecolor=col_noise, alpha=0.3)
plt.fill_between(range(d), noise_10, noise_90, facecolor=col_noise, alpha=0.2)
plt.fill_between(range(d), noise_min, noise_max, facecolor=col_noise, alpha=0.15)

plt.xlabel('Dimension', fontsize=25)
plt.ylabel('Normalized SRW value', fontsize=25)

plt.yticks(fontsize=20)
plt.xticks(range(d),range(1,d+1), fontsize=20)

plt.legend(loc='best', fontsize=20)
plt.grid(ls=':')
plt.show()