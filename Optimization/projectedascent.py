# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

try:
    import cupy as cp
except:
    pass

from .algorithm import Algorithm

import sys
sys.path.insert(0, "../")
from utils import projection_Omega


# Algorithm 1 in https://arxiv.org/pdf/1901.08949.pdf
class ProjectedGradientAscent(Algorithm):
    
    def run(self, a, b, X, Y, Omega, k):
        """Run the ascent iterations."""
        maxmin_values = []
        minmax_values = []
        gap = self.threshold
        step_size = self.step_size_0 # Fixed stepsize for Gradient Ascent
        for t in range(self.max_iter):
            
            if gap < self.threshold:
                if self.verbose:
                    print('Precision', gap, 'attained.')
                break
            
            if self.verbose and t%10 == 0:
                print('Iteration', t)
            
            # Optimal transport computation
            C = self.Mahalanobis(X, Y, Omega)
            OT_val, OT_plan = self.OT(a, b, C)
            pi = OT_plan
            maxmin_values.append(OT_val)
            
            # Gradient step
            V = self.Vpi(X, Y, a, b, OT_plan)
            if self.reg == 0: # Supergradient Method needs diminishing stepsizes
                step_size = self.step_size_0/np.sqrt(t+1)
            Omega = projection_Omega(Omega + step_size*V, k)
            
            # Duality values
            d = V.shape[0]
            if self.use_gpu:
                eigenvalues = cp.linalg.eigvalsh(V)
                eigenvalues = eigenvalues[-k:]
            else:
                eigenvalues = sp.linalg.eigh(V, eigvals=(d-k,d-1), eigvals_only=True)
            
            sum_eigenvalues = np.sum(eigenvalues)
            max_maxmin_values = max(maxmin_values)
            gap = np.abs(sum_eigenvalues - max_maxmin_values)/max_maxmin_values
            minmax_values.append(sum_eigenvalues)
        
        return Omega, pi, maxmin_values, minmax_values