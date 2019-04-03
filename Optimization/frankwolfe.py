# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

try:
    import cupy as cp
except:
    pass

from .algorithm import Algorithm


# Algorithm 2 in https://arxiv.org/pdf/1901.08949.pdf
class FrankWolfe(Algorithm):
    
    def __init__(self, reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu, verbose=False):
        assert reg > 0
        step_size_0 = None
        super().__init__(reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu, verbose)
        
    def run(self, a, b, X, Y, Omega, k):
        """Run the Frank-Wolfe iterations."""   
        maxmin_values = []
        minmax_values = []
        gap = self.threshold
        for t in range(self.max_iter):
            
            if gap < self.threshold:
                if self.verbose:
                    print('Precision', gap, 'attained.')
                break
            
            if self.verbose:
                print('Iteration', t)
            
            # Optimal transport computation (Sinkhorn)
            C = self.Mahalanobis(X, Y, Omega)
            OT_val, OT_plan = self.OT(a, b, C)
            pi = OT_plan
            maxmin_values.append(OT_val)
            
            # Second-order moment of the displacements
            V = self.Vpi(X, Y, a, b, OT_plan)
            
            # Minimization of Linearized objective
            d = V.shape[0]
            if self.use_gpu:
                eigenvalues, eigenvectors = cp.linalg.eigh(V)
                eigenvalues = eigenvalues[-k:]
                eigenvectors = eigenvectors[:,-k:]
            else:
                eigenvalues, eigenvectors = sp.linalg.eigh(V, eigvals=(d-k,d-1))
            
            Omega_hat = eigenvectors.dot(eigenvectors.T)
            
            # Frank-Wolfe step
            step_size = 1/(t+2)
            Omega = (1-step_size)*Omega + step_size*Omega_hat
            
            # Duality values
            sum_eigenvalues = np.sum(eigenvalues)
            max_maxmin_values = max(maxmin_values)
            gap = np.abs(sum_eigenvalues - max_maxmin_values)/max_maxmin_values
            minmax_values.append(sum_eigenvalues)
        
        return Omega, pi, maxmin_values, minmax_values


