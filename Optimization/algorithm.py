# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import ot
from Optimization.sinkhorn import sinkhorn_knopp
try:
    from sinkhornGPU import sinkhorn_knopp_gpu
    import cupy as cp
except:
    print("GPU not found.")
    pass

class Algorithm:
    def __init__(self, reg, step_size_0, max_iter, threshold, max_iter_sinkhorn, threshold_sinkhorn, use_gpu, verbose=False):
        """
        reg : Entropic regularization strength
        step_size_0 : Initial step size for ProjectedGradientAscent
        max_iter : Maximum number of iterations to be run
        threshold : Stopping threshold (stops when precision 'threshold' is attained or 'max_iter' iterations are run)
        max_iter_sinkhorn : Maximum number of iterations to be run in Sinkhorn algorithm
        threshold_sinlhorn : Stopping threshold for Sinkhorn Algorithm
        use_gpu : 'True' to use GPU, 'False' otherwise
        verbose : 'True' to print additional messages, 'False' otherwise
        """
        
        assert reg >= 0
        if step_size_0 is not None:
            assert step_size_0 > 0
        assert isinstance(max_iter, int)
        assert max_iter > 0
        assert threshold > 0
        assert isinstance(max_iter_sinkhorn, int)
        assert max_iter_sinkhorn > 0
        assert threshold_sinkhorn > 0
        assert isinstance(use_gpu, bool)
        assert isinstance(verbose, bool)
        
        self.reg = reg
        self.step_size_0 = step_size_0
        self.max_iter = max_iter
        self.threshold = threshold
        self.max_iter_sinkhorn = max_iter_sinkhorn
        self.threshold_sinkhorn = threshold_sinkhorn
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.u = None
        self.v = None
        
    def Vpi(self, X, Y, a, b, OT_plan):
        """Return the second order matrix of the displacements: sum_ij { (OT_plan)_ij (X_i-Y_j)(X_i-Y_j)^T }."""
        A = X.T.dot(OT_plan).dot(Y)
        if self.use_gpu:
            return X.T.dot(cp.diag(a)).dot(X) + Y.T.dot(cp.diag(b)).dot(Y) - A - A.T
        else:
            return X.T.dot(np.diag(a)).dot(X) + Y.T.dot(np.diag(b)).dot(Y) - A - A.T
        
    def Mahalanobis(self, X, Y, Omega):
        """Return the matrix of Mahalanobis costs."""
        n = X.shape[0]
        m = Y.shape[0]
        if self.use_gpu:
            ones = cp.ones((n,m))
            return cp.diag(cp.diag(X.dot(Omega).dot(X.T))).dot(ones) + ones.dot(cp.diag(cp.diag(Y.dot(Omega).dot(Y.T)))) - 2*X.dot(Omega).dot(Y.T)
        else:
            ones = np.ones((n,m))
            return np.diag(np.diag(X.dot(Omega).dot(X.T))).dot(ones) + ones.dot(np.diag(np.diag(Y.dot(Omega).dot(Y.T)))) - 2*X.dot(Omega).dot(Y.T)
        
    def OT(self, a, b, ground_cost, warm_u=None, warm_v=None):
        """Return the OT cost and plan."""
        if self.reg==0.: # Solve exact OT
            OT_plan, log = ot.emd(a, b, ground_cost, log=True)
            OT_val = log['cost']
        else: # Run Sinkhorn algorithm
            if self.use_gpu:
                OT_plan, log = sinkhorn_knopp_gpu(a, b, ground_cost, self.reg, numItermax=self.max_iter_sinkhorn, stopThr=self.threshold_sinkhorn, warm_u=self.u, warm_v=self.v, log=True, to_numpy=False)
                self.u = log['u']
                self.v = log['v']
            else:
                OT_plan, log = sinkhorn_knopp(a, b, ground_cost, self.reg, numItermax=self.max_iter_sinkhorn, stopThr=self.threshold_sinkhorn, warm_u=self.u, warm_v=self.v, log=True)
                self.u = log['u']
                self.v = log['v']
                #print(log['nb_iterations'])
            OT_val = np.sum(ground_cost*OT_plan)
        return OT_val, OT_plan
    
    def initialize(self, a, b, X, Y, Omega, k):
        """Initialize Omega with the projection onto the subspace spanned by top-k eigenvectors of V_pi*, where pi* (=OT_plan) is the (classical) optimal transport plan."""
        if self.verbose:
            print('Initializing')
            
        n = X.shape[0]
        m = Y.shape[0]
        
        # Compute the cost matrix
        if self.use_gpu:
            ones = cp.ones((n,m))
            C = cp.diag(cp.diag(X.dot(X.T))).dot(ones) + ones.dot(cp.diag(cp.diag(Y.dot(Y.T)))) - 2*X.dot(Y.T)
        else:
            ones = np.ones((n,m))
            C = np.diag(np.diag(X.dot(X.T))).dot(ones) + ones.dot(np.diag(np.diag(Y.dot(Y.T)))) - 2*X.dot(Y.T)
        
        # Compute the OT plan
        _, OT_plan = self.OT(a, b, C)
        V = self.Vpi(X, Y, a, b, OT_plan)
        
        # Eigendecompose V
        d = V.shape[0]
        if self.use_gpu:
            _, eigenvectors = cp.linalg.eigh(V)
            eigenvectors = eigenvectors[:,-k:]
        else:
            _, eigenvectors = sp.linalg.eigh(V, eigvals=(d-k,d-1))
        
        # Return the projection
        Omega = eigenvectors.dot(eigenvectors.T)
        return Omega
        
    def run(self, a, b, X, Y, Omega, k):
        raise NotImplementedError("Method 'run' is not implemented !")
    
    