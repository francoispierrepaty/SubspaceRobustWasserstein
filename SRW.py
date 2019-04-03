# -*- coding: utf-8 -*-#

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

try:
    import cupy as cp
except:
    pass

class SubspaceRobustWasserstein:
    
    def __init__(self, X, Y, a, b, algo, k):
        """
        X    : (number_points_1, dimension) matrix of atoms for the first measure
        Y    : (number_points_2, dimension) matrix of atoms for the second measure
        a    : (number_points_1,) vector of weights for the first measure
        b    : (number_points_2,) vector of weights for the second measure
        algo : algorithm to compute the SRW distance (instance of class 'ProjectedGradientAscent' or 'FrankWolfe')
        k    : dimension parameter (can be of type 'int', 'list' or 'set' in order to compute SRW for several paremeters 'k').
        """
        
        # Check shapes
        d = X.shape[1]
        n = X.shape[0]
        m = Y.shape[0]
        assert d == Y.shape[1]
        assert n == a.shape[0]
        assert m == b.shape[0]
        
        if isinstance(k, int):
            assert k <= d
            assert k == int(k)
            assert 1 <= k
        elif isinstance(k, list) or isinstance(k, set):
            assert len(k) > 0
            k = list(set(k))
            k.sort(reverse=True)
            assert k[0] <= d
            assert k[-1] >= 1
            for l in k:
                assert l == int(l)
        else:
            raise TypeError("Parameter 'k' should be of type 'int' or 'list' or 'set'.")
        
        # Measures
        if algo.use_gpu:
            self.X = cp.asarray(X)
            self.Y = cp.asarray(Y)
            self.a = cp.asarray(a)
            self.b = cp.asarray(b)
        else:
            self.X = X
            self.Y = Y
            self.a = a
            self.b = b
        self.d = d
        
        # Algorithm
        self.algo = algo
        self.k = k
        if self.algo.use_gpu:
            self.Omega = cp.identity(self.d)
        else:
            self.Omega = np.identity(self.d)
        self.pi = None
        self.maxmin_values = []
        self.minmax_values = []

    def run(self):
        """Run algorithm algo on the data."""
        if isinstance(self.k, int):
            self.Omega = self.algo.initialize(self.a, self.b, self.X, self.Y, self.Omega, self.k)
            self.Omega, self.pi, self.maxmin_values, self.minmax_values = self.algo.run(self.a, self.b, self.X, self.Y, self.Omega, self.k)
        elif isinstance(self.k, list):
            #TODO: clean this up
            Omega_0 = self.algo.initialize(self.a, self.b, self.X, self.Y, self.Omega, self.k[0])
            self.Omega, self.pi, self.maxmin_values, self.minmax_values = {}, {}, {}, {}
            for l in self.k:
                if l != self.k[0]:
                    Omega_0 = eigenvectors[:,-l:].dot(eigenvectors[:,-l:].T)
                Omega, pi, maxmin_values, minmax_values = self.algo.run(self.a, self.b, self.X, self.Y, Omega_0, l)
                V = self.algo.Vpi(self.X, self.Y, self.a, self.b, pi)
                if self.algo.use_gpu:
                    _, eigenvectors = cp.linalg.eigh(V)
                else:
                    _, eigenvectors = np.linalg.eigh(V)
                self.Omega[l] = Omega
                self.pi[l] = pi
                self.maxmin_values[l] = maxmin_values
                self.minmax_values[l] = minmax_values
    
    def get_Omega(self):
        return self.Omega
    
    def get_pi(self):
        return self.pi
    
    def get_maxmin_values(self):
        """Get the values of the maxmin problem along the iterations."""
        return self.maxmin_values
    
    def get_minmax_values(self):
        """Get the values of the minmax problem along the iterations."""
        return self.minmax_values
    
    def get_value(self):
        """Return the SRW distance."""
        if isinstance(self.k, int):
            return np.max(self.maxmin_values)
        else:
            return {key: np.max(val) for key, val in self.maxmin_values.items()}
    
    def plot_values(self, real_value=None):
        """Plot values if computed for several dimension parameters 'k'."""
        assert not isinstance(self.k, int)
        values = self.get_value()
        plt.plot(values.keys(), values.values(), lw=4)
        if real_value is not None:
            plt.plot(values.keys(), len(values)*[real_value])
        plt.grid(ls=':')
        plt.xticks(np.sort(list(values.keys())))
        plt.xlabel('Dimension parameter $k$', fontsize=25)
        plt.show()
    
    def get_projected_pushforwards(self, l=None):
        """Return the pushforwards."""
        isnotdict = False # True iff self.Omega and other attributes are NOT dictionnaries
        if isinstance(self.k, int):
            isnotdict = True
        if isinstance(self.k, int) and l is None:
            l = self.k
        elif isinstance(self.k, int) and l != self.k:
            raise ValueError("Argument 'l' should match class attribute 'k'.")
        elif l is None and isinstance(self.k, list):
            raise ValueError("When class attribute 'k' is a list, argument 'l' should be specified.")
        elif isinstance(self.k, list) and l not in self.k:
            raise ValueError("When class attribute 'k' is a list, argument 'l' should be in the list 'k'.")
        
        if isnotdict:
            d = self.Omega.shape[0]
            if self.algo.use_gpu:
                eigenvalues, eigenvectors = cp.linalg.eigh(self.Omega)
                eigenvalues = eigenvalues[-self.k:]
                eigenvectors = eigenvectors[:,-self.k:]
            else:
                eigenvalues, eigenvectors = sp.linalg.eigh(self.Omega, eigvals=(d-self.k,d-1))
        else:
            d = self.Omega[l].shape[0]
            if self.algo.use_gpu:
                eigenvalues, eigenvectors = cp.linalg.eigh(self.Omega[l])
                eigenvalues = eigenvalues[-self.k:]
                eigenvectors = eigenvectors[:,-self.k:]
            else:
                eigenvalues, eigenvectors = sp.linalg.eigh(self.Omega[l], eigvals=(d-self.k,d-1))
            
        eigenvalues[eigenvalues<0]=0.
        eigenvalues = np.sqrt(eigenvalues)
        if self.algo.use_gpu:
            projector = (cp.diag(eigenvalues).dot(eigenvectors.T)).T
        else:
            projector = (np.diag(eigenvalues).dot(eigenvectors.T)).T
        
        proj_X = self.X.dot(projector)
        proj_Y = self.Y.dot(projector)
        
        return proj_X, proj_Y
    
    def plot_projected_pushforwards(self, l=None, path=None):
        """Plot the pushforwards measures under Omega^(1/2)."""
        if isinstance(self.k, int) and l is None:
            l = self.k
        elif isinstance(self.k, int) and l != self.k:
            raise ValueError("Argument 'l' should match class attribute 'k'.")
        elif l is None and isinstance(self.k, list):
            raise ValueError("When class attribute 'k' is a list, argument 'l' should be specified.")
        elif isinstance(self.k, list) and l not in self.k:
            raise ValueError("When class attribute 'k' is a list, argument 'l' should be in the list 'k'.")
        
        proj_X, proj_Y = self.get_projected_pushforwards(l)
        
        plt.scatter(proj_X[:,0], proj_X[:,1], s=self.X.shape[0]*20*self.a, c='r', zorder=10, alpha=0.7)
        plt.scatter(proj_Y[:,0], proj_Y[:,1], s=self.Y.shape[0]*20*self.b, c='b', zorder=10, alpha=0.7)
        plt.title('Optimal projections', fontsize=25)
        plt.axis('equal')
        if path is not None:
            plt.savefig(path)
        plt.show()
    
    def plot_transport_plan(self, l=None, path=None):
        """Plot the transport plan."""
        isnotdict = False
        if isinstance(self.k, int):
            isnotdict = True
        if isinstance(self.k, int) and l is None:
            l = self.k
        elif isinstance(self.k, int) and l != self.k:
            raise ValueError("Argument 'l' should match class attribute 'k'.")
        elif l is None and isinstance(self.k, list):
            raise ValueError("When class attribute 'k' is a list, argument 'l' should be specified.")
        elif isinstance(self.k, list) and l not in self.k:
            raise ValueError("When class attribute 'k' is a list, argument 'l' should be in the list 'k'.")
        
        for i in range(self.X.shape[0]):
            for j in range(self.Y.shape[0]):
                if isnotdict and self.pi[i,j] > 0.:
                    plt.plot([self.X[i,0], self.Y[j,0]], [self.X[i,1], self.Y[j,1]], c='k', lw=30*self.pi[i,j])
                elif not isnotdict and self.pi[l][i,j] > 0.:
                    plt.plot([self.X[i,0], self.Y[j,0]], [self.X[i,1], self.Y[j,1]], c='k', lw=30*self.pi[l][i,j])
        plt.scatter(self.X[:,0], self.X[:,1], s=self.X.shape[0]*20*self.a, c='r', zorder=10, alpha=0.7)
        plt.scatter(self.Y[:,0], self.Y[:,1], s=self.Y.shape[0]*20*self.b, c='b', zorder=10, alpha=0.7)
        plt.title('Optimal SRW transport plan', fontsize=25)
        plt.axis('equal')
        if path is not None:
            plt.savefig(path)
        plt.show()
    
    def plot_convergence(self, l=None, path=None):
        """Plot the convergence of the optimization problem."""
        isnotdict = False # True iff self.Omega and other attributes are NOT dictionnaries
        if isinstance(self.k, int):
            isnotdict = True
        if isinstance(self.k, int) and l is None:
            l = self.k
        elif isinstance(self.k, int) and l != self.k:
            raise ValueError("Argument 'l' should match class attribute 'k'.")
        elif l is None and isinstance(self.k, list):
            raise ValueError("When class attribute 'k' is a list, argument 'l' should be specified.")
        elif isinstance(self.k, list) and l not in self.k:
            raise ValueError("When class attribute 'k' is a list, argument 'l' should be in the list 'k'.")
        
        
        if isnotdict:
            plt.plot(self.minmax_values, label='Sum of the '+str(l)+' largest eigenvalues of $V_\pi$', lw=4)
            plt.plot(self.maxmin_values, label='Optimal transport between the pushforwards', lw=4)
        else:
            plt.plot(self.minmax_values[l], label='Sum of the '+str(l)+' largest eigenvalues of $V_\pi$', lw=4)
            plt.plot(self.maxmin_values[l], label='Optimal transport between the pushforwards', lw=4)
        plt.xlabel('Number of iterations', fontsize=25)
        plt.legend(fontsize=15)
        if path is not None:
            plt.savefig(path)
        plt.show()