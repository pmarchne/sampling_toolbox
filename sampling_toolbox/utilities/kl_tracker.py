import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.special import gamma

class GenericKLTracker:
    def __init__(self, log_evidence=0.0):
        self.log_evidence = log_evidence

    def estimate_kl_mgvi(self, cubature_points, cubature_weights, cached_log_p, mu, R, weights):
        """
        Zero target-evaluation KL tracker for MGVI.
        Resets tracking directly to the existing cubature points.
        """
        # 1. Evaluate log q(x) at the cubature points (cheap)
        log_q = self._evaluate_gmm_logpdf(cubature_points, mu, R, weights)
        # 2. Compute the weighted components
        entropy_term = np.sum(cubature_weights * log_q)
        cross_entropy_term = np.sum(cubature_weights * cached_log_p)
        
        return entropy_term - cross_entropy_term + self.log_evidence

    def estimate_kl_particles(self, particles, cached_log_p):
        """
        Zero target-evaluation KL tracker for SVGD / Langevin SDE.
        Uses k-NN for entropy and cached target values for cross-entropy.
        """
        # Estimate negative entropy using k-NN on current particle positions
        entropy_term = -self._estimate_entropy_knn(particles)
        # Average log target values
        cross_entropy_term = np.mean(cached_log_p)
        return entropy_term - cross_entropy_term + self.log_evidence

    def _estimate_entropy_knn(self, samples, k=2):
        """Kozachenko-Leonenko entropy estimator for spaces up to ~10-20D."""
        N, dim = samples.shape
        if N <= k:
            return 0.0
        tree = cKDTree(samples)
        distances, _ = tree.query(samples, k=k)
        rho = distances[:, k-1]
        rho = np.maximum(rho, 1e-12) # Prevent log(0)
        c_d = (np.pi ** (dim / 2.0)) / gamma(dim / 2.0 + 1)
        euler_gamma = 0.5772156649
        # Continuous entropy estimate
        H = np.mean(dim * np.log(rho)) + np.log(c_d) + np.log(N - 1) + euler_gamma
        return H


    def _evaluate_gmm_logpdf(self, samples, mu, R, weights):
        """Evaluates log q(x) safely using logsumexp."""
        K = len(weights)
        N, dim = samples.shape
        log_probs = np.zeros((N, K))
        for k in range(K):
            diff = samples - mu[k]
            z = solve_triangular(R[k], diff.T, lower=True).T
            log_det = np.sum(np.log(np.diagonal(R[k])))
            log_probs[:, k] = (np.log(weights[k]) 
                               - 0.5 * dim * np.log(2 * np.pi) 
                               - log_det 
                               - 0.5 * np.sum(z**2, axis=1))
        return logsumexp(log_probs, axis=1)