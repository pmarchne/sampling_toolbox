import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, gamma, psi  # Added psi for exact k-NN

class RelativeKLTracker:
    """
    Tracks reverse KL divergence D_KL(mu || pi) using pre-cached unnormalized log-target evaluations.
    When log_evidence is omitted (0.0), the output tracks D_KL(mu || pi) - log Z, which serves as a relative convergence diagnostic.
    """
    def __init__(self, log_evidence=0.0):
        self.log_evidence = log_evidence

    def estimate_kl_mgvi(self, cubature_points, cubature_weights, cached_log_p, mu, R, weights):
        """Zero target-evaluation KL tracker for MixGVI."""
        log_q = self._evaluate_gmm_logpdf(cubature_points, mu, R, weights)
        # E_q[log q(x)] - E_q[log p(x)] + log Z
        entropy_term = np.sum(cubature_weights * log_q)
        cross_entropy_term = np.sum(cubature_weights * cached_log_p)
        return entropy_term - cross_entropy_term + self.log_evidence

    def estimate_kl_particles(self, particles, cached_log_p, k=2):
        """Zero target-evaluation KL tracker for particle methods."""
        entropy_term = -self._estimate_entropy_knn(particles, k=k)
        cross_entropy_term = np.mean(cached_log_p)
        return entropy_term - cross_entropy_term + self.log_evidence

    def _estimate_entropy_knn(self, samples, k=2):
        """Kozachenko-Leonenko entropy estimator corrected for arbitrary k."""
        N, dim = samples.shape
        if N <= k:
            return 0.0
        tree = cKDTree(samples)
        distances, _ = tree.query(samples, k=k)
        rho = np.maximum(distances[:, k-1], 1e-12)
        c_d = (np.pi ** (dim / 2.0)) / gamma(dim / 2.0 + 1)
        # Generalized exact equation using Digamma functions
        H = np.mean(dim * np.log(rho)) + np.log(c_d) + psi(N) - psi(k)
        return H

    def _evaluate_gmm_logpdf(self, samples, mu, R, weights):
        """Evaluates log q(x) safely avoiding zero weight underflow."""
        K = len(weights)
        N, dim = samples.shape
        log_probs = np.zeros((N, K))
        for k in range(K):
            diff = samples - mu[k]
            # Solve L z = diff^T for lower-triangular factor
            z = solve_triangular(R[k], diff.T, lower=True).T
            log_det = np.sum(np.log(np.diagonal(R[k])))
            # Use clipping to prevent crashes
            safe_log_w = np.log(np.maximum(weights[k], 1e-15))
            log_probs[:, k] = (safe_log_w 
                               - 0.5 * dim * np.log(2 * np.pi) 
                               - log_det 
                               - 0.5 * np.sum(z**2, axis=1))
        return logsumexp(log_probs, axis=1)