import numpy as np

class NumericalPreconditioner:
    """
    Computes a regularized 2nd-order preconditioning matrix Q(x) and its 
    analytical divergence div_Q(x) using finite differences.
    """
    def __init__(self, grad_log_posterior, eps=1e-5, e_min=1e-4):
        self.grad_log_posterior = grad_log_posterior
        self.eps = eps
        self.e_min = e_min  
        self._current_x = None
        self._cached_Q = None
        self._cached_div_Q = None

    def _compute_Q_at(self, x):
        dim = len(x)
        g_base = self.grad_log_posterior(x)
        H = np.zeros((dim, dim))
        
        for i in range(dim):
            x_perturbed = x.copy()
            x_perturbed[i] += self.eps
            g_perturbed = self.grad_log_posterior(x_perturbed)
            H[:, i] = (g_perturbed - g_base) / self.eps
            
        H = 0.5 * (H + H.T)
        M = -H  
        w, v = np.linalg.eigh(M)
        w_clipped = np.clip(w, a_min=self.e_min, a_max=None)
        M_spd = v @ np.diag(w_clipped) @ v.T
        return np.linalg.inv(M_spd)

    def _update_cache(self, x):
        if self._current_x is not None and np.allclose(x, self._current_x, rtol=1e-8):
            return
            
        dim = len(x)
        Q_base = self._compute_Q_at(x)
        
        dQ_dx = []
        for i in range(dim):
            x_perturbed = x.copy()
            x_perturbed[i] += self.eps
            Q_perturbed = self._compute_Q_at(x_perturbed)
            dQ_dx.append((Q_perturbed - Q_base) / self.eps)
        
        # Generalized N-Dimensional Divergence Calculation
        div_Q = np.zeros(dim)
        for i in range(dim):
            for j in range(dim):
                div_Q[i] += dQ_dx[j][i, j]
                
        self._current_x = x.copy()
        self._cached_Q = Q_base
        self._cached_div_Q = div_Q

    def get_Q(self, x):
        self._update_cache(x)
        return self._cached_Q

    def get_div_Q(self, x):
        self._update_cache(x)
        return self._cached_div_Q