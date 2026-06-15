class GaussianMixtureODE(VI):
    '''
    Implements Gaussian mixture variational inference using ODE formulation
    for mean and Cholesky factor R of the covariance (Sigma = R @ R.T).
    Hessian-free: only requires gradients of log-likelihood and log-prior.
    Cubature rule approximates expectations.
    '''
    def __init__(self,
                 log_likelihood,
                 grad_log_likelihood,
                 log_prior,
                 grad_log_prior,
                 step_size=0.1,
                 n_iter=50,
                 use_rk4=False,
                 use_adam=False,
                 num_samples=1):
        super().__init__(log_likelihood, grad_log_likelihood, log_prior, grad_log_prior, step_size)
        self.n_iter = n_iter
        self.use_rk4 = use_rk4
        self.use_adam = use_adam
        self.ns = num_samples
        if num_samples==0:
            self.f_increment = self._compute_increments_cub
        else:
            self.f_increment= self._compute_increments

    def _mixture_grad(self, x, means, Rs):
        """
        Compute ∇ₓ log [ (1/K) ∑_i N(x | m_i, Σ_i) ]
        where Σ_i = R_i R_i^T are 2×2 covariances in Cholesky form,
        and all mixture weights are equal (1/K).
        - means is the list with all the individual means
        - Rs is the list with all the individual square root covariance matrices
        """
        #K = len(means)
        logps = []
        grads = []

        for m, R in zip(means, Rs):
            diff = x - m
            # R is lower‐triangular such that Σ = R R^T
            inv_R = solve_triangular(R, np.eye(2), lower=True)
            inv_cov = inv_R.T @ inv_R

            # exponent and log‐normalizer for 2D Gaussian
            exponent = -0.5 * diff @ inv_cov @ diff
            log_norm = - np.log(2 * np.pi) - np.sum(np.log(np.diag(R)))

            # per‐component log‐density (up to the constant log(1/K), which we drop)
            logps.append(log_norm + exponent)
            grads.append(-inv_cov @ diff)

        logps = np.array(logps)           # shape (K,)
        grads = np.array(grads)           # shape (K, 2)

        # softmax over the log‐densities → responsibilities π_i(x)
        ws = np.exp(logps - np.max(logps))
        ws /= ws.sum()

        # return ∑_i π_i(x) * ∇ log N_i(x)
        return np.tensordot(ws, grads, axes=(0, 0))  # shape (2,)
    
    def _compute_increments_cub(self, means, Rs):
        '''Compute ODE increments for all the means and Rs via cubature.'''
        dms, dRs = [], []
        dim = 2
        for _, (m, R) in enumerate(zip(means, Rs)):
            c, alpha = np.sqrt(dim), 1/(2*dim)
            R_inv = solve_triangular(R, np.eye(dim), lower=True)
            dm_t = np.zeros(dim) 
            dm_m = np.zeros(dim)
            M_exp = np.zeros((dim, dim))
            for k in range(dim):
                e = np.zeros(dim)
                e[k] = 1
                delta = c * (R @ e)
                for sign in [+1, -1]:
                    x = m + sign*delta
                    gt = self.grad_log_likelihood(x) + self.grad_log_prior(x)
                    gm = self._mixture_grad(x, means, Rs)
                    dm_t += alpha * gt
                    dm_m += alpha * gm
                    diff = gt - gm
                    M_exp += alpha * (np.outer(sign*delta, diff) + np.outer(diff, sign*delta))
            dms.append(dm_t - dm_m)
            dR = R @ np.tril(R_inv @ M_exp @ R_inv.T)
            dRs.append(dR)
        return dms, dRs

    def _compute_increments(self, means, Rs):
        '''Compute ODE increments for mu and R via MC.'''
        dms, dRs = [], []
        dim = 2
        for _, (mu, R) in enumerate(zip(means, Rs)): 
            grad_mu = np.zeros(dim)
            grad_cov = np.zeros((dim, dim))
            Z = np.random.randn(self.ns, dim)
            for z in Z:
                x = mu + R @ z
                gt = self.grad_log_likelihood(x) + self.grad_log_prior(x)
                gm = self._mixture_grad(x, means, Rs)
                diff = gt - gm
                grad_mu += diff
                grad_cov += np.outer(R @ z, diff) + np.outer(diff, R @ z)

            grad_mu /= self.ns
            grad_cov /= self.ns
            dms.append(grad_mu)
            # R gradient
            R_inv = solve_triangular(R, np.eye(dim), lower=True)
            grad_R = R @ np.tril(R_inv @ grad_cov @ R_inv.T)
            dRs.append(grad_R)
        return dms, dRs
    
    def run(self, mus0, Rs0, cub=False):
        '''Integrate ODE for mu and R over n_iter steps using RK4, Adam, or Euler.'''
        # Initialize means and Cholesky factors
        mus = [m.copy() for m in mus0]
        Rs = [R.copy() for R in Rs0]
        history = []

        # Adam state for each mixture component
        K = len(mus)
        m_mu = [np.zeros_like(m) for m in mus]
        v_mu = [np.zeros_like(m) for m in mus]
        m_R = [np.zeros_like(R) for R in Rs]
        v_R = [np.zeros_like(R) for R in Rs]
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        def add(ms, Rs, kms, kRs, coeff):
            return [m + coeff * dm for m, dm in zip(ms, kms)], \
                [R + coeff * dR for R, dR in zip(Rs, kRs)]

        for n in range(1, self.n_iter + 1):
            history.append(( [m.copy() for m in mus], [R.copy() @ R.T.copy() for R in Rs] ))

            if self.use_rk4:
                # RK4 integration
                k1_m, k1_R = self.f_increment(mus, Rs)
                m2, R2 = add(mus, Rs, k1_m, k1_R, 0.5 * self.step_size)
                k2_m, k2_R = self.f_increment(m2, R2)
                m3, R3 = add(mus, Rs, k2_m, k2_R, 0.5 * self.step_size)
                k3_m, k3_R = self.f_increment(m3, R3)
                m4, R4 = add(mus, Rs, k3_m, k3_R, self.step_size)
                k4_m, k4_R = self.f_increment(m4, R4)
                for i in range(K):
                    mus[i] += (self.step_size/6) * (k1_m[i] + 2*k2_m[i] + 2*k3_m[i] + k4_m[i])
                    Rs[i]  += (self.step_size/6) * (k1_R[i] + 2*k2_R[i] + 2*k3_R[i] + k4_R[i])

            elif self.use_adam:
                # Compute gradients
                grads_mu, grads_R = self.f_increment(mus, Rs)
                # Adam update per component
                for i in range(K):
                    # Update biased first moment
                    m_mu[i] = beta1 * m_mu[i] + (1 - beta1) * grads_mu[i]
                    m_R[i]  = beta1 * m_R[i]  + (1 - beta1) * grads_R[i]
                    # Update biased second moment
                    v_mu[i] = beta2 * v_mu[i] + (1 - beta2) * (grads_mu[i] ** 2)
                    v_R[i]  = beta2 * v_R[i]  + (1 - beta2) * (grads_R[i] ** 2)
                    # Bias correction
                    mhat_mu = m_mu[i] / (1 - beta1**n)
                    mhat_R  = m_R[i]  / (1 - beta1**n)
                    vhat_mu = v_mu[i] / (1 - beta2**n)
                    vhat_R  = v_R[i]  / (1 - beta2**n)
                    # Parameter update
                    mus[i] += self.step_size * mhat_mu / (np.sqrt(vhat_mu) + eps)
                    Rs[i]  += self.step_size * mhat_R  / (np.sqrt(vhat_R)  + eps)

            else:
                # explicit Euler
                dmus, dRs = self.f_increment(mus, Rs)
                for i in range(K):
                    mus[i] += self.step_size * dmus[i]
                    Rs[i]  += self.step_size * dRs[i]

        return history

    def _sample(self, x0s, Sigma0s):
        '''Initialize R from Sigma0's Cholesky and run inference.'''
        #dim = 2
        #x0s = [np.array([4000., 4000.]), np.array([5000., 2000.]), np.array([2300., 5000.]), np.array([2500., 4000.])]
        #Sigma0s = [100.*np.eye(dim), 100.*np.eye(dim), 100.*np.eye(dim), 100.*np.eye(dim)]
        return self.run(x0s, Sigma0s)