import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sampling_toolbox.base_vi import VI
from sampling_toolbox.utilities.gauss_vi_tools import compute_increments, compute_weight_increments
from sampling_toolbox.utilities.time_integration import adam_step_pos, euler_step_pos, rk4_step_pos, init_states, euler_step_w, adam_step_w, rk4_step_w, heun_adaptive_step_pos, heun_step_w
from sampling_toolbox.utilities.kl_tracker import GenericKLTracker
# 1) unify gaussian expectations : midpoint, cubature or MC
# l-bfgs precond option
# 2) FR weight in log-space + Strang splitting
# 3) Time integration : adam or rk4 or Euler
# test on Rosenbrock
# output : 1D wasserstein distance + KL divergence

class GaussianODE(VI):
    '''
    Implements Gaussian variational inference using ODE formulation
    for mean and Cholesky factor R of the covariance (Sigma = R @ R.T).
    Hessian-free: only requires gradients of log-likelihood and log-prior.
    Cubature rule approximates expectations.
    '''
    def __init__(self,
                 log_likelihood,
                 grad_log_likelihood,
                 log_prior,
                 grad_log_prior,
                 step_size: float = 0.1,
                 n_iter = 50,
                 num_samples = 1,
                 method = 'cubature',
                 time_scheme = 'euler',
                 step_size_w: float = 0.,
                 rng = None):
        super().__init__(log_likelihood, grad_log_likelihood, log_prior, grad_log_prior, step_size, rng=rng)
        self.n_iter = n_iter
        self.time_scheme = time_scheme
        self.method = method
        self.ns = num_samples
        self.step_size_w = step_size_w
        self.ll = 1.0 #0.2 # damping parameter 0.3
        self.kl_track = GenericKLTracker(0.)
        # Trackers for analysis
        self.dt_history = []
        self.ll_history = []
        self.acceptance_history = []
        self.kl_hist = False

    def _get_pos_increments(self, mu, R, logws):
        """
        Wrapper for calculating Wasserstein trajectory
        """
        dm, dR = compute_increments(
            means=mu,
            Rs=R,
            logws=logws,
            grad_log_target=self.grad_log_posterior,
            method=self.method, 
            ns=self.ns,
            ll=self.ll
        )
        return dm, dR
    
    def _get_w_increments(self, mu, R, logws):
        """Wrapper for calculating Fisher-Rao Weight trajectory"""
        dlogws = compute_weight_increments(
            means=mu,
            Rs=R,
            logws=logws,
            log_target=self.log_posterior,
            method=self.method,
            ns=self.ns
        )
        return dlogws # 0.01
    

    def _advance_weights(self, mu, R, logws, iter_idx, m_w, v_w, step_size_w):
        """Advances the weight vectors by dt * dt_factor."""
        if self.time_scheme == 'euler':
            logws = euler_step_w(mu, R, logws, self._get_w_increments, step_size_w)
        elif self.time_scheme == 'heun_adaptive':
            effective_dt = min(step_size_w, 0.01)
            logws = euler_step_w(mu, R, logws, self._get_w_increments, effective_dt)
        elif self.time_scheme == 'rk4':
            logws = rk4_step_w(mu, R, logws, self._get_w_increments, step_size_w)
        elif self.time_scheme == 'adam':
            logws = adam_step_w(mu, R, logws, iter_idx, self._get_w_increments, step_size_w, m_w, v_w)
        else:
            raise ValueError(f"Scheme '{self.time_scheme}' not recognized for weights.")
        return logws
    

    def _advance_pos(self, mu, R, logws, iter, m_mu, v_mu, m_R, v_R):
        """
        Internal wrapper to freeze configurations. 
        Ensures you never re-specify targets or methods in the solver loop.
        """
        dt = self.step_size
        accepted = True
        if self.time_scheme == 'euler':
            m1, R1 = euler_step_pos(mu, R, logws, self._get_pos_increments, dt)
        elif self.time_scheme == 'rk4':
            m1, R1 = rk4_step_pos(mu, R, logws, self._get_pos_increments, dt)
        elif self.time_scheme == 'adam':
            m1, R1 = adam_step_pos(mu, R, logws, iter, self._get_pos_increments, dt, m_mu, v_mu, m_R, v_R)
        elif self.time_scheme == 'heun_adaptive':
            # Default tolerances can be made class attributes: self.rtol, self.atol
            m1, R1, dt, accepted = heun_adaptive_step_pos(
                mu, R, logws, self._get_pos_increments, dt, 
                rtol=getattr(self, 'rtol', 1e-3), 
                atol=getattr(self, 'atol', 1e-6)
            )
            self.step_size = dt # Update the stored step size
            # --- ADAPTIVE LAMBDA STRATEGY ---
            target_ll = 0.2 if accepted else 0.50
            self.ll = 0.9 * self.ll + 0.1 * target_ll
        else:
            raise ValueError("time-stepping scheme not implemented !")
        
        self.dt_history.append(dt)
        self.ll_history.append(self.ll)
        self.acceptance_history.append(accepted)

        return m1, R1, accepted, dt

    def plot_diagnostics(self):
        import matplotlib.pyplot as plt
        
        fig, ax1 = plt.subplots(figsize=(10, 4))
        
        # Plot Step Size (dt) on the left axis
        color = 'tab:blue'
        ax1.set_xlabel('Solver Sub-steps (including rejections)')
        ax1.set_ylabel('Step Size (dt)', color=color)
        ax1.plot(self.dt_history, color=color, alpha=0.8, label='dt')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Plot Lambda Damping on the right axis
        ax2 = ax1.twinx()  
        color = 'tab:orange'
        ax2.set_ylabel('Damping (lambda)', color=color)
        ax2.plot(self.ll_history, color=color, linestyle='--', alpha=0.8, label='lambda')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Mark rejections with red vertical dots
        rejection_indices = [i for i, acc in enumerate(self.acceptance_history) if not acc]
        for idx in rejection_indices:
            ax1.axvline(x=idx, color='red', alpha=0.2, linestyle=':')
            
        plt.title('Joint Adaptive Evolution: Step Size vs Geometric Damping')
        fig.tight_layout()
        plt.show()

    def _sample_from_gmm(self, mu, R, weights, num_samples=10000):
        """Draws exact Monte Carlo samples from the current GMM approximation."""
        K = len(weights)
        dim = mu[0].shape[0]
        comp_indices = self.rng.choice(K, size=num_samples, p=weights)
        samples = np.zeros((num_samples, dim))
        for k in range(K):
            mask = (comp_indices == k)
            n_k = np.sum(mask)
            if n_k > 0:
                z = self.rng.normal(size=(n_k, dim))
                samples[mask] = mu[k] + z @ R[k].T
        return samples

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

    def _compute_current_kl(self, mu, R, weights, mc_samples=2000):
        """Computes D_KL(q || p) via robust Monte Carlo simulation."""
        samples = self._sample_from_gmm(mu, R, weights, num_samples=mc_samples)
        log_q = self._evaluate_gmm_logpdf(samples, mu, R, weights)
        log_p = np.array([self.log_posterior(s) for s in samples])
        return np.mean(log_q - log_p)
    

    def _sample(self, x0, R0):
        '''Integrate ODE for mu and R over n_iter steps.
        Initialize R from Sigma0's Cholesky and run inference.
        '''
        K = len(x0)
        mu = x0.copy()
        R = R0.copy()
        means_hist, Rs_hist = [ [m.copy() for m in mu] ], [ [r.copy() for r in R] ]
        # Initialize Uniform weights in log space
        logws = np.array([np.log(1.0 / K) for _ in range(K)])
        weights_hist = [np.exp(logws).copy()]
        kl_hist = []

        m_mu, v_mu, m_R, v_R, m_w, v_w = init_states(mu, R, logws)

        # Flag to check if we should evolve weights (WFR) or keep them fixed (Pure Wasserstein)
        evolve_weights = (K > 1) and (self.step_size_w > 0.0)

        rejected_steps_count = 0
        max_rejections = 500

        i = 1
        while i <= self.n_iter:
            #if evolve_weights:
            #    logws = self._advance_weights(mu, R, logws, i, m_w, v_w, dt_factor=0.5)

            mu_next, R_next, accepted, dt_new = self._advance_pos(mu, R, logws, i, m_mu, v_mu, m_R, v_R)
            if accepted:
                mu, R = mu_next, R_next
                if evolve_weights:
                    logws = self._advance_weights(mu, R, logws, i, m_w, v_w, dt_new)
                current_w = np.exp(logws).copy()
                # Keep trace histories
                means_hist.append([m.copy() for m in mu])
                Rs_hist.append([r.copy() for r in R])
                weights_hist.append(np.exp(logws).copy())
                if self.kl_track:
                    log_p_values = [self.log_posterior(particles[p, :]) for p in range(n)]
                    current_kl = self.kl_track.estimate_kl_mgvi(particles, log_p_values)
                    kl_hist.append(current_kl)
                    # kl_hist.append(self._compute_current_kl(mu, R, current_w))
                
                # Debug tracking diagnostics
                if i % 1 == 0 or i == 1:
                    print(f"Iteration {i:3d} | Step size {self.step_size:3f} | KL: {kl_hist[-1]:.4f}")
                    for k in range(K):
                        cov = R[k] @ R[k].T
                        std = np.sqrt(np.diag(cov))
                        print(f"  Comp {k} | Weight: {np.exp(logws[k]):.4f} | Mean[0]: {mu[k][0]:.4f} | Std[0]: {std[0]:.4f}")
                        
                i += 1  # Securely advance the actual iteration count
            else:
                rejected_steps_count += 1
                if rejected_steps_count > max_rejections:
                    raise RuntimeError(f"Solver failed: Step size reduced too many times without succeeding.")
    
        print("number of rejected steps = ", rejected_steps_count)
        return mu, R, np.exp(logws), means_hist, Rs_hist, weights_hist, kl_hist
