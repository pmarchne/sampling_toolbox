import numpy as np
from scipy.special import logsumexp
from sampling_toolbox.base_vi import VI
from sampling_toolbox.utilities.gauss_vi_tools import compute_increments_generic, compute_weight_increments, compute_Es_cached
from sampling_toolbox.utilities.time_integration import (
    euler_step_pos, euler_step_w, heun_adaptive_step_pos, rmsprop_step_pos
)
from sampling_toolbox.utilities.kl_tracker import RelativeKLTracker
from sampling_toolbox.utilities.gauss_vi_tools import mixture_grad


class GaussianODE(VI):
    def __init__(self,
                 log_and_grad_post,
                 step_size: float = 0.01,
                 n_iter = 50,
                 time_scheme = 'euler',
                 time_scheme_fr = 'euler',
                 precond = 'none',
                 step_size_w: float = 0.,
                 ess_target = None,
                 rng = None):
        super().__init__(log_and_grad_post, step_size, rng=rng)
        self.n_iter = n_iter
        self.time_scheme = time_scheme
        self.time_scheme_fr = time_scheme_fr
        self.max_rejections = 500
        self.max_nan = 10
        self.precond = precond
        self.step_size_w = step_size_w
        self.cumulative_dist = 0.0
        self.kl_track = RelativeKLTracker(0.)
        self.ess_target = ess_target
        
        # Diagnostics
        self.dt_history = []
        self.dt_fr_history = []
        self.acceptance_history = []
        self._last_cub_points = None
        self._last_logs = None
        self._cached_dmu = None
        self._cached_dR = None

        self.reg_lambda = 1e3
        self.lambda_end = 1e-3
        self.lambda_start = 1e3
        self.lambda_history = []
        # RMSprop specific hyperparameters
        self.alpha_rmsprop = 0.9
        self.eps_rmsprop = 1e-8
        self.v_mu = []
        self.v_R = []

    def _get_pos_increments(self, mu, R, logws, update_cache=True):
        """Standard wrapper passed into ODE time integrators."""
        dm, dR, cub_pts, cub_logs = compute_increments_generic(
        means=mu,
        Rs=R,
        logws=logws,
        log_and_grad_post=self.log_and_grad_posterior,
        mixture_grad_fn=mixture_grad,
        precond=self.precond,
        reg_lambda=self.reg_lambda
        )
        if not np.isfinite(dm).all() or not np.isfinite(dR).all():
            raise ValueError("Non-finite increments computed.")
            
        if update_cache:
            self._last_cub_points = cub_pts.copy()
            self._last_logs = cub_logs.copy()
        return dm, dR  # outputs required by time integration routines


    def _get_w_increments(self, mu, R, logws):
        """Reuses cached log evaluations directly from the position pass."""
        if self._last_logs is None:
            raise RuntimeError("Weight increments requested, but no spatial cache exists.")
        dlogws = compute_weight_increments(
            means=mu,
            Rs=R,
            logws=logws,
            cached_log_targets=self._last_logs
        )
        return dlogws


    def _advance_weights(self, mu, R, logws, dt_pos):
        """
        Advances weights using a staggered WFR approach. Uses the accepted 
        spatial step (dt_pos) as a dynamic baseline to scale weight updates.
        """
        K = len(mu)
        if K == 1:
            return logws

        # 1. Classical Explicit Euler Branch
        if self.time_scheme_fr == 'euler':
            effective_dt_w = dt_pos * self.step_size_w
            return euler_step_w(mu, R, logws, self._get_w_increments, effective_dt_w)
        # 2. Staggered WFR Adaptive Branch
        elif self.time_scheme_fr == 'heun_adaptive':
            # Extract expectations and compute component weights
            Es = compute_Es_cached(mu, R, logws, self._last_logs)
            ws = np.exp(logws)
            mean_E = np.sum(ws * Es)
            fr_variance = np.sum(ws * (Es - mean_E)**2)
            delta_max = getattr(self, 'delta_max', 0.02)
            adaptive_dt_w = min(0.1, delta_max / (np.sqrt(fr_variance) + 1e-3))
            unc_logws = logws - adaptive_dt_w * Es
            new_logws = unc_logws - logsumexp(unc_logws)
            self.dt_fr_history.append(adaptive_dt_w)
            return new_logws
        else:
            raise ValueError(f"Scheme '{self.time_scheme_fr}' not implemented for weights.")


    def _advance_pos(self, mu, R, logws, it):
        dt = self.step_size
        accepted = True
        tau = min(1., (it - 1) / (self.n_iter - 1))
        self.reg_lambda = self.lambda_start * (self.lambda_end / self.lambda_start) ** (tau)
        try:
            if self.time_scheme == 'euler':
                m1, R1 = euler_step_pos(mu, R, logws, self._get_pos_increments, dt)
            elif self.time_scheme == 'rmsprop':
                m1, R1, dt_actual, accepted = rmsprop_step_pos(
                    mu, R, logws, self._get_pos_increments,
                    lr=dt,
                    v_mu=self.v_mu,
                    v_R=self.v_R,
                    alpha=self.alpha_rmsprop,
                    eps=self.eps_rmsprop
                )
            elif self.time_scheme == 'heun_adaptive':
                m1, R1, dmu_next, dR_next, dt_actual, accepted = heun_adaptive_step_pos(
                    mu, R, logws, self._get_pos_increments, dt, 
                    dmu1=self._cached_dmu,
                    dR1=self._cached_dR,
                    rtol=getattr(self, 'rtol', 5e-2), #5e-2
                    atol=getattr(self, 'atol', 1e-2) # 1e-2
                )
                self.step_size = dt_actual
                if accepted:
                    self.cumulative_dist += dt_actual
                    self._cached_dmu = dmu_next
                    self._cached_dR = dR_next
            else:
                raise ValueError(f"Scheme '{self.time_scheme}' not implemented.")
                
            # Post-step floor check to keep R healthy for the next iteration
            if accepted:
                for r_mat in R1:
                    dim = mu[0].shape[0]
                    di = np.diag_indices(dim)
                    r_mat[di] = np.clip(r_mat[di], 1e-3, 300.0)
                    r_mat[di] = np.maximum(r_mat[di], 1e-6)

        except (ValueError, np.linalg.LinAlgError) as e:
            # If solve_triangular or any other linalg operation fails:
            print(f" [Step Warning] Matrix instability encountered: {e}. Rejecting step.")
            accepted = False
            # Aggressively shrink the step size for the next attempt
            self.step_size = dt * 0.5 
            # Return current unchanged state so it can retry with the smaller dt
            m1, R1 = mu, R
            dt_actual = self.step_size
        
        self.acceptance_history.append(accepted)
        return m1, R1, accepted, dt_actual


    def _sample(self, x0, R0):
        K = len(x0)
        dim = x0[0].shape[0]
        mu, R = x0.copy(), R0.copy()
        means_hist, Rs_hist = [[m.copy() for m in mu]], [[r.copy() for r in R]]
        logws = np.array([np.log(1.0 / K) for _ in range(K)])
        weights_hist = [np.exp(logws).copy()]
        current_w = weights_hist[0]
        kl_hist = []

        evolve_weights = (K > 1) and (self.step_size_w > 0.0)
        rejected_steps_count = 0
        nan_count = 0
        i = 1
        self._cached_dmu, self._cached_dR = self._get_pos_increments(mu, R, logws, update_cache=True)

        while i <= self.n_iter:
            
            saved_cub_pts = self._last_cub_points
            saved_logs = self._last_logs
            mu_next, R_next, accepted, dt_new = self._advance_pos(mu, R, logws, i)

            if np.isnan(dt_new) or any(np.isnan(m).any() for m in mu_next) or any(np.isnan(r).any() for r in R_next):
                print(f"\n[Warning] NaN/Inf detected at iter {i}. Recovering state, shrinking dt, and continuing.")
                # Shrink step size aggressively
                self.step_size = self.step_size * 0.05
                nan_count += 1
                # Revert to last step's healthy parameters
                if nan_count > self.max_nan:
                    raise RuntimeError("Solver failed: Too many rejected steps due to NaN loops.")
                continue

            if accepted:
                dt_actual = dt_new
                
                if evolve_weights:
                    logws_next = self._advance_weights(mu, R, logws, dt_actual)
                    current_w = np.exp(logws_next).copy()
                    logws = logws_next
                
                # estimate KL through current cubature points
                if self.kl_track and self._last_cub_points is not None:
                    cub_weights = []
                    for k in range(K):
                        for _ in range(2 * dim):
                            cub_weights.append(current_w[k] / (2 * dim))
                    cub_weights = np.array(cub_weights)
                    
                    current_kl = self.kl_track.estimate_kl_mgvi(
                        cubature_points=self._last_cub_points,
                        cubature_weights=cub_weights,
                        cached_log_p=self._last_logs,
                        mu=mu,
                        R=R,
                        weights=current_w
                    )
                    kl_hist.append(current_kl)

                mu, R = mu_next, R_next
                
                self.lambda_history.append(self.reg_lambda)
                means_hist.append([m.copy() for m in mu])
                Rs_hist.append([r.copy() for r in R])
                weights_hist.append(current_w)
                self.dt_history.append(dt_actual)
                        
                if i % 1 == 0 or i == 1:
                    kl_val = kl_hist[-1] if kl_hist else 0.0
                    print(f"Iter {i:3d} | dt: {dt_new:.4f} | KL: {kl_val:.4f}")
                    print(f"weights {current_w}")
                i += 1
            else:
                self._last_cub_points = saved_cub_pts
                self._last_logs = saved_logs
                rejected_steps_count += 1
                if rejected_steps_count > self.max_rejections:
                    raise RuntimeError("Solver failed: Too many rejected steps.")
    
        print("Number of rejected steps = ", rejected_steps_count)
        print("Number of nans = ", nan_count)
        return mu, R, np.exp(logws), means_hist, Rs_hist, weights_hist, kl_hist
