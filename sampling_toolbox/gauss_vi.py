import numpy as np
from scipy.special import logsumexp
from sampling_toolbox.base_vi import VI
from sampling_toolbox.utilities.gauss_vi_tools import compute_increments_generic, compute_weight_increments, compute_Es_cached
from sampling_toolbox.utilities.time_integration import (
    euler_step_pos, euler_step_w, heun_adaptive_step_pos, heun_adaptive_step_pos2
)
from sampling_toolbox.utilities.stopping_criteria import VariationalEarlyStopping
from sampling_toolbox.utilities.kl_tracker import GenericKLTracker
from sampling_toolbox.utilities.gauss_vi_tools import mixture_grad


class GaussianODE(VI):
    def __init__(self,
                 log_and_grad_post,
                 step_size: float = 0.1,
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
        self.max_rejections = 5000
        self.precond = precond
        self.step_size_w = step_size_w
        self.cumulative_dist = 0.0
        self.kl_track = GenericKLTracker(0.)
        self.ess_target = ess_target
        
        # Diagnostics
        self.dt_history = []
        self.dt_fr_history = []
        self.acceptance_history = []
        
        # Free-reuse caches
        self._last_cub_points = None
        self._last_logs = None

        #self.reg_lambda = 1e2
        self.lambda_end = 1e-3
        self.lambda_start = 1e3
        self.lambda_history = []

        self.t_adam = 0
        self.m_mu = []
        self.v_mu = []
        self.m_R  = []
        self.v_R  = []

        self.prev_nrmse = 1.0

    def _get_pos_increments(self, mu, R, logws, update_cache=False):
        """Standard wrapper passed into ODE time integrators."""
        dm, dR, cub_pts, cub_logs = compute_increments_generic(
        means=mu,
        Rs=R,
        logws=logws,
        log_and_grad_post=self.log_and_grad_post,
        mixture_grad_fn=mixture_grad,
        precond=self.precond,
        reg_lambda=self.reg_lambda
        )

        # Intercept and cache structural values evaluated during this pass
        if update_cache:
            self._last_cub_points = cub_pts
            self._last_logs = cub_logs

        # 2. Filter them through the isolated component trust region
        #dm, dR = self._apply_component_trust_region(mu, R, dm_raw, dR_raw)

        return dm, dR  # outputs required by time integration routines
    

    '''def _apply_component_trust_region(self, mu, R, dm, dR):
        """
        Damps spatial increments on a per-component basis if they exceed 
        a local trust-region threshold, keeping healthy components fast.
        """
        K = len(mu)
        dm_controlled = []
        dR_controlled = []
        
        # Trust region threshold: How many multiples of its own scale 
        # can a component change in a single theoretical unit of time?
        # 2.0 to 5.0 allows incredibly aggressive, fast Natural Gradient behavior.
        MAX_GROWTH_RATIO = 50.0

        for k in range(K):
            # Current physical scale of this specific component
            norm_R = np.linalg.norm(R[k], ord='fro')
            
            # Proposed raw changes for this step
            norm_dm = np.linalg.norm(dm[k])
            norm_dR = np.linalg.norm(dR[k], ord='fro')
            
            # Find the maximum relative explosive force of this component
            # We protect against division by zero if a component has completely collapsed
            scale_denominator = max(norm_R, 1e-5)
            rel_mu_speed = norm_dm / scale_denominator
            rel_R_speed = norm_dR / scale_denominator
            
            max_rel_speed = max(rel_mu_speed, rel_R_speed)
            
            # If the component goes crazy, calculate a custom local brake factor
            if max_rel_speed > MAX_GROWTH_RATIO:
                local_brake = MAX_GROWTH_RATIO / max_rel_speed
                
                # Print a warning so you track who is misbehaving
                print(f" [Trust Region] Component {k+1} penalized! Local brake: {local_brake:.4f}")
                
                dm_controlled.append(dm[k] * local_brake)
                dR_controlled.append(dR[k] * local_brake)
            else:
                # Well-behaved component: zero interference, full natural gradient speed!
                dm_controlled.append(dm[k])
                dR_controlled.append(dR[k])
            
        return dm_controlled, dR_controlled'''


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


    def _advance_weights(self, mu, R, logws, dt_pos, it):
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
            # Compute the Fisher-Rao speed (volatility of energies)
            fr_variance = np.sum(ws * (Es - mean_E)**2)
            fr_speed = np.sqrt(fr_variance)
            # Target a maximum geometric displacement on the simplex per step
            delta_max = getattr(self, 'delta_max', 0.02) #between 0.01 (poor init) up to 0.05
            adaptive_dt_w = min(0.1, delta_max / (fr_speed + 1e-3)) # set a max FR to 0.1
            # --- THE WFR COUPLING MECHANISM ---
            # We use the accepted dt_pos as our global time-scale budget.
            # self.step_size_w acts as your global strategy modifier.
            # The final step is the strict minimum of what the spatial budget allows 
            # and what the local weight speed limit demands.
            budget = 0.5*self.n_iter# / 2.
            #max_weight_dt = adaptive_dt_w # min(global_budget_dt, adaptive_dt_w)
            max_weight_dt = min(1.0, it/budget)*adaptive_dt_w
            #max_weight_dt = adaptive_dt_w
            print("FR step = ", max_weight_dt)
            # print("FR speed dt", delta_max / (fr_speed + 1e-3))
            # --- REPLICATOR DYNAMICS UPDATE ---
            # 1. Step forward in unnormalized log space
            unc_logws = logws - max_weight_dt * Es
            # 2. Normalize safely back onto the simplex
            new_logws = unc_logws - logsumexp(unc_logws)
            # Log history and return the updated log weights
            self.dt_fr_history.append(max_weight_dt)
            return new_logws

        else:
            raise ValueError(f"Scheme '{self.time_scheme_fr}' not implemented for weights.")


    def _advance_pos(self, mu, R, logws, it):
        dt = self.step_size
        accepted = True
        tau = min(1., (it - 1) / (self.n_iter - 1))
        self.reg_lambda = self.lambda_start * (self.lambda_end / self.lambda_start) ** (tau)
        
        if self.time_scheme == 'euler':
            m1, R1 = euler_step_pos(mu, R, logws, self._get_pos_increments, dt)
        elif self.time_scheme == 'heun_adaptive':
            m1, R1, dt_actual, accepted = heun_adaptive_step_pos(
                mu, R, logws, self._get_pos_increments, dt, 
                rtol=getattr(self, 'rtol', 5e-2),#1e-2, 5e-2
                atol=getattr(self, 'atol', 1e-2)#1e-3, 1e-2
            )
            #m1, R1, dt_actual, accepted = heun_adaptive_step_pos2(
            #    mu, R, logws, self._get_pos_increments, dt
            #)
            self.step_size = dt_actual
            if accepted:
                self.cumulative_dist += dt_actual
        else:
            raise ValueError(f"Scheme '{self.time_scheme}' not implemented.")
        
        self.dt_history.append(dt)
        self.acceptance_history.append(accepted)
        return m1, R1, accepted, dt


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
        i = 1

        #early_stopper = VariationalEarlyStopping(kl_tol=5e-6, patience_window=8)

        while i <= self.n_iter:
            #dt_actual = self.step_size
            mu_next, R_next, accepted, dt_new = self._advance_pos(mu, R, logws, i)

            if accepted:
                dt_actual = dt_new
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
                    #if early_stopper.update(current_kl, step_accepted=True):
                    #    break

                if evolve_weights:
                    logws_next = self._advance_weights(mu, R, logws, dt_actual, i)
                    current_w = np.exp(logws_next).copy()
                    logws = logws_next
                
                mu, R = mu_next, R_next
                
                self.lambda_history.append(self.reg_lambda)
                means_hist.append([m.copy() for m in mu])
                Rs_hist.append([r.copy() for r in R])
                weights_hist.append(current_w)
                        
                if i % 1 == 0 or i == 1:
                    kl_val = kl_hist[-1] if kl_hist else 0.0
                    print(f"Iter {i:3d} | dt: {dt_new:.4f} | KL: {kl_val:.4f}")
                    cov0 = R[0] @ R[0].T
                    std0 = np.sqrt(np.diag(cov0))
                    if K > 1:
                        cov1 = R[1] @ R[1].T
                        std1 = np.sqrt(np.diag(cov1))
                        cov2 = R[2] @ R[2].T
                        std2 = np.sqrt(np.diag(cov2))
                        print('------------')
                        print(f"weights {current_w}")
                        print(f"mean 0 {mu[0]}")
                        print(f"std 0 {std0}")
                        print(f"mean 1 {mu[1]}")
                        print(f"std 1 {std1}")
                        print(f"mean 2 {mu[2]}")
                        print(f"std 2 {std2}")
                i += 1
            else:
                rejected_steps_count += 1
                #if early_stopper.update(None, step_accepted=False):
                #    break
                if rejected_steps_count > self.max_rejections:
                    raise RuntimeError("Solver failed: Too many rejected steps.")
    
        print("Number of rejected steps = ", rejected_steps_count)
        return mu, R, np.exp(logws), means_hist, Rs_hist, weights_hist, kl_hist
