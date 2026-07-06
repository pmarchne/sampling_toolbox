import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_result_gmm, plot_vi_diagnostics
from benchmarks_2D import (
    gauss2d_log, 
    gauss2d_grad_log, 
    four_mode2D_asymmetric_log, 
    four_mode2D_asymmetric_grad_log
)
from scipy.stats import multivariate_normal, norm
from sampling_toolbox.gauss_vi import GaussianODE
from sampling_toolbox.utilities.kl_tracker import GenericKLTracker

def compute_trajectory_metrics(X, Y, logPOST, m_hist, R_hist):
    """
    Computes both KL and TV distance for every step in the trajectory history.
    """
    # 1. Normalize true grid posterior to a valid probability mass function (PMF)
    POST = np.exp(logPOST - np.max(logPOST))
    p_pmf = POST / np.sum(POST)
    
    kl_history = []
    tv_history = []
    
    pos = np.dstack((X, Y))
    eps = 1e-10 # Prevent numerical log(0) issues
    
    # 2. Iterate through every saved optimization step
    for mu, R in zip(m_hist, R_hist):
        mu = mu[0]  
        R = R[0]
        cov = R @ R.T
        q_pdf = multivariate_normal(mean=mu, cov=cov).pdf(pos)
        q_pmf = q_pdf / np.sum(q_pdf)
        
        # Compute forward metric history: KL(q || p)
        kl = np.sum(q_pmf * np.log((q_pmf + eps) / (p_pmf + eps)))
        tv = 0.5 * np.sum(np.abs(p_pmf - q_pmf))
        
        kl_history.append(kl)
        tv_history.append(tv)
        
    return np.array(kl_history), np.array(tv_history)


def get_experiment_config(case_id):
    """
    Returns the target functions, initialization, and weight step size (dtw)
    tailored exactly for each benchmark case.
    """
    np.random.seed(42)  # Set seed for reproducible initializations
    
    if case_id == 1:
        print("\n=== RUNNING CASE 1: Single Gaussian Ideal VI ===")
        # Ideal Target: Standard Single 2D Gaussian centered at (0,0)
        f_target = lambda x: gauss2d_log(x, 0.05)
        df_target = lambda x: gauss2d_grad_log(x, 0.05)
        
        # Initialize 1 component significantly offset to verify trajectory convergence
        mu = [np.array([3.5, -1.5])]
        sigma = [0.4 * np.eye(2)]
        dtw = 0.0  # Weights irrelevant for K=1
        
    elif case_id == 2:
        print("\n=== RUNNING CASE 2: Four Asymmetric Modes (Fixed Weights) ===")
        f_target = lambda x: four_mode2D_asymmetric_log(x, eta=2.5, bias=np.array([0.4, 0.2]))
        df_target = lambda x: four_mode2D_asymmetric_grad_log(x, eta=2.5, bias=np.array([0.4, 0.2]))
        
        # Explicitly initialize 4 components near the known mode coordinates
        mode_centers = [np.array([0.4, 3.]), np.array([0.4, -3.]), np.array([-3., -0.4]), np.array([3., -0.4])]
        mu = [m + np.random.uniform(-0.2, 0.2, size=2) for m in mode_centers]
        sigma = [0.1 * np.eye(2) for _ in range(4)]
        dtw = 0.0
        
    elif case_id == 3:
        print("\n=== RUNNING CASE 3: Four Asymmetric Modes (Dynamic Weights) ===")
        f_target = lambda x: four_mode2D_asymmetric_log(x, eta=2.5, bias=np.array([0.4, 0.2]))
        df_target = lambda x: four_mode2D_asymmetric_grad_log(x, eta=2.5, bias=np.array([0.4, 0.2]))
        
        # Use identical nearby initialization as Case 2 to isolate weight behavior
        mode_centers = [np.array([0.4, 3.]), np.array([0.4, -3.]), np.array([-3., -0.4]), np.array([3., -0.4])]
        mu = [m + np.random.uniform(-0.2, 0.2, size=2) for m in mode_centers]
        sigma = [0.1 * np.eye(2) for _ in range(4)]
        dtw = 0.1
        
    else:
        raise ValueError("Invalid Case ID")
        
    return f_target, df_target, mu, sigma, dtw


def compute_grid_log_evidence(f_target, xlim=4., ylim=4., n_grid=300):
    """Dynamically computes Log Z for the specific active target."""
    x = np.linspace(-xlim, xlim, n_grid)
    y = np.linspace(-ylim, ylim, n_grid)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="xy")
    
    log_post_grid = f_target((X, Y))
    max_log_post = np.max(log_post_grid)
    log_sum_exp_grid = max_log_post + np.log(np.sum(np.exp(log_post_grid - max_log_post)))
    logZ = log_sum_exp_grid + np.log(dx * dy)
    return X, Y, log_post_grid, logZ


if __name__ == "__main__":
    # Select which cases you want to run: [1, 2, 3]
    cases_to_test = [1, 2, 3]
    
    # Global Solver Hyperparameters
    dt = 0.002
    nit = 400
    integrator = 'heun_adaptive'
    
    for case in cases_to_test:
        # 1. Setup target landscape and initialization properties
        f, df, mu_init, sigma_init, dtw_val = get_experiment_config(case)
        log_and_grad_post = lambda x: (f(x), df(x))
        
        # 2. Compute true normalized baseline evidence for this specific target
        X, Y, logPOST, logZ = compute_grid_log_evidence(f)
        print(f"Computed Log Evidence (log Z): {logZ:.4f}")
        
        # 3. Handle base-level matrix/vector allocations safely via deep copies
        mu_in = [m.copy() for m in mu_init]
        R_in = [np.linalg.cholesky(s) for s in sigma_init]
        
        # 4. Instantiate Solver instances
        gvi_id = GaussianODE(log_and_grad_post, step_size=dt, n_iter=nit, time_scheme=integrator, time_scheme_fr=integrator, precond='identity', step_size_w=dtw_val)
        gvi_nat = GaussianODE(log_and_grad_post, step_size=dt, n_iter=nit, time_scheme=integrator, time_scheme_fr=integrator, precond='natural', step_size_w=dtw_val)
        
        gvi_id.kl_track = GenericKLTracker(logZ)
        gvi_nat.kl_track = GenericKLTracker(logZ)
        
        # 5. Execute Trajectories using distinct copies of input components
        print("Running Identity Flow...")
        final_m_id, final_R_id, final_ws_id, m_hist_id, R_hist_id, _, kl_hist_id = gvi_id.sample(
            [m.copy() for m in mu_in], [r.copy() for r in R_in]
        )
        
        print("Running Natural Flow...")
        final_m_nat, final_R_nat, final_ws_nat, m_hist_nat, R_hist_nat, w_hist_nat, kl_hist_nat = gvi_nat.sample(
            [m.copy() for m in mu_in], [r.copy() for r in R_in]
        )
        
        # 6. Display Diagnostics and Plots per Case
        print(f"Final Weights (Natural): {final_ws_nat}")
        
        # Convergence Trajectory Graph
        plt.figure(figsize=(6, 4))
        plt.plot(kl_hist_id, 'r', label='Identity Flow')
        plt.plot(kl_hist_nat, 'b', label='Natural Gradient Flow')
        plt.xlabel('Iteration')
        plt.ylabel('KL Divergence')
        plt.yscale('log')
        plt.title(f'Case {case}: KL History Profile')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.show()
        
        # Spatial Contour Overlays
        plot_result_gmm(X, Y, logPOST, mus_hist=m_hist_nat, Rs=final_R_id, weights=final_ws_id, method=f'Case {case} - Identity', mu0=mu_in)
        plot_result_gmm(X, Y, logPOST, mus_hist=m_hist_id, Rs=final_R_nat, weights=final_ws_nat, method=f'Case {case} - Natural', mu0=mu_in)
        
        # 1. Compute full histories post-optimization
        kl_hist_id, tv_hist_id = compute_trajectory_metrics(X, Y, logPOST, m_hist_id, R_hist_id)
        
        import matplotlib.pyplot as plt
        plt.plot(kl_hist_id, "b-")
        plt.plot(tv_hist_id, "r-")
        plt.yscale('log')
        plt.show()


        # 2. Extract final values for testing
        final_mu = m_hist_id[-1]
        final_R = R_hist_id[-1]

        if case > 1:
            plot_vi_diagnostics(m_hist_nat[1:], R_hist_nat[1:], w_hist_nat[1:], kl_hist_nat)