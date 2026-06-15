import numpy as np

from sampling_toolbox.svgd import SVGD
from sampling_toolbox.langevin import ULA
from sampling_toolbox.aldi import ALDI
from plotting import plot_result, plot_result_gmm, plot_vi_diagnostics
from benchmarks_2D import rosenbrock2d_log, rosenbrock2d_grad_log, rosenbrock2d_hessian_gn_log, rosenbrock2d_div_Q
from sampling_toolbox.gauss_vi import GaussianODE
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

def log_prior(x):
    return 0.0

def grad_log_prior(x):
    return np.zeros_like(x)

def f(x):
    return rosenbrock2d_log(x, alpha=0.5)

def df(x):
    return rosenbrock2d_grad_log(x, alpha=0.5)

def df_hessian_preconditioned(x):
    # Compute base gradient
    base_grad = rosenbrock2d_grad_log(x, alpha=0.5)
    # Compute Hessian matrix at current point
    hess = rosenbrock2d_hessian_gn_log(x)
    hess_inv = np.linalg.inv(-hess)
    return hess_inv @ base_grad
    
def initialize():
    # Setup the target
    rng = np.random.default_rng(1)
    dt = 0.0005
    nsteps = 4000

    mu = np.array([-2., 10.])
    sigma = np.array([2., 2.])
    nparticles = 200
    initial_particles = rng.normal(loc=mu, scale=np.sqrt(sigma), size=(nparticles, len(mu)))

    print(f"init particles max: {np.max(initial_particles, axis=0)}")
    print(f"init particles min: {np.min(initial_particles, axis=0)}")
    print(f"Initial mean 0: {np.mean(initial_particles, axis=0)}")
    print(f"Initial std 0: {np.std(initial_particles, axis=0)}")

    return initial_particles, dt, nsteps, rng

def run_svgd(particles, dt, nsteps, rng):
    svgd = SVGD(log_likelihood=f, log_prior=log_prior,
                grad_log_likelihood=df, grad_log_prior=grad_log_prior,
                step_size=dt, n_iter=nsteps, rng=rng, tol=1e-5)
    print("\nRunning SVGD...")
    s_svgd, s_history = svgd.sample(particles, num_samples=0)
    svgd.report_calls()
    svgd.print_statistics(s_svgd)
    print("svgd finished")
    return s_history

def dynamic_preconditioner(x):
    hess = rosenbrock2d_hessian_gn_log(x)
    return -np.linalg.inv(hess)

def run_langevin(particles, dt, nsteps, rng):
    x_map = np.array([1., 1.])
    hess = rosenbrock2d_hessian_gn_log(x_map)
    hess_inv = -np.linalg.inv(hess)
    print(hess_inv)
    # hess_inv = np.eye(2)
    #print("hessian inverse at map", hess_inv)
    ula = ULA(log_likelihood=f, log_prior=log_prior,
                grad_log_likelihood=df, grad_log_prior=grad_log_prior,
                step_size=dt, n_iter=nsteps, rng=rng, preconditioner=dynamic_preconditioner,
                div_preconditioner=rosenbrock2d_div_Q)
    print("\nRunning ULA...")
    s_ula, s_history = ula.sample(particles, num_samples=0)
    ula.report_calls()
    ula.print_statistics(s_ula)
    print("ula finished")
    return s_history

def run_aldi(particles, dt, nsteps, rng):
    aldi = ALDI(log_likelihood=f, log_prior=log_prior,
                grad_log_likelihood=df, grad_log_prior=grad_log_prior,
                step_size=dt, n_iter=nsteps, rng=rng)
    print("\nRunning ALDI...")
    s_aldi, s_history = aldi.sample(particles, num_samples=0)
    aldi.report_calls()
    aldi.print_statistics(s_aldi)
    print("aldi finished")
    return s_history

'''if __name__ == "__main__":
    init_particles, dt, nsteps, rng = initialize()
    samples_history_svgd = run_svgd(init_particles, dt, nsteps, rng)
    samples_history_ula = run_langevin(init_particles, dt, nsteps, rng)
    samples_history_aldi = run_aldi(init_particles, dt, nsteps, rng)

    print(samples_history_svgd[0].shape)
    n_grid = 200
    xlim = 10 
    ylim = 5 
    x = np.linspace(-xlim, xlim, n_grid) 
    y = np.linspace(-ylim, ylim+60, n_grid) 
    X, Y = np.meshgrid(x, y, indexing="xy") # shape (n_gridy, n_gridx)
    logPOST = f((X, Y))
    plot_result(
        X,
        Y,
        logPOST,
        samples_history_svgd[-1],  # optional
        method='svgd'
    )
    plot_result(
        X,
        Y,
        logPOST,
        samples_history_ula[-1],  # optional
        method='ula'
    )
    plot_result(
        X,
        Y,
        logPOST,
        samples_history_aldi[-1],  # optional
        method='aldi'
    )'''

def evaluate_and_compare_evidence(X, Y, logPOST, gvi, final_mean, final_R, final_ws, mc_samples=20000):
    """
    Computes log Z from the grid and compares it with the ELBO and 
    Importance Sampling estimates derived from the trained GMM.
    """
    # ----------------------------------------------------
    # 1. Compute Ground Truth log Z from the 2D Grid
    # ----------------------------------------------------
    # Extract coordinate vectors from grid to find step sizes
    x_vec = X[0, :]
    y_vec = Y[:, 0]
    dx = x_vec[1] - x_vec[0]
    dy = y_vec[1] - y_vec[0]
    
    # 2D integration via Log-Sum-Exp
    log_Z_grid = logsumexp(logPOST) + np.log(dx) + np.log(dy)
    
    # ----------------------------------------------------
    # 2. Compute ELBO and Importance Sampling from GMM
    # ----------------------------------------------------
    # Draw exact Monte Carlo samples from the converged GMM
    samples = gvi._sample_from_gmm(final_mean, final_R, final_ws, num_samples=mc_samples)
    
    # Evaluate log q(x) and log p*(x) for those samples
    log_q = gvi._evaluate_gmm_logpdf(samples, final_mean, final_R, final_ws)
    log_p = np.array([gvi.log_posterior(s) for s in samples])
    
    # Difference vector
    log_weights = log_p - log_q
    
    # Calculate Metrics
    elbo = np.mean(log_weights)
    log_Z_gmm_is = logsumexp(log_weights) - np.log(mc_samples)
    
    # ----------------------------------------------------
    # 3. Print Comparison Report
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("           LOG EVIDENCE (log Z) COMPARISON")
    print("="*50)
    print(f"Ground Truth (2D Grid Integration):   {log_Z_grid:11.5f}")
    print(f"GMM Importance Sampling Estimate:     {log_Z_gmm_is:11.5f}")
    print(f"GMM Variational Lower Bound (ELBO):   {elbo:11.5f}")
    print("-"*50)
    print(f"True KL Gap (log Z - ELBO):           {log_Z_grid - elbo:11.5f}")
    print(f"IS Residual Error (|Grid - IS|):       {np.abs(log_Z_grid - log_Z_gmm_is):11.5f}")
    print("="*50)
    
    return log_Z_grid, log_Z_gmm_is, elbo


def compute_distribution_distances(X, Y, logPOST, gvi, final_mean, final_R, final_ws):
    """
    Computes the 2D Total Variation distance and the 1D Wasserstein distances 
    for the X and Y marginal distributions by comparing the GMM to the grid reference.
    """
    # 1. Extract the underlying 1D coordinate axes
    # For indexing="xy", X varies across columns, Y varies across rows
    x_vec = X[0, :]
    y_vec = Y[:, 0]
    
    # 2. Create normalized PMF for the Reference Target PDF
    # Normalize in log-space first to protect against underflow vulnerabilities
    P_grid = np.exp(logPOST - logsumexp(logPOST))
    
    # 3. Evaluate and Normalize GMM PDF over the exact same grid coordinate space
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    log_q = gvi._evaluate_gmm_logpdf(grid_points, final_mean, final_R, final_ws)
    log_q_grid = log_q.reshape(X.shape)
    Q_grid = np.exp(log_q_grid - logsumexp(log_q_grid))
    
    # 4. Calculate 2D Total Variation (TV) Distance
    # Formula: 0.5 * sum(|P - Q|)
    tv_distance = 0.5 * np.sum(np.abs(P_grid - Q_grid))
    
    # 5. Extract 1D Marginal PMFs by summing out the opposite dimension
    # Summing over axis=0 (rows/Y) leaves the X marginal profile
    P_x = np.sum(P_grid, axis=0)
    Q_x = np.sum(Q_grid, axis=0)
    
    # Summing over axis=1 (columns/X) leaves the Y marginal profile
    P_y = np.sum(P_grid, axis=1)
    Q_y = np.sum(Q_grid, axis=1)
    
    # 6. Calculate 1D Wasserstein Distances via Scipy
    # Uses the coordinate grids as spatial positions and marginal sums as weights
    w1_x = wasserstein_distance(x_vec, x_vec, u_weights=P_x, v_weights=Q_x)
    w1_y = wasserstein_distance(y_vec, y_vec, u_weights=P_y, v_weights=Q_y)
    
    # Print Metrics Summary
    print("\n" + "="*50)
    print("         GEOMETRIC & STATISTICAL DISTANCES")
    print("="*50)
    print(f"Total Variation (TV) Distance (2D):   {tv_distance:.5f}")
    print(f"1D Wasserstein Distance (X marginal): {w1_x:.5f}")
    print(f"1D Wasserstein Distance (Y marginal): {w1_y:.5f}")
    print("="*50)
    
    return tv_distance, w1_x, w1_y


def generate_random_gmm_init(n_components=3, mode='standard', seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    mu = []
    sigma = []
    
    # 1. Define the geometric domain based on your target plots
    # x1 (horizontal) spans roughly -6 to 6
    # x2 (vertical) spans roughly 0 to 45
    
    for _ in range(n_components):
        if mode == 'stiff':
            # STRESS TEST 1: Components start ultra-sharp (tiny variances)
            # This triggers massive initial gradients
            x1 = np.random.uniform(-5.0, 5.0)
            x2 = np.random.uniform(5.0, 35.0)
            std_x1 = np.random.uniform(0.05, 0.2)
            std_x2 = np.random.uniform(0.05, 0.2)
            
        elif mode == 'far_away':
            # STRESS TEST 2: Components start completely trapped out in the cold
            x1 = np.random.uniform(-7.0, 7.0)
            x2 = np.random.uniform(30.0, 50.0) 
            std_x1 = np.random.uniform(0.5, 2.0)
            std_x2 = np.random.uniform(0.5, 2.0)
            
        else:
            # STANDARD RANDOM: A healthy, diverse spread
            x1 = np.random.uniform(-6.0, 6.0)
            x2 = np.random.uniform(2.0, 30.0)
            # Log-uniform sampling for robust scale diversity
            std_x1 = np.exp(np.random.uniform(np.log(0.1), np.log(3.0)))
            std_x2 = np.exp(np.random.uniform(np.log(0.1), np.log(3.0)))
            
        mu.append(np.array([x1, x2]))
        sigma.append(np.diag([std_x1**2, std_x2**2]))
        
    return mu, sigma


if __name__ == "__main__":
    dt = 0.00015 # 0.08
    nit = 300
    n_comp = 10
    mu, sigma = generate_random_gmm_init(n_components=n_comp, mode='far_away', seed=2)
    print(mu)
    print(sigma)
    #mu = [np.array([-0.5, 15.]), np.array([-3, 40.]), np.array([3., 5.])]
    #sigma = [np.diag([1**2, 1**2]), np.diag([1.**2, 1.**2]), np.diag([0.1**2, 0.1**2])]

    gvi = GaussianODE(f, df, log_prior, grad_log_prior, step_size=dt, n_iter=nit, method='cubature_hess', time_scheme='heun_adaptive', num_samples=100, step_size_w=0.)
    # gvi = GaussianODE(f, df, log_prior, grad_log_prior, step_size=dt, n_iter=nit, method='cubature', time_scheme='heun_adaptive', num_samples=500, step_size_w=0.1)

    R = [np.linalg.cholesky(sigma[i]) for i in range(n_comp)]

    print(f"current mean {mu}")
    cov = R[0] @ R[0].T
    std = np.sqrt(np.diag(cov))
    print(f"current std {std}")

    final_mean, final_R, final_ws, means, Rs, ws, kl_h = gvi.sample(mu, R)
    gvi.plot_diagnostics()
    #final_cov = final_R @ final_R.T
    #std = np.sqrt(np.diag(final_cov))
    #print(final_mean)
    #print(std)
    gvi.report_calls()

    n_grid = 200
    xlim = 10 
    ylim = 15 
    x = np.linspace(-xlim, xlim, n_grid) 
    y = np.linspace(-ylim, ylim+60, n_grid) 
    X, Y = np.meshgrid(x, y, indexing="xy") # shape (n_gridy, n_gridx)
    logPOST = f((X, Y))

    evaluate_and_compare_evidence(X, Y, logPOST, gvi, final_mean, final_R, final_ws, mc_samples=20000)
    compute_distribution_distances(X, Y, logPOST, gvi, final_mean, final_R, final_ws)

    plot_result_gmm(X, Y, logPOST, mus=final_mean, Rs=final_R, weights=final_ws)
    plot_vi_diagnostics(means, Rs, ws, kl_h)

    