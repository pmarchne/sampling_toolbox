import numpy as np

from plotting import plot_result_gmm
from benchmarks_2D import rosenbrock2d_log, rosenbrock2d_grad_log
from sampling_toolbox.gauss_vi import GaussianODE
from sampling_toolbox.utilities.kl_tracker import RelativeKLTracker

def log_prior(x):
    return 0.0

def grad_log_prior(x):
    return np.zeros_like(x)

def f(x):
    return rosenbrock2d_log(x, 0.5)

def df(x):
    return rosenbrock2d_grad_log(x, 0.5)

def log_and_grad_post(x):
    return f(x), df(x)

from scipy.stats import multivariate_normal

def recompute_gmm_kl(means_hist, Rs_hist, weights_hist, log_post_fn, logZ, n_samples=40000, seed=42):
    """
    Recomputes the true KL history using Monte Carlo sampling.
    """
    np.random.seed(seed)
    kl_history = []
    
    # Iterate through each saved step of the optimizer
    for t in range(len(means_hist)):
        mus = means_hist[t]
        Rs = Rs_hist[t]
        weights = weights_hist[t]
        
        n_comp = len(weights)
        dim = len(mus[0])
        
        # 1. Sample component indices based on weights
        comp_choices = np.random.choice(n_comp, size=n_samples, p=weights)
        
        # 2. Draw samples from the chosen GMM components
        samples = np.zeros((n_samples, dim))
        for i in range(n_comp):
            idx = (comp_choices == i)
            n_idx = np.sum(idx)
            if n_idx > 0:
                z = np.random.normal(size=(n_idx, dim))
                samples[idx] = mus[i] + z @ Rs[i].T
                
        # 3. Evaluate log q(x) for all samples
        q_densities = np.zeros(n_samples)
        for i in range(n_comp):
            Sigma = Rs[i] @ Rs[i].T
            q_densities += weights[i] * multivariate_normal.pdf(samples, mean=mus[i], cov=Sigma)
        log_q = np.log(q_densities + 1e-15)
        
        # 4. Evaluate target log p*(x)
        log_p_star = log_post_fn((samples[:, 0], samples[:, 1]))
        
        # 5. Compute expectation: E_q[log q - log p*] + logZ
        kl_step = np.mean(log_q - log_p_star) + logZ
        kl_history.append(kl_step)
        
    return np.array(kl_history)

def generate_random_gmm_init(n_components=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    mu = []
    sigma = []
    for i in range(n_components):
        x1 = np.random.uniform(-9, 9)
        x2 = np.random.uniform(-10, 60)
        std_x1 = 1.
        std_x2 = std_x1
        mu.append(np.array([x1, x2]))
        sigma.append(np.diag([std_x1**2, std_x2**2]))
        print("mean", i, "is", mu)
    return mu, sigma

def generate_grid_gmm_init():
    """
    Initializes exactly 6 components on a uniform 2D grid:
    x = [-5.0, 0.0, 5.0]
    y = [-5.0, 20.0]
    """
    mu = []
    sigma = []
    
    # Define the exact grid coordinates specified
    x_coords = [-4.0, 0.0, 4.0] #  [-4.0, 0.0, 4.0]
    y_coords =  [-8.0, 50.0] # [-8.0, 40.0] #50.
    
    # Iterate systematically to generate 3 * 2 = 6 coordinates
    for y in y_coords:
        for x in x_coords:
            # Set the mean vector
            mean_vector = np.array([x, y])
            mu.append(mean_vector)
            
            # Spherical initial standard deviation
            std_val = 2.#1.25 #1.5
            cov_matrix = np.diag([std_val**2, std_val**2])
            sigma.append(cov_matrix)
            
            print(f"Component {len(mu)-1} initialized at grid point: [{x}, {y}]")
            
    return mu, sigma


if __name__ == "__main__":
    savefig = False
    n_grid = 400
    xlim = 10.
    ylim = 15.
    x = np.linspace(-xlim, xlim, 3000) 
    dx = x[1] - x[0]
    y = np.linspace(-ylim, ylim+70, 3000)
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="xy")
    logPOST = f((X, Y))
    max_logPOST = np.max(logPOST)
    log_sum_exp_grid = max_logPOST + np.log(np.sum(np.exp(logPOST - max_logPOST)))
    logZ = log_sum_exp_grid + np.log(dx*dy)
    print(f"Log Evidence (log Z): {logZ}")

    dt = 0.01
    dtw = 0.05 #0.2 # 0.5 #0.01 # 0.01
    nit = 200 # 400
    n_comp = 6
    mu, sigma = generate_grid_gmm_init() #generate_random_gmm_init(n_components=n_comp, seed=12)
    mu_in = [m.copy() for m in mu]
    R = [np.linalg.cholesky(sigma[i]) for i in range(n_comp)]
    R_in  = [r.copy() for r in R]

    integrator ='heun_adaptive'
    integrator_fr = 'heun_adaptive'
    precond = ['None', 'natural']
    gvi_id = GaussianODE(log_and_grad_post, step_size=dt, n_iter=nit, time_scheme=integrator, time_scheme_fr=integrator_fr, precond=precond[0], step_size_w=dtw)
    gvi_nat = GaussianODE(log_and_grad_post, step_size=dt, n_iter=nit, time_scheme=integrator, time_scheme_fr=integrator_fr, precond=precond[1], step_size_w=dtw)
    
    gvi_id.kl_track = RelativeKLTracker(logZ)
    gvi_nat.kl_track = RelativeKLTracker(logZ)

    final_mean_id, final_R_id, final_ws_id, means_id, Rs_id, ws_id, kl_hist_id = gvi_id.sample([m.copy() for m in mu_in], [r.copy() for r in R])
    final_mean_nat, final_R_nat, final_ws_nat, means_nat, Rs_nat, ws_nat, kl_hist_nat = gvi_nat.sample([m.copy() for m in mu_in], [r.copy() for r in R])

    path = ''
    colors = {
        'id': 'green',
        'nat': 'red',
    }
    labels = {
        'id': r'Identity',
        'nat': r'Natural',
    }
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",       # Classic publication serif style
        "text.usetex": True,         # Set to True if your system has a full local LaTeX installation
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "grid.alpha": 0.25,
        "grid.linestyle": "--"
    })

    plt.figure(figsize=(6, 2.25))
    plt.plot(gvi_id.dt_history, color=colors['id'], label=labels['id'], lw=1.7)
    plt.plot(gvi_nat.dt_history, color=colors['nat'], label=labels['nat'], lw=1.7, linestyle='--')
    plt.xlabel(r'Iteration')
    plt.ylabel(r'$\Delta t_{W}$')
    plt.grid(True)
    plt.xlim([0, nit])
    #plt.savefig(path+'time_steps.pdf', dpi=300, bbox_inches='tight')
    #plt.close()
    plt.show()

    plt.figure()
    plt.plot(gvi_id.dt_fr_history, color=colors['id'], label=labels['id'], lw=1.7)
    plt.plot(gvi_nat.dt_fr_history, color=colors['nat'], label=labels['nat'], lw=1.7, linestyle='--')
    plt.title('step size FR')
    plt.legend()
    plt.show()
    #

    estimated_KL_id = np.maximum(1e-4, np.asarray(kl_hist_id))
    estimated_KL_nat = np.maximum(1e-4, np.asarray(kl_hist_nat))
    plt.figure(figsize=(6, 2.25))
    plt.plot(estimated_KL_id, color=colors['id'], label=labels['id'], lw=1.7)
    plt.plot(estimated_KL_nat, color=colors['nat'], label=labels['nat'], lw=1.7, linestyle='--')
    plt.xlabel(r'Iteration')
    plt.ylabel(r'KL$(\mu \parallel \pi)$')
    plt.yscale('log')
    plt.grid(True)
    plt.xlim([0, nit])
    plt.ylim([1e-2, 50.])
    plt.legend()
    #plt.savefig(path+'KL_gmm.pdf', dpi=300, bbox_inches='tight')
    #plt.close()
    plt.show()
    plt.close()

    print("Recomputing True KL history for Identity...")
    true_kl_id = recompute_gmm_kl(means_id, Rs_id, ws_id, f, logZ)
    
    print("Recomputing True KL history for Natural...")
    true_kl_nat = recompute_gmm_kl(means_nat, Rs_nat, ws_nat, f, logZ)

    # Plotting the corrected curves
    plt.figure(figsize=(6, 2.25))
    plt.plot(true_kl_id, color=colors['id'], label=labels['id'], lw=1.7)
    plt.plot(true_kl_nat, color=colors['nat'], label=labels['nat'], lw=1.7, linestyle='--')
    plt.xlabel(r'Iteration')
    plt.ylabel(r'KL$(\mu \parallel \pi)$')
    plt.yscale('log')
    plt.grid(True)
    plt.xlim([0, nit])
    plt.legend()
    if savefig == True:
        plt.savefig(path+'KL_mc_gmm.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    plot_result_gmm(X, Y, logPOST, mus_hist=means_nat, Rs=final_R_nat, weights=final_ws_nat, method='Natural', mu0=mu_in)
    if savefig == True:
        plt.savefig(path+'Rosenbrock_gmm.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    plot_result_gmm(X, Y, logPOST, mus_hist=means_id, Rs=final_R_id, weights=final_ws_id, method='Identity', mu0=mu_in)
    plt.show()
    #plt.close()

    plt.figure(figsize=(6, 2.25))
    weights_arr = np.array(ws_nat[1:])
    colors = plt.cm.Reds(np.linspace(0.2, 1, max(n_comp, 5)))
    for k in range(n_comp):
        plt.plot(weights_arr[:, k], color=colors[k], lw=2, label=f"Comp {k}")
    plt.xlabel(r'Iteration')
    plt.ylabel(r'$w_k$')
    plt.grid(True)
    plt.title('Natural')
    plt.xlim([0, nit])
    if savefig == True:
        plt.savefig(path+'weights_gmm.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
