import numpy as np
from plotting import plot_optimization_trajectories, set_up_plots
from benchmarks_2D import rosenbrock2d_log, rosenbrock2d_grad_log
from sampling_toolbox.gauss_vi import GaussianODE
from sampling_toolbox.utilities.kl_tracker import RelativeKLTracker
from benchmark_rosenbrock import recompute_gmm_kl

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

if __name__ == "__main__":
    set_up_plots()
    savefig = False
    n_grid = 400
    xlim = 10
    ylim = 15
    x = np.linspace(-xlim, xlim, n_grid) 
    dx = x[1] - x[0]
    y = np.linspace(-ylim, ylim+70, n_grid)
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="xy")
    logPOST = f((X, Y))
    max_logPOST = np.max(logPOST)
    log_sum_exp_grid = max_logPOST + np.log(np.sum(np.exp(logPOST - max_logPOST)))
    logZ = log_sum_exp_grid + np.log(dx*dy)
    print(f"Log Evidence (log Z): {logZ}")

    dt = 1. #0.002
    nit = 150
    n_comp = 1
    sigma = 7.
    mu, sigma = [np.array([-9., 40.])], [np.diag([sigma**2, sigma**2])]
    mu_in = [m.copy() for m in mu]
    
    R = [np.linalg.cholesky(sigma[0])]
    R_in  = [R.copy()]
    dtw = 0.0

    #integrator = 'heun_adaptive'
    integrator = 'heun_adaptive'
    #integrator = 'adam'
    precond = ['None', 'natural', 'hessian']
    gvi_id = GaussianODE(log_and_grad_post, step_size=dt, n_iter=nit, time_scheme=integrator, precond=precond[0], step_size_w=dtw)
    gvi_nat = GaussianODE(log_and_grad_post, step_size=dt, n_iter=nit, time_scheme=integrator, precond=precond[1], step_size_w=dtw)
    gvi_hess = GaussianODE(log_and_grad_post, step_size=dt, n_iter=nit, time_scheme=integrator, precond=precond[2], step_size_w=dtw)
    gvi_hess.lambda_start = 1e-6
    gvi_hess.lambda_end = 1e-6
    gvi_id.kl_track = RelativeKLTracker(logZ)
    gvi_nat.kl_track = RelativeKLTracker(logZ)
    gvi_hess.kl_track = RelativeKLTracker(logZ)

    final_mean_id, final_R_id, final_ws_id, means_id, Rs_id, ws_id, kl_hist_id = gvi_id.sample([m.copy() for m in mu_in], [r.copy() for r in R])
    final_mean_nat, final_R_nat, final_ws_nat, means_nat, Rs_nat, ws_nat, kl_hist_nat = gvi_nat.sample([m.copy() for m in mu_in], [r.copy() for r in R])
    final_mean_hess, final_R_hess, final_ws_hess, means, Rs, ws, kl_hist_hess = gvi_hess.sample([m.copy() for m in mu_in], [r.copy() for r in R])

    path = ''
    colors = {
        'id': 'green',
        'nat': 'red',
        'new': 'blue'
    }

    labels = {
        'id': r'Identity',
        'nat': r'Natural',
        'new': r'Newton-like'
    }
    import matplotlib.pyplot as plt
  

    plt.figure(figsize=(6, 2.25))
    plt.plot(gvi_id.dt_history, color=colors['id'], label=labels['id'], lw=1.7)
    plt.plot(gvi_hess.dt_history, color=colors['new'], label=labels['new'], lw=1.7, linestyle='-.')
    plt.plot(gvi_nat.dt_history, color=colors['nat'], label=labels['nat'], lw=1.7, linestyle='--')
    plt.xlabel(r'Iteration')
    plt.ylabel(r'$\Delta t_{W}$')
    plt.grid(True)
    plt.xlim([0, nit])
    if savefig == True:
        plt.savefig(path+'time_steps.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(6, 2.25))
    plt.plot(kl_hist_id, color=colors['id'], label=labels['id'], lw=1.7)
    plt.plot(kl_hist_hess, color=colors['new'], label=labels['new'], lw=1.7, linestyle='-.')
    plt.plot(kl_hist_nat, color=colors['nat'], label=labels['nat'], lw=1.7, linestyle='--')
    plt.xlabel(r'Iteration')
    plt.ylabel(r'KL$(\mu \parallel \pi)$')
    plt.yscale('log')
    plt.grid(True)
    plt.xlim([0, nit])
    plt.legend()
    plt.show()
    #

    print("Recomputing True KL history...")
    true_kl_id = recompute_gmm_kl(means_id, Rs_id, ws_id, f, logZ)
    true_kl_nat = recompute_gmm_kl(means_nat, Rs_nat, ws_nat, f, logZ)
    true_kl_hess = recompute_gmm_kl(means, Rs, ws, f, logZ)
    # Plotting the corrected curves
    plt.figure(figsize=(6, 2.25))
    plt.plot(true_kl_id, color=colors['id'], label=labels['id'], lw=1.7)
    plt.plot(true_kl_hess, color=colors['new'], label=labels['new'], lw=1.7, linestyle='-.')
    plt.plot(true_kl_nat, color=colors['nat'], label=labels['nat'], lw=1.7, linestyle='--')
    plt.xlabel(r'Iteration')
    plt.ylabel(r'KL$(\mu \parallel \pi)$')
    plt.yscale('log')
    plt.grid(True)
    plt.xlim([0, nit])
    plt.legend()
    if savefig == True:
        plt.savefig(path+'KL_mc.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    plot_optimization_trajectories(X, Y, logPOST,
        means_nat,          # Trajectory of means for 'natural' method (shape: [T, 2] or list)
        final_R_nat,           # Final R matrix (or factor) for 'natural' method (shape: [2, 2])
        means_id,         # Trajectory of means for 'identity' method (shape: [T2, 2] or list)
        means,         # Trajectory of means for 'Newton-like' method (shape: [T3, 2] or list)
        mu_in,          # Initial mean vector (shape: [2,])
        R_in,           # Initial R matrix/covariance factor for the initial state (shape: [2, 2])
        cmap="Blues"
    )
    if savefig == True:
        plt.savefig(path+'Rosenbrock.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()