import numpy as np

from sampling_toolbox.svgd import SVGD
from sampling_toolbox.langevin import ULA
from sampling_toolbox.aldi import ALDI
from plotting import plot_result, plot_kl
from benchmarks_2D import rosenbrock2d_log, rosenbrock2d_grad_log, rosenbrock2d_hessian_gn_log, rosenbrock2d_div_Q, rosenbrock2d_div_Q_gn_fast

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

    mu = np.array([-1., 3.])
    sigma = np.array([0.1, 0.1])
    nparticles = 1000
    initial_particles = rng.normal(loc=mu, scale=np.sqrt(sigma), size=(nparticles, len(mu)))

    print(f"init particles max: {np.max(initial_particles, axis=0)}")
    print(f"init particles min: {np.min(initial_particles, axis=0)}")
    print(f"Initial mean 0: {np.mean(initial_particles, axis=0)}")
    print(f"Initial std 0: {np.std(initial_particles, axis=0)}")

    return initial_particles, rng

def run_svgd(particles, dt, nsteps, rng):
    svgd = SVGD(log_likelihood=f, log_prior=log_prior,
                grad_log_likelihood=df, grad_log_prior=grad_log_prior,
                step_size=dt, n_iter=nsteps, rng=rng, tol=1e-5)
    print("\nRunning SVGD...")
    s_svgd, s_history, kl_hist = svgd.sample(particles, num_samples=0)
    svgd.report_calls()
    svgd.print_statistics(s_svgd)
    print("svgd finished")
    return s_history, kl_hist

def dynamic_preconditioner(x):
    hess = rosenbrock2d_hessian_gn_log(x)
    return -np.linalg.inv(hess)

def run_langevin(particles, dt, nsteps, rng, precond=True):
    x_map = np.array([1., 1.])
    hess = rosenbrock2d_hessian_gn_log(x_map)
    hess_inv = -np.linalg.inv(hess)
    # print(hess_inv)
    # hess_inv = np.eye(2)
    #print("hessian inverse at map", hess_inv)
    if precond == 'gn': 
        ula = ULA(log_likelihood=f, log_prior=log_prior,
                grad_log_likelihood=df, grad_log_prior=grad_log_prior,
                step_size=dt, n_iter=nsteps, rng=rng, preconditioner=dynamic_preconditioner, div_preconditioner=rosenbrock2d_div_Q_gn_fast
            )
    elif precond == 'numeric_fd':
        ula = ULA(log_likelihood=f, log_prior=log_prior,
                grad_log_likelihood=df, grad_log_prior=grad_log_prior,
                step_size=dt, n_iter=nsteps, rng=rng, preconditioner='numerical', div_preconditioner='numerical'
            )
    elif precond == 'map':
        ula = ULA(log_likelihood=f, log_prior=log_prior,
                grad_log_likelihood=df, grad_log_prior=grad_log_prior,
                step_size=dt, n_iter=nsteps, rng=rng, preconditioner=hess_inv
            )
    else:
        ula = ULA(log_likelihood=f, log_prior=log_prior,
                grad_log_likelihood=df, grad_log_prior=grad_log_prior,
                step_size=dt, n_iter=nsteps, rng=rng
            )
    print("\nRunning ULA...")
    s_ula, s_history, kl_hist = ula.sample(particles, num_samples=0)
    ula.report_calls()
    ula.print_statistics(s_ula)
    print("ula finished")
    return s_history, kl_hist

def run_aldi(particles, dt, nsteps, rng):
    aldi = ALDI(log_likelihood=f, log_prior=log_prior,
                grad_log_likelihood=df, grad_log_prior=grad_log_prior,
                step_size=dt, n_iter=nsteps, rng=rng)
    print("\nRunning ALDI...")
    s_aldi, s_history, kl_hist = aldi.sample(particles, num_samples=0)
    aldi.report_calls()
    aldi.print_statistics(s_aldi)
    print("aldi finished")
    return s_history, kl_hist

if __name__ == "__main__":
    init_particles, rng = initialize()

    #samples_history_svgd, kl_svgd = run_svgd(init_particles, dt=0.1, nsteps=200, rng=rng)
    samples_history_ula_gn,  kl_ula_gn  = run_langevin(init_particles, dt=0.005, nsteps=15000, rng=rng, precond='gn')
    #samples_history_ula_num,  kl_ula_num  = run_langevin(init_particles, dt=0.0001, nsteps=500, rng=rng, precond='numeric_fd')
    samples_history_ula_map,  kl_ula_map  = run_langevin(init_particles, dt=0.005, nsteps=15000, rng=rng, precond='map')
    samples_history_ula,  kl_ula  = run_langevin(init_particles, dt=0.005, nsteps=15000, rng=rng, precond=False)
    # Store everything in dictionaries
    results = {
        #'svgd': samples_history_svgd[-1],
        'ula_gn':  samples_history_ula_gn[-1],
        #'ula_num':  samples_history_ula_num[-1],
        'ula_map':  samples_history_ula_map[-1],
        'ula':  samples_history_ula[-1]
    }
    kl_tracks = {
        #'svgd': kl_svgd,
        'ula_gn':  kl_ula_gn,
        #'ula_num':  kl_ula_num,
        'ula_map':  kl_ula_map,
        'ula':  kl_ula,
    }

    #print(samples_history_svgd[0].shape)
    n_grid = 200
    xlim = 10 
    ylim = 15 
    x = np.linspace(-xlim, xlim, n_grid) 
    y = np.linspace(-ylim, ylim+80, n_grid) 
    X, Y = np.meshgrid(x, y, indexing="xy") # shape (n_gridy, n_gridx)
    logPOST = f((X, Y))

    max_logPOST = np.max(logPOST)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    # 3. Compute log(Z) stably
    log_Z = max_logPOST + np.log(np.sum(np.exp(logPOST - max_logPOST))) + np.log(dx) + np.log(dy)

    print(f"Computed Normalizing Constant log(Z): {log_Z:.6f}")

    for method_name in ['ula_gn', 'ula_map', 'ula']:
        # Extract the final step's particles dynamically
        final_samples = results[method_name]
        plot_result(
            X=X,
            Y=Y,
            logPOST=logPOST,
            particles=final_samples,
            method=method_name
        )

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(kl_ula_gn, label='gn')
    #plt.plot(kl_ula_num, label='num')
    plt.plot(kl_ula_map, label='map')
    plt.plot(kl_ula, label='ula')
    plt.xlabel('Iteration')
    plt.ylabel('KL')
    plt.legend()
    plt.show() 