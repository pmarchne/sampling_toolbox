import numpy as np
import matplotlib.pyplot as plt

from sampling_toolbox.svgd import SVGD
from sampling_toolbox.langevin import ULA
from sampling_toolbox.aldi import ALDI
from plotting import plot_result
from benchmarks_2D import rosenbrock2d_log, rosenbrock2d_grad_log


def log_prior(x):
    return 0.0

def grad_log_prior(x):
    return np.zeros_like(x)

def f(x):
    return rosenbrock2d_log(x, alpha=0.5)

def df(x):
    return rosenbrock2d_grad_log(x, alpha=0.5)


def initialize():
    # Setup the target
    rng = np.random.default_rng(1)

    mu = np.array([-1.0, 30.0])
    sigma = np.array([0.25, 3.5])
    nparticles = 1000
    initial_particles = rng.normal(loc=mu, scale=np.sqrt(sigma), size=(nparticles, len(mu)))

    print(f"init particles max: {np.max(initial_particles, axis=0)}")
    print(f"init particles min: {np.min(initial_particles, axis=0)}")
    print(f"Initial mean: {np.mean(initial_particles, axis=0)}")
    print(f"Initial std: {np.std(initial_particles, axis=0)}")

    return initial_particles, rng


def run_svgd(particles, dt, nsteps, rng, log_evidence=0.0):
    svgd = SVGD(
        log_likelihood=f,
        log_prior=log_prior,
        grad_log_likelihood=df,
        grad_log_prior=grad_log_prior,
        step_size=dt,
        n_iter=nsteps,
        rng=rng,
        tol=1e-5,
        verbose=True,
    )
    svgd.kl_track.log_evidence = log_evidence
    print("\nRunning SVGD...")
    s_svgd, s_history, diagnostics = svgd.sample(particles)
    svgd.report_calls()
    svgd.print_statistics(s_svgd)
    print("svgd finished")
    return s_history, diagnostics


def run_langevin(particles, dt, nsteps, rng, log_evidence=0.0):
    ula = ULA(
        log_likelihood=f,
        log_prior=log_prior,
        grad_log_likelihood=df,
        grad_log_prior=grad_log_prior,
        step_size=dt,
        n_iter=nsteps,
        rng=rng,
        verbose=True,
    )
    ula.kl_track.log_evidence = log_evidence
    print("\nRunning ULA...")
    s_ula, s_history, diagnostics = ula.sample(particles)
    ula.report_calls()
    ula.print_statistics(s_ula)
    print("ula finished")
    return s_history, diagnostics


def run_aldi(particles, dt, nsteps, rng, log_evidence=0.0):
    aldi = ALDI(
        log_likelihood=f,
        log_prior=log_prior,
        grad_log_likelihood=df,
        grad_log_prior=grad_log_prior,
        step_size=dt,
        n_iter=nsteps,
        rng=rng,
        verbose=True,
    )
    aldi.kl_track.log_evidence = log_evidence
    print("\nRunning ALDI...")
    s_aldi, s_history, diagnostics = aldi.sample(particles)
    aldi.report_calls()
    aldi.print_statistics(s_aldi)
    print("aldi finished")
    return s_history, diagnostics


if __name__ == "__main__":
    init_particles, rng = initialize()

    n_grid = 400
    xlim = 10
    ylim = 15
    x = np.linspace(-xlim, xlim, n_grid)
    y = np.linspace(-ylim, ylim + 80, n_grid)
    X, Y = np.meshgrid(x, y, indexing="xy")
    logPOST = f((X, Y))
    max_logPOST = np.max(logPOST)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    log_Z = max_logPOST + np.log(np.sum(np.exp(logPOST - max_logPOST))) + np.log(dx) + np.log(dy)
    print(f"\nComputed Normalizing Constant log(Z): {log_Z:.6f}")

    samples_history_svgd, diag_svgd = run_svgd(
        init_particles, dt=0.15, nsteps=1000, rng=rng, log_evidence=log_Z
    )
    samples_history_aldi, diag_aldi = run_aldi(
        init_particles, dt=0.025, nsteps=1000, rng=rng, log_evidence=log_Z
    )
    samples_history_ula, diag_ula = run_langevin(
        init_particles, dt=0.1, nsteps=1000, rng=rng, log_evidence=log_Z
    )

    results = {
        "svgd": samples_history_svgd[-1],
        "aldi": samples_history_aldi[-1],
        "ula": samples_history_ula[-1],
    }

    kl_tracks = {
        "svgd": (diag_svgd["kl"], "-"),  # Solid line for deterministic
        "ula":  (diag_ula["kl"], ":"),  # Dotted line for stochastic
        "aldi": (diag_aldi["kl"], ":"), # Dotted line for stochastic
    }

    for method_name in ["svgd", "ula", "aldi"]:
        final_samples = results[method_name]
        plot_result(
            X=X,
            Y=Y,
            logPOST=logPOST,
            particles=final_samples,
            method=method_name,
            initial_particles=init_particles
        )

    plt.figure(figsize=(8, 5))
    for method_name, (kl_vals, linestyle) in kl_tracks.items():
        plt.plot(kl_vals, label=method_name.upper(), linestyle=linestyle, lw=2)
    
    plt.xlabel("Iteration")
    plt.ylabel(r"$\mathrm{KL}(\mu \parallel \pi)$")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.title("Exact KL Divergence Convergence")
    plt.show()