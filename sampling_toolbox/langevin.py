import numpy as np
from sampling_toolbox.base import ParticleMethod
from sampling_toolbox.utilities.kl_tracker import RelativeKLTracker


class ULA(ParticleMethod):
    """Unadjusted Langevin Algorithm (ULA) with optional preconditioning."""

    def __init__(
        self,
        log_likelihood=None,
        log_prior=None,
        grad_log_likelihood=None,
        grad_log_prior=None,
        log_and_grad_post=None,
        step_size=1e-2,
        n_iter=1000,
        rng=None,
        verbose=False,
        preconditioner=None,
        div_preconditioner=None,
    ):
        super().__init__(
            log_likelihood,
            log_prior,
            grad_log_likelihood,
            grad_log_prior,
            log_and_grad_post,
            step_size,
            rng=rng,
        )
        self.step_size = step_size
        self.n_iter = n_iter
        self.precond = preconditioner
        self.div_precond = div_preconditioner
        self.kl_track = RelativeKLTracker(0.0)
        self.verbose = verbose
        self.diagnostics = {
            "kl": [],
        }

    def _Q(self, x):
        if self.precond is None:
            return np.eye(len(x))
        elif callable(self.precond):
            return self.precond(x)
        elif isinstance(self.precond, np.ndarray):
            return self.precond
        else:
            raise ValueError("Preconditioner type or value unknown!")

    def _div_Q(self, x):
        if self.div_precond is None:
            return np.zeros_like(x)
        elif callable(self.div_precond):
            return self.div_precond(x)
        elif isinstance(self.precond, np.ndarray):
            return np.zeros_like(x)
        else:
            raise ValueError("div Q preconditioner unknown or unhandled!")

    def _langevin_update(self, particles):
        n, dim = particles.shape
        grad = np.zeros((n, dim))
        logp = np.zeros(n)
        drift = np.zeros((n, dim))
        noise = np.zeros((n, dim))

        xi_all = self.rng.normal(size=particles.shape)

        # Single pass for target density evaluation + gradient computation
        if self.log_and_grad_post is not None:
            for p in range(n):
                logp[p], grad[p] = self.log_and_grad_posterior(particles[p])
        else:
            for p in range(n):
                grad[p] = self.grad_log_posterior(particles[p])
                logp[p] = self.log_posterior(particles[p])

        # Preconditioned drift and stochastic noise computation
        for p in range(n):
            x_p = particles[p]
            Q_p = self._Q(x_p)
            divQ_p = self._div_Q(x_p)

            if divQ_p.shape != x_p.shape:
                raise ValueError("div_preconditioner shape mismatch.")

            try:
                chol_Q_p = np.linalg.cholesky(Q_p)
            except np.linalg.LinAlgError:
                raise ValueError("Preconditioner matrix is not SPD!")

            drift[p] = Q_p @ grad[p] + divQ_p
            noise[p] = np.sqrt(2.0 * self.step_size) * (chol_Q_p @ xi_all[p])

        return drift, noise, logp

    def _sample(self, x0: np.ndarray):
        # x0 shape should be (num_particles, dim)
        particles = np.asarray(x0).copy()
        samples_history = [particles.copy()]

        for i in range(self.n_iter):
            old_particles = particles.copy()
            drift, noise, log_p_values = self._langevin_update(particles)

            # Update particle states
            particles += self.step_size * drift + noise

            if not np.all(np.isfinite(particles)):
                raise FloatingPointError(
                    "ULA particles diverged: non-finite values detected."
                )

            # Track KL divergence before applying the step
            kl_before_update = self.kl_track.estimate_kl_particles(
                old_particles, log_p_values
            )
            self.diagnostics["kl"].append(kl_before_update)
            samples_history.append(particles.copy())

            if self.verbose:
                print(
                    f"Iter {i+1:5d} | "
                    f"drift_norm={np.linalg.norm(drift)/particles.shape[0]:.3e} | "
                    f"KL={kl_before_update:.3e}"
                )

        return particles, samples_history, self.diagnostics