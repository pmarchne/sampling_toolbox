import numpy as np
from sampling_toolbox.base import ParticleMethod
from sampling_toolbox.utilities.kl_tracker import RelativeKLTracker


class ALDI(ParticleMethod):
    """
    Affine-Invariant Langevin Interactive Dynamics (ALDI).

    Langevin dynamics with an ensemble-dependent empirical covariance preconditioner:
        C = (1/N) * Xc^T @ Xc
    Includes the required structural divergence correction term:
        div(C) = ((dim + 1) / N) * Xc
    """

    def __init__(
        self,
        log_likelihood=None,
        log_prior=None,
        grad_log_likelihood=None,
        grad_log_prior=None,
        log_and_grad_post=None,
        step_size=0.1,
        n_iter=50,
        sqrt_type="cholesky",
        rng=None,
        verbose=False,
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
        self.sqrt_type = sqrt_type
        self.kl_track = RelativeKLTracker(0.0)
        self.verbose = verbose
        self.diagnostics = {
            "kl": [],
        }

    def _covariance(self, xc, n):
        """Computes dxd empirical covariance and square root factor."""
        cov = (xc.T @ xc) / n
        if self.sqrt_type == "cholesky":
            # Lower-triangular Cholesky factor L (d x d)
            L = np.linalg.cholesky(cov)
            return cov, L
        elif self.sqrt_type == "ensemble":
            # Ensemble matrix factor Xc / sqrt(N) (N x d)
            E = xc / np.sqrt(n)
            return cov, E
        else:
            raise ValueError("sqrt_type must be 'cholesky' or 'ensemble'.")

    def _aldi_update(self, particles):
        n, dim = particles.shape
        grad = np.zeros((n, dim))
        logp = np.zeros(n)

        # Single pass for target density evaluation + gradient computation
        if self.log_and_grad_post is not None:
            for p in range(n):
                logp[p], grad[p] = self.log_and_grad_posterior(particles[p])
        else:
            for p in range(n):
                grad[p] = self.grad_log_posterior(particles[p])
                logp[p] = self.log_posterior(particles[p])

        # Centered ensemble and empirical covariance
        xc = particles - particles.mean(axis=0)
        cov, cov_half = self._covariance(xc, n)

        # Divergence correction term: div(C) = ((dim + 1) / N) * Xc
        correction = ((dim + 1) / n) * xc

        # Preconditioned drift: C @ grad + div(C)
        drift = grad @ cov + correction

        # Stochastic noise component
        noise = self.rng.normal(size=particles.shape)
        if self.sqrt_type == "cholesky":
            diffusion = noise @ cov_half.T
        elif self.sqrt_type == "ensemble":
            diffusion = noise * cov_half

        return drift, diffusion, logp

    def _sample(self, x0: np.ndarray):
        # x0 shape should be (num_particles, dim)
        particles = np.asarray(x0).copy()
        samples_history = [particles.copy()]

        for i in range(self.n_iter):
            old_particles = particles.copy()
            drift, diffusion, log_p_values = self._aldi_update(particles)

            # Update particle states
            particles += (
                self.step_size * drift
                + np.sqrt(2.0 * self.step_size) * diffusion
            )

            if not np.all(np.isfinite(particles)):
                raise FloatingPointError(
                    "ALDI particles diverged: non-finite values detected."
                )

            # Track KL divergence before applying the update step
            kl_before_update = self.kl_track.estimate_kl_particles(
                old_particles, log_p_values
            )
            self.diagnostics["kl"].append(kl_before_update)
            samples_history.append(particles.copy())

            if self.verbose:
                drift_norm = np.linalg.norm(drift) / particles.shape[0]
                print(
                    f"Iter {i+1:5d} | "
                    f"drift_norm={drift_norm:.3e} | "
                    f"KL={kl_before_update:.3e}"
                )

        return particles, samples_history, self.diagnostics