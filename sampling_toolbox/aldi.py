import numpy as np
from sampling_toolbox.base import ParticleMethod

class ALDI(ParticleMethod):
    """
    Affine-Invariant Langevin Interactive Dynamics.
    Langevin dynamics with a density dependent preconditioner
    the preconditioner is build from the empirical covariance of current samples
    - requires a correction term in the SDE
    """
    
    def __init__(self, log_likelihood, log_prior,
                 grad_log_likelihood, grad_log_prior,
                 step_size=0.1, n_iter=50, rng=None,
                 sqrt_type='cholesky',
                ):
        super().__init__(log_likelihood, log_prior,
                         grad_log_likelihood, grad_log_prior,
                         step_size, rng=rng)
        self.n_iter = n_iter
        self.sqrt_type = sqrt_type
    
    def covariance(self, xc, n):
        # Compute dxd covariance
        cov = (xc.T @ xc) / n
        if self.sqrt_type == 'cholesky':
            # dxd lower-triangular
            L = np.linalg.cholesky(cov)
            return cov, L
        elif self.sqrt_type == 'ensemble':
            # Nx d ensemble factor: Xc / sqrt(N)
            E = xc / np.sqrt(n)
            return cov, E
        else:
            raise ValueError("sqrt_type must be 'cholesky' or 'ensemble'.")

    def _sample(self, x0: np.ndarray, num_samples=0, save_step=100000):
        particles = x0.copy()
        n, dim = particles.shape
        samples_history = []

        particles_mean = particles - particles.mean(axis=0)
        cov, cov_half = self.covariance(particles_mean, n)

        # correction term = (D+1)/N * Xc
        correction = ((dim + 1) / n) * particles_mean
        grads = np.zeros((n, dim))

        for i in range(self.n_iter):
            grads = np.array([self.grad_log_posterior(p) for p in particles])
            # ∇_m log π(Z) = C @ ∇_m log π(m) + div(C)
            drift = self.step_size * (grads@cov + correction)
            noise = self.rng.normal(size=particles.shape)

            if self.sqrt_type == 'cholesky':
                particles += ( drift
                + np.sqrt(2.0 * self.step_size) * (noise @ cov_half.T)
                )
            elif self.sqrt_type == 'ensemble':
                particles += ( drift
                + np.sqrt(2.0 * self.step_size) * noise * cov_half
                )

            if i % save_step == 0:
                samples_history.append(particles)
            if (i + 1) % 100 == 0 or i == 0:
                print(
                    f"Iteration {i+1:5d} | "
                )
        return particles, samples_history