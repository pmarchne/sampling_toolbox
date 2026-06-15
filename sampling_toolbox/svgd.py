import numpy as np
from sampling_toolbox.base import ParticleMethod
from sampling_toolbox.utilities.kde_kernels import rbf_kernel
from sampling_toolbox.utilities.kl_tracker import GenericKLTracker

class SVGD(ParticleMethod):
    ''' implements the Stein Variational gradient descent '''
    def __init__(self, log_likelihood, log_prior,
                 grad_log_likelihood, grad_log_prior,
                 step_size=0.1, n_iter=50, tol=1e-5, rng=None):
        super().__init__(log_likelihood, log_prior,
                         grad_log_likelihood, grad_log_prior,
                         step_size, rng=rng)
        self.step_size = step_size
        self.n_iter = n_iter
        self.tol = tol
        self.kl_track = GenericKLTracker(0.)

    def _svgd_phi(self, particles):
        n, dim = particles.shape
        K, dK = rbf_kernel(particles, h=-1)
        grad = np.zeros((n, dim))
        for p in range(n):
            grad[p, :] = self.grad_log_posterior(particles[p, :])
        phi = (K @ grad + dK) / n
        return phi

    def _sample(self, x0: np.ndarray, num_samples=0, save_step=1, print_step=1):
        # x0 shape should be (num_particles, dim)
        particles = x0.copy()
        historical_grad = np.zeros_like(particles)
        alpha = 0.9
        fudge_factor = 1e-6
        samples_history = [particles.copy()]
        kl_hist = []
        n = particles.shape[0]
        log_p_values = np.zeros(n)

        for i in range(self.n_iter):
            phi = self._svgd_phi(particles)
            phi_norm = np.linalg.norm(phi) / n
            # RMSprop style initialization
            if i == 0:
                historical_grad = phi ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (phi ** 2)
                
            phi_adj = np.divide(phi, fudge_factor + np.sqrt(historical_grad))
            particles += self.step_size * phi_adj
            # track KL estimate
            if self.kl_track:
                log_p_values = [self.log_posterior(particles[p, :]) for p in range(n)]
                current_kl = self.kl_track.estimate_kl_particles(particles, log_p_values)
                kl_hist.append(current_kl)
            # Reporting & Early Stopping
            if i % save_step == 0:
                samples_history.append(particles.copy())
            if i % print_step == 0 or i == 0:
                print(f"Iteration {i+1:3d} | Update Norm: {phi_norm:.6f}")
            if phi_norm < self.tol:
                print(f"\nConvergence reached at iteration {i+1} (Norm < {self.tol})")
                break
        return particles, samples_history, kl_hist
    
