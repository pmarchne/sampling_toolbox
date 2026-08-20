import numpy as np
from sampling_toolbox.base import ParticleMethod
from sampling_toolbox.utilities.kde_kernels import rbf_kernel
from sampling_toolbox.utilities.kl_tracker import RelativeKLTracker

class SVGD(ParticleMethod):
    ''' implements the Stein Variational gradient descent '''
    def __init__(self, 
        log_likelihood=None,
        log_prior=None,
        grad_log_likelihood=None,
        grad_log_prior=None,
        log_and_grad_post=None,
        step_size=0.1,
        kernel_scale=1.0,
        n_iter=50,
        tol=1e-5,
        rng=None,
        verbose=False,
        transform=None
        ):
        super().__init__(
            log_likelihood, 
            log_prior,
            grad_log_likelihood,
            grad_log_prior,
            log_and_grad_post,
            step_size,
            rng=rng
        )
        self.step_size = step_size
        self.n_iter = n_iter
        self.tol = tol
        self.kl_track = RelativeKLTracker(0.)
        self.verbose = verbose
        self.kernel_scale = kernel_scale
        self.transform = transform
        self.diagnostics = {
            "kl": [],
            "bandwidth": [],
            "z_median_displacement": [],
            "physical_median_displacement": [],
            "physical_spread": [],
            "physical_mean": [],
            "physical_std": [],
            "max_particle_norm": [],
        }

    def _svgd_phi(self, particles):
        n, dim = particles.shape
        K, dK, bandwidth = rbf_kernel(particles, h=-1, scale=self.kernel_scale)
        grad = np.zeros((n, dim))
        logp = np.zeros(n)

        if self.log_and_grad_post is not None:
            for p in range(n): # only one pass through the adjoint solver
                logp[p], grad[p] = self.log_and_grad_posterior(particles[p])
        else:
            for p in range(n):
                grad[p] = self.grad_log_posterior(particles[p])
                logp[p] = self.log_posterior(particles[p])

        #  grad[p, :] = self.grad_log_posterior(particles[p, :])
        phi = (K @ grad + dK) / n
        return phi, logp, bandwidth

    def _sample(self, x0: np.ndarray):
        # x0 shape should be (num_particles, dim)
        particles = x0.copy()
        historical_grad = np.zeros_like(particles)
        alpha = 0.9
        fudge_factor = 1e-6
        samples_history = [particles.copy()]
        n = particles.shape[0]
        log_p_values = np.zeros(n)

        for i in range(self.n_iter):
            old_particles = particles.copy()
            phi, log_p_values, bandwidth = self._svgd_phi(particles)
            velocity_norm = np.linalg.norm(phi) / n
            # RMSprop style initialization
            if i == 0:
                historical_grad = phi ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (phi ** 2)

            phi_adj = np.divide(phi, fudge_factor + np.sqrt(historical_grad))
            particles += self.step_size * phi_adj

            if not np.all(np.isfinite(particles)):
                raise FloatingPointError(
                    "SVGD particles diverged: non-finite values detected."
                )

            # track and estimate the KL from the previous iteration
            kl_before_update = self.kl_track.estimate_kl_particles(old_particles, log_p_values)
            self.diagnostics["kl"].append(kl_before_update)
            samples_history.append(particles.copy())

            # Diagnostics
            z_displacements = np.linalg.norm(
                particles - old_particles,
                axis=1
            )
            self.diagnostics["bandwidth"].append(bandwidth)
            self.diagnostics["z_median_displacement"].append(
                np.median(z_displacements)
            )
            self.diagnostics["max_particle_norm"].append(
                np.max(np.linalg.norm(particles, axis=1))
            )
            # Physical space diagnostics
            if self.transform is not None:
                old_v = self.transform(old_particles)
                v = self.transform(particles)
                v_displacements = np.linalg.norm(
                    v - old_v,
                    axis=1
                )
            else:
                v = particles
                v_displacements = z_displacements

            physical_spread = np.median(
                np.linalg.norm(v - np.mean(v, axis=0), axis=1)
            )
            self.diagnostics["physical_median_displacement"].append(np.median(v_displacements))
            self.diagnostics["physical_spread"].append(physical_spread)
            v_mean = np.mean(v, axis=0)
            v_std = np.std(v, axis=0)
            self.diagnostics["physical_mean"].append(v_mean)
            self.diagnostics["physical_std"].append(v_std)

            if self.verbose:
                log_str = (
                    f"Iter {i+1:4d} | "
                    f"phi={velocity_norm:.3e} | "
                    f"h={bandwidth:.3e} | "
                    f"dz={np.median(z_displacements):.3e} | "
                    f"KL={kl_before_update:.3e}"
                )
                # append physical velocity/space diagnostics
                if self.transform is not None:
                    log_str += (
                        f" | dv={np.median(v_displacements):.2f} m/s"
                        f" | v_spread={physical_spread:.1f}"
                        f" | v_mean={v_mean.round(1)}"
                        f" | v_std={v_std.round(1)}"
                    )
                print(log_str)
            if velocity_norm < self.tol:
                print(f"\nConvergence reached at iteration {i+1} (Norm < {self.tol})")
                break
        return particles, samples_history, self.diagnostics

