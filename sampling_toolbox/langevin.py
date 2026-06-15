import numpy as np
from sampling_toolbox.base import ParticleMethod
from sampling_toolbox.utilities.kl_tracker import GenericKLTracker
from sampling_toolbox.utilities.numerical_precond import NumericalPreconditioner

class ULA(ParticleMethod):
    """
    Unadjusted Langevin Algorithm (ULA) with optional preconditioning.

    Dynamics:
        x_{k+1} = x_k
                  + step_size * ( Q(x_k) grad log pi(x_k)
                                  + div Q(x_k) )
                  + sqrt(2 * step_size * Q(x_k)) * xi_k

    Parameters
    ----------
    preconditioner : callable, optional
        Function Q(x) returning a (d,d) SPD matrix.
        If None, Q(x)=I.

    div_preconditioner : callable, optional
        Function returning divergence of Q:
            div Q(x) in R^d
        If None, assumed zero.
    """

    def __init__(
        self,
        log_likelihood,
        log_prior,
        grad_log_likelihood,
        grad_log_prior,
        step_size=1e-2,
        n_iter=1000,
        rng=None,
        preconditioner=None,
        div_preconditioner=None,
    ):

        super().__init__(
            log_likelihood,
            log_prior,
            grad_log_likelihood,
            grad_log_prior,
            step_size,
            rng=rng,
        )
        self.n_iter = n_iter
        self.precond = preconditioner
        self.div_precond = div_preconditioner
        self.kl_track = GenericKLTracker(0.)
        if isinstance(self.precond, str) and self.precond == 'numerical':
            self.fd_precond = NumericalPreconditioner(self.grad_log_posterior)
        else:
            self.fd_precond = None
            

    def _Q(self, x):
        if self.precond is None:
            return np.eye(len(x) if x is not None else 2)
        elif isinstance(self.precond, str):
            if self.precond == 'numerical':
                return self.fd_precond.get_Q(x)
            else:
                raise ValueError(f"Unknown string preconditioner option: {self.precond}")
        elif callable(self.precond):
            return self.precond(x)
        elif isinstance(self.precond, np.ndarray):
            return self.precond
        else:
            raise ValueError('Preconditioner type or value unknown!')

    def _div_Q(self, x):
        if self.div_precond is None:
            return np.zeros_like(x)
        elif isinstance(self.precond, str) and self.precond == 'numerical':
            return self.fd_precond.get_div_Q(x)
        elif callable(self.div_precond):
            return self.div_precond(x)
        elif isinstance(self.precond, np.ndarray):
            return np.zeros_like(x)
            
        else:
            raise ValueError('div Q preconditioner unknown or unhandled!')

    # Main sampler
    def _sample(self, x0: np.ndarray, num_samples=None, save_step=1, print_step=500):
        particles = np.asarray(x0).copy()
        n, dim = particles.shape
        samples_history = [particles.copy()]
        kl_hist = []

        for k in range(self.n_iter):
            drift_all = np.zeros_like(particles)
            noise_all = np.zeros_like(particles)
            xi_all = self.rng.normal(size=particles.shape)

            for p in range(n):
                x_p = particles[p, :]
                grad_p = self.grad_log_posterior(x_p)
                # get preconditioner
                Q_p = self._Q(x_p)                   
                divQ_p = self._div_Q(x_p)   
                
                if divQ_p.shape != x_p.shape:
                    raise ValueError(f"div_preconditioner shape mismatch.")          
                try:
                    chol_Q_p = np.linalg.cholesky(Q_p)
                except np.linalg.LinAlgError:
                    raise ValueError("Preconditioner matrix is not SPD!")
                
                drift_all[p, :] = Q_p @ grad_p + divQ_p
                noise_all[p, :] = np.sqrt(2.0 * self.step_size) * (chol_Q_p @ xi_all[p, :])

            particles += self.step_size * drift_all + noise_all

            if self.kl_track:
                log_p_values = [self.log_posterior(particles[p, :]) for p in range(n)]
                current_kl = self.kl_track.estimate_kl_particles(particles, log_p_values)
                kl_hist.append(current_kl)
            if k % save_step == 0:
                samples_history.append(particles.copy())
            if k % print_step == 0:
                print(f"Iteration {k+1:5d} complete.")
        return particles, samples_history, kl_hist