import time
import numpy as np
from abc import ABC, abstractmethod 
from tabulate import tabulate

class ParticleMethod(ABC):
    '''
    Abstract base for particle based samplers.

    Parameters
    ----------
    log_likelihood : callable
        Computes the log-likelihood at x.
    log_prior : callable
        Computes the log-prior at x.
    grad_log_likelihood : callable
        Computes ∇ log-likelihood.
    grad_log_prior : callable
        Computes ∇ log-prior.
    step_size : float
        Numerical step size for proposals.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.
    '''
    def __init__(self, log_likelihood=None, log_prior=None,
                 grad_log_likelihood=None, grad_log_prior=None,
                 log_and_grad_post=None,
                 step_size=1e-1, rng=None):
        self.log_likelihood = log_likelihood
        self.log_prior      = log_prior
        self.grad_log_likelihood = grad_log_likelihood
        self.grad_log_prior      = grad_log_prior
        self.log_and_grad_post = log_and_grad_post
        self.step_size = step_size
        self.rng = rng or np.random.default_rng()

        # Diagnostic counters
        self.log_calls = 0
        self.grad_calls = 0
        self.last_sample_time = None

    def log_posterior(self, x):
        ''' Evaluate the log-posterior at x and increment call counter.'''
        self.log_calls += 1
        return self.log_likelihood(x) + self.log_prior(x)

    def grad_log_posterior(self, x):
        ''' Evaluate the gradient of the log-posterior at x and increment counter.'''
        if self.grad_log_likelihood is None or self.grad_log_prior is None:
            raise NotImplementedError("Gradients not provided.")
        self.grad_calls += 1
        return self.grad_log_likelihood(x) + self.grad_log_prior(x)
    
    def log_and_grad_posterior(self, x):
        """
        Evaluate log pi(x) and grad log pi(x) simultaneously
        """
        if self.log_and_grad_post is None:
            raise NotImplementedError("log_and_grad_post was not provided.")
        self.log_calls += 1
        self.grad_calls += 1
        return self.log_and_grad_post(x)

    @abstractmethod
    def _sample(self, x0: np.ndarray):
        pass

    def sample(self, x0):
        ''' run the chain and record timing '''
        start = time.time()
        samples, history, diagnostics = self._sample(np.asarray(x0))
        elapsed = time.time() - start
        self.last_sample_time = elapsed
        print(f"Sampling completed in {elapsed:.3f} seconds.")
        return samples, history, diagnostics
    
    def report_calls(self):
        ''' report number of function calls '''
        print(f"Log evaluations: {self.log_calls}")
        print(f"Gradient log evaluations: {self.grad_calls}")

    def print_statistics(self, samples, param_names=None):
        ''' print chain statistics '''
        n_params = samples.shape[1]
        if param_names is None:
            param_names = [f"param_{i+1}" for i in range(n_params)]
        stats = []
        for i, name in enumerate(param_names):
            mean = np.mean(samples[:, i])
            std = np.std(samples[:, i])
            stats.append([name, mean, std])
        print(tabulate(stats, headers=["Parameter", "Mean", "Std Dev"], tablefmt="grid"))

    def autocorrelation(self, samples, param_index, max_lag=None):
        ''' computes autocorrelation for a given parameter index '''
        n = samples.shape[0]
        param_samples = samples[:, param_index]
        if max_lag is None:
            max_lag = n - 1
        mean = np.mean(param_samples)
        var = np.var(param_samples)
        autocorr = np.correlate(param_samples - mean,
                                param_samples - mean, mode='full') / (var * n)
        autocorr = autocorr[n-1:n+max_lag]  # Extract only positive lags
        lags = np.arange(0, max_lag + 1)
        return lags, autocorr

    def effective_sample_size(self, samples):
        '''
        Compute Effective Sample Size (ESS) for each parameter.

        ESS per parameter = N / (1 + 2 * sum_{k=1 to K} rho_k),
        where K is max lag until autocorrelation becomes non-positive.
        '''
        samples = np.asarray(samples)
        n, d = samples.shape
        ess = np.empty(d)
        for i in range(d):
            # compute autocorrelation
            _, ac = self.autocorrelation(samples, i)
            # find lag where ac becomes negative
            positive_ac = ac[1:]
            k_max = np.where(positive_ac <= 0)[0]
            if k_max.size > 0:
                max_lag = k_max[0]
            else:
                max_lag = len(positive_ac)
            # sum only up to max_lag
            tau = 1 + 2 * np.sum(positive_ac[:max_lag])
            ess[i] = n / tau
        return ess
