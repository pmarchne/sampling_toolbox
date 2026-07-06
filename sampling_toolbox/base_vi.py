import time
from abc import ABC, abstractmethod 
from tabulate import tabulate
import numpy as np


class VI(ABC):
    ''' 
    base class for Variational inference sampling algorithms 
    '''
    def __init__(self, log_and_grad_post, step_size=0.1, rng=None):
        self.log_and_grad_post = log_and_grad_post
        self.step_size = step_size
        self.grad_calls = 0
        self.log_calls = 0
        self.rng = rng or np.random.default_rng()

    
    def log_and_grad_posterior(self, x):
        ''' returns log and grad of posterior pdf as a tuple (log,grad) '''
        self.log_calls += 1
        self.grad_calls += 1
        return self.log_and_grad_post(x)

    @abstractmethod
    def _sample(self, x0, num_samples):
        """Abstract method that must be implemented by subclasses."""

    def sample(self, x0, num_samples):
        ''' implement the sampling algorithm and time it '''
        if not callable(getattr(self, '_sample', None)):
            raise NotImplementedError("Subclasses must implement the '_sample' method.")

        start_time = time.time()
        result = self._sample(x0, num_samples)
        elapsed_time = time.time() - start_time
        print(f"sampling took {elapsed_time:.3f} seconds.")
        return result

    def report_calls(self):
        ''' report number of function calls '''
        print(f"Log evaluations: {self.log_calls}")
        print(f"Gradient log evaluations: {self.grad_calls}")

    def print_statistics(self, samples, burn_in=0):
        ''' print a table with mean and stds of all parameters '''
        stats = []
        for i in range(samples.shape[1]):
            mean = np.mean(samples[burn_in:, i])
            std = np.std(samples[burn_in:, i])
            stats.append([f"vp {i+1}", mean, std])

        # Print table
        headers = ["Parameter", "Mean", "Std Dev"]
        print(tabulate(stats, headers=headers, tablefmt="grid"))