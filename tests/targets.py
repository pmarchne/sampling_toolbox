import numpy as np


def make_gaussian_target():

    mean = np.array([3.0, -2.0])

    cov = np.array([
        [1.5, 0.8],
        [0.8, 2.0]
    ])

    precision = np.linalg.inv(cov)

    def log_likelihood(x):
        dx = x - mean
        return -0.5 * dx @ precision @ dx

    def log_prior(x):
        return 0.0

    def grad_log_likelihood(x):
        dx = x - mean
        return -precision @ dx

    def grad_log_prior(x):
        return np.zeros_like(x)

    return (
        mean,
        cov,
        log_likelihood,
        log_prior,
        grad_log_likelihood,
        grad_log_prior,
    )