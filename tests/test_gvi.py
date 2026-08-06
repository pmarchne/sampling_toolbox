import numpy as np

from sampling_toolbox import GaussianVI
from tests.targets import make_gaussian_target


def test_gaussian_vi_gaussian_target():

    (
        true_mean,
        true_cov,
        _,
        _,
        _,
        _,
    ) = make_gaussian_target()

    def log_and_grad_post(x):

        dx = x - true_mean
        precision = np.linalg.inv(true_cov)

        logp = -0.5 * dx @ precision @ dx
        grad = -precision @ dx

        return logp, grad

    sampler = GaussianVI(
        log_and_grad_post,
        step_size=0.02,
        n_iter=200,
        time_scheme="heun_adaptive"
    )

    # Start far away
    mu0 = [
        np.array([-8.0, 8.0])
    ]
    R0 = [
        np.eye(2)*3.0
    ]

    mu, R, weights, *_ = sampler.sample(mu0, R0)
    estimated_mean = mu[0]
    estimated_cov = R[0] @ R[0].T

    print(estimated_mean)
    print(true_mean)
    print("\n")
    print(estimated_cov)
    print(true_cov)
    assert np.allclose(
        estimated_mean,
        true_mean,
        atol=0.3
    )

    assert np.allclose(
        estimated_cov,
        true_cov,
        atol=0.5
    )