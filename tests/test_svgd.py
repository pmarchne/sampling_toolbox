import numpy as np

from sampling_toolbox import SVGD
from tests.targets import make_gaussian_target


def test_svgd():

    (
        true_mean,
        true_cov,
        log_likelihood,
        log_prior,
        grad_log_likelihood,
        grad_log_prior,
    ) = make_gaussian_target()

    rng = np.random.default_rng(123)

    n_particles = 200

    # deliberately bad initialization
    x0 = rng.normal(
        loc=-8.0,
        scale=1.0,
        size=(n_particles, 2)
    )

    sampler = SVGD(
        log_likelihood,
        log_prior,
        grad_log_likelihood,
        grad_log_prior,
        step_size=0.15,
        n_iter=200,
        rng=rng,
    )

    particles, _, _ = sampler._sample(x0)
    estimated_mean = np.mean(particles, axis=0)
    estimated_cov = np.cov(particles.T)
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