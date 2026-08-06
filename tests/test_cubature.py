import numpy as np
from scipy.linalg import solve_triangular
from sampling_toolbox.utilities.gauss_vi_tools import compute_increments_generic


def test_cubature_exactness_linear_gradient():
    """
    Tests that the cubature rule is exact for a Gaussian target
    with linear gradient.

    Target:
        pi(x) = N(0, I)

    Therefore:
        grad log pi(x) = -x
        log pi(x) = -0.5*x^T*x + constant
    """

    dim = 2

    def log_and_grad_target(x):
        logp = -0.5 * np.dot(x, x)
        grad = -x
        return logp, grad

    # Gaussian q = N(m, RR^T)
    m = np.array([1.5, -0.5])

    R = np.array([
        [1.2, 0.0],
        [0.4, 0.9]
    ])

    # Analytical expectation:
    # dm = E_q[-x] = -m
    expected_dm = -m

    cov = R @ R.T

    # For the Cholesky update:
    # M = E[(x-m)(grad)^T + grad(x-m)^T] + 2I
    expected_M = -2.0 * cov + 2.0 * np.eye(dim)

    R_inv = solve_triangular(
        R,
        np.eye(dim),
        lower=True
    )

    expected_dR = R @ np.tril(
        R_inv @ expected_M @ R_inv.T,
        -1
    )
    # Keep diagonal convention used by implementation
    expected_dR += (
        R @ (0.5 * np.diag(np.diag(
            R_inv @ expected_M @ R_inv.T
        )))
    )

    dms, dRs, cub_points, cub_logs = compute_increments_generic(
        means=[m],
        Rs=[R],
        logws=np.array([0.0]),
        log_and_grad_post=log_and_grad_target,
        mixture_grad_fn=None,
        precond='none'
    )

    dm = dms[0]
    dR = dRs[0]

    assert np.allclose(
        dm,
        expected_dm,
        rtol=1e-12,
        atol=1e-12
    )

    assert np.allclose(
        dR,
        expected_dR,
        rtol=1e-12,
        atol=1e-12
    )

    # Also check that cubature bookkeeping is produced
    assert cub_points.shape == (2 * dim, dim)
    assert cub_logs.shape == (2 * dim,)