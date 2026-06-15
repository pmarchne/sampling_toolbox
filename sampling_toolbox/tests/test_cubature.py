import pytest
import numpy as np
from scipy.linalg import solve_triangular
from sampling_toolbox.utilities.gauss_vi_tools import compute_increments

def test_cubature_exactness_linear_gradient():
    """
    Tests that the cubature rule is exact for a target with a linear gradient.
    Target: N(0, I) -> grad_log_target(x) = -x
    """
    dim = 2
    
    # Target gradient is exactly -x
    def grad_log_target(x):
        return -x

    # Initialize a single arbitrary Gaussian Q = N(m, R R^T)
    m = np.array([1.5, -0.5])
    R = np.array([[1.2, 0.0], 
                  [0.4, 0.9]])
    
    # --- Analytical Expectations ---
    # dm = E[-x] = -m
    expected_dm = -m
    
    # For K=1, M_exp = E[(x-m)(-x)^T + (-x)(x-m)^T]
    # Let y = x-m. Then x = y+m.
    # M_exp = E[ y(-y-m)^T + (-y-m)y^T ] = -E[yy^T] - E[yy^T] = -2 \Sigma
    cov = R @ R.T
    expected_M_exp = -2.0 * cov
    expected_M = expected_M_exp + 2.0 * np.eye(dim)
    
    R_inv = solve_triangular(R, np.eye(dim), lower=True)
    expected_dR = R @ np.tril(R_inv @ expected_M @ R_inv.T)

    # --- Cubature Computation ---
    dm, dR = compute_increments(
        means=[m], 
        Rs=[R], 
        grad_log_target=grad_log_target,
        logws=[0.0], 
        method='cubature'
    )

    # Asserts
    assert np.allclose(dm, expected_dm, rtol=1e-12, atol=1e-12), \
        f"Mean increment failed. Expected {expected_dm}, got {dm}"
        
    assert np.allclose(dR, expected_dR, rtol=1e-12, atol=1e-12), \
        f"Covariance increment failed. Expected {expected_dR}, got {dR}"