import pytest
import numpy as np
from scipy.stats import multivariate_normal
from sampling_toolbox.utilities.gauss_vi_tools import mixture, mixture_grad, mixture_logpdf_and_grad


def generate_random_gmm(K, dim):
    """Helper to generate valid random GMM parameters."""
    np.random.seed(42)
    means = [np.random.uniform(-2, 2, size=dim) for _ in range(K)]
    Rs = []
    covs = []
    for _ in range(K):
        # Generate random positive-definite covariance matrix via lower Cholesky
        A = np.random.uniform(0.5, 1.5, size=(dim, dim))
        L = np.tril(A)
        np.fill_diagonal(L, np.abs(np.diag(L)) + 0.5) # Keep away from zero
        Rs.append(L)
        covs.append(L @ L.T)
        
    # Normalized log weights
    w = np.random.uniform(0.1, 1.0, size=K)
    w /= np.sum(w)
    logws = np.log(w)
    
    return means, Rs, covs, logws

@pytest.mark.parametrize("dim", [2, 5])
@pytest.mark.parametrize("K", [1, 3]) # Tests both single and multi-component conditions
def test_log_pdf_against_scipy(dim, K):
    means, Rs, covs, logws = generate_random_gmm(K, dim)
    x = np.random.uniform(-1, 1, size=dim)
    
    computed_logpdf = mixture(x, means, Rs, logws)
    
    # Ground truth calculation using SciPy standard library
    weights = np.exp(logws)
    scipy_pdf = 0.0
    for w, m, cov in zip(weights, means, covs):
        scipy_pdf += w * multivariate_normal.pdf(x, mean=m, cov=cov)
    expected_logpdf = np.log(scipy_pdf)
    
    assert np.isclose(computed_logpdf, expected_logpdf, rtol=1e-9, atol=1e-9)

@pytest.mark.parametrize("dim", [2, 4])
@pytest.mark.parametrize("K", [1, 3])
def test_gradients_via_finite_differences(dim, K):
    means, Rs, _, logws = generate_random_gmm(K, dim)
    x = np.random.uniform(-1, 1, size=dim)
    
    analytical_grad = mixture_grad(x, means, Rs, logws)
    
    # Numerical gradient calculation using central finite differences
    eps = 1e-6
    numerical_grad = np.zeros(dim)
    
    for i in range(dim):
        x_plus = x.copy()
        x_plus[i] += eps
        log_p_plus = mixture(x_plus, means, Rs, logws)

        x_minus = x.copy()
        x_minus[i] -= eps
        log_p_minus = mixture(x_minus, means, Rs, logws)

        numerical_grad[i] = (log_p_plus - log_p_minus) / (2 * eps)

    assert np.allclose(analytical_grad, numerical_grad, rtol=1e-5, atol=1e-5)

def test_joint_and_separate_match():
    """Ensures separate methods return identical values to the unified call."""
    means, Rs, _, logws = generate_random_gmm(K=3, dim=3)
    x = np.array([0.1, -0.5, 0.2])

    log_pdf_sep = mixture(x, means, Rs, logws)
    grad_sep = mixture_grad(x, means, Rs, logws)

    log_pdf_joint, grad_joint = mixture_logpdf_and_grad(x, means, Rs, logws)

    assert np.isclose(log_pdf_sep, log_pdf_joint)
    assert np.allclose(grad_sep, grad_joint)