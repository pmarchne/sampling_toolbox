import numpy as np
import matplotlib.pyplot as plt

# 1) Gaussian posterior (up to additive constant)
def gauss2d_log(x, alpha=5.):
    """
    Returns log p(x) ∝ -0.5 * x^T Σ^{-1} x,
    where Σ = diag(1, alpha).
    """
    y = np.asarray(x)
    inv_cov = np.diag([1.0, 1.0/alpha])
    return -0.5 * y.T.dot(inv_cov).dot(y)

# Gradient of the Gaussian log-density
def gauss2d_grad_log(x, alpha=5.):
    """
    ∇_x log p(x) = -Σ^{-1} x
    """
    y = np.asarray(x)
    inv_cov = np.diag([1.0, 1.0/alpha])
    return -inv_cov.dot(y)

# 2) Log-concave (convex) 2D posterior
def convave2d_log(alpha=0.5):
    """
    φ(x) = ((√α x0 − x1)^2 + x1^4) / 20
    Returns log p(x) ∝ −φ(x)
    """
    def f(x):
        x0, x1 = x
        phi = (np.sqrt(alpha)*x0 - x1)**2 + x1**4
        return -phi / 20.0
    return f

def convave2d_grad_log(alpha=0.5):
    """
    -∇_x φ(x) / 20
    """
    def grad_f(x):
        x0, x1 = x
        A = np.sqrt(alpha)*x0 - x1
        grad_x0 = 2.0*np.sqrt(alpha) * A
        grad_x1 = -2.0 * A + 4.0 * x1**3
        return -np.array([grad_x0, grad_x1]) / 20.0
    return grad_f

def convave2d_hessian_log(alpha=0.5):

    def H(x):
        x0, x1 = x

        H00 = -alpha/10.0
        H01 = np.sqrt(alpha)/10.0
        H11 = -1.0/10.0 - 3.0*x1**2/5.0

        return np.array([
            [H00, H01],
            [H01, H11]
        ])

    return H

def stationary_gaussian_cov(alpha=0.5):
    """
    Stationary Gaussian covariance for the 2D log-concave posterior.
    """
    v2 = np.sqrt(5.0 / 3.0)
    v12 = np.sqrt(15.0) / (3.0 * np.sqrt(alpha))
    v1 = (np.sqrt(15.0) + 30.0) / (3.0 * alpha)
    Cinv = np.array([
        [v1, v12],
        [v12, v2]
    ])
    return Cinv

# 3) Rosenbrock “banana” posterior (up to constant)
def rosenbrock2d_log(x, alpha=10.0):
    x = np.asarray(x)
    if x.ndim == 1:
        x0, x1 = x
        return (-(1 - x0)**2 - alpha * (x1 - x0**2)**2) / 20.0
    # grid case: x = (X, Y)
    X, Y = x
    return (-(1 - X)**2 - alpha * (Y - X**2)**2) / 20.0

def rosenbrock2d_grad_log(x, alpha=10.):
    """
    -∇_x φ(x):
      ∂φ/∂x0 = -2(1 - x0) - 4α x0 (x1 - x0^2)
      ∂φ/∂x1 =  2α (x1 - x0^2)
    """
    x0, x1 = x
    dphi_dx0 = -2.0*(1 - x0) - 4.0*alpha*x0*(x1 - x0**2)
    dphi_dx1 =  2.0*alpha*(x1 - x0**2)
    return -np.array([dphi_dx0, dphi_dx1])/20.

def rosenbrock2d_hessian_log(x, alpha=10.):
    """
    Returns the Hessian of the log-probability for the Rosenbrock function.
    H = -(1/20) * Hessian(phi)
    """
    x0, x1 = x
    # Second derivatives of phi
    # d2phi/dx0^2
    h00 = 2.0 - 4.0 * alpha * (x1 - 3.0 * x0**2)
    # d2phi/(dx0 dx1)
    h01 = -4.0 * alpha * x0
    # d2phi/dx1^2
    h11 = 2.0 * alpha

    # Assemble and scale by -1/20
    H = np.array([
        [h00, h01],
        [h01, h11]
    ])
    return -H / 20.0


def rosenbrock2d_hessian_gn_log(x, alpha=10.0):
    """
    Gauss-Newton Hessian of the log-probability.
    Always (negative) semidefinite; J^T J is PSD.
    """
    x0, x1 = x

    h00 = 1.0 + 4.0 * alpha * x0**2
    h01 = -2.0 * alpha * x0
    h11 = alpha

    H_gn = np.array([
        [h00, h01],
        [h01, h11]
    ])

    return -H_gn / 10.0

def rosenbrock2d_div_Q_gn_fast(x, alpha=10.0):
    """
    Ultra-cheap exact analytical divergence of Q(x) for the GN Hessian.
    Total cost: O(1) space and time.
    """
    # The divergence completely collapses to a constant vector!
    return np.array([0.0, 20.0])

def rosenbrock2d_div_Q(x, alpha=10.0):
    """
    Computes the analytical divergence of Q(x) = (-Hessian_log)^-1 
    for the 2D Rosenbrock function.
    """
    x0, x1 = x
    
    # 1. Compute the Hessian of Phi (scaled by 1/20)
    # This matches the core of your Hessian function before negation
    h00 = (2.0 - 4.0 * alpha * (x1 - 3.0 * x0**2)) / 20.0
    h01 = (-4.0 * alpha * x0) / 20.0
    h11 = (2.0 * alpha) / 20.0
    
    H_phi = np.array([[h00, h01], 
                      [h01, h11]])
    
    # 2. Compute Q(x) by inverting H_phi
    Q = np.linalg.inv(H_phi)
    
    # 3. Third derivatives (Derivatives of the Hessian matrix elements)
    # d(H_phi)/dx0
    dH_dx0 = np.array([
        [24.0 * alpha * x0 / 20.0, -4.0 * alpha / 20.0],
        [-4.0 * alpha / 20.0,      0.0]
    ])
    
    # d(H_phi)/dx1
    dH_dx1 = np.array([
        [-4.0 * alpha / 20.0, 0.0],
        [0.0,                 0.0]
    ])
    
    # 4. Compute derivatives of Q using the identity: dQ = -Q @ dH @ Q
    dQ_dx0 = -Q @ dH_dx0 @ Q
    dQ_dx1 = -Q @ dH_dx1 @ Q
    
    # 5. Extract divergence components:
    # div_0 = d(Q_00)/dx0 + d(Q_01)/dx1
    # div_1 = d(Q_10)/dx0 + d(Q_11)/dx1
    div_x0 = dQ_dx0[0, 0] + dQ_dx1[0, 1]
    div_x1 = dQ_dx0[1, 0] + dQ_dx1[1, 1]
    
    return np.array([div_x0, div_x1])

# ——— 4) Disk “ring” distribution ———
def disk2D_log(X, Y=None, sigma=0.5):
    """
    Log-probability for either a single 1D array/list or 2D meshgrids.
    
    If Y is None: X is assumed to be a single point or array of shape (2,) -> [x0, x1]
    If Y is provided: X and Y are assumed to be meshgrids of shape (M, N)
    """
    if Y is None:
        # Single input case: X = [x0, x1]
        x0, x1 = X[0], X[1]
    else:
        # Meshgrid case
        x0, x1 = X, Y
        
    r2 = x0**2 + x1**2
    phi = (1.0 - r2)**2 / (2.0 * sigma**2)
    return -phi

def disk2D_grad_log(X, Y=None, sigma=0.5):
    """
    Gradient of log p(x) for either a single 1D array or 2D meshgrids.
    
    Returns:
        If Y is None: A 1D numpy array [grad_x0, grad_x1]
        If Y is provided: A tuple of 2D arrays (grad_X, grad_Y)
    """
    if Y is None:
        # Single input case: X = [x0, x1]
        x0, x1 = X[0], X[1]
    else:
        # Meshgrid case
        x0, x1 = X, Y
        
    r2 = x0**2 + x1**2
    scalar_multiplier = 2.0 * (1.0 - r2) / (sigma**2)
    
    grad_x0 = scalar_multiplier * x0
    grad_y0 = scalar_multiplier * x1
    
    if Y is None:
        return np.array([grad_x0, grad_y0])
    else:
        return grad_x0, grad_y0

# ——— 5) 2D distribution with four Gaussian-like modes ———
def four_mode2D_log(x, eta=2.5, y=np.array([5.0, 5.0])):
    """
    g(x) = [ (x0 - x1)^2, (x0 + x1)^2 ]
    φ(x) = || y - g(x) ||^2 / (2 η)
    log p(x) ∝ −φ(x)
    """
    x0, x1 = x
    g = np.array([ (x0 - x1)**2, (x0 + x1)**2 ])
    diff = y - g
    phi = diff.dot(diff) / (2.0 * eta)
    return -phi

def four_mode2D_grad_log(x, eta=2.5, y=np.array([5.0, 5.0])):
    """
    Jacobian J_ij = ∂ g_i / ∂ x_j:
      ∂g1/∂x0 =  2 (x0 - x1),  ∂g1/∂x1 = -2 (x0 - x1)
      ∂g2/∂x0 =  2 (x0 + x1),  ∂g2/∂x1 =  2 (x0 + x1)
    ∇ φ = (J^T (g(x) - y)) / η
    ∇ log p = −∇ φ
    """
    x0, x1 = x
    g1 = (x0 - x1)**2
    g2 = (x0 + x1)**2
    diff = np.array([g1, g2]) - y

    # build J
    J = np.array([
        [ 2.0*(x0 - x1),  -2.0*(x0 - x1) ],
        [ 2.0*(x0 + x1),   2.0*(x0 + x1) ]
    ])  # shape (2,2)

    grad_phi = J.T.dot(diff) / eta
    return -grad_phi

# 6) double “banana” posterior
def double_banana2d_log(x, eta=1.):
    """
    Returns log p(x) ∝ −φ(x)
    """
    x0, x1 = x
    g1 = np.log( (100. * (x1 - x0**2)**2 + (1. - x0)**2)/0.3 )
    g2 = x0
    g3 = x1
    y = np.array([np.log(101), 0. , 0.])
    diff = y - np.array([g1, g2, g3])
    phi = diff.dot(diff) / (2.0 * eta)
    return -phi

def double_banana2d_grad_log(x, eta=1.0):
    """
    Gradient of log p(x) = -∇_x φ(x)

    φ(x) = ((g1 - log(101))^2 + x0^2 + x1^2) / (2 η)
    where g1 = log((100 (x1 - x0^2)^2 + (1 - x0)^2) / 0.3)

    Returns:
      array([∂ log p/∂x0, ∂ log p/∂x1])
    """
    x0, x1 = x
    # intermediate terms
    u = x1 - x0**2
    v = 1.0 - x0
    A = 100.0 * u**2 + v**2
    # g1 and constant
    g1 = np.log(A / 0.3)
    c = np.log(101.0)
    # derivatives of g1
    dg1_dx0 = (1.0 / A) * (-400.0 * x0 * u - 2.0 * v)
    dg1_dx1 = (1.0 / A) * (200.0 * u)
    # components of diff = [g1 - c, x0, x1]
    d1 = g1 - c
    # gradient of φ: ∂φ/∂xi = [(d1 * ∂g1/∂xi) + xi] / η
    dphi_dx0 = (d1 * dg1_dx0 + x0) / eta
    dphi_dx1 = (d1 * dg1_dx1 + x1) / eta
    # gradient of log p = -∇φ
    return -np.array([dphi_dx0, dphi_dx1])

def main():
    grid_size = 200
    x_vals = np.linspace(-3, 3, grid_size)
    y_vals = np.linspace(-3, 3, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    positions = np.stack([X.ravel(), Y.ravel()]).T

    # List of log-posterior functions
    funcs = [
        ("Gaussian", gauss2d_log),
        ("Convex", convave2d_log),
        ("Rosenbrock", rosenbrock2d_log),
        ("Disk", disk2D_log),
        ("Four-Mode", four_mode2D_log),
        ("Double-banana", double_banana2d_log)
    ]

    # Compute densities and plot
    _, axes = plt.subplots(1, 6, figsize=(25, 5))
    for ax, (title, func) in zip(axes, funcs):
        # Evaluate log-posterior on the grid
        log_vals = np.array([func(pos) for pos in positions])
        density = np.exp(log_vals - np.max(log_vals))  # normalize for visualization
        Z = density.reshape(grid_size, grid_size)
        
        # Plot
        ax.contourf(X, Y, Z, levels=10)
        ax.contour(X, Y, Z, levels=10, colors="black", linewidths=0.8, linestyles="dotted")
        #ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")
        ax.set_title(f"{title} Posterior")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()