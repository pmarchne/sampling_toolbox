import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# UNIFORMIZED 2D BENCHMARKS
# All signatures accept x = [x0, x1] OR x = (X, Y)
# =====================================================================

# 1) Gaussian posterior
def gauss2d_log(x, alpha=5., angle_deg=30.):
    """
    Returns log p(x) for a 2D Gaussian rotated counter-clockwise
    """
    x0, x1 = x
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    # Transform coordinates to the unaligned alignment frame (-theta)
    x0_rot =  x0 * c + x1 * s
    x1_rot = -x0 * s + x1 * c
    return -0.5 * (x0_rot**2 + (x1_rot**2 / alpha))

def gauss2d_grad_log(x, alpha=5., angle_deg=30.):
    """
    ∇_x log p(x) for a 2D Gaussian rotated counter-clockwise.
    """
    x0, x1 = x
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    # Transform coordinates to the unaligned alignment frame
    x0_rot =  x0 * c + x1 * s
    x1_rot = -x0 * s + x1 * c
    # Calculate gradients inside the unaligned frame
    g0_rot = -x0_rot
    g1_rot = -x1_rot / alpha
    # Rotate the gradient vector back to the base spatial frame (+theta)
    g0 = g0_rot * c - g1_rot * s
    g1 = g0_rot * s + g1_rot * c
    return np.array([g0, g1])


# 2) Log-concave (convex) 2D posterior
def convave2d_log(x, alpha=0.5):
    """
    φ(x) = ((√α x0 − x1)^2 + x1^4) / 20
    Returns log p(x) ∝ −φ(x)
    """
    x0, x1 = x
    phi = (np.sqrt(alpha) * x0 - x1)**2 + x1**4
    return -phi / 20.0

def convave2d_grad_log(x, alpha=0.5):
    """
    -∇_x φ(x) / 20
    """
    x0, x1 = x
    A = np.sqrt(alpha) * x0 - x1
    grad_x0 = 2.0 * np.sqrt(alpha) * A
    grad_x1 = -2.0 * A + 4.0 * x1**3
    return -np.array([grad_x0, grad_x1]) / 20.0


# 3) Rosenbrock “banana” posterior
def rosenbrock2d_log(x, alpha=10.0):
    x0, x1 = x
    return (-(1.0 - x0)**2 - alpha * (x1 - x0**2)**2) / 20.0

def rosenbrock2d_grad_log(x, alpha=10.):
    """
    -∇_x φ(x)
    """
    x0, x1 = x
    dphi_dx0 = -2.0 * (1.0 - x0) - 4.0 * alpha * x0 * (x1 - x0**2)
    dphi_dx1 = 2.0 * alpha * (x1 - x0**2)
    return -np.array([dphi_dx0, dphi_dx1]) / 20.0


# 4) Disk “ring” distribution
def disk2D_log(x, sigma=0.5):
    """
    Log-probability for either a single 1D array or 2D meshgrids.
    """
    x0, x1 = x
    r2 = x0**2 + x1**2
    phi = (1.0 - r2)**2 / (2.0 * sigma**2)
    return -phi

def disk2D_grad_log(x, sigma=0.5):
    """
    Gradient of log p(x) for either a single 1D array or 2D meshgrids.
    """
    x0, x1 = x
    r2 = x0**2 + x1**2
    scalar_multiplier = 2.0 * (1.0 - r2) / (sigma**2)
    
    grad_x0 = scalar_multiplier * x0
    grad_x1 = scalar_multiplier * x1
    return np.array([grad_x0, grad_x1])


# 5) 2D distribution with four Gaussian-like modes
def four_mode2D_log(x, eta=2.5, y=np.array([5.0, 5.0])):
    """
    g(x) = [ (x0 - x1)^2, (x0 + x1)^2 ]
    φ(x) = || y - g(x) ||^2 / (2 η)
    """
    x0, x1 = x
    g1 = (x0 - x1)**2
    g2 = (x0 + x1)**2
    phi = ((y[0] - g1)**2 + (y[1] - g2)**2) / (2.0 * eta)
    return -phi

def four_mode2D_grad_log(x, eta=2.5, y=np.array([5.0, 5.0])):
    """
    ∇ log p = −∇ φ
    """
    x0, x1 = x
    g1 = (x0 - x1)**2
    g2 = (x0 + x1)**2
    diff1 = g1 - y[0]
    diff2 = g2 - y[1]

    # Explicit element-wise Jacobian-transposed multiplication
    grad_phi_x0 = (2.0 * (x0 - x1) * diff1 + 2.0 * (x0 + x1) * diff2) / eta
    grad_phi_x1 = (-2.0 * (x0 - x1) * diff1 + 2.0 * (x0 + x1) * diff2) / eta
    return -np.array([grad_phi_x0, grad_phi_x1])

def four_mode2D_asymmetric_log(x, eta=2.5, y=np.array([5.0, 5.0]), bias=np.array([0.4, 0.2])):
    """
    Symmetric 4-mode potential modified with a directional linear tilt 
    to assign unequal weights to different modes.
    
    Parameters:
        bias: [A, B] controlling the weight redistribution along x0 and x1 axes.
    """
    x0, x1 = x
    g1 = (x0 - x1)**2
    g2 = (x0 + x1)**2
    phi = ((y[0] - g1)**2 + (y[1] - g2)**2) / (2.0 * eta)
    
    # Linear bias scales mode heights: log p_new(x) = log p_old(x) + A*x0 + B*x1
    log_tilt = bias[0] * x0 + bias[1] * x1
    return -phi + log_tilt

def four_mode2D_asymmetric_grad_log(x, eta=2.5, y=np.array([5.0, 5.0]), bias=np.array([0.4, 0.2])):
    """
    ∇ log p_asymmetric = ∇(-phi) + bias
    """
    x0, x1 = x
    g1 = (x0 - x1)**2
    g2 = (x0 + x1)**2
    diff1 = g1 - y[0]
    diff2 = g2 - y[1]

    # Original spatial gradients
    grad_phi_x0 = (2.0 * (x0 - x1) * diff1 + 2.0 * (x0 + x1) * diff2) / eta
    grad_phi_x1 = (-2.0 * (x0 - x1) * diff1 + 2.0 * (x0 + x1) * diff2) / eta
    
    # Adding the constant gradient of our linear weight tilt
    return -np.array([grad_phi_x0, grad_phi_x1]) + bias

# 6) Double “banana” posterior
def double_banana2d_log(x, eta=1.):
    x0, x1 = x
    g1 = np.log((100.0 * (x1 - x0**2)**2 + (1.0 - x0)**2) / 0.3)
    phi = ((np.log(101.0) - g1)**2 + x0**2 + x1**2) / (2.0 * eta)
    return -phi

def double_banana2d_grad_log(x, eta=1.0):
    """
    Gradient of log p(x) = -∇_x φ(x)
    """
    x0, x1 = x
    u = x1 - x0**2
    v = 1.0 - x0
    A = 100.0 * u**2 + v**2
    
    g1 = np.log(A / 0.3)
    c = np.log(101.0)
    
    dg1_dx0 = (1.0 / A) * (-400.0 * x0 * u - 2.0 * v)
    dg1_dx1 = (1.0 / A) * (200.0 * u)
    
    d1 = g1 - c
    dphi_dx0 = (d1 * dg1_dx0 + x0) / eta
    dphi_dx1 = (d1 * dg1_dx1 + x1) / eta
    return -np.array([dphi_dx0, dphi_dx1])


def main():
    grid_size = 500  # Increased for much smoother contour definitions
    x_vals = np.linspace(-3, 3, grid_size)
    y_vals = np.linspace(-3, 3, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Dictionary listing target names alongside their log-posterior functions
    funcs = [
        ("Gaussian", gauss2d_log),
        ("Convex", convave2d_log),
        ("Rosenbrock", rosenbrock2d_log),
        ("Disk", disk2D_log),
        ("Four-Mode", four_mode2D_log),
        ("Double-Banana", double_banana2d_log)
    ]

    _, axes = plt.subplots(1, 6, figsize=(25, 5))
    
    for ax, (title, func) in zip(axes, funcs):
        # FAST EVALUATION: Pass the whole grid tuple (X, Y) directly to the function
        log_vals = func((X, Y))
        
        # Safe normalization transformation to get un-scaled density fields
        density = np.exp(log_vals - np.max(log_vals)) 
        
        # Render fields
        ax.contourf(X, Y, density, levels=10, cmap='viridis')
        ax.contour(X, Y, density, levels=10, colors="black", linewidths=0.5, linestyles="dotted")
        
        ax.set_title(f"{title} Posterior")
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()