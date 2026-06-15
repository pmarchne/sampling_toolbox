import numpy as np
from scipy.linalg import solve_triangular

# --- STANDARD BFGS UPDATE ---
def apply_bfgs_update(W_base, s, y, dim):
    """
    Applies a rank-2 BFGS update to an inverse Hessian proxy.
    Returns the updated matrix.
    """
    s_dot_y = np.dot(s, y)
    # Curvature condition: only update if the secant contains valid convex information
    if s_dot_y > 1e-10 * np.dot(s, s):
        rho = 1.0 / s_dot_y
        I_mat = np.eye(dim)
        mat_L = I_mat - rho * np.outer(s, y)
        mat_R = I_mat - rho * np.outer(y, s)
        return mat_L @ W_base @ mat_R + rho * np.outer(s, s)
    else:
        # If flat or locally concave, fall back to the base matrix to ensure stability
        return W_base


def update_global_persistent_matrices(persistent_W_list, m_old, m_new, g_mean_old, g_mean_new):
    """
    Called only when the Heun ODE step is ACCEPTED.
    Updates the temporal macro-curvature using the step taken across the landscape.
    """
    K = len(persistent_W_list)
    dim = m_old[0].shape[0]
    updated_W_list = []
    
    for k in range(K):
        W_old = persistent_W_list[k]
        S_t = m_new[k] - m_old[k]          # Macro spatial step
        Y_t = g_mean_new[k] - g_mean_old[k] # Macro gradient step
        # Apply standard BFGS to track the center trajectory
        W_new = apply_bfgs_update(W_old, S_t, Y_t, dim)
        updated_W_list.append(W_new)
        
    return updated_W_list


def compute_increments_spatial_bfgs(means, Rs, grad_log_target, logws, persistent_W_list, method='cubature_spatial'):
    """
    Evaluates E[H^-1(x) * g(x)] using a global base + spatial secants.
    """
    K = len(means)
    dim = means[0].shape[0]
    dms, dRs, mean_gradients = [], [], []

    for k in range(K):
        m, R = means[k], Rs[k]
        R_inv = solve_triangular(R, np.eye(dim), lower=True)
        
        # 1. Get the global base matrix (approximate H^-1 at the mean)
        W_global = persistent_W_list[k]
        
        # 2. Evaluate the gradient at the center (Anchor point for spatial secants)
        g_mean = grad_log_target(m)
        mean_gradients.append(g_mean) # Store to return for the global macro-update
        
        dm = np.zeros(dim)
        M = np.zeros((dim, dim))
        
        c = np.sqrt(dim)
        alpha = 1.0 / (2 * dim) # Standard 2d cubature weights
        
        # === THE CUBATURE LOOP ===
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            delta = c * (R @ e)

            for sign in [1, -1]:
                x_j = m + sign * delta
                
                # Evaluate physics adjoint at the edge point
                g_j = grad_log_target(x_j)
                
                # 3. Define the spatial secant
                s_j = sign * delta          # x_j - m
                y_j = g_j - g_mean          # Change in gradient from center to edge
                
                # 4. Create the local inverse Hessian proxy for this specific point
                W_local = apply_bfgs_update(W_global, s_j, y_j, dim)
                
                # 5. Warp the coordinate exactly at the edge
                gt_warped = W_local @ g_j
                
                # Accumulate step vectors (handling K=1 vs K>1 mixture cases)
                if K == 1:
                    dm += alpha * gt_warped
                    cross = np.outer(sign * delta, gt_warped)
                    M += alpha * (cross + cross.T)
                else:
                    # Note: You may want to warp gm by W_local as well to keep the geometry consistent
                    gm = mixture_grad(x_j, means, Rs, logws)
                    gm_warped = W_local @ gm 
                    diff = gt_warped - gm_warped
                    dm += alpha * diff
                    cross = np.outer(sign * delta, diff)
                    M += alpha * (cross + cross.T)

        if K == 1:
            # We add the global proxy here to represent the baseline curvature volume expansion
            M += 2.0 * W_global
            
        # Standard covariance matrix increment formulation
        A = R_inv @ M @ R_inv.T
        Phi = np.tril(A, -1) + 0.5 * np.diag(np.diag(A))
        dR = R @ Phi
            
        dms.append(dm)
        dRs.append(dR)

    # Return increments AND the evaluated mean gradients needed for the global update later
    return dms, dRs, mean_gradients