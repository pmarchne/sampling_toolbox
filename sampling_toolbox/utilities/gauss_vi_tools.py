import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

def mixture(x, means, Rs, logws):
    """Compute log mu(x) for the mixture."""
    log_pdf, _ = mixture_logpdf_and_grad(x, means, Rs, logws, compute_grad=False)
    return log_pdf


def mixture_grad(x, means, Rs, logws):
    """Compute ∇_x log mu(x) for the mixture."""
    _, grad = mixture_logpdf_and_grad(x, means, Rs, logws, compute_grad=True)
    return grad


def mixture_logpdf_and_grad(x, means, Rs, logws, compute_grad=True):
    """
    Computes log mu(x) and its gradient
    handles K=1, when mu is a single Gaussian
    """
    dim = x.shape[0]
    K = len(means)
    
    # Single Gaussian (K=1)
    if K == 1:
        y = solve_triangular(Rs[0], x - means[0], lower=True)
        log_norm = -0.5 * (dim * np.log(2 * np.pi)) - np.sum(np.log(np.diag(Rs[0])))
        log_pdf = log_norm - 0.5 * np.dot(y, y)
        
        if compute_grad:
            grad = -solve_triangular(Rs[0], y, lower=True, trans='T')
            return log_pdf, grad
        return log_pdf, None

    # General Case
    logps = np.empty(K)
    grads = np.empty((K, dim)) if compute_grad else None
    
    for k, (m, R) in enumerate(zip(means, Rs)):
        y = solve_triangular(R, x - m, lower=True)
        exponent = -0.5 * np.dot(y, y)
        log_norm = -0.5 * (dim * np.log(2 * np.pi)) - np.sum(np.log(np.diag(R)))
        logps[k] = log_norm + exponent
        
        if compute_grad:
            grads[k] = -solve_triangular(R, y, lower=True, trans='T')
    
    # Total log PDF using stable logsumexp directly with log-weights
    log_mix = logsumexp(logps + logws)
    
    if compute_grad:
        # Responsibilities (posterior probabilities in log-space)
        pis = np.exp(logps + logws - log_mix)
        # Blending gradients: \sum_k \gamma_k * \nabla \log p_k(x)
        grad_mix = np.dot(pis, grads)
        return log_mix, grad_mix
        
    return log_mix, None


def compute_increments_generic(means, Rs, logws, log_and_grad_post, mixture_grad_fn, precond='none', reg_lambda=1e2):
    """
    Unified factory to calculate Wasserstein geometry updates.
    Supports precond: 'none', 'natural', or 'hessian'
    """
    K = len(means)
    dim = means[0].shape[0]
    dms, dRs = [], []
    
    all_cub_points = []
    all_cub_logs = []

    for k in range(K):
        m, R = means[k], Rs[k]
        R_inv = solve_triangular(R, np.eye(dim), lower=True)
        dm = np.zeros(dim)
        M = np.zeros((dim, dim))
        c = np.sqrt(dim)
        alpha = 1.0 / (2 * dim)

        cub_cache = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            delta = R @ e # Step along the Cholesky geometry vectors
            
            for sign in [1, -1]:
                x_j = m + sign * c * delta
                # Evaluate the target log posterior density and gradient
                log_target, g_j = log_and_grad_post(x_j)
                
                cub_cache.append((sign, delta, x_j, g_j))
                all_cub_points.append(x_j)
                all_cub_logs.append(log_target)

        if precond == 'natural':
            Q = R @ R.T
        elif precond == 'hessian':
            J = np.zeros((dim,dim))
            for i in range(dim):
                g_pos = cub_cache[2*i][3]
                g_neg = cub_cache[2*i+1][3]
                J[:, i] = (g_pos - g_neg)/(2*c)
            H_app = J @ R_inv
            H_sym = -0.5*(H_app + H_app.T)
            A = H_sym + reg_lambda * np.eye(dim)
            Q = np.linalg.solve(A, np.eye(dim))
        else:
            Q = np.eye(dim)

        # --- compute trajectories
        for sign, delta, x_j, g_j in cub_cache:
            gt_warped = Q @ g_j
            if K == 1:
                dm += alpha * gt_warped
                #cross = np.outer(sign * delta, gt_warped)
                cross = np.outer(x_j - m, gt_warped)
                M += alpha * (cross + cross.T)
            else:
                gm = mixture_grad_fn(x_j, means, Rs, logws)
                gm_warped = Q @ gm 
                diff = gt_warped - gm_warped
                dm += alpha * diff
                #cross = np.outer(sign * delta, diff)
                cross = np.outer(x_j - m, diff)
                M += alpha * (cross + cross.T)

        if K == 1:
            M += 2.0 * Q
            
        A = R_inv @ M @ R_inv.T
        #A = 0.05*A
        # dR = R @ np.tril(R_inv @ M @ R_inv.T)
        Phi = np.tril(A, -1) + 0.5 * np.diag(np.diag(A))
        dR = R @ Phi
        dms.append(dm)
        dRs.append(dR)

    return dms, dRs, np.array(all_cub_points), np.array(all_cub_logs)


def compute_weight_increments(means, Rs, logws, cached_log_targets):
    """
    Computes the Fisher-Rao ODE increments (dlogws) for the component weights.
    Reuses cached log target evaluations to save computational cost.
    """
    K = len(means)
    dim = means[0].shape[0]
    
    # If it's a single Gaussian, weights don't change
    if K == 1:
        return np.array([0.0])
    
    Es = np.zeros(K)
    eval_idx = 0  # Track our exact linear position in the cached logs
    
    for k in range(K):
        m, R = means[k], Rs[k]
        Ei_m = 0.0
        Ei_t = 0.0
        c = np.sqrt(dim)
        alpha = 1.0 / (2 * dim)
        
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            delta = c * (R @ e)
            for sign in [1, -1]:
                x = m + sign * delta
                
                # Evaluate log q_mix(x) at the cubature point (computationally cheap)
                Ei_m += alpha * mixture(x, means, Rs, logws)
                
                # Fetch the zero-cost cached log \pi(x) from the position step
                Ei_t += alpha * cached_log_targets[eval_idx]
                eval_idx += 1
                
        Es[k] = Ei_m - Ei_t
            
    # Calculate the global expectation under the mixture: E_qmix[...]
    ws = np.exp(logws)
    Es_mean = np.sum(ws * Es)
    
    # Fisher-Rao update step
    dlogws = Es_mean - Es
    return -dlogws

def compute_Es_cached(means, Rs, logws, cached_log_targets):
    """
    Computes the component-specific expectations: 
    Es[k] = E_{q_k}[ log q_mix - log \pi ]
    """
    K = len(means)
    dim = means[0].shape[0]
    Es = np.zeros(K)
    eval_idx = 0 
    
    for k in range(K):
        m, R = means[k], Rs[k]
        Ei_m = 0.0
        Ei_t = 0.0
        c = np.sqrt(dim)
        alpha = 1.0 / (2 * dim)
        
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            delta = c * (R @ e)
            for sign in [1, -1]:
                x = m + sign * delta
                Ei_m += alpha * mixture(x, means, Rs, logws)
                Ei_t += alpha * cached_log_targets[eval_idx]
                eval_idx += 1
        Es[k] = Ei_m - Ei_t
    return Es