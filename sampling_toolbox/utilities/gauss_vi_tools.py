import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sampling_toolbox.examples.benchmarks_2D import rosenbrock2d_hessian_gn_log

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
            # Solve R^T @ g = -y -> g = -R^{-T} @ y
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


def compute_increments(means, Rs, grad_log_target, logws, method='cubature', ns=100, ll=0.5):
    """
    Computes ODE increments dm and dR for GMM.
    Methods: 'cubature', 'mc', 'linearization', 'cubature_hess' (Free GN Curvature Extraction).
    """
    K = len(means)
    dim = means[0].shape[0]
    dms, dRs = [], []

    for k in range(K):
        m, R = means[k], Rs[k]
        R_inv = solve_triangular(R, np.eye(dim), lower=True)
        
        dm = np.zeros(dim)
        M = np.zeros((dim, dim))

        if method == 'linearization':
            # Pre-existing analytical GN Hessian path
            hess = rosenbrock2d_hessian_gn_log(m)
            blended_matrix = (1.0 - ll) * (-hess) + ll * np.eye(dim)
            hess_inv = np.linalg.inv(blended_matrix)
            
            gt = grad_log_target(m)
            if K == 1:
                dm = hess_inv @ gt
                M = 2.0 * hess_inv 
            else:
                gm = mixture_grad(m, means, Rs, logws)
                dm = hess_inv @ (gt - gm)
        elif method == 'cubature':
            # Legacy method utilizing analytical rosenbrock2d_hessian_gn_log
            #hess = rosenbrock2d_hessian_gn_log(m)
            #blended_matrix = (1.0 - ll) * (-hess) + ll * np.eye(dim)
            #hess_inv = np.linalg.inv(blended_matrix)
            hess_inv = np.eye(dim)
            c = np.sqrt(dim)
            alpha = 1.0 / (2 * dim)
            for i in range(dim):
                e = np.zeros(dim)
                e[i] = 1.0
                delta = c * (R @ e)
                    
                for sign in [1, -1]:
                    x = m + sign * delta 
                    gt = hess_inv @ grad_log_target(x)
                    if K == 1:
                        dm += alpha * gt
                        cross = np.outer(sign * delta, gt)
                        M += alpha * (cross + cross.T)
                    else:
                        gm = hess_inv @ mixture_grad(x, means, Rs, logws)
                        diff = gt - gm
                        dm += alpha * diff
                        cross = np.outer(sign * delta, diff)
                        M += alpha * (cross + cross.T)
            if K == 1:
                M += 2.0 * hess_inv
        elif method == 'cubature_hess':
            c     = np.sqrt(dim)
            alpha = 1.0 / (2 * dim)
            hess_inv_avg = np.zeros((dim, dim))
            for i in range(dim):
                e     = np.zeros(dim)
                e[i]  = 1.0
                delta = c * (R @ e)
                for sign in [1, -1]:
                    x             = m + sign * delta
                    hess_x        = rosenbrock2d_hessian_gn_log(x)
                    blended_x     = -hess_x
                    hess_inv_avg += alpha * np.linalg.inv(blended_x)
                    
            # --- Pass 2: cubature integrals with the averaged preconditioner ---
            for i in range(dim):
                e     = np.zeros(dim)
                e[i]  = 1.0
                delta = c * (R @ e)

                for sign in [1, -1]:
                    x  = m + sign * delta
                    gt = hess_inv_avg @ grad_log_target(x)

                    if K == 1:
                        dm   += alpha * gt
                        cross = np.outer(sign * delta, gt)
                        M    += alpha * (cross + cross.T)
                    else:
                        gm   = hess_inv_avg @ mixture_grad(x, means, Rs, logws)
                        diff = gt - gm
                        dm  += alpha * diff
                        cross = np.outer(sign * delta, diff)
                        M    += alpha * (cross + cross.T)

            if K == 1:
                M += 2.0 * hess_inv_avg
        elif method == 'mc':
            Z = np.random.randn(ns, dim)
            for z in Z:
                delta = R @ z
                x = m + delta
                hess = rosenbrock2d_hessian_gn_log(x)
                blended_matrix = (1.0 - ll) * (-hess) + ll * np.eye(dim)
                hess_inv = np.linalg.inv(blended_matrix)
                gt = hess_inv @ grad_log_target(x)
                    
                if K == 1:
                    dm += gt
                    cross = np.outer(delta, gt)
                    M += cross + cross.T
                else:
                    gm = hess_inv @ mixture_grad(x, means, Rs, logws)
                    diff = gt - gm
                    dm += diff
                    cross = np.outer(delta, diff)
                    M += cross + cross.T
                        
            dm /= ns
            M /= ns
            if K == 1:
                hess = rosenbrock2d_hessian_gn_log(m)
                blended_matrix = (1.0 - ll) * (-hess) + ll * np.eye(dim)
                hess_inv = np.linalg.inv(blended_matrix)
                M += 2.0 * hess_inv#np.eye(dim)
        else:
            raise ValueError("method must be 'cubature', 'mc', 'linearization', or 'cubature_hess'")

        # Covariance increment update via Cholesky factor matrix projection
        A = R_inv @ M @ R_inv.T
        Phi = np.tril(A, -1) + 0.5 * np.diag(np.diag(A))
        dR = R @ Phi
        #dR = R @ np.tril(R_inv @ M @ R_inv.T)
            
        dms.append(dm)
        dRs.append(dR)

    return dms, dRs


def compute_weight_increments(means, Rs, logws, log_target, method='cubature', ns=100):
    """
    Computes the Fisher-Rao ODE increments (dlogws) for the component weights.
    d(log w_k)/dt = E_qmix[log qmix - log π] - E_qk[log qmix - log π]
    
    Methods: 'cubature', 'mc', or 'linearization'.
    """
    K = len(means)
    dim = means[0].shape[0]
    
    # If it's a single Gaussian, weights don't change
    if K == 1:
        return np.array([0.0])
    
    # Es holds the component-specific expectations: E_qk[ log qmix(x) - log π(x) ]
    Es = np.zeros(K)
    
    for k in range(K):
        m, R = means[k], Rs[k]
        Ei_m = 0.0
        Ei_t = 0.0
        if method == 'linearization':
            # 1-point evaluation at the mean
            Ei_m = mixture(m, means, Rs, logws) # log qmix(m)
            Ei_t = log_target(m)                            # log π(m)
            Es[k] = Ei_m - Ei_t
            
        elif method == 'cubature' or method == 'cubature_hess':
            c = np.sqrt(dim)
            alpha = 1.0 / (2 * dim)
            for i in range(dim):
                e = np.zeros(dim)
                e[i] = 1.0
                delta = c * (R @ e)
                for sign in [1, -1]:
                    x = m + sign * delta
                    Ei_m += alpha * mixture(x, means, Rs, logws)
                    Ei_t += alpha * log_target(x)
            Es[k] = Ei_m - Ei_t
            
        elif method == 'mc':
            Z = np.random.randn(ns, dim)
            for z in Z:
                x = m + (R @ z)
                Ei_m += mixture(x, means, Rs, logws)
                Ei_t += log_target(x)
            Es[k] = (Ei_m - Ei_t) / ns
            
        else:
            raise ValueError("method must be 'cubature', 'mc', or 'linearization'")
            
    # Calculate the global expectation under the mixture: E_qmix[...]
    ws = np.exp(logws)
    Es_mean = np.sum(ws * Es)
    
    # Fisher-Rao update step
    dlogws = Es_mean - Es
    return dlogws