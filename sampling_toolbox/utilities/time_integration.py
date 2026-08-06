import numpy as np

# --- Euler Steps ---
def euler_step_pos(mu, R, logws, get_pos_increments, step_size):
    dmu, dR = get_pos_increments(mu, R, logws, update_cache=True)
    for m, r, dm, dr in zip(mu, R, dmu, dR):
        m += step_size * dm
        r += step_size * dr
    return mu, R


def euler_step_w(mu, R, logws, get_w_increments, step_size):
    dlogws = get_w_increments(mu, R, logws)
    logws += step_size * dlogws
    # Renormalize and clip for stability
    logws = np.clip(logws, -20, 700)
    logws -= np.logaddexp.reduce(logws)
    return logws

# --- Adaptive RMSprop Step ---
def rmsprop_step_pos(mu, R, logws, get_pos_increments, lr, v_mu, v_R, alpha=0.9, eps=1e-8):
    """
    Stateless RMSprop step for Gaussian parameters.
    Updates running squared gradient caches v_mu and v_R in-place.
    """
    dmu, dR = get_pos_increments(mu, R, logws, update_cache=True)
    
    # Initialize cache structures if empty
    if len(v_mu) == 0:
        v_mu.extend([np.zeros_like(m) for m in mu])
        v_R.extend([np.zeros_like(r) for r in R])

    next_mu, next_R = [], []

    for k in range(len(mu)):
        # Mean update
        v_mu[k] = alpha * v_mu[k] + (1.0 - alpha) * (dmu[k] ** 2)
        m_new = mu[k] + lr * dmu[k] / (np.sqrt(v_mu[k]) + eps)
        
        # Cholesky factor update
        v_R[k] = alpha * v_R[k] + (1.0 - alpha) * (dR[k] ** 2)
        R_new = R[k] + lr * dR[k] / (np.sqrt(v_R[k]) + eps)

        next_mu.append(m_new)
        next_R.append(R_new)

    accepted = True
    return next_mu, next_R, lr, accepted


def heun_adaptive_step_pos(mu, R, logws, get_pos_increments, step_size, dmu1=None, dR1=None, rtol=1e-3, atol=1e-6):
    """
    Performs a single adaptive Heun step.
    Returns:
        next_mu, next_R: The accepted or rejected states
        dt_new: The suggested next step size
        accepted: Boolean indicating if the step met the tolerance
    """
    # Reuse stage 1 increments if available
    if dmu1 is None or dR1 is None:
        dmu1, dR1 = get_pos_increments(mu, R, logws, update_cache=True)
        
    # --- 1. Predictor Step (Euler) ---
    #dmu1, dR1 = get_pos_increments(mu, R, logws, update_cache=True)
    mu_pred = [m + step_size * dm for m, dm in zip(mu, dmu1)]
    R_pred = [r + step_size * dr for r, dr in zip(R, dR1)]
    
    # --- 2. Corrector Step ---
    dmu2, dR2 = get_pos_increments(mu_pred, R_pred, logws, update_cache=True)
    mu_corr = [m + 0.5 * step_size * (dm1 + dm2) for m, dm1, dm2 in zip(mu, dmu1, dmu2)]
    R_corr = [r + 0.5 * step_size * (dr1 + dr2) for r, dr1, dr2 in zip(R, dR1, dR2)]
    
    # --- 3. Error Estimation ---
    err_mu = 0.0
    err_R = 0.0
    total_elements = 0
    for m_c, m_p, r_c, r_p in zip(mu_corr, mu_pred, R_corr, R_pred):
        # Calculate tolerable error per element
        scale_m = atol + rtol * np.maximum(np.abs(m_c), np.abs(m_p))
        scale_r = atol + rtol * np.maximum(np.abs(r_c), np.abs(r_p))
        # Calculate raw squared errors
        err_mu += np.sum(((m_c - m_p) / scale_m) ** 2)
        err_R += np.sum(((r_c - r_p) / scale_r) ** 2)
        total_elements += m_c.size + r_c.size

    # Compute final global NRMSE to pass to the adaptive step engine
    nrmse = np.sqrt((err_mu + err_R) / total_elements)
    safety = 0.9
    if nrmse == 0:
        factor = 1.2  # Gentle growth instead of 2.0
    else:
        factor = safety * (1.0 / nrmse) ** 0.5
    if nrmse <= 1.0:
        factor = np.clip(factor, 0.5, 1.2)#1.15
        step_size_new = step_size * factor
        #print(f"--- [ACCEPTED] Global NRMSE: {nrmse:.4f} | Next dt: {step_size_new:.6f} ---")
        #dmu_next, dR_next = get_pos_increments(mu_corr, R_corr, logws, update_cache=True)
        return mu_corr, R_corr, dmu2, dR2, step_size_new, True
    else:
        # Step REJECTED: Allow drastic shrinking to restore stability quickly
        factor = np.clip(factor, 0.2, 0.8)  # Force at least a 10% drop, 0.1, 0.9
        step_size_new = step_size * factor
        print(f"--- [REJECTED] Global NRMSE: {nrmse:.4f} | Next dt: {step_size_new:.6f} ---")
        return mu, R, None, None, step_size_new, False
