import numpy as np

def init_states(mu, R, logws):
    m_mu = [np.zeros_like(m) for m in mu]
    v_mu = [np.zeros_like(m) for m in mu]
    m_R  = [np.zeros_like(r) for r in R]
    v_R  = [np.zeros_like(r) for r in R]
    
    m_w = np.zeros_like(logws)
    v_w = np.zeros_like(logws)
    return m_mu, v_mu, m_R, v_R, m_w, v_w

# --- Euler Steps (Decoupled) ---

def euler_step_pos(mu, R, logws, get_pos_increments, step_size):
    dmu, dR = get_pos_increments(mu, R, logws)
    for m, r, dm, dr in zip(mu, R, dmu, dR):
        m += step_size * dm
        r += step_size * dr
    return mu, R


def heun_adaptive_step_pos(mu, R, logws, get_pos_increments, step_size, rtol=1e-3, atol=1e-6):
    """
    Performs a single adaptive Heun step.
    Returns:
        next_mu, next_R: The accepted or rejected states
        dt_new: The suggested next step size
        accepted: Boolean indicating if the step met the tolerance
    """
    # --- 1. Predictor Step (Euler) ---
    dmu1, dR1 = get_pos_increments(mu, R, logws)
    mu_pred = [m + step_size * dm for m, dm in zip(mu, dmu1)]
    R_pred = [r + step_size * dr for r, dr in zip(R, dR1)]
    
    # --- 2. Corrector Step ---
    dmu2, dR2 = get_pos_increments(mu_pred, R_pred, logws)
    mu_corr = [m + 0.5 * step_size * (dm1 + dm2) for m, dm1, dm2 in zip(mu, dmu1, dmu2)]
    R_corr = [r + 0.5 * step_size * (dr1 + dr2) for r, dr1, dr2 in zip(R, dR1, dR2)]
    
    # --- 3. Error Estimation ---
    # Local error is the difference between Predictor (O(h^1)) and Corrector (O(h^2))
    err_mu = 0.0
    err_R = 0.0
    total_elements = 0
    
    for m_c, m_p, r_c, r_p in zip(mu_corr, mu_pred, R_corr, R_pred):
        # Calculate tolerable error per element
        scale_m = atol + rtol * np.maximum(np.abs(m_c), np.abs(m_p))
        scale_r = atol + rtol * np.maximum(np.abs(r_c), np.abs(r_p))
        
        err_mu += np.sum(((m_c - m_p) / scale_m) ** 2)
        err_R += np.sum(((r_c - r_p) / scale_r) ** 2)
        total_elements += m_c.size + r_c.size

    # Normalized Root Mean Squared Error
    nrmse = np.sqrt((err_mu + err_R) / total_elements)
    
    # --- 4. Step Size Control ---
    # Safety factors to prevent drastic changes
    safety = 0.9
    if nrmse == 0:
        factor = 2.0
    else:
        # Heun is a 2nd order method, error scales with h^2, so step adjustment scales with h^(1/2)
        factor = safety * (1.0 / nrmse) ** 0.5
    
    # Clamp the adjustment factor to avoid extreme jumps
    factor = np.clip(factor, 0.2, 5.0)
    step_size_new = step_size * factor
    
    if nrmse <= 1.0:
        # Error is within tolerance -> Accept step
        return mu_corr, R_corr, step_size_new, True
    else:
        # Error is too large -> Reject step, return original states and shrunken dt
        return mu, R, step_size_new, False
    

def euler_step_w(mu, R, logws, get_w_increments, step_size):
    dlogws = get_w_increments(mu, R, logws)
    logws += step_size * dlogws
    # Renormalize and clip for stability
    logws = np.clip(logws, -20, 700)
    logws -= np.logaddexp.reduce(logws)
    return logws

def heun_step_w(mu, R, logws, get_w_increments, step_size):
    # --- STEP 1: The Predictor (Euler Step) ---
    k1 = get_w_increments(mu, R, logws)
    logws_pred = logws + step_size * k1
    
    # Keep the predictor inside stable bounds before evaluating the corrector
    logws_pred = np.clip(logws_pred, -20, 700)
    logws_pred -= np.logaddexp.reduce(logws_pred)
    
    # --- STEP 2: The Corrector (Trapezoidal Average) ---
    # Evaluate the gradients at the predicted future state
    k2 = get_w_increments(mu, R, logws_pred)
    
    # Update using the average of both slopes (2nd-order accurate)
    logws_next = logws + (step_size / 2.0) * (k1 + k2)
    
    # Final normalization and clipping
    logws_next = np.clip(logws_next, -20, 700)
    logws_next -= np.logaddexp.reduce(logws_next)
    
    return logws_next
# --- RK4 Steps (Decoupled) ---

def rk4_step_pos(mu, R, logws, get_pos_increments, step_size):
    h = step_size
    
    k1_mu, k1_R = get_pos_increments(mu, R, logws)
    
    mu_k2 = [m + 0.5 * h * dk for m, dk in zip(mu, k1_mu)]
    R_k2  = [r + 0.5 * h * dk for r, dk in zip(R, k1_R)]
    k2_mu, k2_R = get_pos_increments(mu_k2, R_k2, logws)
    
    mu_k3 = [m + 0.5 * h * dk for m, dk in zip(mu, k2_mu)]
    R_k3  = [r + 0.5 * h * dk for r, dk in zip(R, k2_R)]
    k3_mu, k3_R = get_pos_increments(mu_k3, R_k3, logws)
    
    mu_k4 = [m + h * dk for m, dk in zip(mu, k3_mu)]
    R_k4  = [r + h * dk for r, dk in zip(R, k3_R)]
    k4_mu, k4_R = get_pos_increments(mu_k4, R_k4, logws)
    
    for m, r, k1m, k2m, k3m, k4m, k1r, k2r, k3r, k4r in zip(
        mu, R, k1_mu, k2_mu, k3_mu, k4_mu, k1_R, k2_R, k3_R, k4_R
    ):
        m += (h / 6.0) * (k1m + 2 * k2m + 2 * k3m + k4m)
        r += (h / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
    return mu, R

def rk4_step_w(mu, R, logws, get_w_increments, step_size):
    h = step_size
    
    k1 = get_w_increments(mu, R, logws)
    
    logws_k2 = logws + 0.5 * h * k1
    k2 = get_w_increments(mu, R, logws_k2)
    
    logws_k3 = logws + 0.5 * h * k2
    k3 = get_w_increments(mu, R, logws_k3)
    
    logws_k4 = logws + h * k3
    k4 = get_w_increments(mu, R, logws_k4)
    
    logws += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    logws = np.clip(logws, -20, 700)
    logws -= np.logaddexp.reduce(logws)
    return logws

# --- Adam Steps (Decoupled) ---

def adam_step_pos(mu, R, logws, iter_idx, get_pos_increments, step_size, m_mu, v_mu, m_R, v_R):
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    grad_mu, grad_R = get_pos_increments(mu, R, logws)
    t = iter_idx + 1 
    
    for i in range(len(mu)):
        m_mu[i] = beta1 * m_mu[i] + (1 - beta1) * grad_mu[i]
        v_mu[i] = beta2 * v_mu[i] + (1 - beta2) * (grad_mu[i] ** 2)
        m_hat_mu = m_mu[i] / (1 - beta1 ** t)
        v_hat_mu = v_mu[i] / (1 - beta2 ** t)
        mu[i] += step_size * m_hat_mu / (np.sqrt(v_hat_mu) + eps)

    for i in range(len(R)):
        m_R[i] = beta1 * m_R[i] + (1 - beta1) * grad_R[i]
        v_R[i] = beta2 * v_R[i] + (1 - beta2) * (grad_R[i] ** 2)
        m_hat_R = m_R[i] / (1 - beta1 ** t)
        v_hat_R = v_R[i] / (1 - beta2 ** t)
        R[i] += step_size * m_hat_R / (np.sqrt(v_hat_R) + eps)
    
    return mu, R

def adam_step_w(mu, R, logws, iter_idx, get_w_increments, step_size, m_w, v_w):
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    grad_w = get_w_increments(mu, R, logws)
    t = iter_idx + 1
    
    m_w[:] = beta1 * m_w + (1 - beta1) * grad_w
    v_w[:] = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
    m_hat_w = m_w / (1 - beta1 ** t)
    v_hat_w = v_w / (1 - beta2 ** t)
    
    logws += step_size * m_hat_w / (np.sqrt(v_hat_w) + eps)
    logws = np.clip(logws, -20, 700)
    logws -= np.logaddexp.reduce(logws)
    return logws