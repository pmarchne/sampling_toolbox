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


def heun_adaptive_step_pos(mu, R, logws, get_pos_increments, step_size, rtol=1e-3, atol=1e-6):
    """
    Performs a single adaptive Heun step.
    Returns:
        next_mu, next_R: The accepted or rejected states
        dt_new: The suggested next step size
        accepted: Boolean indicating if the step met the tolerance
    """
    # --- 1. Predictor Step (Euler) ---
    dmu1, dR1 = get_pos_increments(mu, R, logws, update_cache=True)
    mu_pred = [m + step_size * dm for m, dm in zip(mu, dmu1)]
    R_pred = [r + step_size * dr for r, dr in zip(R, dR1)]
    
    # --- 2. Corrector Step ---
    dmu2, dR2 = get_pos_increments(mu_pred, R_pred, logws, update_cache=False)
    mu_corr = [m + 0.5 * step_size * (dm1 + dm2) for m, dm1, dm2 in zip(mu, dmu1, dmu2)]
    R_corr = [r + 0.5 * step_size * (dr1 + dr2) for r, dr1, dr2 in zip(R, dR1, dR2)]
    
    # --- 3. Error Estimation ---
    # Local error is the difference between Predictor (O(h^1)) and Corrector (O(h^2))
    err_mu = 0.0
    err_R = 0.0
    #total_error_sum = 0.0
    total_elements = 0
    #ws = np.exp(logws)
    for m_c, m_p, r_c, r_p in zip(mu_corr, mu_pred, R_corr, R_pred):
        # Calculate tolerable error per element
        scale_m = atol + rtol * np.maximum(np.abs(m_c), np.abs(m_p))
        scale_r = atol + rtol * np.maximum(np.abs(r_c), np.abs(r_p))
        
        # Calculate raw squared errors
        err_mu += np.sum(((m_c - m_p) / scale_m) ** 2)
        err_R += np.sum(((r_c - r_p) / scale_r) ** 2)
        total_elements += m_c.size + r_c.size

        #comp_elements = m_c.size + r_c.size
        #comp_error_sum = np.sum(local_err_mu) + np.sum(local_err_R)
        
        # 3. Calculate this component's isolated NRMSE
        #comp_nrmse = np.sqrt(comp_error_sum / comp_elements)
        
        # --- THE TRUNCATION OVERRIDE ---
        # If a component is a zombie (low mass) OR an extreme outlier on a cliff,
        # we cap its personal NRMSE at 1.0. This tells the global solver:
        # "This component is running at max safe speed, do not freeze the others because of it."
        '''if comp_nrmse > 1.0:
            # Condition A: It's a low-mass spectator (zombie). We completely ignore its noise.
            is_zombie = ws[k] < 0.005
            # Condition B: It's a heavy component experiencing a massive unphysical explosion 
            # (Set to 1000.0 so Rosenbrock can naturally turn, but FWI cliffs are still suppressed)
            print("comp_nrmse = ", comp_nrmse)
            is_catastrophic_explosion = comp_nrmse > 100.0
            if is_zombie or is_catastrophic_explosion:
                print(f"force neutral! Component {k+1} (Zombie={is_zombie}, Explosion={is_catastrophic_explosion})")
                comp_error_sum = 1.0 * comp_elements  # Force a neutral 1.0 NRMSE'''
                
        # 4. Accumulate into global pools
        #total_error_sum += comp_error_sum
        #total_elements += comp_elements

    # Compute final global NRMSE to pass to the adaptive step engine
    nrmse = np.sqrt((err_mu + err_R) / total_elements)
    
    # --- SINGLE COMPONENT DIAGNOSTIC CRADLE ---
    # 1. Compute unweighted raw errors
    #raw_err_mu = np.abs(np.array(mu_corr) - np.array(mu_pred))
    #raw_err_R = np.abs(np.array(R_corr) - np.array(R_pred))
    
    # 2. Compute the tolerances being applied
    #scale_m2 = atol + rtol * np.maximum(np.abs(mu_corr), np.abs(mu_pred))
    #scale_r2 = atol + rtol * np.maximum(np.abs(R_corr), np.abs(R_pred))
    
    # 3. Calculate isolated NRMSE contributions
    #nrmse_mu = np.sqrt(np.sum((raw_err_mu / scale_m2) ** 2) / raw_err_mu.size)
    #nrmse_R = np.sqrt(np.sum((raw_err_R / scale_r2) ** 2) / raw_err_R.size)
    
    #status = "ACCEPTED" if nrmse <= 1.0 else "REJECTED"
    
    #print(f"\n=== Step Status: {status} | Global NRMSE: {nrmse:.4f} | Current dt: {step_size:.6f} ===")
    #print(f"Mean (mu)  -> Max Raw Error: {np.max(raw_err_mu):.4f} | Mean Scale Vol: {np.mean(scale_m2):.2f} | NRMSE_mu: {nrmse_mu:.4f}")
    #print(f"Covariance -> Max Raw Error: {np.max(raw_err_R):.4f} | Mean Scale Vol: {np.mean(scale_r2):.2f} | NRMSE_R:  {nrmse_R:.4f}")
    #print("=========================================================================")



    # --- 4. Step Size Control ---
    # Safety factors to prevent drastic changes
    '''
    # Clamp the adjustment factor to avoid extreme jumps
    factor = np.clip(factor, 0.2, 5.0)
    step_size_new = step_size * factor
    
    if nrmse <= 1.0:
        # Error is within tolerance -> Accept step
        return mu_corr, R_corr, step_size_new, True
    else:'''
    
    safety = 0.9
    if nrmse == 0:
        factor = 1.2  # Gentle growth instead of 2.0
    else:
        factor = safety * (1.0 / nrmse) ** 0.5

    if nrmse <= 1.0:
        # Step ACCEPTED: Allow cautious growth to prevent chatter
        factor = np.clip(factor, 0.2, 1.5)  # Cap growth at 20% per step, 0.5 ,1.2
        step_size_new = step_size * factor
        print(f"--- [ACCEPTED] Global NRMSE: {nrmse:.4f} | Next dt: {step_size_new:.6f} ---")
        return mu_corr, R_corr, step_size_new, True
    else:
        # Step REJECTED: Allow drastic shrinking to restore stability quickly
        factor = np.clip(factor, 0.2, 0.8)  # Force at least a 10% drop, 0.1, 0.9
        step_size_new = step_size * factor
        print(f"--- [REJECTED] Global NRMSE: {nrmse:.4f} | Next dt: {step_size_new:.6f} ---")
        return mu, R, step_size_new, False
        

def heun_pi_adaptive_step_pos(mu, R, logws, get_pos_increments, step_size, prev_nrmse=1.0, rtol=1e-3, atol=1e-5):
    """
    Performs a single adaptive Heun step stabilized by a PI step-size controller.
    """
    # --- 1. Predictor Step (Euler) ---
    dmu1, dR1 = get_pos_increments(mu, R, logws, update_cache=True)
    mu_pred = [m + step_size * dm for m, dm in zip(mu, dmu1)]
    R_pred = [r + step_size * dr for r, dr in zip(R, dR1)]
    
    # --- 2. Corrector Step ---
    dmu2, dR2 = get_pos_increments(mu_pred, R_pred, logws, update_cache=False)
    mu_corr = [m + 0.5 * step_size * (dm1 + dm2) for m, dm1, dm2 in zip(mu, dmu1, dmu2)]
    R_corr = [r + 0.5 * step_size * (dr1 + dr2) for r, dr1, dr2 in zip(R, dR1, dR2)]
    
    # --- 3. Error Estimation ---
    err_mu = 0.0
    err_R = 0.0
    total_elements = 0
    
    for m_c, m_p, r_c, r_p in zip(mu_corr, mu_pred, R_corr, R_pred):
        # HELPFUL MODIFICATION: Raised atol acts as a noise floor 
        # protecting against micro-scale FWI numerical ripples
        scale_m = atol + rtol * np.maximum(np.abs(m_c), np.abs(m_p))
        scale_r = atol + rtol * np.maximum(np.abs(r_c), np.abs(r_p))
        
        err_mu += np.sum(((m_c - m_p) / scale_m) ** 2)
        err_R += np.sum(((r_c - r_p) / scale_r) ** 2)
        total_elements += m_c.size + r_c.size

    nrmse = np.sqrt((err_mu + err_R) / total_elements)
    nrmse = max(nrmse, 1e-8)  # Prevent division by zero
    
    # --- 4. PI Step Size Control (Gustafsson Acceleration) ---
    safety = 0.9
    # Standard 2nd-order PI exponents (kP = 0.3, kI = 0.15)
    # kP controls response to current error; kI dampens abrupt step changes
    kP = 0.3 / 2.0
    kI = 0.15 / 2.0
    
    # PI Formula: step_new = step_old * safety * (1/nrmse)^kP * (prev_nrmse/nrmse)^kI
    factor = safety * (1.0 / nrmse) ** kP * (prev_nrmse / nrmse) ** kI
    
    # Strictly bound the changes per step to eliminate step-size chatter
    if nrmse <= 1.0:
        # ACCEPTED
        factor = np.clip(factor, 0.5, 2.0)  # Max 50% growth prevents wild over-shooting
        step_size_new = step_size * factor
        return mu_corr, R_corr, step_size_new, True, nrmse
    else:
        # REJECTED
        factor = np.clip(factor, 0.1, 0.9)  # Force a decisive but bounded shrink
        step_size_new = step_size * factor
        return mu, R, step_size_new, False, nrmse


def heun_adaptive_step_pos2(
    mu, R, logws, get_pos_increments, step_size, 
    rtol_mu=1e-2, atol_mu=1e-3, 
    rtol_R=2e-2, atol_R=5e-3
    #rtol_mu=1e-1, atol_mu=1e-2, 
    #rtol_R=2e-1, atol_R=5e-2
):
    """
    Performs a single adaptive Heun step with split tolerances for Mu and R.
    
    Returns:
        next_mu, next_R: The accepted or rejected states
        dt_new: The suggested next step size
        accepted: Boolean indicating if the step met the tolerance
    """
    # --- 1. Predictor Step (Euler) ---
    dmu1, dR1 = get_pos_increments(mu, R, logws, update_cache=True)
    mu_pred = [m + step_size * dm for m, dm in zip(mu, dmu1)]
    R_pred = [r + step_size * dr for r, dr in zip(R, dR1)]
    
    # --- 2. Corrector Step ---
    dmu2, dR2 = get_pos_increments(mu_pred, R_pred, logws, update_cache=False)
    mu_corr = [m + 0.5 * step_size * (dm1 + dm2) for m, dm1, dm2 in zip(mu, dmu1, dmu2)]
    R_corr = [r + 0.5 * step_size * (dr1 + dr2) for r, dr1, dr2 in zip(R, dR1, dR2)]
    
    # --- 3. Error Estimation ---
    err_mu = 0.0
    err_R = 0.0
    total_elements = 0
    ws = np.exp(logws)
    
    # We will track the aggregate metrics for our print statements
    sum_scale_m = 0.0
    sum_scale_r = 0.0
    count_m = 0
    count_r = 0
    
    for m_c, m_p, r_c, r_p in zip(mu_corr, mu_pred, R_corr, R_pred):
        # Apply the specific tolerance scales independently
        scale_m = atol_mu + rtol_mu * np.maximum(np.abs(m_c), np.abs(m_p))
        scale_r = atol_R + rtol_R * np.maximum(np.abs(r_c), np.abs(r_p))
        
        # Track scale volumes for the diagnostic printout
        sum_scale_m += np.sum(scale_m)
        sum_scale_r += np.sum(scale_r)
        count_m += m_c.size
        count_r += r_c.size
        
        # Calculate localized raw squared errors
        comp_err_mu = np.sum(((m_c - m_p) / scale_m) ** 2)
        comp_err_R = np.sum(((r_c - r_p) / scale_r) ** 2)
        comp_elements = m_c.size + r_c.size
            
        # Accumulate into global pools
        err_mu += comp_err_mu
        err_R += comp_err_R
        total_elements += comp_elements

    # Compute individual component NRMSEs for clean diagnostic tracking
    nrmse_mu = np.sqrt(err_mu / count_m) if count_m > 0 else 0.0
    nrmse_R = np.sqrt(err_R / count_r) if count_r > 0 else 0.0
    
    # Compute final global unified NRMSE to pass to the adaptive engine
    nrmse = np.sqrt((err_mu + err_R) / total_elements)
    
    # --- 4. Step Size Control ---
    safety = 0.9
    if nrmse == 0:
        factor = 1.2  
    else:
        factor = safety * (1.0 / nrmse) ** 0.5
    
    status = "ACCEPTED" if nrmse <= 1.0 else "REJECTED"
    
    # --- 5. Diagnostic Printing ---
    mean_scale_m = sum_scale_m / count_m if count_m > 0 else 0.0
    mean_scale_r = sum_scale_r / count_r if count_r > 0 else 0.0
    
    # Grab the maximum raw error for printout reference
    max_raw_err_mu = max(np.max(np.abs(mc - mp)) for mc, mp in zip(mu_corr, mu_pred))
    max_raw_err_R = max(np.max(np.abs(rc - rp)) for rc, rp in zip(R_corr, R_pred))

    print(f"\n=== Step Status: {status} | Global NRMSE: {nrmse:.4f} | Current dt: {step_size:.6f} ===")
    print(f"Mean (mu)  -> Max Raw Error: {max_raw_err_mu:.4f} | Mean Scale Vol: {mean_scale_m:.2f} | NRMSE_mu: {nrmse_mu:.4f}")
    print(f"Covariance -> Max Raw Error: {max_raw_err_R:.4f} | Mean Scale Vol: {mean_scale_r:.2f} | NRMSE_R:  {nrmse_R:.4f}")
    print("=========================================================================")
    
    if nrmse <= 1.0:
        factor = np.clip(factor, 0.2, 1.5)  
        step_size_new = step_size * factor
        print(f"--- [ACCEPTED] Global NRMSE: {nrmse:.4f} | Next dt: {step_size_new:.6f} ---")
        return mu_corr, R_corr, step_size_new, True
    else:
        factor = np.clip(factor, 0.2, 0.8)  
        step_size_new = step_size * factor
        print(f"--- [REJECTED] Global NRMSE: {nrmse:.4f} | Next dt: {step_size_new:.6f} ---")
        return mu, R, step_size_new, False