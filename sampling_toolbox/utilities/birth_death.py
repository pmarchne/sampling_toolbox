import numpy as np
'''
def apply_birth_death(mu, R, logws, threshold=0.005, rng=None):
        """
        Identifies starved components (weight < threshold), kills them, and 
        regenerates them near the highest-mass component. Keeps K fixed.
        """
        ws = np.exp(logws)
        K = len(mu)
        dim = mu[0].shape[0]
        # Avoid birth-death if there's only 1 component
        if K <= 1:
            return mu, R, logws, False
        # Find the starved components and the strongest mode
        dead_indices = np.where(ws < threshold)[0]
        best_idx = np.argmax(ws)
        
        if len(dead_indices) == 0 or best_idx in dead_indices:
            return mu, R, logws, False  # Nothing to do
            
        # Initialize a fresh random generator if not present
        if rng is None:
            rng = np.random.default_rng()

        did_mutate = False
        for idx in dead_indices:
            print(f" [Birth-Death] Component {idx} killed (mass: {ws[idx]:.4f}). Regenerating near component {best_idx} (mass: {ws[best_idx]:.4f})")
            # 1. Allocate a fresh baseline weight (e.g., 10%) 
            birth_weight = 0.1
            if ws[best_idx] > birth_weight * 2:
                # Steal mass strictly from the best component to preserve sum(weights) == 1
                ws[idx] = birth_weight
                ws[best_idx] -= birth_weight
                did_mutate = True
            else:
                # Fallback
                continue
            # 2. Inherit the geometric shape (Cholesky) of the successful mode
            R[idx] = R[best_idx].copy() / 4.0
            random_direction = rng.normal(size=dim)
            random_direction /= (np.linalg.norm(random_direction) + 1e-12)
            perturbation = (R[best_idx] @ random_direction) * 0.25
            mu[idx] = mu[best_idx] + perturbation
            cov = R[idx] @ R[idx].T
            print('new mean:', mu[idx])
            print('new std:', np.sqrt(np.diag(cov)) )
            
        # 4. Convert safely back into unnormalized log-space and update the cache
        ws = np.clip(ws, 1e-15, 1.0)
        ws /= np.sum(ws)
        logws = np.log(ws + 1e-15)
        
        return mu, R, logws, did_mutate'''


'''def apply_birth_death(mu, R, logws, log_and_grad_post, threshold=0.01, rng=None):
    """
    A hybrid birth-death mechanism.
    - 70% chance: Local Exploitation (Scale-bounded gradient step from best mode).
    - 30% chance: Global Exploration (Sample a fresh candidate directly from the prior).
    """
    ws = np.exp(logws)
    K = len(mu)
    dim = mu[0].shape[0]
    
    if K <= 1:
        return mu, R, logws, False
        
    dead_indices = np.where(ws < threshold)[0]
    best_idx = np.argmax(ws)
    
    if len(dead_indices) == 0 or best_idx in dead_indices:
        return mu, R, logws, False  
        
    if rng is None:
        rng = np.random.default_rng()

    # Pre-evaluate the raw gradient at the best component's mean for exploitation
    _, parent_grad = log_and_grad_post(mu[best_idx])

    did_mutate = False
    for idx in dead_indices:
        # 1. Reallocate weight (take 10% from the best mode)
        birth_weight = 0.10
        if ws[best_idx] > birth_weight * 2:
            ws[idx] = birth_weight
            ws[best_idx] -= birth_weight
            did_mutate = True
        else:
            continue
        # 2. Decide Strategy via a simple 70/30 coin flip
        mu_prior = np.array([3000.0] * 7)
        std = 500.0
        cov_in = np.diag(np.full(len(mu_prior), std**2))
        R_prior = np.linalg.cholesky(cov_in)
        if rng.random() < 0.30:
            # --- BRANCH A: GLOBAL EXPLORATION (Sample from the Prior) ---
            print(f" [Birth-Death] Component {idx} killed. Exploring: Birthing fresh from the PRIOR.")
            # Draw a standard normal vector and transform it by the prior's Cholesky factor
            z = rng.normal(size=dim)
            mu[idx] = mu_prior + R_prior @ z
            # Give it a wider, fresh exploratory footprint (reset to the prior's shape or scaled down)
            R[idx] = R_prior.copy() / 1.5
        else:
            # --- BRANCH B: LOCAL EXPLOITATION (Gradient-guided from Best Gaussian) ---
            print(f" [Birth-Death] Component {idx} killed. Exploiting: Birthing near component {best_idx}.")
            # Inherit half-scale geometry to avoid volume/entropy starvation penalties
            R[idx] = R[best_idx].copy() / 2.0
            # Pure gradient direction step, scaled between 50 and 500 m/s
            grad_norm = np.linalg.norm(parent_grad)
            if grad_norm > 1e-8:
                direction = parent_grad / grad_norm
                step_magnitude = np.clip(grad_norm, 300.0, 800.0)
                mu[idx] = mu[best_idx] + direction * step_magnitude
            else:
                mu[idx] = mu[best_idx] + 50.0  # Fallback symmetry breaker
                print("Fallback")
            
        cov = R[idx] @ R[idx].T
        print('new mean:', mu[idx])
        print('new std:', np.sqrt(np.diag(cov)) )

    print('best component mean', mu[best_idx] )
    cov_best = R[best_idx] @ R[best_idx].T
    print('best component std:', np.sqrt(np.diag(cov_best)) )

    # 3. Strict normalization cleanup to keep weights precisely on the simplex
    ws = np.clip(ws, 1e-15, 1.0)
    ws /= np.sum(ws)
    logws = np.log(ws + 1e-15)
    
    return mu, R, logws, did_mutate'''


def apply_birth_death(mu, R, logws, log_and_grad_post,
                      threshold=0.01,
                      rng=None,
                      n_trials=20):
    """
    Nested-sampling inspired birth-death.

    For each starved Gaussian:
      - kill it
      - take the best existing mode
      - generate n_trials local proposals
      - proposals are Gaussian + gradient guided
      - keep the highest posterior candidate
    """
    ws = np.exp(logws)
    K = len(mu)
    dim = mu[0].shape[0]
    if K <= 1:
        return mu, R, logws, False
    dead_indices = np.where(ws < threshold)[0]
    # choose best basin using posterior, not weight
    logp = np.zeros(K)
    grads = []

    for k in range(K):
        logp[k], g = log_and_grad_post(mu[k])
        grads.append(g)

    best_idx = np.argmax(logp)

    if len(dead_indices) == 0:
        return mu, R, logws, False

    if rng is None:
        rng = np.random.default_rng()

    did_mutate = False
    for idx in dead_indices:
        print(
            f"[Birth] replacing component {idx}, "
            f"using mode {best_idx}, "
            f"logpi={logp[best_idx]:.3f}"
        )
        # Weight transfer
        birth_weight = 0.10
        if ws[best_idx] <= birth_weight*2:
            continue
        ws[idx] = birth_weight
        ws[best_idx] -= birth_weight
        # Prepare local search
        parent = mu[best_idx]
        R_parent = R[best_idx]
        grad = grads[best_idx]
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad_dir = grad / grad_norm
        else:
            grad_dir = np.zeros(dim)
        best_candidate = None
        best_value = -np.inf
        # Nested sampling style replacement
        for _ in range(n_trials):
            # random exploration in Gaussian volume
            z = rng.normal(size=dim)
            # local covariance exploration
            local_move = R_parent @ z
            # gradient push
            step = rng.uniform(0., 1.)
            diag_cov = np.sqrt(np.diag(R_parent @ R_parent.T))
            grad_move = (
                step
                * diag_cov
                * grad_dir
            )
            print("grad move", grad_move)
            candidate = (
                parent
                + 0.5*local_move
                + grad_move
            )
            candidate = np.clip(candidate, 1000., 5000.)
            print("candidate", candidate)
            lp, _ = log_and_grad_post(candidate)
            if lp > best_value:
                best_value = lp
                best_candidate = candidate.copy()

        # Replace dead component
        mu[idx] = best_candidate
        # inherit smaller covariance
        R[idx] = R_parent.copy()/2.0
        print(
            f"   accepted trial logpi={best_value:.3f}"
        )
        cov = R[idx] @ R[idx].T
        print(
            "   new std:",
            np.sqrt(np.diag(cov))
        )
        did_mutate = True
    # normalize weights
    ws = np.clip(ws,1e-15,None)
    ws /= np.sum(ws)
    logws = np.log(ws)
    return mu, R, logws, did_mutate