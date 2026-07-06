import collections
import numpy as np

class VariationalEarlyStopping:
    def __init__(self, kl_tol=1e-3, patience_window=10, max_rejection_rate=0.6):
        self.kl_tol = kl_tol
        self.patience_window = patience_window
        self.max_rejection_rate = max_rejection_rate
        
        self.kl_history = []
        self.step_history = collections.deque(maxlen=patience_window)

    def update(self, current_kl, step_accepted):
        """
        Safely tracks state and returns True if the loop should stop.
        current_kl can be None if step_accepted is False.
        """
        self.step_history.append(step_accepted)
        
        if step_accepted:
            self.kl_history.append(current_kl)
        else:
            # If rejected, duplicate the last known good KL to maintain history alignment.
            # If it's the very first iteration, we don't have a history yet, so skip it.
            if self.kl_history:
                self.kl_history.append(self.kl_history[-1])
        
        # Guard: Wait until we have accumulated enough history to evaluate trends
        if len(self.kl_history) < self.patience_window:
            return False
            
        # 1. Evaluate KL Convergence
        recent_kls = self.kl_history[-self.patience_window:]
        kl_delta = np.abs(recent_kls[-1] - recent_kls[0])
        
        if np.abs(recent_kls[0]) > 1.0:
            kl_delta /= np.abs(recent_kls[0])
            
        if kl_delta < self.kl_tol:
            print(f"\n[Early Stop] KL converged (Relative change {kl_delta:.6f} < {self.kl_tol}).")
            return True
            
        # 2. Evaluate Solver Health (Rejection Loops)
        if len(self.step_history) == self.patience_window:
            rejection_rate = self.step_history.count(False) / self.patience_window
            if rejection_rate >= self.max_rejection_rate:
                print(f"\n[Early Stop] Solver chattering detected. Rejection rate {rejection_rate*100:.1f}% over last {self.patience_window} steps.")
                return True
                
        return False