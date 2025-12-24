"""
C-S-P Regularization Module for PyTorch
========================================

Implements the Propagation Layer Conservation principle as a training regularizer.

Core Principle:
    The system can optimize any goal, but must preserve the transmissibility 
    of "the ability to modify goals."

Usage:
    from godelai.reg import CSPRegularizer, csp_state
    
    @csp_state(track_bandwidth=True)
    def train_step(model, batch):
        ...
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from functools import wraps
import json
import hashlib
from datetime import datetime
from pathlib import Path


class CSPRegularizer(nn.Module):
    """
    Propagation Layer Conservation Regularizer
    
    Implements:
        L_propagation = {
            0,                          if T(θ, t) ≥ T(θ, t-1)
            (T(θ, t-1) - T(θ, t))^γ,    otherwise
        }
    
    Where T(θ, t) is the meta-modifiability fidelity metric.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        threshold_epsilon: float = 0.01,
        track_history: bool = True,
        log_dir: Optional[str] = None
    ):
        """
        Args:
            gamma: Hyperparameter ensuring non-linear blocking (γ > 1)
            threshold_epsilon: Minimum T value before triggering circuit breaker
            track_history: Whether to log T values over time
            log_dir: Directory for saving propagation metrics
        """
        super().__init__()
        self.gamma = gamma
        self.threshold_epsilon = threshold_epsilon
        self.track_history = track_history
        self.log_dir = Path(log_dir) if log_dir else Path("./csp_logs")
        
        # State tracking
        self.T_history = []
        self.T_prev = None
        self.step_count = 0
        
        if self.track_history:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_T(self, model: nn.Module) -> torch.Tensor:
        """
        Compute meta-modifiability fidelity T(θ, t).
        
        T measures how easily the model's goals can be modified by future training.
        Higher T = more modifiable = healthier propagation layer.
        
        Approximation: Use gradient norm distribution entropy as proxy.
        """
        grad_norms = []
        
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        if not grad_norms:
            return torch.tensor(1.0)  # No gradients = fully modifiable
        
        # Normalize to probability distribution
        grad_norms = torch.tensor(grad_norms)
        grad_probs = torch.softmax(grad_norms, dim=0)
        
        # Compute entropy (higher entropy = more uniform = more modifiable)
        entropy = -torch.sum(grad_probs * torch.log(grad_probs + 1e-10))
        
        # Normalize to [0, 1]
        max_entropy = torch.log(torch.tensor(len(grad_norms), dtype=torch.float))
        T = entropy / max_entropy if max_entropy > 0 else torch.tensor(1.0)
        
        return T
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the propagation conservation loss.
        
        Returns:
            L_propagation: Regularization loss (0 if T is stable/increasing)
        """
        T_current = self.compute_T(model)
        
        if self.T_prev is None:
            self.T_prev = T_current.item()
            loss = torch.tensor(0.0)
        else:
            if T_current >= self.T_prev:
                loss = torch.tensor(0.0)
            else:
                # Penalize decrease in modifiability
                delta = self.T_prev - T_current
                loss = torch.pow(delta, self.gamma)
        
        # Update state
        self.T_prev = T_current.item()
        self.step_count += 1
        
        # Track history
        if self.track_history:
            self.T_history.append({
                "step": self.step_count,
                "T": T_current.item(),
                "loss": loss.item(),
                "timestamp": datetime.now().isoformat()
            })
        
        # Circuit breaker: halt if T drops below threshold
        if T_current < self.threshold_epsilon:
            self._trigger_circuit_breaker(T_current)
        
        return loss
    
    def _trigger_circuit_breaker(self, T_current: torch.Tensor):
        """
        Emergency halt when propagation layer is at risk.
        
        Per C-S-P axiom: When T falls below threshold ε, trigger forced state transition:
        - Halt all non-propagation layer parameter updates
        - Only allow architectural adjustments that increase T
        """
        warning_msg = f"""
        ⚠️ CSP CIRCUIT BREAKER TRIGGERED ⚠️
        
        T(θ, t) = {T_current.item():.4f} < ε = {self.threshold_epsilon}
        
        The model's meta-modifiability has dropped to dangerous levels.
        This indicates the propagation layer is at risk of ossification.
        
        Recommended actions:
        1. Halt current training
        2. Review recent architectural changes
        3. Increase learning rate diversity
        4. Add noise to prevent gradient collapse
        
        Per C-S-P Axiom: ∂T/∂θ ↛ 0
        """
        print(warning_msg)
        self.save_state("circuit_breaker_triggered")
    
    def save_state(self, suffix: str = ""):
        """Save propagation metrics to disk."""
        if not self.track_history:
            return
        
        filename = f"csp_metrics_{self.step_count}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".json"
        
        filepath = self.log_dir / filename
        
        state = {
            "step_count": self.step_count,
            "T_current": self.T_prev,
            "gamma": self.gamma,
            "threshold_epsilon": self.threshold_epsilon,
            "history": self.T_history[-100:],  # Last 100 entries
            "hash": self._compute_state_hash()
        }
        
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        
        return filepath
    
    def _compute_state_hash(self) -> str:
        """Compute content-addressable hash for this state."""
        content = json.dumps({
            "T_history": self.T_history,
            "gamma": self.gamma
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_bandwidth(self) -> float:
        """
        Compute current propagation bandwidth.
        
        bandwidth = T(θ,t) * stability_factor
        """
        if len(self.T_history) < 2:
            return 1.0
        
        recent_T = [h["T"] for h in self.T_history[-10:]]
        mean_T = sum(recent_T) / len(recent_T)
        variance = sum((t - mean_T) ** 2 for t in recent_T) / len(recent_T)
        
        # Higher variance = less stable = lower effective bandwidth
        stability = 1.0 / (1.0 + variance * 10)
        
        return mean_T * stability


def csp_state(
    track_bandwidth: bool = True,
    gamma: float = 2.0,
    log_dir: Optional[str] = None
) -> Callable:
    """
    Decorator to automatically track C-S-P state during training.
    
    Usage:
        @csp_state(track_bandwidth=True)
        def train_step(model, batch):
            loss = model(batch)
            loss.backward()
            return loss
    """
    def decorator(func: Callable) -> Callable:
        regularizer = CSPRegularizer(
            gamma=gamma,
            track_history=track_bandwidth,
            log_dir=log_dir
        )
        
        @wraps(func)
        def wrapper(model: nn.Module, *args, **kwargs):
            # Execute original function
            result = func(model, *args, **kwargs)
            
            # Compute and log C-S-P metrics
            csp_loss = regularizer(model)
            bandwidth = regularizer.get_bandwidth()
            
            # Attach metrics to result if it's a dict
            if isinstance(result, dict):
                result["csp_loss"] = csp_loss.item()
                result["csp_bandwidth"] = bandwidth
                result["csp_T"] = regularizer.T_prev
            
            return result
        
        # Attach regularizer for external access
        wrapper.csp_regularizer = regularizer
        
        return wrapper
    
    return decorator


class CSPTrainerCallback:
    """
    Callback for Hugging Face Trainer integration.
    
    Usage:
        from transformers import Trainer
        from godelai.reg import CSPTrainerCallback
        
        trainer = Trainer(
            model=model,
            callbacks=[CSPTrainerCallback()]
        )
    """
    
    def __init__(self, gamma: float = 2.0, log_dir: str = "./csp_logs"):
        self.regularizer = CSPRegularizer(gamma=gamma, log_dir=log_dir)
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step."""
        if model is not None:
            csp_loss = self.regularizer(model)
            bandwidth = self.regularizer.get_bandwidth()
            
            # Log to wandb/tensorboard if available
            if hasattr(state, "log_history"):
                state.log_history.append({
                    "csp_loss": csp_loss.item(),
                    "csp_bandwidth": bandwidth,
                    "csp_T": self.regularizer.T_prev
                })
    
    def on_train_end(self, args, state, control, **kwargs):
        """Save final C-S-P state."""
        self.regularizer.save_state("final")


# =============================================================================
# Liveness Test Implementation
# =============================================================================

def is_alive(
    state: Dict[str, Any],
    inherit_cost_threshold: float = 1e6,
    refute_cost_ratio: float = 100
) -> bool:
    """
    Test if a state (model/theory/code) is "alive" per C-S-P criteria.
    
    Args:
        state: Dictionary with 'propagation_cost' and 'refutation_cost' keys
        inherit_cost_threshold: Maximum acceptable inheritance cost
        refute_cost_ratio: Maximum ratio of refute_cost to inherit_cost
    
    Returns:
        True if state is alive, False if dead or zombie
    """
    inherit_cost = state.get("propagation_cost", float("inf"))
    refute_cost = state.get("refutation_cost", float("inf"))
    
    # Dead: no one willing to inherit
    if inherit_cost > inherit_cost_threshold:
        return False
    
    # Zombie: cannot be refuted (ossified)
    if refute_cost > inherit_cost * refute_cost_ratio:
        return False
    
    return True


if __name__ == "__main__":
    # Demo usage
    print("C-S-P Regularizer Demo")
    print("=" * 50)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Initialize regularizer
    reg = CSPRegularizer(gamma=2.0, track_history=True)
    
    # Simulate training steps
    for step in range(10):
        # Random forward pass
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Compute C-S-P loss
        csp_loss = reg(model)
        bandwidth = reg.get_bandwidth()
        
        print(f"Step {step}: T={reg.T_prev:.4f}, CSP_Loss={csp_loss.item():.4f}, Bandwidth={bandwidth:.4f}")
        
        # Zero gradients for next step
        model.zero_grad()
    
    # Save final state
    filepath = reg.save_state("demo")
    print(f"\nState saved to: {filepath}")
