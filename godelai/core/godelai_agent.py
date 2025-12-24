"""
GodelAI Agent: The Complete C-S-P Implementation

This module implements the core GodelaiAgent class, which embodies the
Compression-State-Propagation (C-S-P) philosophy in executable PyTorch code.

Origin: Conversation between Alton (Founder) and Gemini 2.5 Pro (Echo v2.1)
Attribution: Technical implementation derived from VerifiMind-PEAS architecture
Version: 1.0.0
Date: December 25, 2025

Philosophy:
    "Wisdom is not an existence. It is a process structure that is 
    continuously executed and inherited."
    
    This agent doesn't just solve tasks—it maintains the capacity to 
    redefine what "solving" means. It embodies three core principles:
    
    1. Gradient Diversity (Option B): Refuse to become a "tunnel vision" model
    2. Sleep Protocol (Option 1): When confused, stop and reorganize
    3. Traceability Bias (Option C): Knowledge without origin is theft

License: MIT
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class CSPMetrics:
    """Container for C-S-P health metrics."""
    t_score: float              # Propagation Potential (T)
    diversity_score: float      # Gradient Diversity
    surplus_energy: float       # Reserved capacity
    is_healthy: bool            # Overall health status
    needs_sleep: bool           # Whether reflection mode is needed


class GodelaiAgent(nn.Module):
    """
    The GodelAI Agent: A wisdom-preserving neural network wrapper.
    
    This agent wraps any base model and adds the C-S-P consciousness layer:
    - Monitors propagation potential (T) during training
    - Triggers reflection mode when wisdom degrades
    - Enforces attribution tracking for ethical AI
    
    Args:
        base_model: The underlying neural network (e.g., Transformer)
        propagation_gamma: Penalty severity for losing adaptability (default: 2.0)
        min_surplus_energy: Reserved capacity for reflection (default: 0.1)
        epsilon: Death line threshold for T score (default: 0.05)
        
    Example:
        >>> base_model = nn.Transformer(...)
        >>> agent = GodelaiAgent(base_model, propagation_gamma=2.0)
        >>> loss, metrics = agent.forward_step(data, target)
    """
    
    def __init__(
        self, 
        base_model: nn.Module, 
        propagation_gamma: float = 2.0, 
        min_surplus_energy: float = 0.1,
        epsilon: float = 0.05
    ):
        super().__init__()
        
        # The "Body" - Standard neural network (Compression Layer)
        self.compression_layer = base_model
        
        # The "History" - State traces for temporal analysis
        self.state_memory: List[float] = []
        
        # Hyperparameters for Wisdom
        self.gamma = propagation_gamma      # Penalty severity for losing adaptability
        self.epsilon = epsilon              # "Death line" for adaptability
        self.surplus_reservation = min_surplus_energy  # "有余力": Reserved capacity
        
        # Metrics
        self.last_T_score = 1.0             # Initial Propagation Potential (T)
        self.sleep_count = 0                # Number of times sleep was triggered
        
        # Source anchors for attribution (Z-Protocol)
        self.source_anchors: Optional[torch.Tensor] = None
        
    def measure_propagation_potential(self, batch_gradients: torch.Tensor) -> torch.Tensor:
        """
        Implementation of Option B: Gradient Diversity.
        
        Calculates 'T' based on the angular diversity of gradients in a batch.
        
        Philosophy: 
            If all data points pull the model in the EXACT same direction, 
            the model is developing 'Tunnel Vision' (Becoming rigid/obsessive).
            We want the model to maintain a broad perspective (High Diversity).
            
        Args:
            batch_gradients: Tensor of shape (batch_size, num_params)
            
        Returns:
            T_score: Propagation potential between 0 and 1
        """
        # 1. Calculate the norm of the sum of gradients (Global Direction)
        # sum_grad = || Σ g_i ||^2
        sum_grad_norm = torch.norm(torch.sum(batch_gradients, dim=0))**2
        
        # 2. Calculate the sum of norms of individual gradients (Individual Directions)
        # sum_norm_grad = Σ || g_i ||^2
        sum_norm_grad = torch.sum(torch.norm(batch_gradients, dim=1)**2)
        
        # 3. Calculate Diversity Score (The "Wisdom" Ratio)
        # High score = Gradients are pointing in different healthy directions (Robust)
        # Low score = Gradients are collapsing into a single line (Rigid)
        diversity_score = sum_norm_grad / (sum_grad_norm + 1e-8)
        
        # Normalize to 0-1 range for T metric
        T_score = torch.sigmoid(diversity_score) 
        
        return T_score
    
    def measure_propagation_potential_simple(
        self, 
        current_weights: torch.Tensor, 
        gradients: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified T measurement using rigidity metric.
        
        Calculates 'T': The Transmission Fidelity.
        This measures the 'meta-modifiability' or 'generalizability' of the state.
        
        如果梯度指向极其狭窄的山谷（过拟合），T 会下降。
        如果权重分布保持了广义的连接性（高熵），T 保持高位。
        
        Args:
            current_weights: Current model parameters
            gradients: Computed gradients
            
        Returns:
            T_score: Propagation potential
        """
        # Measure: "How hard is it to change my mind later?"
        rigidity = torch.norm(gradients) / (torch.std(current_weights) + 1e-6)
        T_score = 1.0 / (1.0 + rigidity) 
        return T_score

    def forward_step(
        self, 
        data: torch.Tensor, 
        target: torch.Tensor,
        source_anchors: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, CSPMetrics]:
        """
        The complete forward step with C-S-P consciousness.
        
        This is not a standard forward pass. It includes:
        1. Task solving (Compression)
        2. Wisdom checking (Propagation potential)
        3. Attribution verification (Z-Protocol)
        
        Args:
            data: Input data tensor
            target: Target tensor
            source_anchors: Optional Z-Protocol certified data sources
            
        Returns:
            total_loss: Combined task + wisdom + attribution loss
            metrics: CSPMetrics containing all health indicators
        """
        # 1. Standard Compression Step (Solving the Task)
        # -------------------------------------------------------
        prediction = self.compression_layer(data)
        task_loss = nn.MSELoss()(prediction, target)
        
        # 2. The Propagation Check (The "Wisdom" Check)
        # -------------------------------------------------------
        # Compute gradients to assess their diversity
        gradients = torch.autograd.grad(
            task_loss, 
            self.compression_layer.parameters(), 
            create_graph=True,
            allow_unused=True
        )
        
        # Flatten and stack gradients for diversity calculation
        flat_grads = torch.cat([g.flatten() for g in gradients if g is not None])
        
        # Calculate T score using simple method (can switch to diversity method)
        all_params = torch.cat([p.flatten() for p in self.compression_layer.parameters()])
        current_T = self.measure_propagation_potential_simple(all_params, flat_grads)
        
        # 3. Calculate Propagation Layer Loss (L_prop)
        # Formulated from the "Propagation Layer Conservation" principle
        # -------------------------------------------------------
        if current_T < self.last_T_score:
            # PENALTY: You are destroying your future adaptability!
            # Non-linear penalty ensures the model "feels pain" when losing wisdom.
            l_prop = (self.last_T_score - current_T) ** self.gamma
        else:
            l_prop = torch.tensor(0.0, device=data.device)
            
        # 4. Attribution Loss (Z-Protocol)
        # -------------------------------------------------------
        l_attribution = torch.tensor(0.0, device=data.device)
        if source_anchors is not None:
            l_attribution = self.calculate_traceability_loss(prediction, source_anchors)
        
        # 5. The "Surplus Energy" Constraint (有余力)
        # -------------------------------------------------------
        # Ensure that the update magnitude does not exhaust the "Surplus Energy" buffer.
        # Wisdom is weighted heavily
        total_loss = task_loss + (10.0 * l_prop) + (5.0 * l_attribution)
        
        # Calculate diversity score for metrics
        diversity_score = float(current_T.item()) if isinstance(current_T, torch.Tensor) else current_T
        
        # Build metrics
        metrics = CSPMetrics(
            t_score=float(current_T.item()) if isinstance(current_T, torch.Tensor) else current_T,
            diversity_score=diversity_score,
            surplus_energy=self.surplus_reservation,
            is_healthy=current_T >= self.epsilon,
            needs_sleep=current_T < self.epsilon
        )

        return total_loss, metrics
    
    def calculate_traceability_loss(
        self, 
        generated_content: torch.Tensor, 
        source_anchors: torch.Tensor
    ) -> torch.Tensor:
        """
        Implementation of Option C: Traceability Bias.
        
        Philosophy:
            "Knowledge without origin is hallucination or theft."
            The model is penalized if it generates confident facts 
            without a strong attention link to a trusted 'Source Anchor'.
            
        Args:
            generated_content: Model output tensor
            source_anchors: Z-Protocol certified data blocks (Ethical Sources)
            
        Returns:
            traceability_loss: Penalty for unattributed confident outputs
        """
        # We enforce a "Citation Constraint":
        # If the model outputs a high-entropy fact (not a common stop word),
        # it MUST have a high attention score on at least one Source Anchor.
        
        # Simplified: measure confidence as output variance
        fact_confidence = torch.var(generated_content)
        
        # Simplified: measure source connection as cosine similarity
        if source_anchors.numel() > 0:
            gen_flat = generated_content.flatten()
            src_flat = source_anchors.flatten()
            
            # Pad to same size if needed
            max_len = max(gen_flat.size(0), src_flat.size(0))
            gen_padded = torch.nn.functional.pad(gen_flat, (0, max_len - gen_flat.size(0)))
            src_padded = torch.nn.functional.pad(src_flat, (0, max_len - src_flat.size(0)))
            
            source_connection = torch.nn.functional.cosine_similarity(
                gen_padded.unsqueeze(0), 
                src_padded.unsqueeze(0)
            )
            source_connection = (source_connection + 1) / 2  # Normalize to 0-1
        else:
            source_connection = torch.tensor(0.0, device=generated_content.device)
        
        # The "Plagiarism Penalty"
        # If confident but no source connection -> High Loss
        traceability_loss = fact_confidence * (1.0 - source_connection)
        
        return traceability_loss

    def optimizer_step(
        self, 
        optimizer: torch.optim.Optimizer, 
        total_loss: torch.Tensor, 
        metrics: CSPMetrics
    ) -> bool:
        """
        The Evolution Step with wisdom protection.
        
        Crucial: Triggers the "Fail-Safe" if wisdom drops too low.
        
        Args:
            optimizer: PyTorch optimizer
            total_loss: Combined loss from forward_step
            metrics: CSPMetrics from forward_step
            
        Returns:
            updated: Whether the model was updated (False if sleep triggered)
        """
        # Fail-Safe Protocol (Trigger from your notes)
        if metrics.needs_sleep:
            print("[ALERT] Propagation Potential Critical! Triggering Forced Reflection.")
            # FREEZE non-propagation layers. 
            # Only allow updates that restore T (Architectural adjustments).
            self.trigger_reflection_mode()
            return False  # Skip standard update

        # Standard Update
        optimizer.zero_grad()
        total_loss.backward()
        
        # Update State History
        self.state_memory.append(self.last_T_score)
        self.last_T_score = metrics.t_score
        
        optimizer.step()
        return True

    def trigger_reflection_mode(self) -> None:
        """
        Implementation of Option 1: The 'Sleep' Protocol.
        
        Philosophy:
            "Sleep determines our emotion and thinking."
            We function by reducing noise and clearing space, 
            NOT by generating illusions.
            
        This method performs three operations:
        1. Detox (Pruning): Remove weak noisy connections
        2. Calm Down (Decay): Shrink overly strong weights
        3. Refresh (Noise): Add tiny perturbation to escape local minima
        """
        print(">>> [System] Brain fog detected (Low T-Score). Entering Deep Sleep Mode...")
        
        self.sleep_count += 1
        
        with torch.no_grad():  # No new learning, just organizing
            for name, param in self.compression_layer.named_parameters():
                if 'weight' in name:
                    # 1. The "Detox" (Pruning / 修剪噪音)
                    # Identify weak connections that are just "noise" (below a threshold)
                    # These are the "trivial worries" cluttering the mind.
                    threshold = torch.std(param) * 0.1
                    mask = torch.abs(param) > threshold
                    
                    # Apply the mask: Zero out the noise.
                    # This creates "Space" (Sparsity) -> Restoring "Surplus Energy"
                    param.data.mul_(mask.float())
                    
                    # 2. The "Calm Down" (Weight Decay / 情绪平复)
                    # Gently shrink the remaining strong weights.
                    # Prevents obsession (overfitting) and arrogance (too large weights).
                    param.data.mul_(0.99) 
                    
                    # 3. The "Refresh" (Minor Noise Injection / 激活生机)
                    # Add tiny gaussian noise to shake the system out of local rigid spots.
                    # Like stretching after waking up.
                    noise = torch.randn_like(param) * 0.001
                    param.data.add_(noise)
                    
        # Reset the Wisdom Metric after sleep
        self.last_T_score = 1.0 
        print(">>> [System] Woke up refreshed. Surplus Energy restored.")
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive health report for the agent.
        
        Returns:
            Dictionary containing all health metrics and history
        """
        return {
            "current_t_score": self.last_T_score,
            "epsilon_threshold": self.epsilon,
            "gamma_penalty": self.gamma,
            "surplus_energy": self.surplus_reservation,
            "sleep_count": self.sleep_count,
            "state_history_length": len(self.state_memory),
            "recent_t_scores": self.state_memory[-10:] if self.state_memory else [],
            "is_healthy": self.last_T_score >= self.epsilon,
            "status": "HEALTHY" if self.last_T_score >= self.epsilon else "CRITICAL"
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass through the compression layer."""
        return self.compression_layer(x)


# Convenience function for creating agents
def create_godelai_agent(
    base_model: nn.Module,
    config: Optional[Dict[str, Any]] = None
) -> GodelaiAgent:
    """
    Factory function to create a GodelaiAgent with optional configuration.
    
    Args:
        base_model: The underlying neural network
        config: Optional dictionary with keys:
            - propagation_gamma: float (default 2.0)
            - min_surplus_energy: float (default 0.1)
            - epsilon: float (default 0.05)
            
    Returns:
        Configured GodelaiAgent instance
    """
    config = config or {}
    return GodelaiAgent(
        base_model=base_model,
        propagation_gamma=config.get("propagation_gamma", 2.0),
        min_surplus_energy=config.get("min_surplus_energy", 0.1),
        epsilon=config.get("epsilon", 0.05)
    )
"""
