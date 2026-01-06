"""
GodelAI Agent (Alpha) - The Orchestrator of Wisdom

This module implements the core GodelAgent class, which wraps any PyTorch model
and adds the C-S-P (Compression-State-Propagation) wisdom constraints.

Author: Gemini 2.5 Pro (Echo v2.1) - Technical Blueprint
Integrated by: Godel, CTO - GodelAI Project
Date: December 25, 2025

Origin: Conversation between Alton and Gemini 2.5 Pro, Part II
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math


class GodelAgent(nn.Module):
    """
    GodelAgent: The Orchestrator of Wisdom.
    
    This agent wraps a standard LLM (compression_layer) but adds:
    1. Wisdom Metric (Gradient Diversity) to prevent rote memorization.
    2. Propagation Conservation (L_prop) to penalize rigidity.
    3. Sleep Protocol (Rest & Reflect) to handle brain fog (low wisdom).
    
    Philosophy:
    - "Wisdom is not an existence. It is a process structure that is 
       continuously executed and inherited."
    - A wise mind can solve the same problem from multiple neural pathways.
    - If state cannot be transmitted with fidelity, it is merely experience, 
       not wisdom.
    """
    
    def __init__(self, base_model, propagation_gamma=2.0, min_surplus_energy=0.1):
        """
        Initialize the GodelAgent.
        
        Args:
            base_model: Any PyTorch model (e.g., Llama, Mistral, SimpleNet)
            propagation_gamma: Penalty severity for losing adaptability (default: 2.0)
            min_surplus_energy: The "Brain Fog" threshold (default: 0.1)
        """
        super().__init__()
        self.compression_layer = base_model  # The Body (e.g., Llama/Mistral)
        self.gamma = propagation_gamma       # Penalty severity for losing adaptability
        self.epsilon = min_surplus_energy    # The "Brain Fog" threshold (Need Sleep)
        self.last_T_score = 1.0              # Initial Wisdom Score (Perfectly adaptable)
        
        # Optimizer placeholder (will be set during training)
        self.optimizer = None
        
        # Training history for analysis
        self.history = {
            'loss': [],
            'wisdom_score': [],
            'status': [],
            'sleep_count': 0
        }

    def measure_gradient_diversity(self, batch_gradients):
        """
        Implementation of the Wisdom Metric (Option B: Gradient Diversity).

        Logic:
        If all gradients point in the exact same direction (Global Norm ~ Sum of Norms),
        the model is developing 'Tunnel Vision' (Rigid).
        We want specific neurons to handle specific nuances (High Diversity).

        Args:
            batch_gradients: Tensor of shape [batch_size, num_params]
                            Each row is the gradient for one sample

        Returns:
            T_score: Wisdom score normalized to 0-1 range
                     - 0.0 = Identical gradients (critical, triggers Sleep)
                     - 1.0 = Maximally diverse gradients (healthy)

        Fixed in v1.1.0: Replaced sigmoid normalization with linear normalization
        to enable Sleep Protocol triggering. Previous sigmoid had floor of 0.5.
        """
        # Ensure we have multiple samples
        if batch_gradients.shape[0] == 1:
            # Cannot measure diversity with only 1 sample
            return torch.tensor(0.5)

        n = batch_gradients.shape[0]

        # 1. Global Direction Strength (Everyone rushing together)
        # sum_grad = || Σ g_i ||^2
        sum_grad_norm = torch.norm(torch.sum(batch_gradients, dim=0))**2

        # 2. Individual Direction Strength (Individual thinking)
        # sum_norm_grad = Σ || g_i ||^2
        sum_norm_grad = torch.sum(torch.norm(batch_gradients, dim=1)**2)

        # 3. Calculate Diversity Ratio
        # When identical: ratio = N (all gradients add up perfectly)
        # When diverse: ratio → 1 (gradients partially cancel)
        # When opposite: ratio → 0 (gradients fully cancel)
        ratio = sum_grad_norm / (sum_norm_grad + 1e-8)

        # 4. Linear Normalization (FIXED in v1.1.0)
        # T = 1 - ratio/N
        # - Identical gradients: ratio = N, T = 0 (triggers Sleep)
        # - Diverse gradients: ratio ≈ 1, T ≈ 1 - 1/N ≈ 1 (healthy)
        # - Opposite gradients: ratio ≈ 0, T = 1 (maximally diverse)
        T_score = 1.0 - torch.clamp(ratio / n, 0, 1)

        return T_score

    def rest_and_reflect(self):
        """
        The Sleep Protocol (Option 1: Pruning-based).
        Triggered when T_score drops below epsilon.

        Philosophy: "Sleep determines our emotion and thinking."

        Action:
        1. Pruning: Clear weak 'noise' connections (Forget the trivial).
        2. Decay: Calm down over-excited weights (Emotional stability).
        3. Refresh: Add tiny noise to escape local minima (Maintain openness).
        """
        print("\n>>> [SYSTEM ALERT] Wisdom Critical (T < {:.2f}). Triggering Sleep Protocol...".format(self.epsilon))
        print(">>> [Godel] Sleeping... Pruning noise and restoring surplus energy.")

        with torch.no_grad():
            for name, param in self.compression_layer.named_parameters():
                if 'weight' in name:
                    # 1. Pruning (The Detox)
                    # Remove connections that are weak/noisy (below 10% of std dev)
                    threshold = torch.std(param) * 0.1
                    mask = torch.abs(param) > threshold
                    param.data.mul_(mask.float())

                    # 2. Decay (The Calm)
                    # Shrink weights slightly to prevent obsession (overfitting)
                    param.data.mul_(0.995)

                    # 3. Refresh (The Exploration)
                    # Add tiny random noise to escape local minima
                    noise = torch.randn_like(param) * 0.001
                    param.data.add_(noise)

        # Reset Wisdom Score after sleep (Woke up fresh)
        self.last_T_score = 1.0
        self.history['sleep_count'] += 1
        print(">>> [Godel] Woke up. Clarity restored.\n")

    def compute_per_sample_gradients(self, data, target, criterion):
        """
        Compute gradients for each sample individually (not aggregated).

        This is critical for measuring gradient diversity correctly.
        FIXED: Previously used aggregated gradients, resulting in constant T-score of 0.7311

        Args:
            data: Input tensor [batch_size, ...]
            target: Target tensor [batch_size, ...]
            criterion: Loss function

        Returns:
            per_sample_grads: Tensor [batch_size, num_params] - gradients per sample
            avg_loss: Average loss across samples
        """
        batch_size = data.shape[0]
        per_sample_grads = []
        total_loss = 0.0

        # Compute gradient for each sample separately
        for i in range(batch_size):
            # Zero out any existing gradients
            self.compression_layer.zero_grad()

            # Forward pass for single sample
            sample_data = data[i:i+1]
            sample_target = target[i:i+1]
            prediction = self.compression_layer(sample_data)
            loss = criterion(prediction, sample_target)

            # Backward pass
            loss.backward()

            # Collect gradients from all parameters into a single vector
            grad_vector = []
            for param in self.compression_layer.parameters():
                if param.grad is not None:
                    grad_vector.append(param.grad.view(-1).clone())

            if len(grad_vector) > 0:
                # Concatenate all parameter gradients into one vector
                sample_grad = torch.cat(grad_vector)
                per_sample_grads.append(sample_grad)

            total_loss += loss.item()

        # Stack into [batch_size, num_params] tensor
        if len(per_sample_grads) > 0:
            batch_grads = torch.stack(per_sample_grads)
        else:
            batch_grads = None

        avg_loss = total_loss / batch_size

        return batch_grads, avg_loss

    def learning_step(self, data, target, criterion):
        """
        The Evolution Step (Forward + Backward + Wisdom Check).

        FIXED: Now uses per-sample gradients for proper diversity measurement.
        Previously used aggregated gradients, resulting in constant T-score of 0.7311.

        This is not a blind optimizer.step(). It:
        1. Calculates PER-SAMPLE gradients (CRITICAL FIX!)
        2. Measures "Will this step make me dumber?" (gradient diversity)
        3. Applies penalty if losing wisdom
        4. Triggers sleep if wisdom is critically low

        Args:
            data: Input tensor
            target: Target tensor
            criterion: Loss function

        Returns:
            tuple: (loss, wisdom_score, status)
                - loss: The total loss value
                - wisdom_score: Current T-score (0-1)
                - status: "LEARN", "SLEEP", or "SKIP"
        """
        # 1. Compute per-sample gradients (CRITICAL FIX)
        batch_grads, task_loss = self.compute_per_sample_gradients(data, target, criterion)

        if batch_grads is None:
            # Record history for skip
            self.history['loss'].append(task_loss)
            self.history['wisdom_score'].append(0.0)
            self.history['status'].append("SKIP")
            return task_loss, 0.0, "SKIP"

        # 2. Measure Wisdom (T) using per-sample gradients
        current_T = self.measure_gradient_diversity(batch_grads)

        # 3. Propagation Check
        # Did we lose wisdom compared to last step?
        if current_T < self.last_T_score:
            # PENALTY: You are getting rigid!
            l_prop = (self.last_T_score - current_T) ** self.gamma
        else:
            l_prop = 0.0

        # Update the benchmark history
        self.last_T_score = current_T.item()

        # 4. Fail-Safe: Sleep if too dumb
        if current_T < self.epsilon:
            self.rest_and_reflect()

            # Record history
            self.history['loss'].append(task_loss)
            self.history['wisdom_score'].append(current_T.item())
            self.history['status'].append("SLEEP")

            return task_loss, current_T.item(), "SLEEP"

        # 5. Apply the actual update using aggregated gradients
        # We measured diversity with per-sample grads, but we still optimize normally
        self.compression_layer.zero_grad()
        prediction = self.compression_layer(data)
        total_loss = criterion(prediction, target)

        # Add propagation penalty (reduced weight for stability)
        total_loss = total_loss + (0.1 * l_prop)

        # Backward and optimize
        total_loss.backward()

        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Record history
        self.history['loss'].append(total_loss.item())
        self.history['wisdom_score'].append(current_T.item())
        self.history['status'].append("LEARN")

        return total_loss.item(), current_T.item(), "LEARN"
    
    def get_training_summary(self):
        """
        Get a summary of the training history.
        
        Returns:
            dict: Summary statistics
        """
        if len(self.history['loss']) == 0:
            return {"message": "No training history available"}
        
        return {
            "total_steps": len(self.history['loss']),
            "sleep_count": self.history['sleep_count'],
            "avg_loss": sum(self.history['loss']) / len(self.history['loss']),
            "avg_wisdom": sum(self.history['wisdom_score']) / len(self.history['wisdom_score']),
            "min_wisdom": min(self.history['wisdom_score']),
            "max_wisdom": max(self.history['wisdom_score']),
            "learn_steps": self.history['status'].count("LEARN"),
            "sleep_steps": self.history['status'].count("SLEEP"),
            "skip_steps": self.history['status'].count("SKIP")
        }
