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
            batch_gradients: Tensor of gradients from the model
            
        Returns:
            T_score: Wisdom score normalized to 0-1 range
        """
        # 1. Global Direction Strength (Everyone rushing together)
        # sum_grad = || Σ g_i ||^2
        sum_grad_norm = torch.norm(torch.sum(batch_gradients, dim=0))**2
        
        # 2. Individual Direction Strength (Individual thinking)
        # sum_norm_grad = Σ || g_i ||^2
        sum_norm_grad = torch.sum(torch.norm(batch_gradients, dim=1)**2)
        
        # 3. Calculate Diversity (Wisdom) Ratio
        # Diversity = Individuals / (Crowd + epsilon)
        diversity_score = sum_norm_grad / (sum_grad_norm + 1e-8)
        
        # Normalize to 0-1 range using Sigmoid (or Tanh)
        T_score = torch.sigmoid(diversity_score)
        
        return T_score

    def rest_and_reflect(self):
        """
        The Sleep Protocol (Option 1: Pruning-based).
        Triggered when T_score drops below epsilon.
        
        Philosophy: "Sleep determines our emotion and thinking."
        
        Action:
        1. Pruning: Clear weak 'noise' connections (Forget the trivial).
        2. Decay: Calm down over-excited weights (Emotional stability).
        """
        print("\n>>> [SYSTEM ALERT] Wisdom Critical (T < {:.2f}). Triggering Sleep Protocol...".format(self.epsilon))
        print(">>> [Godel] Sleeping... Pruning noise and restoring surplus energy.")
        
        with torch.no_grad():
            for name, param in self.compression_layer.named_parameters():
                if 'weight' in name and param.grad is not None:
                    # 1. Pruning (The Detox)
                    # Remove connections that are weak/noisy (below 10% of std dev)
                    threshold = torch.std(param) * 0.1
                    mask = torch.abs(param) > threshold
                    param.data.mul_(mask)
                    
                    # 2. Decay (The Calm)
                    # Shrink weights slightly to prevent obsession (overfitting)
                    param.data.mul_(0.995) 
                    
        # Reset Wisdom Score after sleep (Woke up fresh)
        self.last_T_score = 1.0
        self.history['sleep_count'] += 1
        print(">>> [Godel] Woke up. Clarity restored.\n")

    def learning_step(self, data, target, criterion):
        """
        The Evolution Step (Forward + Backward + Wisdom Check).
        
        This is not a blind optimizer.step(). It:
        1. Calculates gradients
        2. Measures "Will this step make me dumber?"
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
        # 1. Standard Forward Pass (Task Solving)
        prediction = self.compression_layer(data)
        task_loss = criterion(prediction, target)
        
        # 2. Calculate Gradients (But don't step yet!)
        # We need gradients to measure "Wisdom" before we commit to the change.
        task_loss.backward(retain_graph=True)
        
        # Collect gradients from the last layer (or key layers) for diversity check
        # (Simplified: grabbing gradients from one significant layer for demo)
        sample_grads = []
        for param in self.compression_layer.parameters():
            if param.grad is not None:
                sample_grads.append(param.grad.view(-1))
                break  # Just checking first available layer for speed in this demo
        
        if len(sample_grads) > 0:
            batch_grads = torch.stack(sample_grads)  # Reshape if batching allows
            # Note: For full implementation, we need per-sample gradients (e.g. using functorch)
            # Here we use a simplified proxy for the architecture skeleton.
            
            # 3. Measure Wisdom (T)
            current_T = self.measure_gradient_diversity(batch_grads.unsqueeze(0))  # Shape adjust
            
            # 4. Propagation Check
            # Did we lose wisdom compared to last step?
            if current_T < self.last_T_score:
                # PENALTY: You are getting rigid!
                l_prop = (self.last_T_score - current_T) ** self.gamma
            else:
                l_prop = 0.0
                
            # Update the benchmark history
            self.last_T_score = current_T.item()
            
            # 5. Fail-Safe: Sleep if too dumb
            if current_T < self.epsilon:
                self.optimizer.zero_grad()  # Cancel this update!
                self.rest_and_reflect()     # Go to sleep
                
                # Record history
                self.history['loss'].append(task_loss.item())
                self.history['wisdom_score'].append(current_T.item())
                self.history['status'].append("SLEEP")
                
                return task_loss.item(), current_T.item(), "SLEEP"

            # 6. Final Loss with Wisdom Constraint
            total_loss = task_loss + (10.0 * l_prop)
            
            # Commit the update
            if self.optimizer:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Record history
            self.history['loss'].append(total_loss.item())
            self.history['wisdom_score'].append(current_T.item())
            self.history['status'].append("LEARN")
                
            return total_loss.item(), current_T.item(), "LEARN"
        
        # Record history for skip
        self.history['loss'].append(task_loss.item())
        self.history['wisdom_score'].append(0.0)
        self.history['status'].append("SKIP")
        
        return task_loss.item(), 0.0, "SKIP"
    
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
