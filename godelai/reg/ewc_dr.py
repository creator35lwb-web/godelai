"""
EWC-DR: Elastic Weight Consolidation with Dead Rectification (Logits Reversal)
=============================================================================
Based on the principle that standard EWC has fundamental importance estimation
flaws — specifically, it over-penalizes "dead" parameters (near-zero Fisher
information) which should be free to adapt.

The Logits Reversal fix:
    1. Compute Fisher Information Matrix (FIM) as in standard EWC
    2. Identify "dead" parameters: those with Fisher < dead_threshold
    3. For dead parameters, REVERSE the penalty direction — instead of
       penalizing deviation from old values, ENCOURAGE deviation
       (these weights are not important and should be free to adapt)
    4. For "alive" parameters (high Fisher), apply standard EWC penalty

This approach addresses the core flaw: vanilla EWC treats all parameters
equally in terms of "importance direction," but parameters with near-zero
Fisher information are NOT important to old tasks and should be plastic.

Reference: EWC-DR paper (March 2026, arxiv:2603.18596)
Author: L (GodelAI CEO) — MACP v2.2 "Identity"
Date: April 3, 2026
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import copy


class EWCDR(nn.Module):
    """
    EWC with Dead Rectification (Logits Reversal).

    Improves upon vanilla EWC by:
    - Identifying "dead" parameters (low Fisher information)
    - Reversing the penalty for dead parameters (encouraging plasticity)
    - Applying standard EWC penalty only to "alive" parameters

    This prevents the over-regularization of unimportant weights that
    causes vanilla EWC to underperform compared to simple L2 regularization.
    """

    def __init__(
        self,
        ewc_lambda: float = 0.4,
        dead_threshold: float = 1e-4,
        reversal_strength: float = 0.1,
        normalize_fisher: bool = True,
        log_dir: Optional[str] = None,
    ):
        """
        Args:
            ewc_lambda: Regularization strength for alive parameters (standard EWC weight)
            dead_threshold: Fisher information below this = "dead" parameter
            reversal_strength: Penalty strength for dead parameters (encourages deviation)
            normalize_fisher: Whether to normalize Fisher values to [0, 1]
            log_dir: Directory for saving EWC-DR metrics
        """
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.dead_threshold = dead_threshold
        self.reversal_strength = reversal_strength
        self.normalize_fisher = normalize_fisher
        self.log_dir = Path(log_dir) if log_dir else Path("./ewc_dr_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Stored after consolidation
        self.fisher: Dict[str, torch.Tensor] = {}
        self.old_params: Dict[str, torch.Tensor] = {}
        self.is_consolidated = False

        # Metrics
        self.metrics_history = []

    def consolidate(
        self,
        model: nn.Module,
        dataloader: List[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        criterion: nn.Module,
        n_samples: int = 200,
    ) -> Dict[str, float]:
        """
        Compute Fisher Information Matrix and store old parameters.

        Args:
            model: The trained model after Task A
            dataloader: Task A data batches [(inputs, targets), ...]
            device: Computation device
            criterion: Loss function
            n_samples: Max samples to use for Fisher estimation

        Returns:
            dict: Consolidation statistics (alive/dead parameter counts)
        """
        model.eval()

        # Initialize Fisher accumulators
        fisher_accum = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        sample_count = 0
        for inputs, targets in dataloader:
            if sample_count >= n_samples:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            model.zero_grad()
            outputs = model(inputs)
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)

            loss = criterion(outputs_flat, targets_flat)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_accum[name] += param.grad.data.pow(2)

            sample_count += 1

        # Average Fisher over samples
        n = max(sample_count, 1)
        self.fisher = {
            name: fisher_accum[name] / n
            for name in fisher_accum
        }

        # Normalize Fisher to [0, 1] if requested
        if self.normalize_fisher:
            all_fisher_values = torch.cat([f.flatten() for f in self.fisher.values()])
            fisher_max = all_fisher_values.max().item()
            if fisher_max > 0:
                self.fisher = {
                    name: f / fisher_max
                    for name, f in self.fisher.items()
                }

        # Store old parameters (detached copy)
        self.old_params = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        self.is_consolidated = True

        # Compute statistics
        total_params = sum(f.numel() for f in self.fisher.values())
        dead_params = sum(
            (f < self.dead_threshold).sum().item()
            for f in self.fisher.values()
        )
        alive_params = total_params - dead_params

        stats = {
            "total_parameters": total_params,
            "alive_parameters": alive_params,
            "dead_parameters": dead_params,
            "dead_fraction": dead_params / total_params,
            "alive_fraction": alive_params / total_params,
            "fisher_mean": all_fisher_values.mean().item() if self.normalize_fisher else 0.0,
            "fisher_max": fisher_max if self.normalize_fisher else 0.0,
            "samples_used": sample_count,
        }

        self._log_consolidation(stats)
        return stats

    def compute_penalty(self, model: nn.Module) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the EWC-DR penalty.

        For alive parameters (Fisher >= dead_threshold):
            penalty += ewc_lambda * Fisher * (param - old_param)^2

        For dead parameters (Fisher < dead_threshold):
            penalty -= reversal_strength * (param - old_param)^2
            (Logits Reversal: encourage these to change freely)

        Returns:
            Tuple of (penalty_loss, metrics_dict)
        """
        if not self.is_consolidated:
            return torch.tensor(0.0), {}

        alive_penalty = torch.tensor(0.0)
        dead_reversal = torch.tensor(0.0)
        alive_count = 0
        dead_count = 0

        for name, param in model.named_parameters():
            if name not in self.fisher or not param.requires_grad:
                continue

            fisher = self.fisher[name]
            old_param = self.old_params[name]
            delta = (param - old_param).pow(2)

            # Separate alive and dead parameters
            alive_mask = fisher >= self.dead_threshold
            dead_mask = ~alive_mask

            # Alive: standard EWC penalty (preserve important weights)
            if alive_mask.any():
                alive_penalty = alive_penalty + (
                    self.ewc_lambda * (fisher * delta)[alive_mask].sum()
                )
                alive_count += alive_mask.sum().item()

            # Dead: reversal penalty (encourage plasticity of unimportant weights)
            if dead_mask.any():
                dead_reversal = dead_reversal + (
                    self.reversal_strength * delta[dead_mask].sum()
                )
                dead_count += dead_mask.sum().item()

        # Net penalty: alive regularization minus dead reversal
        # (reversal reduces total penalty, encouraging dead params to adapt)
        net_penalty = alive_penalty - dead_reversal
        # Clamp to 0 minimum (don't create negative loss that could destabilize)
        net_penalty = torch.clamp(net_penalty, min=0.0)

        metrics = {
            "alive_penalty": alive_penalty.item(),
            "dead_reversal": dead_reversal.item(),
            "net_penalty": net_penalty.item(),
            "alive_params_penalized": alive_count,
            "dead_params_reversed": dead_count,
        }

        return net_penalty, metrics

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC-DR penalty (for use in training loop)."""
        penalty, metrics = self.compute_penalty(model)
        self.metrics_history.append({
            "timestamp": datetime.now().isoformat(),
            **metrics
        })
        return penalty

    def _log_consolidation(self, stats: Dict):
        """Save consolidation statistics to log directory."""
        log_file = self.log_dir / f"consolidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w") as f:
            json.dump(stats, f, indent=2)

    def save_state(self, tag: str = "checkpoint") -> str:
        """Save EWC-DR state for later resumption."""
        state = {
            "ewc_lambda": self.ewc_lambda,
            "dead_threshold": self.dead_threshold,
            "reversal_strength": self.reversal_strength,
            "is_consolidated": self.is_consolidated,
            "metrics_history": self.metrics_history,
            "tag": tag,
            "timestamp": datetime.now().isoformat(),
        }
        save_path = self.log_dir / f"ewc_dr_state_{tag}.json"
        with open(save_path, "w") as f:
            json.dump(state, f, indent=2)
        return str(save_path)

    def get_summary(self) -> Dict:
        """Return a summary of EWC-DR performance metrics."""
        if not self.metrics_history:
            return {"message": "No metrics recorded yet"}

        net_penalties = [m["net_penalty"] for m in self.metrics_history]
        alive_penalties = [m["alive_penalty"] for m in self.metrics_history]
        dead_reversals = [m["dead_reversal"] for m in self.metrics_history]

        return {
            "steps_recorded": len(self.metrics_history),
            "avg_net_penalty": sum(net_penalties) / len(net_penalties),
            "avg_alive_penalty": sum(alive_penalties) / len(alive_penalties),
            "avg_dead_reversal": sum(dead_reversals) / len(dead_reversals),
            "max_net_penalty": max(net_penalties),
            "is_consolidated": self.is_consolidated,
        }


class VanillaEWC(nn.Module):
    """
    Standard Elastic Weight Consolidation (Kirkpatrick et al., 2017).
    Kept as baseline for comparison with EWC-DR.
    """

    def __init__(self, ewc_lambda: float = 0.4):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.fisher: Dict[str, torch.Tensor] = {}
        self.old_params: Dict[str, torch.Tensor] = {}
        self.is_consolidated = False

    def consolidate(
        self,
        model: nn.Module,
        dataloader: List[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        criterion: nn.Module,
        n_samples: int = 200,
    ):
        """Compute Fisher Information Matrix."""
        model.eval()
        fisher_accum = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        sample_count = 0
        for inputs, targets in dataloader:
            if sample_count >= n_samples:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            model.zero_grad()
            outputs = model(inputs)
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)
            loss = criterion(outputs_flat, targets_flat)
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_accum[name] += param.grad.data.pow(2)
            sample_count += 1

        n = max(sample_count, 1)
        self.fisher = {name: fisher_accum[name] / n for name in fisher_accum}
        self.old_params = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.is_consolidated = True

    def forward(self, model: nn.Module) -> torch.Tensor:
        if not self.is_consolidated:
            return torch.tensor(0.0)
        penalty = torch.tensor(0.0)
        for name, param in model.named_parameters():
            if name in self.fisher and param.requires_grad:
                penalty = penalty + (
                    self.ewc_lambda
                    * self.fisher[name]
                    * (param - self.old_params[name]).pow(2)
                ).sum()
        return penalty
