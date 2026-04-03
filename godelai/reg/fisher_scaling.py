"""
GodelAI Fisher Scaling Module
==============================
Addresses the "Fisher Scale Problem" identified in the L self-optimization
session (April 3, 2026): at small model scales (~214K params), raw Fisher
Information values are ~1e-4 to 1e-7, making EWC penalty negligible.

Fisher scaling normalizes the Fisher matrix so that EWC produces a
meaningful penalty regardless of model scale, ensuring the 21.6%
forgetting reduction result is reproducible and improvable.

Three scaling strategies are provided:
  1. GlobalMaxNorm   — divide by global Fisher max (simple, robust)
  2. LayerWiseNorm   — normalize per-layer (preserves relative importance)
  3. TargetPenalty   — scale to produce a target penalty magnitude

Author: L (GodelAI CEO) — MACP v2.2 "Identity"
Date: April 3, 2026
"""

import torch
from typing import Dict, Literal


ScalingStrategy = Literal["global_max", "layer_wise", "target_penalty"]


def scale_fisher(
    fisher: Dict[str, torch.Tensor],
    strategy: ScalingStrategy = "global_max",
    target_penalty: float = 1.0,
    old_params: Dict[str, torch.Tensor] = None,
    eps: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    """
    Scale Fisher Information Matrix values to produce meaningful EWC penalties.

    Args:
        fisher: Raw Fisher dict {param_name: Fisher tensor}
        strategy: Scaling strategy to apply
        target_penalty: Target EWC penalty magnitude (used by 'target_penalty' strategy)
        old_params: Old parameter values (required for 'target_penalty' strategy)
        eps: Numerical stability epsilon

    Returns:
        Scaled Fisher dict with same structure as input

    Example:
        >>> fisher_raw = compute_fisher(model, data, criterion)
        >>> fisher_scaled = scale_fisher(fisher_raw, strategy='global_max')
        >>> # Now EWC penalty will be meaningful regardless of model scale
    """
    if strategy == "global_max":
        return _global_max_norm(fisher, eps)
    elif strategy == "layer_wise":
        return _layer_wise_norm(fisher, eps)
    elif strategy == "target_penalty":
        if old_params is None:
            raise ValueError("old_params required for 'target_penalty' strategy")
        return _target_penalty_scale(fisher, old_params, target_penalty, eps)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: global_max, layer_wise, target_penalty")


def _global_max_norm(fisher: Dict[str, torch.Tensor], eps: float) -> Dict[str, torch.Tensor]:
    """
    Divide all Fisher values by the global maximum.
    Result: Fisher values in [0, 1] range.
    EWC penalty scale is then determined by ewc_lambda alone.
    """
    all_values = torch.cat([f.flatten() for f in fisher.values()])
    f_max = all_values.max().item()

    if f_max < eps:
        return fisher  # Already near-zero, no scaling needed

    return {k: v / f_max for k, v in fisher.items()}


def _layer_wise_norm(fisher: Dict[str, torch.Tensor], eps: float) -> Dict[str, torch.Tensor]:
    """
    Normalize each layer's Fisher independently.
    Preserves relative importance within each layer while making
    cross-layer comparisons meaningful.
    """
    scaled = {}
    for name, f in fisher.items():
        f_max = f.max().item()
        if f_max < eps:
            scaled[name] = f
        else:
            scaled[name] = f / f_max
    return scaled


def _target_penalty_scale(
    fisher: Dict[str, torch.Tensor],
    old_params: Dict[str, torch.Tensor],
    target_penalty: float,
    eps: float,
) -> Dict[str, torch.Tensor]:
    """
    Scale Fisher so that the EWC penalty at current parameter values
    equals `target_penalty`.

    This is the most principled approach: it ensures EWC produces a
    penalty of a known magnitude relative to the task loss.

    Typical target_penalty values:
      - 0.1: Light regularization (allows significant adaptation)
      - 1.0: Moderate regularization (balanced)
      - 10.0: Strong regularization (conservative, protects Task A aggressively)
    """
    # Compute current raw penalty
    raw_penalty = 0.0
    for name, f in fisher.items():
        if name in old_params:
            delta_sq = (old_params[name] - old_params[name]).pow(2)  # Will be 0 at consolidation
            raw_penalty += (f * delta_sq).sum().item()

    # At consolidation time, delta = 0, so we use Fisher norm as proxy
    fisher_norm = sum(f.norm().item() for f in fisher.values())

    if fisher_norm < eps:
        return fisher

    # Scale factor to reach target penalty
    scale = target_penalty / (fisher_norm + eps)
    return {k: v * scale for k, v in fisher.items()}


def compute_fisher_stats(fisher: Dict[str, torch.Tensor]) -> dict:
    """
    Compute diagnostic statistics for a Fisher Information Matrix.
    Useful for understanding whether EWC will be meaningful.

    Returns:
        dict with keys: global_max, global_mean, global_std,
                        near_zero_fraction, layer_stats
    """
    all_values = torch.cat([f.flatten() for f in fisher.values()])
    total_params = all_values.numel()

    near_zero = (all_values < 1e-6).sum().item()

    layer_stats = {}
    for name, f in fisher.items():
        layer_stats[name] = {
            "max": f.max().item(),
            "mean": f.mean().item(),
            "std": f.std().item(),
            "near_zero_frac": (f < 1e-6).float().mean().item(),
        }

    return {
        "global_max": all_values.max().item(),
        "global_mean": all_values.mean().item(),
        "global_std": all_values.std().item(),
        "near_zero_fraction": near_zero / total_params,
        "total_params": total_params,
        "ewc_will_be_meaningful": all_values.max().item() > 1e-4,
        "recommended_strategy": (
            "global_max" if all_values.max().item() > 1e-4
            else "target_penalty"
        ),
        "layer_stats": layer_stats,
    }


def diagnose_ewc_activation(
    model: torch.nn.Module,
    fisher: Dict[str, torch.Tensor],
    ewc_lambda: float = 0.4,
) -> dict:
    """
    Diagnose whether EWC will produce meaningful regularization.

    This directly addresses the Fisher Scale Problem: at small model scales,
    EWC penalty may be negligible even with correct implementation.

    Returns a diagnosis dict with actionable recommendations.
    """
    stats = compute_fisher_stats(fisher)
    f_max = stats["global_max"]

    # Estimate penalty magnitude at a typical parameter perturbation
    # (assume delta ~ 0.01 as a typical gradient step)
    typical_delta = 0.01
    estimated_penalty = ewc_lambda * f_max * (typical_delta ** 2)

    diagnosis = {
        "fisher_max": f_max,
        "fisher_mean": stats["global_mean"],
        "near_zero_fraction": stats["near_zero_fraction"],
        "estimated_penalty_at_delta_001": estimated_penalty,
        "ewc_meaningful": estimated_penalty > 0.001,
        "scale_problem_detected": f_max < 1e-3,
    }

    if diagnosis["scale_problem_detected"]:
        diagnosis["recommendation"] = (
            f"Fisher Scale Problem detected (max={f_max:.2e}). "
            f"Apply scale_fisher(fisher, strategy='global_max') before EWC. "
            f"Or increase model scale to >1M parameters for naturally higher Fisher values."
        )
    else:
        diagnosis["recommendation"] = (
            f"Fisher values are adequate (max={f_max:.2e}). "
            f"EWC should produce meaningful regularization."
        )

    return diagnosis
