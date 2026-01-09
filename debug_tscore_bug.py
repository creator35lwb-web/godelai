#!/usr/bin/env python3
"""
T-Score Bug Investigation
==========================

Compare Shadow mode vs Active mode T-Score computation on IDENTICAL data.

Author: Claude Code
Date: January 9, 2026
"""
import torch
import torch.nn as nn

print("="*80)
print("T-SCORE BUG INVESTIGATION")
print("="*80)
print()

# Create synthetic gradient data with known properties
batch_size = 32
num_params = 1000

print(f"Synthetic Data: {batch_size} samples, {num_params} parameters")
print()

# ============================================================================
# TEST 1: IDENTICAL GRADIENTS (should give T-Score ≈ 0)
# ============================================================================

print("TEST 1: IDENTICAL GRADIENTS")
print("-" * 40)

# All gradients point in same direction
identical_grad = torch.randn(num_params)
batch_gradients_identical = identical_grad.unsqueeze(0).repeat(batch_size, 1)

print(f"All {batch_size} gradients are identical")
print()

# ACTIVE MODE (GodelAI agent.py implementation)
def compute_tscore_active(batch_gradients):
    """Active mode: From godelai/agent.py (measure_gradient_diversity)"""
    n = batch_gradients.shape[0]

    # 1. Global Direction Strength (squared norm)
    sum_grad_norm = torch.norm(torch.sum(batch_gradients, dim=0))**2

    # 2. Individual Direction Strength (sum of squared norms)
    sum_norm_grad = torch.sum(torch.norm(batch_gradients, dim=1)**2)

    # 3. Calculate ratio
    ratio = sum_grad_norm / (sum_norm_grad + 1e-8)

    # 4. T-Score
    T_score = 1.0 - torch.clamp(ratio / n, 0, 1)

    return T_score.item()


# SHADOW MODE (A/B test implementation - BUGGY)
def compute_tscore_shadow_BUGGY(grad_matrix):
    """Shadow mode: From run_ab_comparison.py (BUGGY VERSION)"""
    n = grad_matrix.shape[0]
    sum_grad = torch.sum(grad_matrix, dim=0)
    sum_norm_grad = torch.norm(sum_grad)  # NOT squared!
    sum_grad_norm = torch.sum(torch.norm(grad_matrix, dim=1))  # NOT squared!
    ratio = sum_grad_norm / (sum_norm_grad + 1e-8)
    t_score = 1.0 - torch.clamp(ratio / n, 0, 1)
    return t_score.item()


# SHADOW MODE (FIXED - should match Active)
def compute_tscore_shadow_FIXED(grad_matrix):
    """Shadow mode: CORRECTED to match agent.py"""
    n = grad_matrix.shape[0]

    # 1. Global Direction Strength (squared norm) - FIXED
    sum_grad_norm = torch.norm(torch.sum(grad_matrix, dim=0))**2

    # 2. Individual Direction Strength (sum of squared norms) - FIXED
    sum_norm_grad = torch.sum(torch.norm(grad_matrix, dim=1)**2)

    # 3. Calculate ratio
    ratio = sum_grad_norm / (sum_norm_grad + 1e-8)

    # 4. T-Score
    t_score = 1.0 - torch.clamp(ratio / n, 0, 1)
    return t_score.item()


# Compute with all three methods
t_active = compute_tscore_active(batch_gradients_identical)
t_shadow_buggy = compute_tscore_shadow_BUGGY(batch_gradients_identical)
t_shadow_fixed = compute_tscore_shadow_FIXED(batch_gradients_identical)

print(f"Active mode (GodelAI):  {t_active:.4f}")
print(f"Shadow mode (BUGGY):    {t_shadow_buggy:.4f}")
print(f"Shadow mode (FIXED):    {t_shadow_fixed:.4f}")
print()

# Expected: T ~= 0 for identical gradients
print(f"Expected: T ~= 0.0 (identical gradients should have low diversity)")
print(f"Active correct? {abs(t_active) < 0.01}")
print(f"Shadow (buggy) correct? {abs(t_shadow_buggy) < 0.01}")
print(f"Shadow (fixed) correct? {abs(t_shadow_fixed) < 0.01}")
print()

# ============================================================================
# TEST 2: DIVERSE GRADIENTS (should give T-Score ≈ 1)
# ============================================================================

print("="*80)
print("TEST 2: DIVERSE GRADIENTS")
print("-" * 40)

# Random gradients (diverse, some cancellation)
batch_gradients_diverse = torch.randn(batch_size, num_params)

print(f"{batch_size} random diverse gradients")
print()

t_active = compute_tscore_active(batch_gradients_diverse)
t_shadow_buggy = compute_tscore_shadow_BUGGY(batch_gradients_diverse)
t_shadow_fixed = compute_tscore_shadow_FIXED(batch_gradients_diverse)

print(f"Active mode (GodelAI):  {t_active:.4f}")
print(f"Shadow mode (BUGGY):    {t_shadow_buggy:.4f}")
print(f"Shadow mode (FIXED):    {t_shadow_fixed:.4f}")
print()

# Expected: T close to 1 for diverse gradients
print(f"Expected: T ~= 0.97 (diverse gradients, some cancellation)")
print(f"Active in range? {0.8 < t_active < 1.0}")
print(f"Shadow (buggy) in range? {0.8 < t_shadow_buggy < 1.0}")
print(f"Shadow (fixed) in range? {0.8 < t_shadow_fixed < 1.0}")
print()

# ============================================================================
# TEST 3: OPPOSITE GRADIENTS (should give T-Score ≈ 1)
# ============================================================================

print("="*80)
print("TEST 3: OPPOSITE GRADIENTS (maximum cancellation)")
print("-" * 40)

# Half positive, half negative (opposite directions)
half_size = batch_size // 2
grad_positive = torch.randn(num_params)
batch_gradients_opposite = torch.zeros(batch_size, num_params)
batch_gradients_opposite[:half_size] = grad_positive
batch_gradients_opposite[half_size:] = -grad_positive

print(f"{half_size} gradients in one direction, {half_size} in opposite direction")
print()

t_active = compute_tscore_active(batch_gradients_opposite)
t_shadow_buggy = compute_tscore_shadow_BUGGY(batch_gradients_opposite)
t_shadow_fixed = compute_tscore_shadow_FIXED(batch_gradients_opposite)

print(f"Active mode (GodelAI):  {t_active:.4f}")
print(f"Shadow mode (BUGGY):    {t_shadow_buggy:.4f}")
print(f"Shadow mode (FIXED):    {t_shadow_fixed:.4f}")
print()

# Expected: T ~= 1 (maximal diversity, perfect cancellation)
print(f"Expected: T ~= 1.0 (opposite gradients cancel perfectly)")
print(f"Active correct? {t_active > 0.99}")
print(f"Shadow (buggy) correct? {t_shadow_buggy > 0.99}")
print(f"Shadow (fixed) correct? {t_shadow_fixed > 0.99}")
print()

# ============================================================================
# DIAGNOSIS
# ============================================================================

print("="*80)
print("DIAGNOSIS")
print("="*80)
print()

print("THE BUG:")
print("-" * 40)
print()
print("Shadow mode (A/B test scripts) uses WRONG formula:")
print()
print("  BUGGY Shadow:")
print("    sum_norm_grad = ||sum(g)||        (linear norm, NOT squared)")
print("    sum_grad_norm = sum(||g||)        (sum of linear norms)")
print("    ratio = sum(||g||) / ||sum(g)||")
print()
print("  CORRECT Active (agent.py):")
print("    sum_grad_norm = ||sum(g)||^2      (SQUARED norm)")
print("    sum_norm_grad = sum(||g||^2)      (sum of SQUARED norms)")
print("    ratio = ||sum(g)||^2 / sum(||g||^2)")
print()
print("These are COMPLETELY DIFFERENT formulas!")
print()
print("For identical gradients:")
print("  Buggy: ratio = sum(||g||) / ||sum(g)|| = n*||g|| / (n*||g||) = 1")
print("         T = 1 - 1/n ~= 0.97 (WRONG! Should be ~=0)")
print()
print("  Correct: ratio = ||sum(g)||^2 / sum(||g||^2) = (n*||g||)^2 / (n*||g||^2) = n")
print("           T = 1 - n/n = 0 (CORRECT!)")
print()
print("="*80)
print()
print("CONCLUSION:")
print("  - Active mode (GodelAI agent.py): [CORRECT]")
print("  - Shadow mode (A/B test scripts): [WRONG - INVERTED!]")
print()
print("FIX: Replace Shadow mode implementation with correct formula (use squared norms)")
print("="*80)
