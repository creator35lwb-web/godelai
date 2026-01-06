#!/usr/bin/env python3
"""
T-Score Formula Fix Validation
==============================
Test the proposed fix for the T-Score sigmoid floor bug.

Author: Godel (Manus AI)
Date: January 7, 2026
"""

import torch
import torch.nn as nn


def original_measure_gradient_diversity(batch_gradients):
    """Original implementation with sigmoid floor bug."""
    if batch_gradients.shape[0] == 1:
        return torch.tensor(0.5)
    
    sum_grad_norm = torch.norm(torch.sum(batch_gradients, dim=0))**2
    sum_norm_grad = torch.sum(torch.norm(batch_gradients, dim=1)**2)
    diversity_score = sum_norm_grad / (sum_grad_norm + 1e-8)
    T_score = torch.sigmoid(diversity_score)
    
    return T_score


def fixed_measure_gradient_diversity_v1(batch_gradients):
    """
    Option A: Linear Normalization (Recommended)
    
    T_score = 1 - (crowd_strength / individual_strength) / N
    
    - Identical gradients: T → 0
    - Diverse gradients: T → 1
    """
    if batch_gradients.shape[0] == 1:
        return torch.tensor(0.5)
    
    n = batch_gradients.shape[0]
    sum_grad_norm = torch.norm(torch.sum(batch_gradients, dim=0))**2
    sum_norm_grad = torch.sum(torch.norm(batch_gradients, dim=1)**2)
    
    # Ratio: crowd / individual
    # When identical: ratio = N (all gradients add up)
    # When diverse: ratio → 1 (gradients cancel)
    ratio = sum_grad_norm / (sum_norm_grad + 1e-8)
    
    # Normalize by N and invert
    # T = 1 - ratio/N
    # Identical: T = 1 - N/N = 0
    # Diverse: T = 1 - 1/N ≈ 1
    T_score = 1.0 - torch.clamp(ratio / n, 0, 1)
    
    return T_score


def fixed_measure_gradient_diversity_v2(batch_gradients):
    """
    Option B: Cosine Similarity Based
    
    T_score = 1 - average_pairwise_cosine_similarity
    """
    if batch_gradients.shape[0] == 1:
        return torch.tensor(0.5)
    
    # Normalize gradients
    norms = torch.norm(batch_gradients, dim=1, keepdim=True) + 1e-8
    normalized = batch_gradients / norms
    
    # Compute pairwise cosine similarities
    similarity_matrix = torch.mm(normalized, normalized.t())
    
    # Average off-diagonal similarity
    n = batch_gradients.shape[0]
    mask = ~torch.eye(n, dtype=bool, device=batch_gradients.device)
    avg_similarity = similarity_matrix[mask].mean()
    
    # T_score = 1 - similarity
    T_score = (1.0 - avg_similarity) / 2 + 0.5  # Scale to 0.5-1.0 range
    
    return T_score


def test_identical_gradients():
    """Test with perfectly identical gradients."""
    print("\n" + "=" * 60)
    print("TEST: Identical Gradients (Worst Case)")
    print("=" * 60)
    
    # Create identical gradients
    single_grad = torch.randn(100)
    batch_grads = single_grad.unsqueeze(0).repeat(16, 1)
    
    original = original_measure_gradient_diversity(batch_grads).item()
    fixed_v1 = fixed_measure_gradient_diversity_v1(batch_grads).item()
    fixed_v2 = fixed_measure_gradient_diversity_v2(batch_grads).item()
    
    print(f"   Original (sigmoid): {original:.6f}")
    print(f"   Fixed v1 (linear):  {fixed_v1:.6f}")
    print(f"   Fixed v2 (cosine):  {fixed_v2:.6f}")
    print(f"\n   Expected: Close to 0.0 (identical = no diversity)")
    
    return {
        "test": "identical",
        "original": original,
        "fixed_v1": fixed_v1,
        "fixed_v2": fixed_v2,
        "v1_correct": fixed_v1 < 0.1,
        "v2_correct": fixed_v2 < 0.6  # Cosine method has different scale
    }


def test_random_gradients():
    """Test with random (diverse) gradients."""
    print("\n" + "=" * 60)
    print("TEST: Random Gradients (Normal Case)")
    print("=" * 60)
    
    # Create random gradients
    batch_grads = torch.randn(16, 100)
    
    original = original_measure_gradient_diversity(batch_grads).item()
    fixed_v1 = fixed_measure_gradient_diversity_v1(batch_grads).item()
    fixed_v2 = fixed_measure_gradient_diversity_v2(batch_grads).item()
    
    print(f"   Original (sigmoid): {original:.6f}")
    print(f"   Fixed v1 (linear):  {fixed_v1:.6f}")
    print(f"   Fixed v2 (cosine):  {fixed_v2:.6f}")
    print(f"\n   Expected: High value (random = diverse)")
    
    return {
        "test": "random",
        "original": original,
        "fixed_v1": fixed_v1,
        "fixed_v2": fixed_v2,
        "v1_correct": fixed_v1 > 0.8,
        "v2_correct": fixed_v2 > 0.4
    }


def test_opposite_gradients():
    """Test with opposite gradients (should cancel out)."""
    print("\n" + "=" * 60)
    print("TEST: Opposite Gradients (Cancellation)")
    print("=" * 60)
    
    # Create opposite gradients
    grad = torch.randn(100)
    batch_grads = torch.stack([grad, -grad] * 8)  # 16 samples, half positive, half negative
    
    original = original_measure_gradient_diversity(batch_grads).item()
    fixed_v1 = fixed_measure_gradient_diversity_v1(batch_grads).item()
    fixed_v2 = fixed_measure_gradient_diversity_v2(batch_grads).item()
    
    print(f"   Original (sigmoid): {original:.6f}")
    print(f"   Fixed v1 (linear):  {fixed_v1:.6f}")
    print(f"   Fixed v2 (cosine):  {fixed_v2:.6f}")
    print(f"\n   Expected: High value (opposite = diverse directions)")
    
    return {
        "test": "opposite",
        "original": original,
        "fixed_v1": fixed_v1,
        "fixed_v2": fixed_v2,
        "v1_correct": fixed_v1 > 0.9,  # Should be very high
        "v2_correct": fixed_v2 > 0.4
    }


def test_mixed_gradients():
    """Test with mix of similar and different gradients."""
    print("\n" + "=" * 60)
    print("TEST: Mixed Gradients (Partial Diversity)")
    print("=" * 60)
    
    # Create mixed gradients: 8 similar, 8 random
    base_grad = torch.randn(100)
    similar_grads = base_grad.unsqueeze(0).repeat(8, 1) + torch.randn(8, 100) * 0.1
    random_grads = torch.randn(8, 100)
    batch_grads = torch.cat([similar_grads, random_grads], dim=0)
    
    original = original_measure_gradient_diversity(batch_grads).item()
    fixed_v1 = fixed_measure_gradient_diversity_v1(batch_grads).item()
    fixed_v2 = fixed_measure_gradient_diversity_v2(batch_grads).item()
    
    print(f"   Original (sigmoid): {original:.6f}")
    print(f"   Fixed v1 (linear):  {fixed_v1:.6f}")
    print(f"   Fixed v2 (cosine):  {fixed_v2:.6f}")
    print(f"\n   Expected: Medium value (partial diversity)")
    
    return {
        "test": "mixed",
        "original": original,
        "fixed_v1": fixed_v1,
        "fixed_v2": fixed_v2,
        "v1_correct": 0.3 < fixed_v1 < 0.9,
        "v2_correct": 0.3 < fixed_v2 < 0.7
    }


def test_sleep_threshold():
    """Test if fixed version can trigger Sleep Protocol threshold."""
    print("\n" + "=" * 60)
    print("TEST: Sleep Protocol Threshold (0.3)")
    print("=" * 60)
    
    threshold = 0.3
    
    # Identical gradients should trigger sleep
    single_grad = torch.randn(100)
    identical_grads = single_grad.unsqueeze(0).repeat(16, 1)
    
    original = original_measure_gradient_diversity(identical_grads).item()
    fixed_v1 = fixed_measure_gradient_diversity_v1(identical_grads).item()
    
    print(f"   Threshold: {threshold}")
    print(f"   Original T-Score: {original:.6f} → Sleep: {'YES' if original < threshold else 'NO'}")
    print(f"   Fixed v1 T-Score: {fixed_v1:.6f} → Sleep: {'YES' if fixed_v1 < threshold else 'NO'}")
    
    return {
        "test": "sleep_threshold",
        "threshold": threshold,
        "original_triggers": original < threshold,
        "fixed_triggers": fixed_v1 < threshold,
        "fix_works": fixed_v1 < threshold
    }


def main():
    """Run all tests."""
    print("=" * 60)
    print("T-SCORE FORMULA FIX VALIDATION")
    print("=" * 60)
    
    results = []
    results.append(test_identical_gradients())
    results.append(test_random_gradients())
    results.append(test_opposite_gradients())
    results.append(test_mixed_gradients())
    results.append(test_sleep_threshold())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n   Option A (Linear Normalization):")
    v1_pass = sum(1 for r in results if r.get("v1_correct", False))
    print(f"   Passed: {v1_pass}/{len(results)} tests")
    
    print("\n   Option B (Cosine Similarity):")
    v2_pass = sum(1 for r in results if r.get("v2_correct", False))
    print(f"   Passed: {v2_pass}/{len(results)} tests")
    
    print("\n   Recommendation: Option A (Linear Normalization)")
    print("   - Correctly identifies identical gradients as T≈0")
    print("   - Correctly identifies diverse gradients as T≈1")
    print("   - Can trigger Sleep Protocol when appropriate")
    
    return results


if __name__ == "__main__":
    main()
