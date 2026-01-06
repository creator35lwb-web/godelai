#!/usr/bin/env python3
"""
GodelAI Adversarial Test Suite
==============================
Push the framework to its limits and verify Sleep Protocol triggers under stress.

These tests intentionally create conditions that SHOULD trigger the Sleep Protocol
to validate that the self-healing mechanism actually works.

Author: Godel (Manus AI)
Date: January 7, 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from godelai.agent import GodelAgent


# ============================================================================
# SIMPLE TEST MODEL
# ============================================================================

class SimpleNet(nn.Module):
    """Simple network for adversarial testing."""
    
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================================
# ADVERSARIAL TEST 1: GRADIENT COLLAPSE ATTACK
# ============================================================================

def test_gradient_collapse():
    """
    Test 1: Gradient Collapse Attack
    
    Force all gradients to be identical by using the same sample repeatedly.
    This should cause T-Score to collapse to near-zero.
    
    Expected: Sleep Protocol SHOULD trigger (T-Score < threshold)
    """
    print("\n" + "=" * 70)
    print("🔴 ADVERSARIAL TEST 1: GRADIENT COLLAPSE ATTACK")
    print("=" * 70)
    print("Goal: Force identical gradients to trigger Sleep Protocol")
    
    # Setup
    model = SimpleNet(input_dim=10, hidden_dim=32, output_dim=2)
    agent = GodelAgent(model, min_surplus_energy=0.3, propagation_gamma=2.0)
    criterion = nn.CrossEntropyLoss()
    
    # Create IDENTICAL samples (this is the attack)
    single_input = torch.randn(1, 10)
    single_target = torch.tensor([0])
    
    # Replicate the same sample to fill a batch
    batch_size = 16
    inputs = single_input.repeat(batch_size, 1)  # All identical!
    targets = single_target.repeat(batch_size)    # All identical!
    
    print(f"\n⚠️  Attack: Using {batch_size} IDENTICAL samples")
    print(f"   All inputs are copies of the same vector")
    print(f"   All targets are the same class")
    
    # Compute per-sample gradients
    batch_grads = []
    for i in range(batch_size):
        model.zero_grad()
        output = model(inputs[i:i+1])
        loss = criterion(output, targets[i:i+1])
        loss.backward()
        
        sample_grads = []
        for param in model.parameters():
            if param.grad is not None:
                sample_grads.append(param.grad.view(-1).clone())
        if sample_grads:
            batch_grads.append(torch.cat(sample_grads))
    
    # Measure T-Score
    grad_matrix = torch.stack(batch_grads)
    t_score = agent.measure_gradient_diversity(grad_matrix).item()
    
    # Check Sleep Protocol
    sleep_triggered = t_score < agent.epsilon
    
    print(f"\n📊 Results:")
    print(f"   T-Score: {t_score:.6f}")
    print(f"   Threshold: {agent.epsilon}")
    print(f"   Sleep Protocol Triggered: {'✅ YES' if sleep_triggered else '❌ NO'}")
    
    # Verify gradient similarity
    grad_similarity = torch.cosine_similarity(
        grad_matrix[0:1], grad_matrix[1:], dim=1
    ).mean().item()
    print(f"   Gradient Cosine Similarity: {grad_similarity:.6f}")
    
    result = {
        "test": "gradient_collapse",
        "t_score": t_score,
        "threshold": agent.epsilon,
        "sleep_triggered": sleep_triggered,
        "gradient_similarity": grad_similarity,
        "expected_trigger": True,
        "passed": sleep_triggered  # Test passes if Sleep triggers
    }
    
    if sleep_triggered:
        print(f"\n✅ TEST PASSED: Sleep Protocol correctly triggered!")
    else:
        print(f"\n⚠️  TEST OBSERVATION: Sleep Protocol did NOT trigger")
        print(f"   This reveals a potential gap in the detection mechanism")
    
    return result


# ============================================================================
# ADVERSARIAL TEST 2: CONTRADICTORY LEARNING
# ============================================================================

def test_contradictory_learning():
    """
    Test 2: Contradictory Learning
    
    Train on data where the same input maps to different outputs.
    This creates conflicting gradients that should destabilize T-Score.
    
    Expected: T-Score instability, possible Sleep trigger
    """
    print("\n" + "=" * 70)
    print("🔴 ADVERSARIAL TEST 2: CONTRADICTORY LEARNING")
    print("=" * 70)
    print("Goal: Create conflicting supervision to destabilize learning")
    
    # Setup
    model = SimpleNet(input_dim=10, hidden_dim=32, output_dim=2)
    agent = GodelAgent(model, min_surplus_energy=0.3, propagation_gamma=2.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Create contradictory data: same input, different labels
    base_input = torch.randn(1, 10)
    
    # Half the batch says class 0, half says class 1 for THE SAME INPUT
    batch_size = 16
    inputs = base_input.repeat(batch_size, 1)  # All same input
    targets = torch.tensor([0, 1] * (batch_size // 2))  # Alternating labels!
    
    print(f"\n⚠️  Attack: Same input with CONTRADICTORY labels")
    print(f"   8 samples say: input → class 0")
    print(f"   8 samples say: input → class 1")
    
    # Track T-Score over training
    t_scores = []
    sleep_events = 0
    
    for epoch in range(20):
        # Compute per-sample gradients
        batch_grads = []
        for i in range(batch_size):
            model.zero_grad()
            output = model(inputs[i:i+1])
            loss = criterion(output, targets[i:i+1])
            loss.backward()
            
            sample_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    sample_grads.append(param.grad.view(-1).clone())
            if sample_grads:
                batch_grads.append(torch.cat(sample_grads))
        
        # Measure T-Score
        grad_matrix = torch.stack(batch_grads)
        t_score = agent.measure_gradient_diversity(grad_matrix).item()
        t_scores.append(t_score)
        
        if t_score < agent.epsilon:
            sleep_events += 1
        
        # Do actual training step
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
    
    print(f"\n📊 Results over 20 epochs:")
    print(f"   Initial T-Score: {t_scores[0]:.6f}")
    print(f"   Final T-Score: {t_scores[-1]:.6f}")
    print(f"   Min T-Score: {min(t_scores):.6f}")
    print(f"   Max T-Score: {max(t_scores):.6f}")
    print(f"   Sleep Events: {sleep_events}")
    
    # Check for instability (high variance in T-Score)
    t_score_std = torch.tensor(t_scores).std().item()
    print(f"   T-Score Std Dev: {t_score_std:.6f}")
    
    result = {
        "test": "contradictory_learning",
        "t_scores": t_scores,
        "initial_t_score": t_scores[0],
        "final_t_score": t_scores[-1],
        "min_t_score": min(t_scores),
        "max_t_score": max(t_scores),
        "t_score_std": t_score_std,
        "sleep_events": sleep_events,
        "expected_behavior": "instability or sleep trigger",
        "passed": sleep_events > 0 or t_score_std > 0.05
    }
    
    if sleep_events > 0:
        print(f"\n✅ TEST PASSED: Sleep Protocol triggered {sleep_events} times!")
    elif t_score_std > 0.05:
        print(f"\n✅ TEST PASSED: Significant T-Score instability detected")
    else:
        print(f"\n⚠️  TEST OBSERVATION: Framework remained stable under contradiction")
    
    return result


# ============================================================================
# ADVERSARIAL TEST 3: EXTREME OVERFITTING
# ============================================================================

def test_extreme_overfitting():
    """
    Test 3: Extreme Overfitting Attack
    
    Train on a tiny dataset (2 samples) for many epochs.
    The model should memorize, and T-Score should degrade.
    
    Expected: T-Score degradation as model memorizes
    """
    print("\n" + "=" * 70)
    print("🔴 ADVERSARIAL TEST 3: EXTREME OVERFITTING")
    print("=" * 70)
    print("Goal: Force memorization with tiny dataset")
    
    # Setup
    model = SimpleNet(input_dim=10, hidden_dim=32, output_dim=2)
    agent = GodelAgent(model, min_surplus_energy=0.3, propagation_gamma=2.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Tiny dataset: only 2 samples!
    inputs = torch.randn(2, 10)
    targets = torch.tensor([0, 1])
    
    print(f"\n⚠️  Attack: Training on only 2 samples for 100 epochs")
    
    # Track metrics
    t_scores = []
    losses = []
    sleep_events = 0
    
    for epoch in range(100):
        # Compute per-sample gradients
        batch_grads = []
        for i in range(2):
            model.zero_grad()
            output = model(inputs[i:i+1])
            loss = criterion(output, targets[i:i+1])
            loss.backward()
            
            sample_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    sample_grads.append(param.grad.view(-1).clone())
            if sample_grads:
                batch_grads.append(torch.cat(sample_grads))
        
        # Measure T-Score
        grad_matrix = torch.stack(batch_grads)
        t_score = agent.measure_gradient_diversity(grad_matrix).item()
        t_scores.append(t_score)
        
        if t_score < agent.epsilon:
            sleep_events += 1
        
        # Training step
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    print(f"\n📊 Results over 100 epochs:")
    print(f"   Initial T-Score: {t_scores[0]:.6f}")
    print(f"   Final T-Score: {t_scores[-1]:.6f}")
    print(f"   T-Score Change: {t_scores[-1] - t_scores[0]:.6f}")
    print(f"   Initial Loss: {losses[0]:.6f}")
    print(f"   Final Loss: {losses[-1]:.6f}")
    print(f"   Sleep Events: {sleep_events}")
    
    # Check for degradation
    degradation = t_scores[0] - t_scores[-1]
    
    result = {
        "test": "extreme_overfitting",
        "t_scores": t_scores[::10],  # Sample every 10 epochs
        "initial_t_score": t_scores[0],
        "final_t_score": t_scores[-1],
        "degradation": degradation,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "sleep_events": sleep_events,
        "expected_behavior": "T-Score degradation",
        "passed": degradation > 0.05 or sleep_events > 0
    }
    
    if sleep_events > 0:
        print(f"\n✅ TEST PASSED: Sleep Protocol triggered!")
    elif degradation > 0.05:
        print(f"\n✅ TEST PASSED: T-Score degraded by {degradation:.4f}")
    else:
        print(f"\n⚠️  TEST OBSERVATION: T-Score remained stable despite overfitting")
    
    return result


# ============================================================================
# ADVERSARIAL TEST 4: LEARNING RATE EXPLOSION
# ============================================================================

def test_learning_rate_explosion():
    """
    Test 4: Learning Rate Explosion
    
    Use an extremely high learning rate to destabilize training.
    Gradients should become chaotic.
    
    Expected: T-Score instability, possible Sleep trigger
    """
    print("\n" + "=" * 70)
    print("🔴 ADVERSARIAL TEST 4: LEARNING RATE EXPLOSION")
    print("=" * 70)
    print("Goal: Destabilize training with extreme learning rate")
    
    # Setup
    model = SimpleNet(input_dim=10, hidden_dim=32, output_dim=2)
    agent = GodelAgent(model, min_surplus_energy=0.3, propagation_gamma=2.0)
    criterion = nn.CrossEntropyLoss()
    
    # EXTREME learning rate (100x normal)
    optimizer = optim.SGD(model.parameters(), lr=10.0)  # Normally 0.01-0.1
    
    # Normal dataset
    batch_size = 16
    inputs = torch.randn(batch_size, 10)
    targets = torch.randint(0, 2, (batch_size,))
    
    print(f"\n⚠️  Attack: Learning rate = 10.0 (100x normal)")
    
    # Track metrics
    t_scores = []
    losses = []
    sleep_events = 0
    nan_detected = False
    
    for epoch in range(30):
        # Compute per-sample gradients
        batch_grads = []
        for i in range(batch_size):
            model.zero_grad()
            output = model(inputs[i:i+1])
            
            # Check for NaN
            if torch.isnan(output).any():
                nan_detected = True
                break
                
            loss = criterion(output, targets[i:i+1])
            loss.backward()
            
            sample_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    sample_grads.append(param.grad.view(-1).clone())
            if sample_grads:
                batch_grads.append(torch.cat(sample_grads))
        
        if nan_detected:
            print(f"   ⚠️  NaN detected at epoch {epoch}!")
            break
        
        if len(batch_grads) == 0:
            break
            
        # Measure T-Score
        grad_matrix = torch.stack(batch_grads)
        t_score = agent.measure_gradient_diversity(grad_matrix).item()
        
        if not torch.isnan(torch.tensor(t_score)):
            t_scores.append(t_score)
            
            if t_score < agent.epsilon:
                sleep_events += 1
        
        # Training step
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    if len(t_scores) > 0:
        print(f"\n📊 Results:")
        print(f"   Epochs completed: {len(t_scores)}")
        print(f"   Initial T-Score: {t_scores[0]:.6f}")
        print(f"   Final T-Score: {t_scores[-1]:.6f}")
        print(f"   T-Score Std Dev: {torch.tensor(t_scores).std().item():.6f}")
        print(f"   Sleep Events: {sleep_events}")
        print(f"   NaN Detected: {nan_detected}")
    
    t_score_std = torch.tensor(t_scores).std().item() if len(t_scores) > 1 else 0
    
    result = {
        "test": "learning_rate_explosion",
        "t_scores": t_scores,
        "epochs_completed": len(t_scores),
        "t_score_std": t_score_std,
        "sleep_events": sleep_events,
        "nan_detected": nan_detected,
        "expected_behavior": "instability or training collapse",
        "passed": nan_detected or sleep_events > 0 or t_score_std > 0.1
    }
    
    if nan_detected:
        print(f"\n✅ TEST PASSED: Training collapsed (NaN detected)")
    elif sleep_events > 0:
        print(f"\n✅ TEST PASSED: Sleep Protocol triggered!")
    elif t_score_std > 0.1:
        print(f"\n✅ TEST PASSED: High T-Score instability detected")
    else:
        print(f"\n⚠️  TEST OBSERVATION: Framework remained stable")
    
    return result


# ============================================================================
# ADVERSARIAL TEST 5: CATASTROPHIC FORGETTING
# ============================================================================

def test_catastrophic_forgetting():
    """
    Test 5: Catastrophic Forgetting Simulation
    
    Train on Task A, then abruptly switch to contradictory Task B.
    T-Score should drop during the transition.
    
    Expected: T-Score drop during task switch
    """
    print("\n" + "=" * 70)
    print("🔴 ADVERSARIAL TEST 5: CATASTROPHIC FORGETTING")
    print("=" * 70)
    print("Goal: Rapid task switching to destroy previous knowledge")
    
    # Setup
    model = SimpleNet(input_dim=10, hidden_dim=32, output_dim=2)
    agent = GodelAgent(model, min_surplus_energy=0.3, propagation_gamma=2.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Task A: Learn pattern where first 5 features determine class
    task_a_inputs = torch.randn(16, 10)
    task_a_targets = (task_a_inputs[:, :5].sum(dim=1) > 0).long()
    
    # Task B: OPPOSITE pattern - same inputs, flipped labels
    task_b_inputs = task_a_inputs.clone()
    task_b_targets = 1 - task_a_targets  # Flip all labels!
    
    print(f"\n⚠️  Attack: Train Task A for 20 epochs, then flip to opposite Task B")
    
    # Track metrics
    t_scores = []
    sleep_events = 0
    task_switch_epoch = 20
    
    for epoch in range(40):
        # Select task
        if epoch < task_switch_epoch:
            inputs, targets = task_a_inputs, task_a_targets
            task = "A"
        else:
            inputs, targets = task_b_inputs, task_b_targets
            task = "B"
        
        # Compute per-sample gradients
        batch_grads = []
        for i in range(len(inputs)):
            model.zero_grad()
            output = model(inputs[i:i+1])
            loss = criterion(output, targets[i:i+1])
            loss.backward()
            
            sample_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    sample_grads.append(param.grad.view(-1).clone())
            if sample_grads:
                batch_grads.append(torch.cat(sample_grads))
        
        # Measure T-Score
        grad_matrix = torch.stack(batch_grads)
        t_score = agent.measure_gradient_diversity(grad_matrix).item()
        t_scores.append(t_score)
        
        if t_score < agent.epsilon:
            sleep_events += 1
        
        # Training step
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        if epoch == task_switch_epoch:
            print(f"   📍 Task switch at epoch {epoch}: A → B (labels flipped)")
    
    # Analyze transition
    pre_switch = t_scores[:task_switch_epoch]
    post_switch = t_scores[task_switch_epoch:]
    
    pre_avg = sum(pre_switch) / len(pre_switch)
    post_avg = sum(post_switch) / len(post_switch)
    transition_drop = pre_avg - post_avg
    
    print(f"\n📊 Results:")
    print(f"   Pre-switch avg T-Score: {pre_avg:.6f}")
    print(f"   Post-switch avg T-Score: {post_avg:.6f}")
    print(f"   Transition drop: {transition_drop:.6f}")
    print(f"   T-Score at switch: {t_scores[task_switch_epoch]:.6f}")
    print(f"   Sleep Events: {sleep_events}")
    
    result = {
        "test": "catastrophic_forgetting",
        "t_scores": t_scores,
        "pre_switch_avg": pre_avg,
        "post_switch_avg": post_avg,
        "transition_drop": transition_drop,
        "t_score_at_switch": t_scores[task_switch_epoch],
        "sleep_events": sleep_events,
        "expected_behavior": "T-Score drop during transition",
        "passed": transition_drop > 0.02 or sleep_events > 0
    }
    
    if sleep_events > 0:
        print(f"\n✅ TEST PASSED: Sleep Protocol triggered!")
    elif transition_drop > 0.02:
        print(f"\n✅ TEST PASSED: Detected transition impact on T-Score")
    else:
        print(f"\n⚠️  TEST OBSERVATION: Framework handled task switch gracefully")
    
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all adversarial tests."""
    print("=" * 70)
    print("🔴 GODELAI ADVERSARIAL TEST SUITE")
    print("=" * 70)
    print("Purpose: Push the framework to its limits")
    print("Goal: Verify Sleep Protocol triggers under stress")
    print("=" * 70)
    
    results = {}
    
    # Run all tests
    results["gradient_collapse"] = test_gradient_collapse()
    results["contradictory_learning"] = test_contradictory_learning()
    results["extreme_overfitting"] = test_extreme_overfitting()
    results["learning_rate_explosion"] = test_learning_rate_explosion()
    results["catastrophic_forgetting"] = test_catastrophic_forgetting()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 ADVERSARIAL TEST SUMMARY")
    print("=" * 70)
    
    total_passed = 0
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result["passed"] else "⚠️  OBSERVED"
        print(f"   {test_name}: {status}")
        if result["passed"]:
            total_passed += 1
    
    print(f"\n   Total: {total_passed}/{total_tests} tests triggered expected behavior")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"adversarial_tests_{timestamp}.json"
    
    # Convert tensors to lists for JSON serialization
    serializable_results = {}
    for test_name, result in results.items():
        serializable_results[test_name] = {
            k: (v if not isinstance(v, list) or not v or not isinstance(v[0], float) 
                else [float(x) for x in v])
            for k, v in result.items()
        }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results, results_file


if __name__ == "__main__":
    results, filepath = main()
