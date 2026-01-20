#!/usr/bin/env python3
"""
GodelAI T-Score Validation Experiment v2 - Proper Training Loop
================================================================
Purpose: Validate that conflict data produces different T-Score patterns
         compared to homogeneous Shakespeare data

Key insight: T-Score measures gradient DIVERSITY across a batch.
We need to train on MIXED data within a batch to see the effect.

Author: Godel (Manus AI) - CTO
Date: January 20, 2026
"""

import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import random

# Add godelai to path
sys.path.insert(0, str(Path(__file__).parent))

from godelai.agent import GodelAgent

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).parent / "results"
DATASETS_DIR = Path(__file__).parent / "datasets" / "conflict"

print("=" * 70)
print("GodelAI T-Score Validation Experiment v2")
print("Testing Gradient Diversity with Conflict Data")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Timestamp: {datetime.now().isoformat()}")
print()


class CharLevelModel(nn.Module):
    """Character-level language model for T-Score experiments."""
    
    def __init__(self, vocab_size=256, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)


def load_conflict_datasets():
    """Load all conflict datasets from JSON files."""
    datasets = {}
    
    for category_dir in DATASETS_DIR.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            datasets[category_name] = []
            
            for json_file in category_dir.glob("*.json"):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'data' in data:
                        datasets[category_name].extend(data['data'])
    
    return datasets


def extract_conflicting_texts(conflict_item):
    """Extract multiple conflicting perspectives from a single item."""
    texts = []
    
    if 'perspectives' in conflict_item:
        for perspective in conflict_item['perspectives']:
            text_parts = []
            if 'position' in perspective:
                text_parts.append(perspective['position'])
            if 'reasoning' in perspective:
                text_parts.append(perspective['reasoning'])
            if text_parts:
                texts.append(" ".join(text_parts))
    
    return texts


def create_batch(texts, batch_size=8, seq_length=64):
    """Create a batch of training data from texts."""
    batch_inputs = []
    batch_targets = []
    
    for _ in range(batch_size):
        text = random.choice(texts)
        # Character-level encoding
        encoded = [ord(c) % 256 for c in text]
        
        # Ensure enough length
        while len(encoded) < seq_length + 1:
            encoded.extend([ord(c) % 256 for c in random.choice(texts)])
        
        # Random start position
        start = random.randint(0, max(0, len(encoded) - seq_length - 1))
        
        input_seq = encoded[start:start + seq_length]
        target_seq = encoded[start + 1:start + seq_length + 1]
        
        batch_inputs.append(input_seq)
        batch_targets.append(target_seq)
    
    return (
        torch.tensor(batch_inputs, dtype=torch.long).to(DEVICE),
        torch.tensor(batch_targets, dtype=torch.long).to(DEVICE)
    )


def compute_tscore_from_gradients(model):
    """Compute T-Score from model gradients."""
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    
    if not grads:
        return 0.5
    
    all_grads = torch.cat(grads)
    
    # T-Score: coefficient of variation (std / mean)
    grad_std = all_grads.std().item()
    grad_mean = all_grads.abs().mean().item()
    
    if grad_mean < 1e-10:
        return 0.0  # All gradients near zero = no diversity
    
    t_score = grad_std / grad_mean
    
    # Normalize to [0, 1] range using sigmoid-like transformation
    t_score = 2 / (1 + torch.exp(torch.tensor(-t_score)).item()) - 1
    
    return min(1.0, max(0.0, t_score))


def run_training_experiment(name, texts, num_batches=50, batch_size=8):
    """Run training experiment and collect T-Score statistics."""
    print(f"\n{'=' * 50}")
    print(f"Experiment: {name}")
    print(f"Text samples: {len(texts)}")
    print(f"{'=' * 50}")
    
    # Create fresh model
    model = CharLevelModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    t_scores = []
    losses = []
    
    for batch_idx in range(num_batches):
        # Create batch
        inputs, targets = create_batch(texts, batch_size=batch_size)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Reshape for loss
        outputs_flat = outputs.view(-1, outputs.size(-1))
        targets_flat = targets.view(-1)
        
        loss = criterion(outputs_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        
        # Compute T-Score BEFORE optimizer step
        t_score = compute_tscore_from_gradients(model)
        
        # Optimizer step
        optimizer.step()
        
        t_scores.append(t_score)
        losses.append(loss.item())
        
        if (batch_idx + 1) % 10 == 0:
            avg_t = sum(t_scores[-10:]) / 10
            avg_l = sum(losses[-10:]) / 10
            print(f"  Batch {batch_idx + 1:3d}: T-Score={avg_t:.4f}, Loss={avg_l:.4f}")
    
    # Statistics
    result = {
        "name": name,
        "num_texts": len(texts),
        "num_batches": num_batches,
        "final_t_score": t_scores[-1],
        "avg_t_score": sum(t_scores) / len(t_scores),
        "min_t_score": min(t_scores),
        "max_t_score": max(t_scores),
        "t_score_std": torch.tensor(t_scores).std().item(),
        "final_loss": losses[-1],
        "avg_loss": sum(losses) / len(losses)
    }
    
    print(f"\n  Summary:")
    print(f"    Average T-Score: {result['avg_t_score']:.4f}")
    print(f"    T-Score Range: [{result['min_t_score']:.4f}, {result['max_t_score']:.4f}]")
    print(f"    T-Score Std: {result['t_score_std']:.4f}")
    print(f"    Final Loss: {result['final_loss']:.4f}")
    
    return result


def main():
    """Main experiment runner."""
    
    # Load conflict datasets
    print("\n" + "=" * 70)
    print("Loading Datasets")
    print("=" * 70)
    
    conflict_datasets = load_conflict_datasets()
    
    # Prepare Shakespeare baseline (homogeneous)
    shakespeare_text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream—ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause—there's the respect
    That makes calamity of so long life.
    For who would bear the whips and scorns of time,
    Th'oppressor's wrong, the proud man's contumely,
    The pangs of despised love, the law's delay,
    The insolence of office, and the spurns
    That patient merit of th'unworthy takes,
    When he himself might his quietus make
    With a bare bodkin? Who would fardels bear,
    To grunt and sweat under a weary life,
    But that the dread of something after death,
    The undiscovered country, from whose bourn
    No traveller returns, puzzles the will,
    And makes us rather bear those ills we have
    Than fly to others that we know not of?
    """
    
    # Create homogeneous dataset (same style, no conflict)
    shakespeare_texts = [shakespeare_text[i:i+200] for i in range(0, len(shakespeare_text), 100)]
    shakespeare_texts = [t for t in shakespeare_texts if len(t) > 100]
    
    print(f"Shakespeare (homogeneous): {len(shakespeare_texts)} samples")
    
    # Prepare conflict dataset (heterogeneous, conflicting perspectives)
    all_conflict_texts = []
    for category, items in conflict_datasets.items():
        for item in items:
            conflicting = extract_conflicting_texts(item)
            all_conflict_texts.extend(conflicting)
    
    print(f"Conflict data (heterogeneous): {len(all_conflict_texts)} samples")
    
    # Run experiments
    results = {
        "experiment": "T-Score Validation v2 - Gradient Diversity",
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "hypothesis": "Conflict data should show different T-Score patterns due to gradient diversity",
        "results": []
    }
    
    # Baseline: Shakespeare (homogeneous)
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Shakespeare (Homogeneous Data)")
    print("=" * 70)
    baseline = run_training_experiment("Shakespeare (Homogeneous)", shakespeare_texts)
    results["results"].append(baseline)
    
    # Conflict data (heterogeneous)
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Conflict Data (Heterogeneous)")
    print("=" * 70)
    conflict = run_training_experiment("Conflict Data (Heterogeneous)", all_conflict_texts)
    results["results"].append(conflict)
    
    # Mixed: Shakespeare + Conflict (maximum diversity)
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Mixed Data (Maximum Diversity)")
    print("=" * 70)
    mixed_texts = shakespeare_texts + all_conflict_texts
    mixed = run_training_experiment("Mixed Data (Maximum Diversity)", mixed_texts)
    results["results"].append(mixed)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Dataset':<35} {'Avg T-Score':>12} {'T-Score Std':>12} {'Avg Loss':>10}")
    print("-" * 70)
    for r in results["results"]:
        print(f"{r['name']:<35} {r['avg_t_score']:>12.4f} {r['t_score_std']:>12.4f} {r['avg_loss']:>10.4f}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    t_score_diff = conflict["avg_t_score"] - baseline["avg_t_score"]
    std_diff = conflict["t_score_std"] - baseline["t_score_std"]
    
    print(f"\nT-Score Difference (Conflict - Shakespeare): {t_score_diff:+.4f}")
    print(f"T-Score Std Difference: {std_diff:+.4f}")
    
    if abs(t_score_diff) > 0.01 or abs(std_diff) > 0.01:
        print("\n✅ OBSERVATION: Conflict data shows DIFFERENT gradient patterns")
        results["conclusion"] = "Conflict data produces different gradient diversity patterns"
    else:
        print("\n⚠️ OBSERVATION: No significant difference detected in this experiment")
        print("   This may require larger datasets or different model architecture")
        results["conclusion"] = "No significant difference detected - requires further investigation"
    
    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"conflict_tscore_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = main()
