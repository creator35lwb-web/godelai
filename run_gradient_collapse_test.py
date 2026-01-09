#!/usr/bin/env python3
"""
Gradient Collapse Detection Test: GodelAI vs Standard Baseline
================================================================

OBJECTIVE:
Test if GodelAI can detect and prevent gradient collapse conditions
that would cause standard training to fail.

TEST DESIGN:
Create adversarial conditions that induce gradient collapse:
1. Very deep network (6+ layers)
2. High learning rate
3. Small batch size
4. Difficult optimization landscape

HYPOTHESIS:
GodelAI's T-Score monitoring should detect low gradient diversity
and trigger Sleep Protocol to prevent collapse.

Author: Claude Code
Date: January 9, 2026
"""
import sys
import io
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path

# Force UTF-8 and unbuffered output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from godelai.agent import GodelAgent

print("="*80, flush=True)
print("GRADIENT COLLAPSE TEST: GodelAI vs Standard Baseline", flush=True)
print("="*80, flush=True)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(flush=True)

# Adversarial configuration designed to cause gradient issues
SEED = 42
CONFIG = {
    "epochs": 10,
    "batch_size": 16,  # Small batch size
    "seq_length": 100,
    "embedding_dim": 128,
    "hidden_dim": 512,  # Large hidden dimension
    "num_layers": 6,  # Very deep network
    "learning_rate": 0.01,  # High learning rate
    "seed": SEED,
    "min_surplus_energy": 0.3,
    "propagation_gamma": 2.0,
    "tscore_sample_batches": 3,
}

print("⚠️  ADVERSARIAL Configuration (Designed to cause gradient collapse):", flush=True)
for key, value in CONFIG.items():
    print(f"  {key}: {value}", flush=True)
print(flush=True)

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Load Mini Shakespeare (small dataset makes collapse more likely)
def load_mini_shakespeare():
    """Load mini Shakespeare dataset (first 5KB)"""
    data_path = Path("data/shakespeare.txt")
    if data_path.exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()[:5000]  # First 5000 characters
    else:
        print("❌ Shakespeare dataset not found. Run main A/B test first.", flush=True)
        sys.exit(1)

    print(f"  Dataset: {len(text):,} characters (mini)", flush=True)
    return text

text = load_mini_shakespeare()
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"  Vocabulary: {vocab_size} unique characters", flush=True)
print(flush=True)

# Create dataset
def create_batches(text, batch_size, seq_length):
    """Create batches from text"""
    data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
    num_batches = len(data) // (batch_size * seq_length)
    data = data[:num_batches * batch_size * seq_length]
    data = data.view(batch_size, -1)

    batches = []
    for i in range(0, data.size(1) - seq_length, seq_length):
        inputs = data[:, i:i+seq_length]
        targets = data[:, i+1:i+seq_length+1]
        batches.append((inputs, targets))

    return batches

batches = create_batches(text, CONFIG['batch_size'], CONFIG['seq_length'])
print(f"Training batches: {len(batches)}", flush=True)
print(flush=True)

# Deep model (prone to gradient issues)
class DeepCharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output)
        return output

# Function to compute gradient norms (detect collapse)
def compute_gradient_stats(model):
    """Compute gradient statistics"""
    grad_norms = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())

    if len(grad_norms) == 0:
        return 0.0, 0.0, 0.0

    avg_norm = sum(grad_norms) / len(grad_norms)
    max_norm = max(grad_norms)
    min_norm = min(grad_norms)

    return avg_norm, max_norm, min_norm

print("="*80, flush=True)
print("RUNNING GRADIENT COLLAPSE TEST", flush=True)
print("="*80, flush=True)
print(flush=True)

# Test Standard Model
print("="*80, flush=True)
print("STANDARD MODEL (Vulnerable to Gradient Collapse)", flush=True)
print("="*80, flush=True)
print(flush=True)

set_seed(SEED)
device = torch.device('cpu')
model_standard = DeepCharRNN(vocab_size, CONFIG['embedding_dim'], CONFIG['hidden_dim'], CONFIG['num_layers']).to(device)
optimizer_standard = optim.Adam(model_standard.parameters(), lr=CONFIG['learning_rate'])
criterion = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model_standard.parameters()):,}", flush=True)
print(f"Model depth: {CONFIG['num_layers']} layers (very deep)", flush=True)
print(flush=True)

gradient_collapse_detected = False
standard_results = {
    "losses": [],
    "grad_avg_norms": [],
    "grad_max_norms": [],
    "grad_min_norms": [],
    "collapse_detected": False,
    "collapse_epoch": None,
}

for epoch in range(CONFIG['epochs']):
    model_standard.train()
    total_loss = 0.0
    epoch_grad_norms = []

    for inputs, targets in batches:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer_standard.zero_grad()
        outputs = model_standard(inputs)
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
        loss = criterion(outputs_flat, targets_flat)
        loss.backward()

        # Compute gradient statistics
        avg_norm, max_norm, min_norm = compute_gradient_stats(model_standard)
        epoch_grad_norms.append(avg_norm)

        # Detect gradient collapse (very small or NaN gradients)
        if avg_norm < 1e-6 or np.isnan(avg_norm):
            gradient_collapse_detected = True
            if standard_results["collapse_epoch"] is None:
                standard_results["collapse_epoch"] = epoch + 1

        optimizer_standard.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(batches)
    avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0

    standard_results["losses"].append(avg_loss)
    standard_results["grad_avg_norms"].append(avg_grad_norm)

    collapse_indicator = " ⚠️ COLLAPSE!" if gradient_collapse_detected else ""
    print(f"Epoch {epoch+1}/{CONFIG['epochs']}: Loss = {avg_loss:.4f}, Avg Grad Norm = {avg_grad_norm:.6f}{collapse_indicator}", flush=True)

standard_results["collapse_detected"] = gradient_collapse_detected

print(flush=True)
if gradient_collapse_detected:
    print(f"❌ Gradient Collapse DETECTED in Standard model at epoch {standard_results['collapse_epoch']}", flush=True)
    print(f"   Gradients became vanishingly small (< 1e-6)", flush=True)
else:
    print(f"✅ No gradient collapse detected in Standard model", flush=True)
    print(f"   Final gradient norm: {standard_results['grad_avg_norms'][-1]:.6f}", flush=True)
print(flush=True)

# TODO: Test GodelAI Model
print("="*80, flush=True)
print("GODELAI MODEL (With T-Score Monitoring)", flush=True)
print("="*80, flush=True)
print(flush=True)

print("⏳ GodelAI gradient collapse test - TO BE IMPLEMENTED", flush=True)
print(f"   This requires testing if Sleep Protocol prevents/recovers from collapse", flush=True)
print(flush=True)

# Save results
results = {
    "config": CONFIG,
    "standard": standard_results,
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
}

output_file = Path(f"results/gradient_collapse_test_{results['timestamp']}.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {output_file}", flush=True)
print(flush=True)
print("="*80, flush=True)
print("GRADIENT COLLAPSE TEST COMPLETE (Standard Only)", flush=True)
print("="*80, flush=True)

# Print summary
print(flush=True)
print("SUMMARY:", flush=True)
print("-" * 80, flush=True)
print(f"Standard Model:", flush=True)
print(f"  Gradient Collapse: {'YES ❌' if standard_results['collapse_detected'] else 'NO ✅'}", flush=True)
print(f"  Final Loss: {standard_results['losses'][-1]:.4f}", flush=True)
print(f"  Final Grad Norm: {standard_results['grad_avg_norms'][-1]:.6f}", flush=True)
print(flush=True)
print("Note: GodelAI test needs to be implemented to compare if Sleep Protocol prevents collapse.", flush=True)
print("="*80, flush=True)
