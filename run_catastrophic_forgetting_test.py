#!/usr/bin/env python3
"""
Catastrophic Forgetting Test: GodelAI vs Standard Baseline
===========================================================

OBJECTIVE:
Test if GodelAI preserves gradient diversity and prevents catastrophic
forgetting when learning multiple tasks sequentially.

TEST DESIGN:
1. Task A: Train on first 50% of Shakespeare (500KB)
2. Task B: Train on last 50% of Shakespeare (different char distribution)
3. Measure: How much Task A knowledge is retained

HYPOTHESIS:
GodelAI's gradient diversity preservation should reduce catastrophic
forgetting compared to standard training.

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
print("CATASTROPHIC FORGETTING TEST: GodelAI vs Standard Baseline", flush=True)
print("="*80, flush=True)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(flush=True)

# Configuration
SEED = 42
CONFIG = {
    "task_a_epochs": 5,  # Train on Task A
    "task_b_epochs": 5,  # Train on Task B
    "batch_size": 64,
    "seq_length": 100,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "learning_rate": 0.002,
    "seed": SEED,
    "min_surplus_energy": 0.3,
    "propagation_gamma": 2.0,
    "tscore_sample_batches": 3,
}

print("Test Configuration:", flush=True)
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

# Load Shakespeare dataset
def load_shakespeare():
    """Load Shakespeare dataset"""
    data_path = Path("data/shakespeare.txt")
    if data_path.exists():
        print(f"✅ Using cached dataset: {data_path}", flush=True)
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("📥 Downloading Shakespeare dataset...", flush=True)
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8')
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✅ Downloaded and cached dataset", flush=True)

    print(f"  Size: {len(text):,} characters", flush=True)
    return text

# Split dataset into Task A and Task B
text = load_shakespeare()
split_point = len(text) // 2

task_a_text = text[:split_point]  # First half
task_b_text = text[split_point:]  # Second half

print(flush=True)
print(f"Task A (First 50%): {len(task_a_text):,} characters", flush=True)
print(f"Task B (Last 50%): {len(task_b_text):,} characters", flush=True)
print(flush=True)

# Build vocabularies
chars_a = sorted(list(set(task_a_text)))
chars_b = sorted(list(set(task_b_text)))
chars_all = sorted(list(set(text)))

print(f"Task A vocabulary: {len(chars_a)} unique characters", flush=True)
print(f"Task B vocabulary: {len(chars_b)} unique characters", flush=True)
print(f"Combined vocabulary: {len(chars_all)} unique characters", flush=True)
print(flush=True)

# Use combined vocabulary for fair comparison
vocab_size = len(chars_all)
char_to_idx = {ch: i for i, ch in enumerate(chars_all)}
idx_to_char = {i: ch for i, ch in enumerate(chars_all)}

# Create datasets
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

task_a_batches = create_batches(task_a_text, CONFIG['batch_size'], CONFIG['seq_length'])
task_b_batches = create_batches(task_b_text, CONFIG['batch_size'], CONFIG['seq_length'])

print(f"Task A batches: {len(task_a_batches)}", flush=True)
print(f"Task B batches: {len(task_b_batches)}", flush=True)
print(flush=True)

# Model definition
class CharRNN(nn.Module):
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

# Evaluation function
def evaluate_on_task(model, batches, criterion, device, task_name):
    """Evaluate model on a specific task"""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, targets in batches[:10]:  # Use first 10 batches
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)
            loss = criterion(outputs_flat, targets_flat)

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss

print("="*80, flush=True)
print("RUNNING CATASTROPHIC FORGETTING TEST", flush=True)
print("="*80, flush=True)
print(flush=True)

# Test Standard Model
print("="*80, flush=True)
print("STANDARD MODEL (Baseline)", flush=True)
print("="*80, flush=True)
print(flush=True)

set_seed(SEED)
device = torch.device('cpu')
model_standard = CharRNN(vocab_size, CONFIG['embedding_dim'], CONFIG['hidden_dim'], CONFIG['num_layers']).to(device)
optimizer_standard = optim.Adam(model_standard.parameters(), lr=CONFIG['learning_rate'])
criterion = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model_standard.parameters()):,}", flush=True)
print(flush=True)

# Phase 1: Train on Task A
print("--- Phase 1: Training on Task A ---", flush=True)
for epoch in range(CONFIG['task_a_epochs']):
    model_standard.train()
    total_loss = 0.0

    for inputs, targets in task_a_batches:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer_standard.zero_grad()
        outputs = model_standard(inputs)
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        optimizer_standard.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(task_a_batches)
    print(f"Epoch {epoch+1}/{CONFIG['task_a_epochs']}: Task A Loss = {avg_loss:.4f}", flush=True)

task_a_loss_after_training = evaluate_on_task(model_standard, task_a_batches, criterion, device, "Task A")
print(f"Task A Loss after Phase 1: {task_a_loss_after_training:.4f}", flush=True)
print(flush=True)

# Phase 2: Train on Task B
print("--- Phase 2: Training on Task B ---", flush=True)
for epoch in range(CONFIG['task_b_epochs']):
    model_standard.train()
    total_loss = 0.0

    for inputs, targets in task_b_batches:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer_standard.zero_grad()
        outputs = model_standard(inputs)
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        optimizer_standard.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(task_b_batches)
    print(f"Epoch {epoch+1}/{CONFIG['task_b_epochs']}: Task B Loss = {avg_loss:.4f}", flush=True)

# Measure catastrophic forgetting
task_a_loss_after_task_b = evaluate_on_task(model_standard, task_a_batches, criterion, device, "Task A")
task_b_loss_final = evaluate_on_task(model_standard, task_b_batches, criterion, device, "Task B")

print(flush=True)
print(f"Task A Loss after Phase 2: {task_a_loss_after_task_b:.4f}", flush=True)
print(f"Task B Loss after Phase 2: {task_b_loss_final:.4f}", flush=True)
print(flush=True)

forgetting_standard = task_a_loss_after_task_b - task_a_loss_after_training
print(f"⚠️ Catastrophic Forgetting (Standard): {forgetting_standard:+.4f}", flush=True)
print(f"   (Positive = worse, Task A knowledge degraded)", flush=True)
print(flush=True)

# TODO: Test GodelAI Model
print("="*80, flush=True)
print("GODELAI MODEL (With Gradient Diversity Preservation)", flush=True)
print("="*80, flush=True)
print(flush=True)

print("⏳ GodelAI catastrophic forgetting test - TO BE IMPLEMENTED", flush=True)
print(f"   This requires integrating GodelAgent with sequential task training", flush=True)
print(flush=True)

# Save results
results = {
    "config": CONFIG,
    "standard": {
        "task_a_loss_after_phase1": task_a_loss_after_training,
        "task_a_loss_after_phase2": task_a_loss_after_task_b,
        "task_b_loss_final": task_b_loss_final,
        "forgetting": forgetting_standard,
    },
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
}

output_file = Path(f"results/catastrophic_forgetting_test_{results['timestamp']}.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {output_file}", flush=True)
print(flush=True)
print("="*80, flush=True)
print("CATASTROPHIC FORGETTING TEST COMPLETE (Standard Only)", flush=True)
print("="*80, flush=True)
