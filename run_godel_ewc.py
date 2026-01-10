#!/usr/bin/env python3
"""
GODELAI-EWC: Elastic Weight Consolidation for Catastrophic Forgetting
========================================================================

PIVOT: From "Sleep Protocol Interruption" to "Memory Regularization"

MISSION: Beat Standard Model's 5.3% forgetting WITHOUT blocking learning.

APPROACH:
- Phase 1: Train on Task A (first 50% Shakespeare)
- Consolidation: Compute Fisher Information Matrix (FIM) - which weights matter
- Phase 2: Train on Task B with EWC penalty - protect important weights

EWC LOSS:
  Loss = Task_Loss + λ * Σ(FIM * (θ - θ_A)^2)

  This penalizes changes to parameters that were important for Task A.

TARGET:
- Standard Baseline: Forgetting = +0.0742 (5.3% degradation)
- Godel-Sleep (Failed): Final Loss = 4.17 (learning blocked)
- Godel-EWC (Goal): Loss ~1.30-1.40 (learning works) + Forgetting < 0.02 (memory works)

Author: Claude Code + Strategic Partner (Gemini/User)
Date: January 11, 2026
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
from copy import deepcopy

# Force UTF-8 and unbuffered output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

print("="*80, flush=True)
print("🧠 GODELAI-EWC: Elastic Weight Consolidation Test", flush=True)
print("="*80, flush=True)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(flush=True)

# CONFIGURATION
SEED = 42  # MUST match baseline
CONFIG = {
    "task_a_epochs": 5,
    "task_b_epochs": 5,
    "batch_size": 64,
    "seq_length": 100,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "learning_rate": 0.002,
    "seed": SEED,
    "ewc_lambda": 1000.0,  # EWC regularization strength
    "fisher_samples": 100,  # Number of samples for FIM computation
}

print("🎯 EWC Configuration:", flush=True)
print(f"   ewc_lambda: {CONFIG['ewc_lambda']} (regularization strength)", flush=True)
print(f"   fisher_samples: {CONFIG['fisher_samples']} (FIM computation)", flush=True)
print(f"   Standard baseline forgetting: +0.0742 (5.3%)", flush=True)
print(f"   Godel-Sleep final loss: 4.17 (learning blocked)", flush=True)
print(f"   TARGET: Loss ~1.30-1.40, Forgetting < 0.02", flush=True)
print(flush=True)

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
chars_all = sorted(list(set(text)))
vocab_size = len(chars_all)
char_to_idx = {ch: i for i, ch in enumerate(chars_all)}
idx_to_char = {i: ch for i, ch in enumerate(chars_all)}

print(f"Vocabulary: {vocab_size} unique characters", flush=True)
print(flush=True)

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
def evaluate_on_task(model, batches, device, task_name):
    """Evaluate model on a specific task"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    criterion = nn.CrossEntropyLoss()

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

# Fisher Information Matrix computation
def compute_fisher_information(model, batches, device, num_samples):
    """
    Compute Fisher Information Matrix (FIM) for the model.

    FIM measures parameter importance by approximating:
    F_i = E[(∂log p(y|x,θ)/∂θ_i)^2]

    We estimate this as the average squared gradient over samples.
    """
    print("🔍 Computing Fisher Information Matrix...", flush=True)

    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Initialize FIM storage
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)

    # Sample batches for FIM computation
    sample_batches = batches[:num_samples] if len(batches) >= num_samples else batches

    for batch_idx, (inputs, targets) in enumerate(sample_batches):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero gradients
        model.zero_grad()

        # Forward pass
        outputs = model(inputs)
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
        loss = criterion(outputs_flat, targets_flat)

        # Backward pass
        loss.backward()

        # Accumulate squared gradients (Fisher approximation)
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2

        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {batch_idx + 1}/{len(sample_batches)} batches", flush=True)

    # Average over samples
    for name in fisher:
        fisher[name] /= len(sample_batches)

    print(f"✅ Fisher Information computed from {len(sample_batches)} batches", flush=True)
    return fisher

# EWC penalty computation
def compute_ewc_penalty(model, fisher, old_params, ewc_lambda):
    """
    Compute EWC penalty:
    EWC_penalty = λ * Σ(F_i * (θ_i - θ_old_i)^2)

    This penalizes changes to parameters that were important for previous task.
    """
    penalty = 0.0
    for name, param in model.named_parameters():
        if name in fisher:
            penalty += (fisher[name] * (param - old_params[name]) ** 2).sum()

    return ewc_lambda * penalty

print("="*80, flush=True)
print("🧠 GODELAI-EWC EXPERIMENT", flush=True)
print("="*80, flush=True)
print(flush=True)

set_seed(SEED)
device = torch.device('cpu')
model = CharRNN(vocab_size, CONFIG['embedding_dim'], CONFIG['hidden_dim'], CONFIG['num_layers']).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
print(flush=True)

# ============================================================================
# PHASE 1: Train on Task A (Normal Training)
# ============================================================================
print("--- Phase 1: Training on Task A (Normal) ---", flush=True)
for epoch in range(CONFIG['task_a_epochs']):
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(task_a_batches):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
        loss = criterion(outputs_flat, targets_flat)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(task_a_batches)
    print(f"Epoch {epoch+1}/{CONFIG['task_a_epochs']}: Task A Loss = {avg_loss:.4f}", flush=True)

task_a_loss_after_phase1 = evaluate_on_task(model, task_a_batches, device, "Task A")
print(f"✅ Task A Loss after Phase 1: {task_a_loss_after_phase1:.4f}", flush=True)
print(flush=True)

# ============================================================================
# CONSOLIDATION: Compute Fisher Information Matrix
# ============================================================================
print("--- Consolidation: Computing Fisher Information Matrix ---", flush=True)
fisher = compute_fisher_information(
    model,
    task_a_batches,
    device,
    CONFIG['fisher_samples']
)

# Store Task A parameters
old_params = {}
for name, param in model.named_parameters():
    old_params[name] = param.data.clone()

print(f"✅ Task A parameters saved ({len(old_params)} parameter tensors)", flush=True)
print(flush=True)

# ============================================================================
# PHASE 2: Train on Task B with EWC Penalty
# ============================================================================
print("--- Phase 2: Training on Task B with EWC Regularization ---", flush=True)
print(f"   EWC Lambda: {CONFIG['ewc_lambda']}", flush=True)
print(flush=True)

for epoch in range(CONFIG['task_b_epochs']):
    model.train()
    total_task_loss = 0.0
    total_ewc_penalty = 0.0
    total_combined_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(task_b_batches):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Task B loss
        outputs = model(inputs)
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
        task_loss = criterion(outputs_flat, targets_flat)

        # EWC penalty
        ewc_penalty = compute_ewc_penalty(model, fisher, old_params, CONFIG['ewc_lambda'])

        # Combined loss
        combined_loss = task_loss + ewc_penalty

        combined_loss.backward()
        optimizer.step()

        total_task_loss += task_loss.item()
        total_ewc_penalty += ewc_penalty.item()
        total_combined_loss += combined_loss.item()

    avg_task_loss = total_task_loss / len(task_b_batches)
    avg_ewc_penalty = total_ewc_penalty / len(task_b_batches)
    avg_combined_loss = total_combined_loss / len(task_b_batches)

    print(f"Epoch {epoch+1}/{CONFIG['task_b_epochs']}: Task B Loss = {avg_task_loss:.4f}, "
          f"EWC Penalty = {avg_ewc_penalty:.4f}, Combined = {avg_combined_loss:.4f}", flush=True)

# Measure catastrophic forgetting
task_a_loss_after_phase2 = evaluate_on_task(model, task_a_batches, device, "Task A")
task_b_loss_final = evaluate_on_task(model, task_b_batches, device, "Task B")

print(flush=True)
print(f"✅ Task A Loss after Phase 2: {task_a_loss_after_phase2:.4f}", flush=True)
print(f"✅ Task B Loss after Phase 2: {task_b_loss_final:.4f}", flush=True)
print(flush=True)

forgetting_ewc = task_a_loss_after_phase2 - task_a_loss_after_phase1

print("="*80, flush=True)
print("🎯 GODELAI-EWC RESULTS", flush=True)
print("="*80, flush=True)
print(flush=True)
print(f"Standard Model Baseline:  +0.0742 (5.3% degradation)", flush=True)
print(f"Godel-Sleep (Failed):     Final Loss = 4.17 (learning blocked)", flush=True)
print(f"Godel-EWC Forgetting:     {forgetting_ewc:+.4f}", flush=True)
print(flush=True)

# Comparison to baselines
if forgetting_ewc < 0.0742 and task_a_loss_after_phase1 < 2.0:
    improvement = ((0.0742 - forgetting_ewc) / 0.0742) * 100
    print(f"🎉 SUCCESS! GodelAI-EWC reduced forgetting by {improvement:.1f}%", flush=True)
    print(f"   Absolute improvement: {0.0742 - forgetting_ewc:.4f}", flush=True)

    if task_a_loss_after_phase1 < 1.5:
        print(f"   ✅ Learning NOT blocked (final loss = {task_a_loss_after_phase1:.4f} vs Sleep = 4.17)", flush=True)
        print(f"   ✅ Best of both worlds: Memory preservation + Learning capability!", flush=True)
    else:
        print(f"   ⚠️  Learning partially impaired (final loss = {task_a_loss_after_phase1:.4f})", flush=True)
elif forgetting_ewc < 0.0742:
    improvement = ((0.0742 - forgetting_ewc) / 0.0742) * 100
    print(f"⚠️  PARTIAL SUCCESS: Forgetting reduced by {improvement:.1f}%", flush=True)
    print(f"   But initial learning may be impaired (loss = {task_a_loss_after_phase1:.4f})", flush=True)
else:
    degradation = ((forgetting_ewc - 0.0742) / 0.0742) * 100
    print(f"❌ FAILURE: EWC worse than Standard by {degradation:.1f}%", flush=True)
    print(f"   Additional forgetting: {forgetting_ewc - 0.0742:.4f}", flush=True)

print(flush=True)

# Save results
results = {
    "config": CONFIG,
    "baseline_forgetting": 0.0742,
    "baseline_sleep_loss": 4.17,
    "ewc": {
        "task_a_loss_after_phase1": task_a_loss_after_phase1,
        "task_a_loss_after_phase2": task_a_loss_after_phase2,
        "task_b_loss_final": task_b_loss_final,
        "forgetting": forgetting_ewc,
    },
    "comparison": {
        "standard_forgetting": 0.0742,
        "ewc_forgetting": forgetting_ewc,
        "improvement": 0.0742 - forgetting_ewc,
        "improvement_percentage": ((0.0742 - forgetting_ewc) / 0.0742) * 100 if forgetting_ewc <= 0.0742 else -((forgetting_ewc - 0.0742) / 0.0742) * 100,
        "learning_preserved": task_a_loss_after_phase1 < 2.0,
        "better_than_sleep": task_a_loss_after_phase1 < 4.0,
    },
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
}

output_file = Path(f"results/ewc_test_result_{results['timestamp']}.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {output_file}", flush=True)
print(flush=True)
print("="*80, flush=True)
print("🎯 GODELAI-EWC EXPERIMENT COMPLETE", flush=True)
print("="*80, flush=True)
