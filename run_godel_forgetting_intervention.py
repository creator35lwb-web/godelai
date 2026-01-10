#!/usr/bin/env python3
"""
CRITICAL EXPERIMENT: GodelAI Intervention on Catastrophic Forgetting
====================================================================

MISSION: Prove that GodelAI can reduce catastrophic forgetting better than
the Standard Model.

CONTEXT:
- Standard Model Baseline: +0.0742 forgetting (5.3% degradation)
- Test: catastrophic_forgetting_test_20260109_092340.json
- Issue: Previous tests had epsilon too low (0.3), Sleep Protocol never triggered
- T-Score naturally high (>0.92), so we need aggressive threshold

INTERVENTION:
- Set epsilon = 0.935 (AGGRESSIVE - will trigger when T-Score dips slightly)
- Use GodelAgent wrapper (active Sleep Protocol)
- Exact same setup as previous test (seed=42, Task A -> Task B)

SUCCESS METRIC:
- Standard Forgetting: +0.0742 (baseline)
- GodelAI Forgetting: ? (MUST be < 0.0742 to prove benefit)

Author: Claude Code + Gemini 2.0 Flash Thinking (Strategy Partner)
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
print("🎯 CRITICAL EXPERIMENT: GodelAI Intervention on Catastrophic Forgetting", flush=True)
print("="*80, flush=True)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(flush=True)

# CRITICAL CONFIGURATION
SEED = 42  # MUST match previous test
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
    "min_surplus_energy": 0.935,  # 🔥 AGGRESSIVE THRESHOLD (was 0.3)
    "propagation_gamma": 2.0,
    "tscore_sample_batches": 3,
}

print("⚠️  INTERVENTION Configuration:", flush=True)
print(f"   min_surplus_energy (ε): {CONFIG['min_surplus_energy']} (AGGRESSIVE)", flush=True)
print(f"   Baseline forgetting: +0.0742 (5.3% degradation)", flush=True)
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

# Evaluation function (uses base criterion directly)
def evaluate_on_task(model, batches, device, task_name):
    """Evaluate model on a specific task"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    base_crit = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in batches[:10]:  # Use first 10 batches
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)
            loss = base_crit(outputs_flat, targets_flat)

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss

print("="*80, flush=True)
print("🧠 GODELAI INTERVENTION TEST", flush=True)
print("="*80, flush=True)
print(flush=True)

set_seed(SEED)
device = torch.device('cpu')
model = CharRNN(vocab_size, CONFIG['embedding_dim'], CONFIG['hidden_dim'], CONFIG['num_layers']).to(device)

# Create a wrapper criterion that handles sequence model reshaping
base_criterion = nn.CrossEntropyLoss()

class SequenceCriterion(nn.Module):
    """Wrapper to handle sequence-to-sequence loss for GodelAgent"""
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion

    def forward(self, prediction, target):
        # prediction: [batch, seq, vocab] or [1, seq, vocab]
        # target: [batch, seq] or [1, seq]
        pred_flat = prediction.reshape(-1, prediction.size(-1))
        target_flat = target.reshape(-1)
        return self.base_criterion(pred_flat, target_flat)

criterion = SequenceCriterion(base_criterion)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
print(f"Aggressive threshold: ε = {CONFIG['min_surplus_energy']}", flush=True)
print(flush=True)

# Initialize GodelAgent
agent = GodelAgent(
    base_model=model,
    min_surplus_energy=CONFIG['min_surplus_energy'],
    propagation_gamma=CONFIG['propagation_gamma']
)

# Set optimizer for agent
optimizer = optim.Adam(agent.parameters(), lr=CONFIG['learning_rate'])
agent.optimizer = optimizer

# Track sleep events
total_sleep_events = 0
sleep_epochs = []

# Phase 1: Train on Task A
print("--- Phase 1: Training on Task A ---", flush=True)
for epoch in range(CONFIG['task_a_epochs']):
    agent.train()
    total_loss = 0.0
    epoch_sleep_events = 0
    epoch_t_scores = []

    for batch_idx, (inputs, targets) in enumerate(task_a_batches):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Use GodelAgent's learning_step (handles T-Score + Sleep Protocol)
        # SequenceCriterion handles flattening internally
        loss, t_score, status = agent.learning_step(inputs, targets, criterion)

        if status == "SLEEP":
            epoch_sleep_events += 1
            total_sleep_events += 1

        total_loss += loss
        epoch_t_scores.append(t_score)

    avg_loss = total_loss / len(task_a_batches)
    avg_t_score = sum(epoch_t_scores) / len(epoch_t_scores) if epoch_t_scores else 0.0

    status_msg = ""
    if epoch_sleep_events > 0:
        status_msg = f" 💤 {epoch_sleep_events} sleep events"
        sleep_epochs.append(epoch + 1)

    print(f"Epoch {epoch+1}/{CONFIG['task_a_epochs']}: Task A Loss = {avg_loss:.4f}, T-Score = {avg_t_score:.4f}{status_msg}", flush=True)

task_a_loss_after_training = evaluate_on_task(agent.compression_layer, task_a_batches, device, "Task A")
print(f"✅ Task A Loss after Phase 1: {task_a_loss_after_training:.4f}", flush=True)
print(f"   Sleep events during Phase 1: {total_sleep_events}", flush=True)
print(flush=True)

# Phase 2: Train on Task B
print("--- Phase 2: Training on Task B ---", flush=True)
phase2_sleep_events = 0
for epoch in range(CONFIG['task_b_epochs']):
    agent.train()
    total_loss = 0.0
    epoch_sleep_events = 0
    epoch_t_scores = []

    for batch_idx, (inputs, targets) in enumerate(task_b_batches):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Use GodelAgent's learning_step (handles T-Score + Sleep Protocol)
        # SequenceCriterion handles flattening internally
        loss, t_score, status = agent.learning_step(inputs, targets, criterion)

        if status == "SLEEP":
            epoch_sleep_events += 1
            total_sleep_events += 1
            phase2_sleep_events += 1

        total_loss += loss
        epoch_t_scores.append(t_score)

    avg_loss = total_loss / len(task_b_batches)
    avg_t_score = sum(epoch_t_scores) / len(epoch_t_scores) if epoch_t_scores else 0.0

    status_msg = ""
    if epoch_sleep_events > 0:
        status_msg = f" 💤 {epoch_sleep_events} sleep events"
        sleep_epochs.append(CONFIG['task_a_epochs'] + epoch + 1)

    print(f"Epoch {epoch+1}/{CONFIG['task_b_epochs']}: Task B Loss = {avg_loss:.4f}, T-Score = {avg_t_score:.4f}{status_msg}", flush=True)

# Measure catastrophic forgetting
task_a_loss_after_task_b = evaluate_on_task(agent.compression_layer, task_a_batches, device, "Task A")
task_b_loss_final = evaluate_on_task(agent.compression_layer, task_b_batches, device, "Task B")

print(flush=True)
print(f"✅ Task A Loss after Phase 2: {task_a_loss_after_task_b:.4f}", flush=True)
print(f"✅ Task B Loss after Phase 2: {task_b_loss_final:.4f}", flush=True)
print(f"   Sleep events during Phase 2: {phase2_sleep_events}", flush=True)
print(flush=True)

forgetting_godel = task_a_loss_after_task_b - task_a_loss_after_training

print("="*80, flush=True)
print("🎯 CRITICAL RESULTS", flush=True)
print("="*80, flush=True)
print(flush=True)
print(f"Standard Model Baseline: +0.0742 (5.3% degradation)", flush=True)
print(f"GodelAI Forgetting:      {forgetting_godel:+.4f}", flush=True)
print(flush=True)

if forgetting_godel < 0.0742:
    improvement = ((0.0742 - forgetting_godel) / 0.0742) * 100
    print(f"🎉 SUCCESS! GodelAI reduced forgetting by {improvement:.1f}%", flush=True)
    print(f"   Absolute improvement: {0.0742 - forgetting_godel:.4f}", flush=True)
elif forgetting_godel == 0.0742:
    print(f"⚠️  NEUTRAL: GodelAI matched Standard (no improvement)", flush=True)
else:
    degradation = ((forgetting_godel - 0.0742) / 0.0742) * 100
    print(f"❌ FAILURE: GodelAI worse than Standard by {degradation:.1f}%", flush=True)
    print(f"   Additional forgetting: {forgetting_godel - 0.0742:.4f}", flush=True)

print(flush=True)
print(f"Total Sleep Events: {total_sleep_events}", flush=True)
if sleep_epochs:
    print(f"Sleep triggered in epochs: {sleep_epochs}", flush=True)
else:
    print(f"⚠️  WARNING: Sleep Protocol never triggered (threshold may still be too high)", flush=True)
print(flush=True)

# Save results
results = {
    "config": CONFIG,
    "baseline_forgetting": 0.0742,
    "godelai": {
        "task_a_loss_after_phase1": task_a_loss_after_training,
        "task_a_loss_after_phase2": task_a_loss_after_task_b,
        "task_b_loss_final": task_b_loss_final,
        "forgetting": forgetting_godel,
        "total_sleep_events": total_sleep_events,
        "sleep_epochs": sleep_epochs,
    },
    "comparison": {
        "standard_forgetting": 0.0742,
        "godelai_forgetting": forgetting_godel,
        "improvement": 0.0742 - forgetting_godel,
        "improvement_percentage": ((0.0742 - forgetting_godel) / 0.0742) * 100 if forgetting_godel <= 0.0742 else -((forgetting_godel - 0.0742) / 0.0742) * 100,
    },
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
}

output_file = Path(f"results/godel_forgetting_intervention_{results['timestamp']}.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {output_file}", flush=True)
print(flush=True)
print("="*80, flush=True)
print("🎯 CRITICAL EXPERIMENT COMPLETE", flush=True)
print("="*80, flush=True)
