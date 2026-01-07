#!/usr/bin/env python3
"""
Diagnostic script to measure Shakespeare benchmark performance on CPU.
Tests just 1 epoch to estimate total time needed.
"""
import sys
import io
import torch
import torch.nn as nn
import time
from pathlib import Path

# Force UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from godelai.agent import GodelAgent

print("=" * 70)
print("SHAKESPEARE BENCHMARK PERFORMANCE DIAGNOSTIC")
print("=" * 70)
print()

# Check system
print("System Configuration:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print()

# Load dataset
print("Loading dataset...")
data_path = Path("data/shakespeare.txt")
if not data_path.exists():
    print("❌ Dataset not found! Run the full benchmark first to download it.")
    sys.exit(1)

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"  Dataset size: {len(text):,} characters ({len(text)/1024/1024:.2f} MB)")
print()

# Create character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"  Vocabulary size: {vocab_size} unique characters")
print()

# Encode text
data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
print(f"  Encoded data shape: {data.shape}")
print()

# Model configuration (same as full benchmark)
config = {
    "seq_length": 100,
    "batch_size": 64,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
}

print("Model Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")
print()

# Create model
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.fc(output)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CharRNN(vocab_size, config['embedding_dim'], config['hidden_dim'], config['num_layers']).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")
print()

# Create GodelAI agent
agent = GodelAgent(
    model,
    min_surplus_energy=0.3,
    propagation_gamma=2.0
)

# Prepare one batch
print("Preparing test batch...")
seq_len = config['seq_length']
batch_size = config['batch_size']

# Get indices for one batch
batch_start = 0
inputs = []
targets = []
for i in range(batch_size):
    start_idx = batch_start + i * 100
    if start_idx + seq_len + 1 < len(data):
        inputs.append(data[start_idx:start_idx + seq_len])
        targets.append(data[start_idx + 1:start_idx + seq_len + 1])

if len(inputs) < batch_size:
    print(f"⚠️  Warning: Only {len(inputs)} sequences available")
    batch_size = len(inputs)

input_batch = torch.stack(inputs).to(device)
target_batch = torch.stack(targets).to(device)

print(f"  Batch shape: {input_batch.shape}")
print()

# Test forward pass
print("Testing forward pass...")
start_time = time.time()
logits = model(input_batch)
forward_time = time.time() - start_time
print(f"  ✅ Forward pass: {forward_time:.3f}s")
print()

# Test loss computation
print("Testing loss computation...")
start_time = time.time()
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits.reshape(-1, vocab_size), target_batch.reshape(-1))
loss_time = time.time() - start_time
print(f"  ✅ Loss computation: {loss_time:.3f}s")
print(f"  Loss value: {loss.item():.4f}")
print()

# Test backward pass
print("Testing backward pass...")
start_time = time.time()
loss.backward()
backward_time = time.time() - start_time
print(f"  ✅ Backward pass: {backward_time:.3f}s")
print()

# **CRITICAL TEST**: T-Score computation (per-sample gradients)
print("Testing T-Score computation (per-sample gradients)...")
print("  ⚠️  This is the expensive operation!")
print()

start_time = time.time()

# Compute per-sample gradients (this is what GodelAI does)
batch_grads = []
for i in range(input_batch.size(0)):
    sample_input = input_batch[i:i+1]
    sample_target = target_batch[i:i+1]

    model.zero_grad()
    logits = model(sample_input)
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = sample_target.view(-1)
    loss = loss_fn(logits_flat, targets_flat)
    loss.backward()

    # Collect gradients
    sample_grads = []
    for param in model.parameters():
        if param.grad is not None:
            sample_grads.append(param.grad.view(-1).clone())
    if sample_grads:
        batch_grads.append(torch.cat(sample_grads))

# Compute T-Score
if len(batch_grads) > 0:
    grad_matrix = torch.stack(batch_grads)
    tscore_value = agent.measure_gradient_diversity(grad_matrix).item()
else:
    tscore_value = 0.0

tscore_time = time.time() - start_time

print(f"  ✅ T-Score computation: {tscore_time:.3f}s")
print(f"  T-Score value: {tscore_value:.4f}")
print(f"  Per-sample gradients computed: {len(batch_grads)}")
print()

# Estimate total time
print("=" * 70)
print("TIME ESTIMATES")
print("=" * 70)
print()

# Calculate batches per epoch
total_sequences = (len(data) - seq_len - 1) // 100
batches_per_epoch = total_sequences // batch_size
print(f"Batches per epoch: {batches_per_epoch:,}")
print()

# Time per batch (training step)
time_per_batch = forward_time + loss_time + backward_time
print(f"Time per training batch: {time_per_batch:.3f}s")
print()

# T-Score is computed once per epoch on 5 sample batches (per the code)
tscore_batches_per_epoch = 5
tscore_time_per_epoch = tscore_time * tscore_batches_per_epoch
print(f"T-Score time per epoch: {tscore_time_per_epoch:.1f}s ({tscore_batches_per_epoch} sample batches)")
print()

# Total time per epoch
training_time_per_epoch = (time_per_batch * batches_per_epoch)
total_time_per_epoch = training_time_per_epoch + tscore_time_per_epoch
print(f"Training time per epoch: {training_time_per_epoch:.1f}s")
print(f"Total time per epoch: {total_time_per_epoch:.1f}s ({total_time_per_epoch/60:.1f} minutes)")
print()

# Full benchmark (30 epochs)
total_epochs = 30
total_time_estimate = total_time_per_epoch * total_epochs
hours = int(total_time_estimate // 3600)
minutes = int((total_time_estimate % 3600) // 60)

print(f"FULL BENCHMARK (30 epochs):")
print(f"  Estimated time: {hours}h {minutes}m ({total_time_estimate/3600:.2f} hours)")
print()

# Analysis
print("=" * 70)
print("ANALYSIS")
print("=" * 70)
print()

print("Per-sample gradient computation (T-Score) breakdown:")
print(f"  Time per batch: {tscore_time:.3f}s")
print(f"  Batches per epoch: {tscore_batches_per_epoch}")
print(f"  Total epochs: {total_epochs}")
print(f"  Total T-Score time: {tscore_time * tscore_batches_per_epoch * total_epochs:.1f}s ({tscore_time * tscore_batches_per_epoch * total_epochs / 60:.1f} minutes)")
print()

if total_time_estimate > 7200:  # More than 2 hours
    print("⚠️  RECOMMENDATION: Full benchmark will take over 2 hours!")
    print()
    print("Options:")
    print("  1. Reduce T-Score sampling (e.g., every 3 epochs instead of every epoch)")
    print("  2. Use fewer T-Score sample batches (e.g., 3 instead of 5)")
    print("  3. Reduce total epochs (e.g., 20 instead of 30)")
    print("  4. Run on a machine with GPU acceleration")
    print()
elif total_time_estimate > 3600:  # More than 1 hour
    print("⚠️  Full benchmark will take over 1 hour.")
    print("   Consider running overnight or reducing configuration.")
    print()
else:
    print("✅ Full benchmark should complete in reasonable time!")
    print()

print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
