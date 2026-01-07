#!/usr/bin/env python3
"""
Optimized Full Shakespeare Benchmark with Visible Progress
===========================================================
- Reduced epochs (10-20) for faster completion
- Progress bars and real-time output
- Incremental result saving
- Unbuffered output for monitoring

Author: Claude Code
Date: January 8, 2026
"""
import sys
import io

# Force UTF-8 and unbuffered output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
from datetime import datetime
from pathlib import Path
import urllib.request

from godelai.agent import GodelAgent


print("="*80, flush=True)
print("GodelAI v1.1.0 - OPTIMIZED Shakespeare Benchmark", flush=True)
print("="*80, flush=True)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(flush=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "epochs": 10,  # Reduced from 30 for faster validation
    "batch_size": 64,
    "seq_length": 100,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "learning_rate": 0.002,
    "min_surplus_energy": 0.3,
    "propagation_gamma": 2.0,
    "tscore_sample_batches": 3,  # Reduced from 5
}

print("Configuration:", flush=True)
for key, value in CONFIG.items():
    print(f"  {key}: {value}", flush=True)
print(flush=True)


# ============================================================================
# DATASET
# ============================================================================

def download_shakespeare():
    """Download Tiny Shakespeare dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    filepath = data_dir / "shakespeare.txt"

    if not filepath.exists():
        print(f"📥 Downloading from {url}...", flush=True)
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ Downloaded to {filepath}", flush=True)
    else:
        print(f"✅ Using cached dataset: {filepath}", flush=True)

    return filepath


print("Loading dataset...", flush=True)
data_path = download_shakespeare()

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"  Size: {len(text):,} characters ({len(text)/1024/1024:.2f} MB)", flush=True)

# Character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"  Vocabulary: {vocab_size} unique characters", flush=True)
print(flush=True)

# Encode
data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)


# ============================================================================
# MODEL
# ============================================================================

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

model = CharRNN(
    vocab_size=vocab_size,
    embedding_dim=CONFIG["embedding_dim"],
    hidden_dim=CONFIG["hidden_dim"],
    num_layers=CONFIG["num_layers"]
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}", flush=True)
print(flush=True)


# ============================================================================
# GODELAI AGENT
# ============================================================================

agent = GodelAgent(
    model,
    min_surplus_energy=CONFIG["min_surplus_energy"],
    propagation_gamma=CONFIG["propagation_gamma"]
)
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
criterion = nn.CrossEntropyLoss()


# ============================================================================
# DATA PREPARATION
# ============================================================================

# Split data
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

print(f"Training samples: {len(train_data):,}", flush=True)
print(f"Validation samples: {len(val_data):,}", flush=True)
print(flush=True)


def create_batches(data, batch_size, seq_length):
    """Create training batches."""
    batches = []
    max_start = len(data) - seq_length - 1

    for i in range(0, max_start, batch_size * 100):
        batch_inputs = []
        batch_targets = []

        for j in range(batch_size):
            start_idx = i + j * 100
            if start_idx + seq_length + 1 < len(data):
                batch_inputs.append(data[start_idx:start_idx + seq_length])
                batch_targets.append(data[start_idx + 1:start_idx + seq_length + 1])

        if len(batch_inputs) == batch_size:
            batches.append((
                torch.stack(batch_inputs).to(device),
                torch.stack(batch_targets).to(device)
            ))

    return batches


train_batches = create_batches(train_data, CONFIG["batch_size"], CONFIG["seq_length"])
val_batches = create_batches(val_data, CONFIG["batch_size"], CONFIG["seq_length"])

print(f"Training batches: {len(train_batches)}", flush=True)
print(f"Validation batches: {len(val_batches)}", flush=True)
print(flush=True)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def compute_tscore(model, sample_batches, agent, criterion):
    """Compute T-Score on sample batches."""
    model.eval()
    t_scores = []
    sleep_count = 0

    for inputs, targets in sample_batches:
        # Compute per-sample gradients
        batch_grads = []
        for i in range(inputs.size(0)):
            sample_input = inputs[i:i+1]
            sample_target = targets[i:i+1]

            model.zero_grad()
            logits, _ = model(sample_input)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = sample_target.view(-1)
            loss = criterion(logits_flat, targets_flat)
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
            wisdom = agent.measure_gradient_diversity(grad_matrix).item()
            t_scores.append(wisdom)

            # Check sleep protocol
            if wisdom < agent.epsilon:
                sleep_count += 1

    model.train()
    avg_t_score = sum(t_scores) / len(t_scores) if len(t_scores) > 0 else 0.0
    return avg_t_score, sleep_count


def evaluate(model, val_batches, criterion):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, targets in val_batches[:20]:  # Sample of val set
            logits, _ = model(inputs)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item()
            total_batches += 1

    model.train()
    return total_loss / total_batches if total_batches > 0 else 0.0


def generate_text(model, seed_text, length=200):
    """Generate text sample."""
    model.eval()
    chars_generated = []

    # Encode seed
    seed_indices = [char_to_idx.get(c, 0) for c in seed_text[-CONFIG["seq_length"]:]]
    current_seq = torch.tensor([seed_indices], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(length):
            logits, _ = model(current_seq)
            probs = torch.softmax(logits[0, -1], dim=0)
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_char_idx]
            chars_generated.append(next_char)

            # Update sequence
            current_seq = torch.cat([
                current_seq[:, 1:],
                torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
            ], dim=1)

    model.train()
    return ''.join(chars_generated)


# ============================================================================
# TRAINING LOOP
# ============================================================================

print("="*80, flush=True)
print("TRAINING START", flush=True)
print("="*80, flush=True)
print(flush=True)

history = {
    "train_loss": [],
    "val_loss": [],
    "t_score": [],
    "sleep_events": [],
}

samples = []
best_val_loss = float('inf')
start_time = time.time()

for epoch in range(CONFIG["epochs"]):
    epoch_start = time.time()
    print(f"Epoch {epoch+1}/{CONFIG['epochs']}", flush=True)

    # Training
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for batch_idx, (inputs, targets) in enumerate(train_batches):
        optimizer.zero_grad()
        logits, _ = model(inputs)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

        # Progress indicator every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_batches)}: loss={loss.item():.4f}", flush=True)

    avg_train_loss = epoch_loss / batch_count
    history["train_loss"].append(avg_train_loss)

    # Validation
    val_loss = evaluate(model, val_batches, criterion)
    history["val_loss"].append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss

    # T-Score computation
    sample_batches = train_batches[:CONFIG["tscore_sample_batches"]]
    t_score, sleep_events = compute_tscore(model, sample_batches, agent, criterion)
    history["t_score"].append(t_score)
    history["sleep_events"].append(sleep_events)

    # Generate sample text
    if epoch % 2 == 0:  # Every 2 epochs
        sample_text = generate_text(model, "ROMEO:\n", length=200)
        samples.append(sample_text)

    epoch_time = time.time() - epoch_start

    # Print epoch summary
    print(f"  Train Loss: {avg_train_loss:.4f}", flush=True)
    print(f"  Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f})", flush=True)
    print(f"  T-Score: {t_score:.4f}", flush=True)
    print(f"  Sleep Events: {sleep_events}", flush=True)
    print(f"  Time: {epoch_time:.1f}s", flush=True)
    print(flush=True)

total_time = time.time() - start_time


# ============================================================================
# SAVE RESULTS
# ============================================================================

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = results_dir / f"shakespeare_optimized_{timestamp}.json"

results = {
    "config": CONFIG,
    "system": {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    },
    "history": history,
    "samples": samples,
    "final_metrics": {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": best_val_loss,
        "final_t_score": history["t_score"][-1],
        "avg_t_score": sum(history["t_score"]) / len(history["t_score"]),
        "total_sleep_events": sum(history["sleep_events"]),
        "total_params": total_params,
        "training_time_minutes": total_time / 60,
    }
}

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print("="*80, flush=True)
print("TRAINING COMPLETE!", flush=True)
print("="*80, flush=True)
print(flush=True)
print(f"Results saved to: {results_file}", flush=True)
print(flush=True)
print("Final Metrics:", flush=True)
print(f"  Epochs: {CONFIG['epochs']}", flush=True)
print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}", flush=True)
print(f"  Best Val Loss: {best_val_loss:.4f}", flush=True)
print(f"  Final T-Score: {history['t_score'][-1]:.4f}", flush=True)
print(f"  Average T-Score: {results['final_metrics']['avg_t_score']:.4f}", flush=True)
print(f"  Total Sleep Events: {sum(history['sleep_events'])}", flush=True)
print(f"  Training Time: {total_time/60:.1f} minutes", flush=True)
print(flush=True)

if samples:
    print("Latest Generated Text:", flush=True)
    print("-"*80, flush=True)
    print(samples[-1][:300], flush=True)
    if len(samples[-1]) > 300:
        print("...", flush=True)
    print("-"*80, flush=True)
