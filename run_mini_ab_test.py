#!/usr/bin/env python3
"""
CRITICAL TEST #1: Mini Shakespeare A/B Comparison
==================================================

HYPOTHESIS:
On small datasets (5KB), GodelAI should provide value:
- Sleep Protocol should trigger (it did 30 times in original mini test)
- Should prevent overfitting / maintain gradient diversity
- Should improve generalization (better val loss)

This is THE test for Sleep Protocol efficacy.

If this test is NEGATIVE, Sleep Protocol may have very limited utility.
If this test is POSITIVE, we've found where GodelAI works!

Author: Claude Code (Scientific Validation)
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

# Force UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from godelai.agent import GodelAgent

print("="*80, flush=True)
print("CRITICAL TEST #1: Mini Shakespeare A/B Test", flush=True)
print("="*80, flush=True)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(flush=True)
print("HYPOTHESIS: GodelAI helps on small datasets (5KB)", flush=True)
print("EXPECTATION: Sleep Protocol triggers, improves generalization", flush=True)
print(flush=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42  # FIXED SEED

CONFIG = {
    "epochs": 10,
    "batch_size": 32,  # Smaller for mini dataset
    "seq_length": 100,
    "embedding_dim": 64,  # Smaller model for small dataset
    "hidden_dim": 128,
    "num_layers": 2,
    "learning_rate": 0.002,
    "seed": SEED,
    "min_surplus_energy": 0.3,
    "propagation_gamma": 2.0,
    "tscore_sample_batches": 3,
}

print("Configuration:", flush=True)
for key, value in CONFIG.items():
    print(f"  {key}: {value}", flush=True)
print(flush=True)


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# MINI DATASET (5KB - FIRST 5000 CHARS OF SHAKESPEARE)
# ============================================================================

print("Loading MINI Shakespeare dataset (5KB)...", flush=True)

# Load full dataset and take first 5000 chars
data_path = Path("data/shakespeare.txt")
if not data_path.exists():
    print("❌ Dataset not found! Run full benchmark first.", flush=True)
    sys.exit(1)

with open(data_path, 'r', encoding='utf-8') as f:
    full_text = f.read()

# Take first 5000 characters (similar to original mini benchmark)
text = full_text[:5000]

print(f"  Size: {len(text):,} characters (mini dataset)", flush=True)

# Character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"  Vocabulary: {vocab_size} unique characters", flush=True)
print(flush=True)

# Encode
data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

# Split
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

print(f"Training samples: {len(train_data):,}", flush=True)
print(f"Validation samples: {len(val_data):,}", flush=True)
print(flush=True)


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


# ============================================================================
# DATA BATCHING
# ============================================================================

def create_batches(data, batch_size, seq_length, seed):
    set_seed(seed)
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
                torch.stack(batch_inputs),
                torch.stack(batch_targets)
            ))

    return batches


# ============================================================================
# T-SCORE COMPUTATION (SHADOW MODE FOR STANDARD)
# ============================================================================

def compute_tscore_shadow(model, sample_batches, criterion, device):
    model.eval()
    t_scores = []

    for inputs, targets in sample_batches:
        inputs = inputs.to(device)
        targets = targets.to(device)

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

            sample_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    sample_grads.append(param.grad.view(-1).clone())
            if sample_grads:
                batch_grads.append(torch.cat(sample_grads))

        if len(batch_grads) > 0:
            grad_matrix = torch.stack(batch_grads)
            n = grad_matrix.shape[0]
            sum_grad = torch.sum(grad_matrix, dim=0)
            sum_norm_grad = torch.norm(sum_grad)
            sum_grad_norm = torch.sum(torch.norm(grad_matrix, dim=1))
            ratio = sum_grad_norm / (sum_norm_grad + 1e-8)
            t_score = 1.0 - torch.clamp(ratio / n, 0, 1)
            t_scores.append(t_score.item())

    model.train()
    return sum(t_scores) / len(t_scores) if len(t_scores) > 0 else 0.0


def evaluate(model, val_batches, criterion, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, targets in val_batches:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, _ = model(inputs)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item()
            total_batches += 1

    model.train()
    return total_loss / total_batches if total_batches > 0 else 0.0


# ============================================================================
# MODEL A: STANDARD BASELINE
# ============================================================================

def train_standard_model():
    print("="*80, flush=True)
    print("MODEL A: STANDARD BASELINE (No GodelAI)", flush=True)
    print("="*80, flush=True)
    print(flush=True)

    set_seed(CONFIG["seed"])
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

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    train_batches = create_batches(train_data, CONFIG["batch_size"], CONFIG["seq_length"], CONFIG["seed"])
    val_batches = create_batches(val_data, CONFIG["batch_size"], CONFIG["seq_length"], CONFIG["seed"])

    print(f"Training batches: {len(train_batches)}", flush=True)
    print(f"Validation batches: {len(val_batches)}", flush=True)
    print(flush=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "t_score_shadow": [],
    }

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}", flush=True)

        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for inputs, targets in train_batches:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_train_loss = epoch_loss / batch_count
        history["train_loss"].append(avg_train_loss)

        val_loss = evaluate(model, val_batches, criterion, device)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        sample_batches = [(inputs.to(device), targets.to(device))
                         for inputs, targets in train_batches[:CONFIG["tscore_sample_batches"]]]
        t_score_shadow = compute_tscore_shadow(model, sample_batches, criterion, device)
        history["t_score_shadow"].append(t_score_shadow)

        print(f"  Train Loss: {avg_train_loss:.4f}", flush=True)
        print(f"  Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f})", flush=True)
        print(f"  T-Score (SHADOW): {t_score_shadow:.4f} [NOT USED]", flush=True)
        print(flush=True)

    total_time = time.time() - start_time

    print("="*80, flush=True)
    print("MODEL A (STANDARD) COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"Best Val Loss: {best_val_loss:.4f}", flush=True)
    print(f"Average T-Score (Shadow): {sum(history['t_score_shadow'])/len(history['t_score_shadow']):.4f}", flush=True)
    print(flush=True)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "training_time": total_time,
        "model_type": "STANDARD",
    }


# ============================================================================
# MODEL B: GODELAI
# ============================================================================

def train_godelai_model():
    print("="*80, flush=True)
    print("MODEL B: GODELAI (Full C-S-P Framework)", flush=True)
    print("="*80, flush=True)
    print(flush=True)

    set_seed(CONFIG["seed"])
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

    agent = GodelAgent(
        model,
        min_surplus_energy=CONFIG["min_surplus_energy"],
        propagation_gamma=CONFIG["propagation_gamma"]
    )

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    train_batches = create_batches(train_data, CONFIG["batch_size"], CONFIG["seq_length"], CONFIG["seed"])
    val_batches = create_batches(val_data, CONFIG["batch_size"], CONFIG["seq_length"], CONFIG["seed"])

    print(f"Training batches: {len(train_batches)}", flush=True)
    print(f"Validation batches: {len(val_batches)}", flush=True)
    print(flush=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "t_score": [],
        "sleep_events": [],
    }

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}", flush=True)

        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for inputs, targets in train_batches:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_train_loss = epoch_loss / batch_count
        history["train_loss"].append(avg_train_loss)

        val_loss = evaluate(model, val_batches, criterion, device)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # T-Score computation with Sleep Protocol
        sample_batches = train_batches[:CONFIG["tscore_sample_batches"]]
        t_scores = []
        sleep_count = 0

        for inputs, targets in sample_batches:
            inputs = inputs.to(device)
            targets = targets.to(device)

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

                sample_grads = []
                for param in model.parameters():
                    if param.grad is not None:
                        sample_grads.append(param.grad.view(-1).clone())
                if sample_grads:
                    batch_grads.append(torch.cat(sample_grads))

            if len(batch_grads) > 0:
                grad_matrix = torch.stack(batch_grads)
                wisdom = agent.measure_gradient_diversity(grad_matrix).item()
                t_scores.append(wisdom)

                if wisdom < agent.epsilon:
                    sleep_count += 1
                    agent.rest_and_reflect()
                    print(f"  💤 SLEEP PROTOCOL TRIGGERED (T-Score: {wisdom:.4f} < {agent.epsilon})", flush=True)

        avg_t_score = sum(t_scores) / len(t_scores) if len(t_scores) > 0 else 0.0
        history["t_score"].append(avg_t_score)
        history["sleep_events"].append(sleep_count)

        print(f"  Train Loss: {avg_train_loss:.4f}", flush=True)
        print(f"  Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f})", flush=True)
        print(f"  T-Score (ACTIVE): {avg_t_score:.4f}", flush=True)
        print(f"  Sleep Events: {sleep_count}", flush=True)
        print(flush=True)

    total_time = time.time() - start_time

    print("="*80, flush=True)
    print("MODEL B (GODELAI) COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"Best Val Loss: {best_val_loss:.4f}", flush=True)
    print(f"Average T-Score: {sum(history['t_score'])/len(history['t_score']):.4f}", flush=True)
    print(f"Total Sleep Events: {sum(history['sleep_events'])}", flush=True)
    print(flush=True)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "training_time": total_time,
        "total_sleep_events": sum(history["sleep_events"]),
        "model_type": "GODELAI",
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("RUNNING MODEL A (STANDARD)...", flush=True)
    print(flush=True)
    results_standard = train_standard_model()

    print("\n\n", flush=True)

    print("RUNNING MODEL B (GODELAI)...", flush=True)
    print(flush=True)
    results_godelai = train_godelai_model()

    # Save results
    results_dir = Path("results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"mini_ab_test_{timestamp}.json"

    ab_results = {
        "config": CONFIG,
        "dataset": "Mini Shakespeare (5KB, first 5000 chars)",
        "standard": results_standard,
        "godelai": results_godelai,
        "timestamp": timestamp,
    }

    with open(results_file, 'w') as f:
        json.dump(ab_results, f, indent=2)

    print("="*80, flush=True)
    print("MINI A/B TEST COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"Results saved to: {results_file}", flush=True)
    print(flush=True)

    # Analysis
    standard_avg_t = sum(results_standard['history']['t_score_shadow'])/len(results_standard['history']['t_score_shadow'])
    godelai_avg_t = sum(results_godelai['history']['t_score'])/len(results_godelai['history']['t_score'])

    val_diff = results_standard['best_val_loss'] - results_godelai['best_val_loss']
    val_improvement = (val_diff / results_standard['best_val_loss']) * 100

    print("COMPARISON:", flush=True)
    print(f"  Standard Best Val Loss: {results_standard['best_val_loss']:.4f}", flush=True)
    print(f"  GodelAI Best Val Loss: {results_godelai['best_val_loss']:.4f}", flush=True)
    print(f"  Difference: {val_diff:.4f} ({val_improvement:+.2f}%)", flush=True)
    print(flush=True)
    print(f"  Standard Avg T-Score: {standard_avg_t:.4f}", flush=True)
    print(f"  GodelAI Avg T-Score: {godelai_avg_t:.4f}", flush=True)
    print(f"  Difference: {godelai_avg_t - standard_avg_t:+.4f}", flush=True)
    print(flush=True)
    print(f"  GodelAI Sleep Events: {results_godelai['total_sleep_events']}", flush=True)
    print(flush=True)

    # Verdict
    if abs(val_improvement) >= 5:
        if val_improvement > 0:
            print("✅ POSITIVE RESULT: GodelAI improves validation loss by >5%", flush=True)
        else:
            print("❌ NEGATIVE RESULT: GodelAI WORSENS validation loss by >5%", flush=True)
    elif abs(val_improvement) < 1:
        print("❌ NEGATIVE RESULT: No meaningful difference in validation loss", flush=True)
    else:
        print("❓ UNCLEAR: Small difference, may need more epochs", flush=True)
