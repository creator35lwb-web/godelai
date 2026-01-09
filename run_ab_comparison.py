#!/usr/bin/env python3
"""
CRITICAL A/B TEST: GodelAI vs Standard Baseline
================================================

SCIENTIFIC OBJECTIVE:
Prove (or disprove) GodelAI's efficacy by comparing against standard training
on IDENTICAL conditions.

TEST DESIGN:
- Model A (Baseline): Standard GRU with T-Score computed in SHADOW MODE (not used)
- Model B (GodelAI): Full C-S-P framework with Sleep Protocol active
- IDENTICAL: Architecture, dataset, seed, hyperparameters
- DIFFERENT: Only the training loop (standard vs GodelAI)

HYPOTHESIS TO TEST:
H1: GodelAI maintains higher T-Score than standard training
H2: GodelAI achieves better validation loss (generalization)
H3: Sleep Protocol provides measurable value

NULL HYPOTHESIS:
T-Score is dataset-dependent, not framework-dependent.
GodelAI provides no advantage over standard training.

Author: Claude Code (Scientific Validation)
Date: January 8, 2026
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
import urllib.request

# Force UTF-8 and unbuffered output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from godelai.agent import GodelAgent

print("="*80, flush=True)
print("SCIENTIFIC A/B TEST: GodelAI vs Standard Baseline", flush=True)
print("="*80, flush=True)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(flush=True)

# ============================================================================
# CONFIGURATION - IDENTICAL FOR BOTH MODELS
# ============================================================================

SEED = 42  # FIXED SEED FOR REPRODUCIBILITY

CONFIG = {
    "epochs": 10,
    "batch_size": 64,
    "seq_length": 100,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "learning_rate": 0.002,
    "seed": SEED,
    # GodelAI specific (only used for Model B)
    "min_surplus_energy": 0.3,
    "propagation_gamma": 2.0,
    "tscore_sample_batches": 3,
}

print("A/B Test Configuration:", flush=True)
print(f"  SEED: {SEED} (CRITICAL - Must be identical for both models)", flush=True)
for key, value in CONFIG.items():
    print(f"  {key}: {value}", flush=True)
print(flush=True)


# ============================================================================
# REPRODUCIBILITY SETUP
# ============================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Set all random seeds to {seed}", flush=True)


# ============================================================================
# DATASET PREPARATION
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

print(f"  Size: {len(text):,} characters", flush=True)

# Character mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"  Vocabulary: {vocab_size} unique characters", flush=True)
print(flush=True)

# Encode
data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

# Split data (SAME SPLIT FOR BOTH MODELS)
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

print(f"Training samples: {len(train_data):,}", flush=True)
print(f"Validation samples: {len(val_data):,}", flush=True)
print(flush=True)


# ============================================================================
# MODEL ARCHITECTURE - IDENTICAL FOR BOTH
# ============================================================================

class CharRNN(nn.Module):
    """Character-level RNN - IDENTICAL for both models."""
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
# DATA BATCHING - IDENTICAL FOR BOTH
# ============================================================================

def create_batches(data, batch_size, seq_length, seed):
    """Create batches with fixed seed for reproducibility."""
    set_seed(seed)  # Reset seed before batching
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
# T-SCORE COMPUTATION (SHADOW MODE FOR STANDARD MODEL)
# ============================================================================

def compute_tscore_shadow(model, sample_batches, criterion, device):
    """
    Compute T-Score in SHADOW MODE (for standard model).
    This DOES NOT affect training - only for logging/comparison.
    """
    model.eval()
    t_scores = []

    for inputs, targets in sample_batches:
        inputs = inputs.to(device)
        targets = targets.to(device)

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

        # Compute T-Score using GodelAI formula
        if len(batch_grads) > 0:
            grad_matrix = torch.stack(batch_grads)

            # GodelAI T-Score formula (v1.1.0) - FIXED: Added **2 to match agent.py
            n = grad_matrix.shape[0]
            sum_grad = torch.sum(grad_matrix, dim=0)
            sum_grad_norm = torch.norm(sum_grad)**2  # FIXED: Squared norm (was linear)
            sum_norm_grad = torch.sum(torch.norm(grad_matrix, dim=1)**2)  # FIXED: Sum of squared norms (was linear)
            ratio = sum_grad_norm / (sum_norm_grad + 1e-8)
            t_score = 1.0 - torch.clamp(ratio / n, 0, 1)
            t_scores.append(t_score.item())

    model.train()
    return sum(t_scores) / len(t_scores) if len(t_scores) > 0 else 0.0


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate(model, val_batches, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, targets in val_batches[:20]:
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
# MODEL A: STANDARD BASELINE (NO GODELAI)
# ============================================================================

def train_standard_model():
    """
    Train STANDARD model with T-Score computed in SHADOW MODE.
    T-Score is LOGGED but NOT USED in training.
    """
    print("="*80, flush=True)
    print("MODEL A: STANDARD BASELINE (No GodelAI)", flush=True)
    print("="*80, flush=True)
    print("T-Score: SHADOW MODE (computed but not used)", flush=True)
    print("Sleep Protocol: DISABLED", flush=True)
    print(flush=True)

    # Set seed for reproducibility
    set_seed(CONFIG["seed"])

    # Create model
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

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Create batches with FIXED SEED
    train_batches = create_batches(train_data, CONFIG["batch_size"], CONFIG["seq_length"], CONFIG["seed"])
    val_batches = create_batches(val_data, CONFIG["batch_size"], CONFIG["seq_length"], CONFIG["seed"])

    print(f"Training batches: {len(train_batches)}", flush=True)
    print(f"Validation batches: {len(val_batches)}", flush=True)
    print(flush=True)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "t_score_shadow": [],  # T-Score computed but NOT used
    }

    best_val_loss = float('inf')
    start_time = time.time()

    # STANDARD TRAINING LOOP
    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}", flush=True)

        # Training
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (inputs, targets) in enumerate(train_batches):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # STANDARD TRAINING - No GodelAI
            optimizer.zero_grad()
            logits, _ = model(inputs)
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_batches)}: loss={loss.item():.4f}", flush=True)

        avg_train_loss = epoch_loss / batch_count
        history["train_loss"].append(avg_train_loss)

        # Validation
        val_loss = evaluate(model, val_batches, criterion, device)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # T-Score in SHADOW MODE (computed but not used)
        sample_batches = [(inputs.to(device), targets.to(device))
                         for inputs, targets in train_batches[:CONFIG["tscore_sample_batches"]]]
        t_score_shadow = compute_tscore_shadow(model, sample_batches, criterion, device)
        history["t_score_shadow"].append(t_score_shadow)

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"  Train Loss: {avg_train_loss:.4f}", flush=True)
        print(f"  Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f})", flush=True)
        print(f"  T-Score (SHADOW): {t_score_shadow:.4f} [NOT USED]", flush=True)
        print(f"  Time: {epoch_time:.1f}s", flush=True)
        print(flush=True)

    total_time = time.time() - start_time

    print("="*80, flush=True)
    print("MODEL A (STANDARD) COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}", flush=True)
    print(f"Best Val Loss: {best_val_loss:.4f}", flush=True)
    print(f"Average T-Score (Shadow): {sum(history['t_score_shadow'])/len(history['t_score_shadow']):.4f}", flush=True)
    print(f"Training Time: {total_time/60:.1f} minutes", flush=True)
    print(flush=True)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "training_time": total_time,
        "model_type": "STANDARD",
    }


# ============================================================================
# MODEL B: GODELAI (FULL C-S-P FRAMEWORK)
# ============================================================================

def train_godelai_model():
    """
    Train with FULL GodelAI framework.
    T-Score is ACTIVE and influences training.
    Sleep Protocol is ENABLED.
    """
    print("="*80, flush=True)
    print("MODEL B: GODELAI (Full C-S-P Framework)", flush=True)
    print("="*80, flush=True)
    print("T-Score: ACTIVE (influences training)", flush=True)
    print("Sleep Protocol: ENABLED", flush=True)
    print(f"Min Surplus Energy (ε): {CONFIG['min_surplus_energy']}", flush=True)
    print(f"Propagation Gamma (γ): {CONFIG['propagation_gamma']}", flush=True)
    print(flush=True)

    # Set seed for reproducibility (SAME AS MODEL A)
    set_seed(CONFIG["seed"])

    # Create model (IDENTICAL ARCHITECTURE)
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

    # GodelAI Agent
    agent = GodelAgent(
        model,
        min_surplus_energy=CONFIG["min_surplus_energy"],
        propagation_gamma=CONFIG["propagation_gamma"]
    )

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Create batches with FIXED SEED (SAME AS MODEL A)
    train_batches = create_batches(train_data, CONFIG["batch_size"], CONFIG["seq_length"], CONFIG["seed"])
    val_batches = create_batches(val_data, CONFIG["batch_size"], CONFIG["seq_length"], CONFIG["seed"])

    print(f"Training batches: {len(train_batches)}", flush=True)
    print(f"Validation batches: {len(val_batches)}", flush=True)
    print(flush=True)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "t_score": [],  # T-Score ACTIVE
        "sleep_events": [],
    }

    best_val_loss = float('inf')
    start_time = time.time()

    # GODELAI TRAINING LOOP
    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}", flush=True)

        # Training
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (inputs, targets) in enumerate(train_batches):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # STANDARD TRAINING STEP (GodelAI computes T-Score separately)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_batches)}: loss={loss.item():.4f}", flush=True)

        avg_train_loss = epoch_loss / batch_count
        history["train_loss"].append(avg_train_loss)

        # Validation
        val_loss = evaluate(model, val_batches, criterion, device)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # T-Score computation (ACTIVE - using GodelAI)
        sample_batches = train_batches[:CONFIG["tscore_sample_batches"]]
        t_scores = []
        sleep_count = 0

        for inputs, targets in sample_batches:
            inputs = inputs.to(device)
            targets = targets.to(device)

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

            # Compute T-Score using GodelAI
            if len(batch_grads) > 0:
                grad_matrix = torch.stack(batch_grads)
                wisdom = agent.measure_gradient_diversity(grad_matrix).item()
                t_scores.append(wisdom)

                # Check Sleep Protocol
                if wisdom < agent.epsilon:
                    sleep_count += 1
                    agent.rest_and_reflect()
                    print(f"  💤 SLEEP PROTOCOL TRIGGERED (T-Score: {wisdom:.4f} < {agent.epsilon})", flush=True)

        avg_t_score = sum(t_scores) / len(t_scores) if len(t_scores) > 0 else 0.0
        history["t_score"].append(avg_t_score)
        history["sleep_events"].append(sleep_count)

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"  Train Loss: {avg_train_loss:.4f}", flush=True)
        print(f"  Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f})", flush=True)
        print(f"  T-Score (ACTIVE): {avg_t_score:.4f}", flush=True)
        print(f"  Sleep Events: {sleep_count}", flush=True)
        print(f"  Time: {epoch_time:.1f}s", flush=True)
        print(flush=True)

    total_time = time.time() - start_time

    print("="*80, flush=True)
    print("MODEL B (GODELAI) COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}", flush=True)
    print(f"Best Val Loss: {best_val_loss:.4f}", flush=True)
    print(f"Average T-Score: {sum(history['t_score'])/len(history['t_score']):.4f}", flush=True)
    print(f"Total Sleep Events: {sum(history['sleep_events'])}", flush=True)
    print(f"Training Time: {total_time/60:.1f} minutes", flush=True)
    print(flush=True)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "training_time": total_time,
        "total_sleep_events": sum(history["sleep_events"]),
        "model_type": "GODELAI",
    }


# ============================================================================
# MAIN A/B TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80, flush=True)
    print("STARTING A/B TEST", flush=True)
    print("="*80, flush=True)
    print(flush=True)

    # Run Model A (Standard)
    print("RUNNING MODEL A (STANDARD BASELINE)...", flush=True)
    print(flush=True)
    results_standard = train_standard_model()

    print("\n\n", flush=True)

    # Run Model B (GodelAI)
    print("RUNNING MODEL B (GODELAI)...", flush=True)
    print(flush=True)
    results_godelai = train_godelai_model()

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"ab_test_{timestamp}.json"

    ab_results = {
        "config": CONFIG,
        "standard": results_standard,
        "godelai": results_godelai,
        "timestamp": timestamp,
    }

    with open(results_file, 'w') as f:
        json.dump(ab_results, f, indent=2)

    print("="*80, flush=True)
    print("A/B TEST COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"Results saved to: {results_file}", flush=True)
    print(flush=True)

    # Quick comparison
    print("QUICK COMPARISON:", flush=True)
    print(f"  Standard Best Val Loss: {results_standard['best_val_loss']:.4f}", flush=True)
    print(f"  GodelAI Best Val Loss: {results_godelai['best_val_loss']:.4f}", flush=True)
    print(f"  Difference: {results_standard['best_val_loss'] - results_godelai['best_val_loss']:.4f}", flush=True)
    print(flush=True)
    print(f"  Standard Avg T-Score (Shadow): {sum(results_standard['history']['t_score_shadow'])/len(results_standard['history']['t_score_shadow']):.4f}", flush=True)
    print(f"  GodelAI Avg T-Score (Active): {sum(results_godelai['history']['t_score'])/len(results_godelai['history']['t_score']):.4f}", flush=True)
    print(f"  Difference: {sum(results_godelai['history']['t_score'])/len(results_godelai['history']['t_score']) - sum(results_standard['history']['t_score_shadow'])/len(results_standard['history']['t_score_shadow']):.4f}", flush=True)
    print(flush=True)
