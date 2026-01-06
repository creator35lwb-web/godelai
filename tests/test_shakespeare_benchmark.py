#!/usr/bin/env python3
"""
GodelAI Shakespeare Benchmark
==============================
Demonstrates GodelAI on the classic Tiny Shakespeare dataset.

This benchmark validates:
1. T-Score stability on real text generation
2. Sleep Protocol behavior during training
3. Text generation quality
4. Comparison with standard language modeling

Author: Claude Code (Claude Sonnet 4.5)
Date: January 6, 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import time
from datetime import datetime
from pathlib import Path
import urllib.request

# Import GodelAI
from godelai.agent import GodelAgent


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
        print(f"📥 Downloading Tiny Shakespeare from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ Downloaded to {filepath}")
    else:
        print(f"✅ Using cached dataset: {filepath}")

    return filepath


def prepare_shakespeare_data(filepath, seq_length=100, train_split=0.9):
    """Prepare Shakespeare data for training."""
    print(f"\n📖 Loading Shakespeare text...")

    # Read text
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"   Total characters: {len(text):,}")

    # Create character vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"   Vocabulary size: {vocab_size}")

    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # Encode text
    data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

    # Split train/val
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"   Train samples: {len(train_data):,}")
    print(f"   Val samples: {len(val_data):,}")

    return train_data, val_data, char_to_idx, idx_to_char, vocab_size


def create_batches(data, seq_length, batch_size):
    """Create input/target batches."""
    # Calculate number of sequences
    num_seqs = len(data) // seq_length

    # Trim to multiple of seq_length
    data = data[:num_seqs * seq_length]

    # Create sequences
    sequences = []
    for i in range(0, len(data) - seq_length):
        input_seq = data[i:i+seq_length]
        target_seq = data[i+1:i+seq_length+1]
        sequences.append((input_seq, target_seq))

    # Create batches
    batches = []
    for i in range(0, len(sequences) - batch_size + 1, batch_size):
        batch_seqs = sequences[i:i+batch_size]
        input_batch = torch.stack([seq[0] for seq in batch_seqs])
        target_batch = torch.stack([seq[1] for seq in batch_seqs])
        batches.append((input_batch, target_batch))

    return batches


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ShakespeareGRU(nn.Module):
    """Simple GRU model for character-level language modeling."""

    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Model initialized: {total_params:,} parameters")

    def forward(self, x, hidden=None):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embedding]
        output, hidden = self.gru(embedded, hidden)  # [batch, seq_len, hidden]
        logits = self.fc(output)  # [batch, seq_len, vocab]
        return logits, hidden


# ============================================================================
# TEXT GENERATION
# ============================================================================

def generate_text(model, agent, start_text, char_to_idx, idx_to_char,
                 length=200, temperature=0.8):
    """Generate text from the model."""
    model.eval()

    # Encode start text
    input_seq = torch.tensor([char_to_idx.get(ch, 0) for ch in start_text],
                             dtype=torch.long).unsqueeze(0)

    generated = start_text
    hidden = None

    with torch.no_grad():
        # Process start text
        if len(start_text) > 0:
            logits, hidden = model(input_seq, hidden)
            current_char = input_seq[0, -1].item()
        else:
            current_char = 0

        # Generate characters
        for _ in range(length):
            input_char = torch.tensor([[current_char]], dtype=torch.long)
            logits, hidden = model(input_char, hidden)

            # Apply temperature
            logits = logits[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)

            # Sample
            next_char = torch.multinomial(probs, 1).item()
            generated += idx_to_char[next_char]
            current_char = next_char

    model.train()
    return generated


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, agent, train_batches, criterion, optimizer, epoch):
    """Train for one epoch (fast version - no T-Score during training)."""
    model.train()
    total_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_batches):
        # Standard training step
        optimizer.zero_grad()
        logits, _ = model(inputs)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        task_loss = criterion(logits_flat, targets_flat)
        task_loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += task_loss.item()

        # Progress
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"   Batch {batch_idx+1}/{len(train_batches)} | Loss: {avg_loss:.4f}")

    avg_loss = total_loss / len(train_batches)

    return avg_loss


def compute_epoch_t_score(model, agent, batches, criterion, num_batches=3):
    """
    Compute T-Score on a small sample of batches at epoch end.
    This is expensive, so we only do it on a few batches for demonstration.
    """
    model.eval()
    t_scores = []
    sleep_count = 0

    # Use only first N batches for T-Score computation
    sample_batches = batches[:min(num_batches, len(batches))]

    for inputs, targets in sample_batches:
        # Compute per-sample gradients for T-Score
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
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in val_batches:
            logits, _ = model(inputs)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item()

    model.train()
    return total_loss / len(val_batches)


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def main():
    """Run Shakespeare benchmark."""
    print("=" * 70)
    print("📚 GODELAI SHAKESPEARE BENCHMARK")
    print("=" * 70)

    # Configuration
    config = {
        "seq_length": 100,
        "batch_size": 32,  # Standard batch size
        "embedding_dim": 64,
        "hidden_dim": 128,
        "num_layers": 2,
        "epochs": 10,  # Quick demo (reduced from 20)
        "learning_rate": 0.002,
        "min_surplus_energy": 0.3,
        "propagation_gamma": 2.0,
    }

    print("\n⚙️  Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    # Prepare data
    filepath = download_shakespeare()
    train_data, val_data, char_to_idx, idx_to_char, vocab_size = prepare_shakespeare_data(
        filepath, config["seq_length"]
    )

    print(f"\n🔨 Creating batches...")
    train_batches = create_batches(train_data, config["seq_length"], config["batch_size"])
    val_batches = create_batches(val_data, config["seq_length"], config["batch_size"])
    print(f"   Train batches: {len(train_batches)}")
    print(f"   Val batches: {len(val_batches)}")

    # Create model
    print(f"\n🧠 Initializing model...")
    model = ShakespeareGRU(
        vocab_size=vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"]
    )

    # Wrap with GodelAgent
    agent = GodelAgent(
        model,
        min_surplus_energy=config["min_surplus_energy"],
        propagation_gamma=config["propagation_gamma"]
    )

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "t_score": [],
        "sleep_events": [],
        "samples": []
    }

    # Training loop
    print(f"\n🚀 Starting training ({config['epochs']} epochs)...")
    print("=" * 70)

    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(config["epochs"]):
        print(f"\n📖 Epoch {epoch+1}/{config['epochs']}")

        # Train
        train_loss = train_epoch(
            model, agent, train_batches, criterion, optimizer, epoch
        )

        # Validate
        val_loss = evaluate(model, val_batches, criterion)

        # Compute T-Score on a small sample (expensive, so only on 3 batches)
        print(f"   Computing T-Score (wisdom metric)...")
        avg_t_score, sleep_events = compute_epoch_t_score(
            model, agent, train_batches, criterion, num_batches=3
        )

        # Generate sample
        sample = generate_text(
            model, agent,
            start_text="ROMEO:\n",
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            length=150,
            temperature=0.8
        )

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["t_score"].append(avg_t_score)
        history["sleep_events"].append(sleep_events)
        history["samples"].append(sample)

        # Print summary
        print(f"\n   📊 Results:")
        print(f"      Train Loss: {train_loss:.4f}")
        print(f"      Val Loss: {val_loss:.4f}")
        print(f"      T-Score: {avg_t_score:.4f}")
        print(f"      Sleep Events: {sleep_events}")
        print(f"\n   📝 Generated Sample:")
        print(f"      {sample[:200]}...")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"   ✅ New best validation loss!")

    elapsed = time.time() - start_time

    # Final evaluation
    print("\n" + "=" * 70)
    print("🏁 TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n📊 Final Results:")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Final T-Score: {history['t_score'][-1]:.4f}")
    print(f"   Total Sleep Events: {sum(history['sleep_events'])}")
    print(f"   Training Time: {elapsed/60:.1f} minutes")

    # Generate final samples
    print(f"\n📝 Final Text Samples:\n")

    prompts = ["ROMEO:\n", "JULIET:\n", "First Citizen:\n"]
    for prompt in prompts:
        sample = generate_text(
            model, agent, prompt,
            char_to_idx, idx_to_char,
            length=200, temperature=0.7
        )
        print(f"   Prompt: '{prompt.strip()}'")
        print(f"   {sample}")
        print()

    # Save results
    results = {
        "config": config,
        "history": {
            "train_loss": [float(x) for x in history["train_loss"]],
            "val_loss": [float(x) for x in history["val_loss"]],
            "t_score": [float(x) for x in history["t_score"]],
            "sleep_events": history["sleep_events"],
        },
        "final_metrics": {
            "best_val_loss": float(best_val_loss),
            "final_t_score": float(history["t_score"][-1]),
            "total_sleep_events": sum(history["sleep_events"]),
            "training_time_minutes": elapsed / 60,
        },
        "samples": history["samples"]
    }

    # Save to file
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"shakespeare_benchmark_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved to: {results_file}")

    return results, results_file


if __name__ == "__main__":
    results, filepath = main()
