"""
EWC-DR vs Vanilla EWC Benchmark
================================
Comprehensive comparison of:
1. Standard training (no EWC) — baseline
2. Vanilla EWC (Kirkpatrick et al., 2017)
3. EWC-DR (Dead Rectification / Logits Reversal, March 2026)

Tests catastrophic forgetting on sequential Task A → Task B learning
using the GodelAI GRU architecture.

Author: L (GodelAI CEO) — MACP v2.2 "Identity"
Date: April 3, 2026
"""

import sys
import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from godelai.reg.ewc_dr import EWCDR, VanillaEWC

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    "seed": 1337,
    "seq_length": 50,
    "batch_size": 32,
    "hidden_size": 128,
    "num_layers": 2,
    "task_a_epochs": 10,
    "task_b_epochs": 10,
    "learning_rate": 0.001,
    "ewc_lambda": 0.4,
    "ewc_dr_lambda": 0.4,
    "ewc_dr_dead_threshold": 1e-4,
    "ewc_dr_reversal_strength": 0.05,
    "fisher_samples": 200,
    "device": "cpu",
}

torch.manual_seed(CONFIG["seed"])


# ============================================================================
# Model Definition
# ============================================================================
class GRULanguageModel(nn.Module):
    """Simple GRU-based character-level language model."""

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        logits = self.fc(output)
        return logits


# ============================================================================
# Data Preparation
# ============================================================================
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def text_to_batches(text: str, vocab: dict, seq_len: int, batch_size: int):
    """Convert text to character-level batches."""
    encoded = [vocab.get(c, 0) for c in text]
    total = len(encoded)
    batches = []
    for i in range(0, total - seq_len - 1, seq_len):
        chunk = encoded[i:i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        x = torch.tensor(chunk[:seq_len], dtype=torch.long).unsqueeze(0)
        y = torch.tensor(chunk[1:seq_len + 1], dtype=torch.long).unsqueeze(0)
        batches.append((x, y))

    # Create mini-batches
    mini_batches = []
    for i in range(0, len(batches), batch_size):
        batch = batches[i:i + batch_size]
        xs = torch.cat([b[0] for b in batch], dim=0)
        ys = torch.cat([b[1] for b in batch], dim=0)
        mini_batches.append((xs, ys))

    return mini_batches


def evaluate(model, batches, device, criterion):
    """Evaluate model loss on given batches."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in batches:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
    return total_loss / max(len(batches), 1)


def train_epoch(model, batches, device, criterion, optimizer, ewc_module=None):
    """Train one epoch, optionally with EWC/EWC-DR penalty."""
    model.train()
    total_task_loss = 0.0
    total_penalty = 0.0

    for inputs, targets in batches:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
        task_loss = criterion(outputs_flat, targets_flat)

        if ewc_module is not None and ewc_module.is_consolidated:
            penalty = ewc_module(model)
            combined_loss = task_loss + penalty
            total_penalty += penalty.item()
        else:
            combined_loss = task_loss

        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_task_loss += task_loss.item()

    n = max(len(batches), 1)
    return total_task_loss / n, total_penalty / n


# ============================================================================
# Main Benchmark
# ============================================================================
def run_condition(name: str, model, task_a_batches, task_b_batches, device,
                  criterion, ewc_module=None):
    """Run a single experimental condition."""
    print(f"\n{'='*60}")
    print(f"  CONDITION: {name}")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # Phase 1: Train on Task A
    print(f"\n  Phase 1: Training on Task A ({CONFIG['task_a_epochs']} epochs)")
    for epoch in range(CONFIG["task_a_epochs"]):
        task_loss, penalty = train_epoch(model, task_a_batches, device, criterion, optimizer)
        if (epoch + 1) % 5 == 0:
            val_loss = evaluate(model, task_a_batches, device, criterion)
            print(f"    Epoch {epoch+1:2d}: Task Loss={task_loss:.4f}, Val={val_loss:.4f}")

    task_a_loss_after_phase1 = evaluate(model, task_a_batches, device, criterion)
    print(f"\n  ✅ Task A final loss: {task_a_loss_after_phase1:.4f}")

    # Consolidate EWC if applicable
    if ewc_module is not None:
        print(f"\n  🔒 Consolidating {name} importance weights...")
        if hasattr(ewc_module, 'consolidate'):
            stats = ewc_module.consolidate(model, task_a_batches, device, criterion,
                                           n_samples=CONFIG["fisher_samples"])
            if isinstance(stats, dict) and "dead_fraction" in stats:
                print(f"     Dead parameters: {stats['dead_fraction']*100:.1f}%")
                print(f"     Alive parameters: {stats['alive_fraction']*100:.1f}%")
        else:
            ewc_module.consolidate(model, task_a_batches, device, criterion,
                                   n_samples=CONFIG["fisher_samples"])

    # Phase 2: Train on Task B (with EWC penalty if applicable)
    print(f"\n  Phase 2: Training on Task B ({CONFIG['task_b_epochs']} epochs)")
    for epoch in range(CONFIG["task_b_epochs"]):
        task_loss, penalty = train_epoch(model, task_b_batches, device, criterion,
                                         optimizer, ewc_module)
        if (epoch + 1) % 5 == 0:
            val_loss = evaluate(model, task_b_batches, device, criterion)
            print(f"    Epoch {epoch+1:2d}: Task Loss={task_loss:.4f}, "
                  f"Penalty={penalty:.4f}, Val={val_loss:.4f}")

    task_a_loss_after_phase2 = evaluate(model, task_a_batches, device, criterion)
    task_b_loss_final = evaluate(model, task_b_batches, device, criterion)

    forgetting = task_a_loss_after_phase2 - task_a_loss_after_phase1

    print(f"\n  📊 Results:")
    print(f"     Task A loss (after Phase 1): {task_a_loss_after_phase1:.4f}")
    print(f"     Task A loss (after Phase 2): {task_a_loss_after_phase2:.4f}")
    print(f"     Task B final loss:           {task_b_loss_final:.4f}")
    print(f"     Forgetting (delta Task A):   {forgetting:+.4f}")

    return {
        "name": name,
        "task_a_loss_phase1": task_a_loss_after_phase1,
        "task_a_loss_phase2": task_a_loss_after_phase2,
        "task_b_loss_final": task_b_loss_final,
        "forgetting": forgetting,
    }


def main():
    print("=" * 70)
    print("GodelAI EWC-DR Benchmark v1.0")
    print("Comparing: Standard | Vanilla EWC | EWC-DR (Logits Reversal)")
    print(f"Author: L (GodelAI CEO) — MACP v2.2 'Identity'")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    device = torch.device(CONFIG["device"])

    # Load text data
    shakespeare_path = Path("datasets") / "shakespeare_full.txt"
    if not shakespeare_path.exists():
        print("\n⚠️  No Shakespeare text found. Using synthetic Task A/B data...")
        task_a_text = ("The quick brown fox jumps over the lazy dog. " * 200 +
                       "Pack my box with five dozen liquor jugs. " * 200 +
                       "How vexingly quick daft zebras jump. " * 200)
        task_b_text = ("To be or not to be that is the question whether tis nobler. " * 200 +
                       "All that glitters is not gold often have you heard that told. " * 200 +
                       "What light through yonder window breaks it is the east and Juliet. " * 200)
    else:
        full_text = load_text(str(shakespeare_path))
        # Use first 200KB for Task A, next 200KB for Task B (manageable on CPU)
        task_a_text = full_text[:200000]
        task_b_text = full_text[200000:400000]

    all_chars = sorted(set(task_a_text + task_b_text))
    vocab = {c: i for i, c in enumerate(all_chars)}
    vocab_size = len(vocab)

    print(f"\n  Vocabulary size: {vocab_size}")
    print(f"  Task A text length: {len(task_a_text):,} chars")
    print(f"  Task B text length: {len(task_b_text):,} chars")

    # Create batches
    task_a_batches = text_to_batches(task_a_text, vocab, CONFIG["seq_length"], CONFIG["batch_size"])
    task_b_batches = text_to_batches(task_b_text, vocab, CONFIG["seq_length"], CONFIG["batch_size"])

    print(f"  Task A batches: {len(task_a_batches)}")
    print(f"  Task B batches: {len(task_b_batches)}")

    criterion = nn.CrossEntropyLoss()
    results = []

    # ── Condition 1: Standard (No EWC) ──────────────────────────────────────
    torch.manual_seed(CONFIG["seed"])
    model_standard = GRULanguageModel(vocab_size, CONFIG["hidden_size"], CONFIG["num_layers"])
    result_standard = run_condition(
        "Standard (No EWC)", model_standard, task_a_batches, task_b_batches,
        device, criterion, ewc_module=None
    )
    results.append(result_standard)

    # ── Condition 2: Vanilla EWC ─────────────────────────────────────────────
    torch.manual_seed(CONFIG["seed"])
    model_ewc = GRULanguageModel(vocab_size, CONFIG["hidden_size"], CONFIG["num_layers"])
    vanilla_ewc = VanillaEWC(ewc_lambda=CONFIG["ewc_lambda"])
    result_ewc = run_condition(
        "Vanilla EWC", model_ewc, task_a_batches, task_b_batches,
        device, criterion, ewc_module=vanilla_ewc
    )
    results.append(result_ewc)

    # ── Condition 3: EWC-DR (Logits Reversal) ────────────────────────────────
    torch.manual_seed(CONFIG["seed"])
    model_ewcdr = GRULanguageModel(vocab_size, CONFIG["hidden_size"], CONFIG["num_layers"])
    ewc_dr = EWCDR(
        ewc_lambda=CONFIG["ewc_dr_lambda"],
        dead_threshold=CONFIG["ewc_dr_dead_threshold"],
        reversal_strength=CONFIG["ewc_dr_reversal_strength"],
    )
    result_ewcdr = run_condition(
        "EWC-DR (Logits Reversal)", model_ewcdr, task_a_batches, task_b_batches,
        device, criterion, ewc_module=ewc_dr
    )
    results.append(result_ewcdr)

    # ── Final Comparison ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    baseline_forgetting = results[0]["forgetting"]

    print(f"\n{'Condition':<30} {'Task A Forgetting':>18} {'Improvement':>12} {'Task B Loss':>12}")
    print("-" * 75)

    for r in results:
        if r["name"] == "Standard (No EWC)":
            improvement_str = "baseline"
        else:
            if baseline_forgetting > 0:
                improvement_pct = ((baseline_forgetting - r["forgetting"]) / baseline_forgetting) * 100
                improvement_str = f"{improvement_pct:+.1f}%"
            else:
                improvement_str = "N/A"

        print(f"{r['name']:<30} {r['forgetting']:>+18.4f} {improvement_str:>12} {r['task_b_loss_final']:>12.4f}")

    # Determine winner
    print("\n" + "=" * 70)
    forgettings = {r["name"]: r["forgetting"] for r in results}
    best = min(forgettings, key=forgettings.get)
    print(f"  🏆 Best forgetting reduction: {best}")

    ewc_improvement = ((baseline_forgetting - results[1]["forgetting"]) / max(abs(baseline_forgetting), 1e-8)) * 100
    ewcdr_improvement = ((baseline_forgetting - results[2]["forgetting"]) / max(abs(baseline_forgetting), 1e-8)) * 100

    print(f"\n  Vanilla EWC improvement over baseline: {ewc_improvement:+.1f}%")
    print(f"  EWC-DR improvement over baseline:      {ewcdr_improvement:+.1f}%")

    if results[2]["forgetting"] < results[1]["forgetting"]:
        delta = results[1]["forgetting"] - results[2]["forgetting"]
        print(f"\n  ✅ EWC-DR outperforms Vanilla EWC by: {delta:.4f} absolute forgetting")
        print(f"     EWC-DR is the recommended approach for GodelAI v3.x")
    else:
        delta = results[2]["forgetting"] - results[1]["forgetting"]
        print(f"\n  ⚠️  Vanilla EWC outperforms EWC-DR by: {delta:.4f} in this run")
        print(f"     Consider tuning EWC-DR hyperparameters (dead_threshold, reversal_strength)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "EWC-DR vs Vanilla EWC Benchmark",
        "timestamp": timestamp,
        "author": "L (GodelAI CEO) — MACP v2.2 'Identity'",
        "config": CONFIG,
        "results": results,
        "summary": {
            "baseline_forgetting": baseline_forgetting,
            "vanilla_ewc_forgetting": results[1]["forgetting"],
            "ewc_dr_forgetting": results[2]["forgetting"],
            "vanilla_ewc_improvement_pct": ewc_improvement,
            "ewc_dr_improvement_pct": ewcdr_improvement,
            "ewc_dr_beats_vanilla": results[2]["forgetting"] < results[1]["forgetting"],
        }
    }

    output_path = Path(f"results/ewc_dr_benchmark_{timestamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  💾 Results saved to: {output_path}")
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
