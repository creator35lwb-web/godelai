#!/usr/bin/env python3
"""
GodelAI Expanded T-Score Experiment v3 — Semantic + Gradient Analysis
=====================================================================
Purpose: Compare gradient-level T-Score with semantic-level T-Score
         to test whether meaning-level conflicts produce different
         gradient patterns than syntactic-only differences.

Builds on: run_conflict_tscore_v2.py (Godel, Jan 20, 2026)
New: SemanticTScore class using sentence-transformers

Author: Claude Code (Opus 4.6) — Lead Engineer
Date: February 7, 2026
Protocol: MACP v2.0
"""

import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import random
import numpy as np

# Add godelai to path
sys.path.insert(0, str(Path(__file__).parent))

from godelai.agent import GodelAgent

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).parent / "results"
DATASETS_DIR = Path(__file__).parent / "datasets" / "conflict"

print("=" * 70)
print("GodelAI Expanded T-Score Experiment v3")
print("Semantic + Gradient Diversity Analysis")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Timestamp: {datetime.now().isoformat()}")
print()


class CharLevelModel(nn.Module):
    """Character-level language model for T-Score experiments."""

    def __init__(self, vocab_size=256, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)


def load_conflict_datasets():
    """Load all conflict datasets from JSON files (supports old and new directory names)."""
    datasets = {}

    for category_dir in DATASETS_DIR.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            datasets[category_name] = []

            for json_file in category_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'data' in data:
                            datasets[category_name].extend(data['data'])
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"  Warning: Could not load {json_file}: {e}")

    return datasets


def extract_texts_from_sample(item):
    """Extract all text fragments from a conflict sample (old + new schema)."""
    texts = []

    # New schema: position_a / position_b with training_text
    if 'position_a' in item and 'position_b' in item:
        pa = item['position_a']
        pb = item['position_b']
        if pa.get('claim'):
            texts.append(pa['claim'])
        if pb.get('claim'):
            texts.append(pb['claim'])
        if pa.get('reasoning'):
            texts.append(pa['reasoning'])
        if pb.get('reasoning'):
            texts.append(pb['reasoning'])
        if item.get('training_text'):
            texts.append(item['training_text'])

    # Old schema: perspectives array
    elif 'perspectives' in item:
        for perspective in item['perspectives']:
            parts = []
            if 'position' in perspective:
                parts.append(perspective['position'])
            if 'reasoning' in perspective:
                parts.append(perspective['reasoning'])
            if parts:
                texts.append(" ".join(parts))

    # Old schema: fact_a / fact_b
    elif 'fact_a' in item and 'fact_b' in item:
        if item['fact_a'].get('statement'):
            texts.append(item['fact_a']['statement'])
        if item['fact_b'].get('statement'):
            texts.append(item['fact_b']['statement'])

    # Old schema: timeline
    elif 'timeline' in item:
        for entry in item['timeline']:
            if entry.get('belief'):
                texts.append(entry['belief'])

    # Fallback: training_prompt
    if not texts and item.get('training_prompt'):
        texts.append(item['training_prompt'])

    return texts


def create_batch(texts, batch_size=8, seq_length=64):
    """Create a batch of character-level training data from texts."""
    batch_inputs = []
    batch_targets = []

    for _ in range(batch_size):
        text = random.choice(texts)
        encoded = [ord(c) % 256 for c in text]

        while len(encoded) < seq_length + 1:
            encoded.extend([ord(c) % 256 for c in random.choice(texts)])

        start = random.randint(0, max(0, len(encoded) - seq_length - 1))
        input_seq = encoded[start:start + seq_length]
        target_seq = encoded[start + 1:start + seq_length + 1]

        batch_inputs.append(input_seq)
        batch_targets.append(target_seq)

    return (
        torch.tensor(batch_inputs, dtype=torch.long).to(DEVICE),
        torch.tensor(batch_targets, dtype=torch.long).to(DEVICE)
    )


def run_gradient_experiment(name, texts, num_batches=50, batch_size=8):
    """Run training experiment measuring gradient T-Score."""
    print(f"\n  [Gradient] {name}: {len(texts)} texts, {num_batches} batches")

    model = CharLevelModel().to(DEVICE)
    agent = GodelAgent(model, propagation_gamma=2.0, min_surplus_energy=0.1)
    agent.optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    t_scores = []
    losses = []
    variances = []

    for batch_idx in range(num_batches):
        inputs, targets = create_batch(texts, batch_size=batch_size)

        # Reshape targets for CrossEntropyLoss
        outputs = model(inputs)
        outputs_flat = outputs.view(-1, outputs.size(-1))
        targets_flat = targets.view(-1)

        loss, t_score, status, metrics = agent.learning_step(
            inputs, targets_flat.view(inputs.shape), criterion
        )

        t_scores.append(t_score)
        losses.append(loss)
        variances.append(metrics.get('t_score_variance', 0.0))

        if (batch_idx + 1) % 25 == 0:
            avg_t = sum(t_scores[-10:]) / min(10, len(t_scores))
            print(f"    Batch {batch_idx + 1:3d}: T={avg_t:.4f}, Loss={loss:.4f}, Status={status}")

    return {
        'avg_t_score': float(np.mean(t_scores)),
        'std_t_score': float(np.std(t_scores)),
        'min_t_score': float(np.min(t_scores)),
        'max_t_score': float(np.max(t_scores)),
        'avg_variance': float(np.mean(variances)),
        'final_loss': float(losses[-1]),
        'avg_loss': float(np.mean(losses)),
        't_scores': [float(t) for t in t_scores],
        'sleep_count': agent.history['sleep_count']
    }


def run_semantic_experiment(name, samples, semantic_scorer):
    """Run semantic T-Score analysis on conflict samples."""
    print(f"\n  [Semantic] {name}: {len(samples)} samples")

    results = semantic_scorer.analyze_dataset(samples)

    print(f"    Avg Pairwise Tension: {results.get('avg_pairwise_tension', 0):.4f}")
    print(f"    Avg Semantic T-Score: {results.get('avg_semantic_tscore', 0):.4f}")

    return results


def main():
    """Main experiment runner."""

    # --- Load Data ---
    print("\n" + "=" * 70)
    print("Phase 1: Loading Datasets")
    print("=" * 70)

    conflict_datasets = load_conflict_datasets()

    # Shakespeare baseline
    shakespeare_text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die-to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream-ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause-there's the respect
    That makes calamity of so long life.
    """
    shakespeare_texts = [shakespeare_text[i:i+200] for i in range(0, len(shakespeare_text), 100)]
    shakespeare_texts = [t for t in shakespeare_texts if len(t) > 100]

    # Flatten conflict data
    all_conflict_samples = []
    all_conflict_texts = []
    for category, items in conflict_datasets.items():
        for item in items:
            all_conflict_samples.append(item)
            extracted = extract_texts_from_sample(item)
            all_conflict_texts.extend(extracted)

    print(f"  Shakespeare: {len(shakespeare_texts)} text chunks")
    print(f"  Conflict samples: {len(all_conflict_samples)}")
    print(f"  Conflict text fragments: {len(all_conflict_texts)}")

    if not all_conflict_texts:
        print("\n  ERROR: No conflict texts found. Check datasets/conflict/ directory.")
        return

    # --- Initialize Semantic Scorer ---
    print("\n" + "=" * 70)
    print("Phase 2: Initializing Semantic T-Score")
    print("=" * 70)

    semantic_scorer = None
    try:
        from godelai.semantic_tscore import SemanticTScore
        semantic_scorer = SemanticTScore()
        print(f"  Model loaded: {semantic_scorer.model_name}")
    except ImportError as e:
        print(f"  WARNING: sentence-transformers not available: {e}")
        print("  Semantic analysis will be skipped.")
        print("  Install with: pip install sentence-transformers")

    # --- Gradient T-Score Experiments ---
    print("\n" + "=" * 70)
    print("Phase 3: Gradient T-Score Experiments")
    print("=" * 70)

    gradient_results = {}

    # Baseline: Shakespeare
    gradient_results['shakespeare'] = run_gradient_experiment(
        "Shakespeare (Homogeneous)", shakespeare_texts
    )

    # Conflict data
    gradient_results['conflict'] = run_gradient_experiment(
        "Conflict Data (Heterogeneous)", all_conflict_texts
    )

    # Mixed
    mixed_texts = shakespeare_texts + all_conflict_texts
    gradient_results['mixed'] = run_gradient_experiment(
        "Mixed (Maximum Diversity)", mixed_texts
    )

    # --- Semantic T-Score Experiments ---
    semantic_results = {}

    if semantic_scorer:
        print("\n" + "=" * 70)
        print("Phase 4: Semantic T-Score Analysis")
        print("=" * 70)

        # Per-category analysis
        for category, items in conflict_datasets.items():
            if items:
                semantic_results[category] = run_semantic_experiment(
                    f"Category: {category}", items, semantic_scorer
                )

        # Overall analysis
        semantic_results['all_conflict'] = run_semantic_experiment(
            "All Conflict Data", all_conflict_samples, semantic_scorer
        )

        # Shakespeare semantic baseline (create fake "samples" for comparison)
        shakespeare_samples = [
            {
                'position_a': {'claim': shakespeare_texts[i]},
                'position_b': {'claim': shakespeare_texts[(i + 1) % len(shakespeare_texts)]}
            }
            for i in range(len(shakespeare_texts) - 1)
        ]
        if shakespeare_samples:
            semantic_results['shakespeare'] = run_semantic_experiment(
                "Shakespeare (Baseline)", shakespeare_samples, semantic_scorer
            )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Dataset':<30} {'Grad T-Score':>12} {'Grad Std':>10} {'Sleep':>6}")
    print("-" * 62)
    for name, r in gradient_results.items():
        print(f"  {name:<28} {r['avg_t_score']:>12.4f} {r['std_t_score']:>10.4f} {r['sleep_count']:>6}")

    if semantic_results:
        print(f"\n{'Dataset':<30} {'Sem T-Score':>12} {'Pairwise':>10} {'Samples':>8}")
        print("-" * 62)
        for name, r in semantic_results.items():
            print(f"  {name:<28} {r.get('avg_semantic_tscore', 0):>12.4f} "
                  f"{r.get('avg_pairwise_tension', 0):>10.4f} "
                  f"{r.get('count', 0):>8}")

    # Key comparisons
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    shk = gradient_results['shakespeare']
    con = gradient_results['conflict']

    t_diff = con['avg_t_score'] - shk['avg_t_score']
    std_diff = con['std_t_score'] - shk['std_t_score']
    var_diff = con['avg_variance'] - shk['avg_variance']

    print(f"\n  Gradient T-Score Diff (Conflict - Shakespeare): {t_diff:+.4f}")
    print(f"  Gradient Std Diff: {std_diff:+.4f}")
    print(f"  Variance Diff: {var_diff:+.6f}")

    if semantic_results and 'all_conflict' in semantic_results and 'shakespeare' in semantic_results:
        sem_con = semantic_results['all_conflict']
        sem_shk = semantic_results['shakespeare']
        sem_t_diff = sem_con.get('avg_semantic_tscore', 0) - sem_shk.get('avg_semantic_tscore', 0)
        sem_p_diff = sem_con.get('avg_pairwise_tension', 0) - sem_shk.get('avg_pairwise_tension', 0)
        print(f"  Semantic T-Score Diff: {sem_t_diff:+.4f}")
        print(f"  Pairwise Tension Diff: {sem_p_diff:+.4f}")

    # --- Save Results ---
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = RESULTS_DIR / f"expanded_tscore_v3_{timestamp}.json"

    output = {
        'experiment': 'Expanded T-Score v3 — Semantic + Gradient',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'hypothesis': 'Semantic conflict produces different gradient diversity patterns',
        'data_summary': {
            'shakespeare_chunks': len(shakespeare_texts),
            'conflict_samples': len(all_conflict_samples),
            'conflict_text_fragments': len(all_conflict_texts),
            'categories': list(conflict_datasets.keys())
        },
        'gradient_results': gradient_results,
        'semantic_results': semantic_results,
        'key_findings': {
            'gradient_tscore_diff': t_diff,
            'gradient_std_diff': std_diff,
            'gradient_variance_diff': var_diff,
        }
    }

    # Remove raw t_scores list from saved output (too large)
    for key in output['gradient_results']:
        output['gradient_results'][key].pop('t_scores', None)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    return output


if __name__ == "__main__":
    results = main()
