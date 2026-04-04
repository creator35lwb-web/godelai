"""
GodelAI Make-or-Break Benchmark: PermutedMNIST (Domain-Incremental)

This is the test that determines whether GodelAI's value generalizes
beyond our own conflict data. PermutedMNIST is:
- Domain-incremental (same output head, different input distributions)
- Community-standard (used in EWC, SI, PackNet, GEM papers)
- The EXACT setting GodelAI claims to excel at

If GodelPlugin shows >20% forgetting reduction here, we write the paper.
If <5%, we pivot to replay combination.

Author: L (GodelAI CEO)
Date: April 4, 2026
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import random
import copy

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- Avalanche imports ---
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger

# --- GodelAI imports ---
import sys
sys.path.insert(0, '/home/ubuntu/godelai')
from godelai.avalanche_plugin import GodelPlugin


def build_model():
    """Simple MLP for PermutedMNIST — standard architecture from CL literature."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


def compute_forgetting(acc_matrix):
    """
    Compute average forgetting from accuracy matrix.
    
    acc_matrix[i][j] = accuracy on task j after training on task i.
    Forgetting for task j = max accuracy on j before final - final accuracy on j.
    
    Returns average forgetting across all tasks except the last.
    """
    n_tasks = len(acc_matrix)
    if n_tasks < 2:
        return 0.0
    
    forgetting_values = []
    for j in range(n_tasks - 1):  # For each task except the last
        # Best accuracy on task j across all evaluations up to (but not including) final
        best_acc = max(acc_matrix[i][j] for i in range(j, n_tasks - 1))
        # Final accuracy on task j
        final_acc = acc_matrix[n_tasks - 1][j]
        forgetting_values.append(max(0, best_acc - final_acc))
    
    return np.mean(forgetting_values)


def run_condition(name, model, benchmark, plugins, n_experiences, lr=0.001, epochs=2):
    """Run a single experimental condition and return metrics."""
    print(f"\n{'='*60}")
    print(f"CONDITION: {name}")
    print(f"{'='*60}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loss_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )
    
    all_plugins = [eval_plugin] + plugins
    
    strategy = Naive(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=64,
        train_epochs=epochs,
        eval_mb_size=128,
        plugins=all_plugins,
        device=torch.device('cpu'),
    )
    
    # Accuracy matrix: acc_matrix[i][j] = acc on task j after training task i
    acc_matrix = []
    
    start_time = time.time()
    
    for exp_idx, experience in enumerate(benchmark.train_stream[:n_experiences]):
        print(f"\n--- Training Experience {exp_idx} ---")
        strategy.train(experience)
        
        print(f"\n--- Evaluating after Experience {exp_idx} ---")
        results = strategy.eval(benchmark.test_stream[:n_experiences])
        
        # Extract per-task accuracies from this evaluation
        task_accs = []
        for j in range(n_experiences):
            key = f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{j:03d}'
            acc = results.get(key, 0.0)
            task_accs.append(acc)
        acc_matrix.append(task_accs)
    
    elapsed = time.time() - start_time
    
    # Compute metrics
    final_accs = acc_matrix[-1] if acc_matrix else []
    avg_final_acc = np.mean(final_accs) if final_accs else 0.0
    forgetting = compute_forgetting(acc_matrix)
    
    # Get stream-level metrics from last eval
    stream_acc = results.get('Top1_Acc_Stream/eval_phase/test_stream', avg_final_acc)
    stream_forgetting = results.get('StreamForgetting/eval_phase/test_stream', forgetting)
    
    print(f"\n--- {name} Summary ---")
    print(f"  Avg Final Accuracy: {avg_final_acc:.4f}")
    print(f"  Computed Forgetting: {forgetting:.4f}")
    print(f"  Stream Accuracy: {stream_acc:.4f}")
    print(f"  Stream Forgetting: {stream_forgetting:.4f}")
    print(f"  Per-task final accs: {[f'{a:.4f}' for a in final_accs]}")
    print(f"  Time: {elapsed:.1f}s")
    
    return {
        "name": name,
        "avg_final_accuracy": float(avg_final_acc),
        "computed_forgetting": float(forgetting),
        "stream_accuracy": float(stream_acc),
        "stream_forgetting": float(stream_forgetting),
        "per_task_final_accs": [float(a) for a in final_accs],
        "acc_matrix": [[float(a) for a in row] for row in acc_matrix],
        "time_seconds": float(elapsed),
    }


def main():
    N_EXPERIENCES = 5  # 5 permutations — standard in CL literature
    EPOCHS = 2         # 2 epochs per experience — enough for MLP on MNIST
    LR = 0.001
    
    print("=" * 60)
    print("GodelAI Make-or-Break: PermutedMNIST (Domain-Incremental)")
    print("=" * 60)
    print(f"Experiences: {N_EXPERIENCES}")
    print(f"Epochs per experience: {EPOCHS}")
    print(f"Model: MLP (784→256→256→10)")
    print(f"This is a DOMAIN-INCREMENTAL benchmark — GodelAI's claimed territory.")
    print()
    
    # Create benchmark
    benchmark = PermutedMNIST(n_experiences=N_EXPERIENCES, seed=42)
    
    results = {}
    
    # --- Condition 1: Naive (No Protection) ---
    model_naive = build_model()
    results["naive"] = run_condition(
        "Naive (No Protection)", model_naive, benchmark,
        plugins=[], n_experiences=N_EXPERIENCES, lr=LR, epochs=EPOCHS
    )
    
    # --- Condition 2: Naive + GodelPlugin ---
    # Reset seeds for fair comparison
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    model_godel = build_model()
    godel_plugin = GodelPlugin(
        ewc_lambda=400.0,
        fisher_scaling_strategy="global_max",
        dead_threshold=0.01,
        fisher_samples=200,
        propagation_gamma=2.0,
        min_surplus_energy=0.1,
        t_score_window=50,
        verbose=True,
    )
    results["godelai"] = run_condition(
        "Naive + GodelPlugin (Full C-S-P)", model_godel, benchmark,
        plugins=[godel_plugin], n_experiences=N_EXPERIENCES, lr=LR, epochs=EPOCHS
    )
    
    # --- Condition 3: Naive + GodelPlugin (T-Score Only, No EWC) ---
    # This isolates the monitoring value from the regularization value
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    model_tscore = build_model()
    tscore_plugin = GodelPlugin(
        ewc_lambda=0.0,  # Disable EWC — monitoring only
        fisher_scaling_strategy="global_max",
        dead_threshold=0.01,
        fisher_samples=200,
        propagation_gamma=2.0,
        min_surplus_energy=0.1,
        t_score_window=50,
        verbose=True,
    )
    results["tscore_only"] = run_condition(
        "Naive + GodelPlugin (T-Score Monitor Only)", model_tscore, benchmark,
        plugins=[tscore_plugin], n_experiences=N_EXPERIENCES, lr=LR, epochs=EPOCHS
    )
    
    # --- Final Comparison ---
    print("\n" + "=" * 60)
    print("FINAL COMPARISON — PermutedMNIST (Domain-Incremental)")
    print("=" * 60)
    print(f"{'Condition':<45} {'Avg Acc':>8} {'Forgetting':>11} {'Time':>8}")
    print("-" * 72)
    for key in ["naive", "godelai", "tscore_only"]:
        r = results[key]
        print(f"{r['name']:<45} {r['avg_final_accuracy']:>8.4f} "
              f"{r['computed_forgetting']:>11.4f} {r['time_seconds']:>7.1f}s")
    
    # --- Compute improvement ---
    naive_forg = results["naive"]["computed_forgetting"]
    godel_forg = results["godelai"]["computed_forgetting"]
    
    if naive_forg > 0:
        improvement = (naive_forg - godel_forg) / naive_forg * 100
    else:
        improvement = 0.0
    
    print(f"\n{'='*60}")
    print(f"FORGETTING REDUCTION: {improvement:.1f}%")
    print(f"  Naive forgetting:   {naive_forg:.4f}")
    print(f"  GodelAI forgetting: {godel_forg:.4f}")
    
    if improvement > 20:
        print(f"\n  VERDICT: GO — Write the paper.")
    elif improvement > 5:
        print(f"\n  VERDICT: MARGINAL — Needs replay combination.")
    else:
        print(f"\n  VERDICT: NO-GO on PermutedMNIST — Pivot to replay.")
    print(f"{'='*60}")
    
    # --- GodelPlugin diagnostics ---
    if hasattr(godel_plugin, 'history'):
        print(f"\nGodelPlugin Diagnostics:")
        print(f"  T-Scores: {godel_plugin.history['experience_t_scores']}")
        print(f"  Sleep counts: {godel_plugin.history['experience_sleep_counts']}")
        print(f"  EWC-DR active: {godel_plugin.ewc_dr is not None}")
    
    # Save results
    output_path = "/home/ubuntu/godelai/results/permuted_mnist_benchmark.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
