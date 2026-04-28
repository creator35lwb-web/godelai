"""
PermutedMNIST GodelReplay Experiment
=====================================
4-strategy comparison on PermutedMNIST (10 tasks, 5 epochs each):

  1. Naive          — no continual learning strategy (~90% forgetting, sanity baseline)
  2. Replay-only    — Avalanche Replay, mem_size=500 (~8-12% forgetting, literature baseline)
  3. EWC-only       — GodelPlugin without Replay (~31.5% forgetting, reproduces prior result)
  4. GodelReplay    — Replay + GodelPlugin (PRIMARY: target < Replay-only forgetting)

Results are logged to log_{strategy}.txt and printed as a summary table.

Author: Rk (Claude Code) + RNA — FLYWHEEL TEAM
Orchestrator: T (CTO, Manus AI)
Kaggle Notebook: godelai-replay-permutedmnist-v1
Date: April 2026
MACP: v2.3.1
"""

import sys
import os
import torch
import torch.nn as nn

# Kaggle path injection (no-op locally if already installed)
_kaggle_path = "/kaggle/working/godelai-repo"
if os.path.exists(_kaggle_path) and _kaggle_path not in sys.path:
    sys.path.insert(0, _kaggle_path)

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.training.supervised import Naive, Replay
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger, TextLogger

from godelai.avalanche_plugin import GodelPlugin
from godelai.strategies.godel_replay import create_godel_replay_strategy


# ── Model ────────────────────────────────────────────────────────────────────

class GodelMLP(nn.Module):
    """218K-param MLP matching prior GodelAI PermutedMNIST experiments."""
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# ── Config ───────────────────────────────────────────────────────────────────

def _resolve_device() -> str:
    """P100 is sm_60 — PyTorch ≥2.0 requires sm_70+. Fall back to CPU."""
    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 7:
        return "cuda"
    print(f"[Warning] GPU sm_{major}{minor} < sm_70 ({torch.cuda.get_device_name(0)}) "
          f"— falling back to CPU (P100 known incompatibility with current PyTorch).")
    return "cpu"


CONFIG = {
    "n_experiences": 10,
    "seed": 42,
    "train_epochs": 5,
    "train_mb_size": 128,
    "eval_mb_size": 256,
    "lr": 0.001,
    "device": _resolve_device(),
    "ewc_lambda": 400.0,
    "fisher_scaling": "global_max",
    "propagation_gamma": 2.0,
    "t_score_window": 50,
    "mem_size": 500,
}


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_strategy(strategy_name: str, config: dict):
    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy_name.upper()}")
    print(f"{'='*60}")

    scenario = PermutedMNIST(
        n_experiences=config["n_experiences"],
        seed=config["seed"],
    )

    model = GodelMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    log_path = f"log_{strategy_name}.txt"
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger(), TextLogger(open(log_path, "w"))],
    )

    device = config["device"]

    if strategy_name == "naive":
        strategy = Naive(
            model=model, optimizer=optimizer, criterion=criterion,
            train_epochs=config["train_epochs"],
            train_mb_size=config["train_mb_size"],
            eval_mb_size=config["eval_mb_size"],
            device=device,
            plugins=[eval_plugin],
        )

    elif strategy_name == "replay_only":
        strategy = Replay(
            model=model, optimizer=optimizer, criterion=criterion,
            mem_size=config["mem_size"],
            train_epochs=config["train_epochs"],
            train_mb_size=config["train_mb_size"],
            eval_mb_size=config["eval_mb_size"],
            device=device,
            plugins=[eval_plugin],
        )

    elif strategy_name == "ewc_only":
        godel_plugin = GodelPlugin(
            ewc_lambda=config["ewc_lambda"],
            fisher_scaling_strategy=config["fisher_scaling"],
            propagation_gamma=config["propagation_gamma"],
            t_score_window=config["t_score_window"],
        )
        strategy = Naive(
            model=model, optimizer=optimizer, criterion=criterion,
            train_epochs=config["train_epochs"],
            train_mb_size=config["train_mb_size"],
            eval_mb_size=config["eval_mb_size"],
            device=device,
            plugins=[godel_plugin, eval_plugin],
        )

    elif strategy_name == "godel_replay":
        strategy, _ = create_godel_replay_strategy(
            model=model, optimizer=optimizer, criterion=criterion,
            mem_size=config["mem_size"],
            ewc_lambda=config["ewc_lambda"],
            fisher_scaling_strategy=config["fisher_scaling"],
            propagation_gamma=config["propagation_gamma"],
            t_score_window=config["t_score_window"],
            train_epochs=config["train_epochs"],
            train_mb_size=config["train_mb_size"],
            eval_mb_size=config["eval_mb_size"],
            device=device,
            eval_plugin=eval_plugin,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    for exp_id, experience in enumerate(scenario.train_stream):
        print(f"\n  Task {exp_id + 1}/{config['n_experiences']}")
        strategy.train(experience)
        strategy.eval(scenario.test_stream)

    metrics = eval_plugin.get_all_metrics()
    forgetting_key = [k for k in metrics if "StreamForgetting" in k]
    avg_forgetting = metrics[forgetting_key[0]][1][-1] if forgetting_key else None

    accuracy_key = [k for k in metrics if "Top1_Acc_Stream/eval_phase/test_stream" in k]
    final_accuracy = metrics[accuracy_key[0]][1][-1] if accuracy_key else None

    print(f"\n  [{strategy_name}] Final stream accuracy: {final_accuracy}")
    print(f"  [{strategy_name}] Avg stream forgetting: {avg_forgetting}")

    return {
        "strategy": strategy_name,
        "final_accuracy": final_accuracy,
        "avg_forgetting": avg_forgetting,
        "log": log_path,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nGodelReplay PermutedMNIST Experiment")
    print(f"Device: {CONFIG['device']} | Tasks: {CONFIG['n_experiences']} | Epochs/task: {CONFIG['train_epochs']}")
    print(f"Replay mem_size: {CONFIG['mem_size']} | EWC lambda: {CONFIG['ewc_lambda']} | Fisher: {CONFIG['fisher_scaling']}")

    strategies = ["naive", "replay_only", "ewc_only", "godel_replay"]
    results = []

    for name in strategies:
        r = run_strategy(name, CONFIG)
        results.append(r)

    # Summary table
    print("\n" + "="*70)
    print("  GODELREPLAY RESULTS — PermutedMNIST (10 tasks, seed=42)")
    print("="*70)
    print(f"  {'Strategy':<20} {'Final Acc':>12} {'Avg Forgetting':>16}")
    print(f"  {'-'*20} {'-'*12} {'-'*16}")
    for r in results:
        acc = f"{r['final_accuracy']:.4f}" if r["final_accuracy"] is not None else "N/A"
        forg = f"{r['avg_forgetting']:.4f}" if r["avg_forgetting"] is not None else "N/A"
        print(f"  {r['strategy']:<20} {acc:>12} {forg:>16}")
    print("="*70)

    replay_forg = next((r["avg_forgetting"] for r in results if r["strategy"] == "replay_only"), None)
    godelreplay_forg = next((r["avg_forgetting"] for r in results if r["strategy"] == "godel_replay"), None)

    if replay_forg is not None and godelreplay_forg is not None and replay_forg > 0:
        delta_pct = (replay_forg - godelreplay_forg) / replay_forg * 100
        verdict = "HYPOTHESIS CONFIRMED" if godelreplay_forg < replay_forg else "HYPOTHESIS REJECTED"
        print(f"\n  GodelReplay vs Replay-only: {delta_pct:+.1f}% forgetting reduction")
        print(f"  Verdict: {verdict}")

    print("\n  Detailed logs: log_naive.txt, log_replay_only.txt, log_ewc_only.txt, log_godel_replay.txt")


if __name__ == "__main__":
    main()
