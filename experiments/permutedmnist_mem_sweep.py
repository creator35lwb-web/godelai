"""
PermutedMNIST Memory Buffer Sweep
===================================
Tests whether GodelPlugin's contribution to forgetting reduction grows
as the replay buffer shrinks.

Hypothesis: The delta (Replay-only - GodelReplay forgetting) increases as
mem_size decreases, proving GodelPlugin provides complementary protection
when replay alone is insufficient.

Runs Replay-only vs GodelReplay at three buffer sizes:
  mem_size = 50   (small — replay insufficient alone)
  mem_size = 200  (medium — partial coverage)
  mem_size = 500  (large — replay near-saturated, already benchmarked)

Results feed the Two-Layer Architecture paper:
"GodelPlugin's value as safety net is greatest when replay is constrained."

Author: Rk (Claude Code) — FLYWHEEL TEAM
Date: April 2026
MACP: v2.3.1
"""

import sys
import os
import torch
import torch.nn as nn

_kaggle_path = "/kaggle/working/godelai-repo"
if os.path.exists(_kaggle_path) and _kaggle_path not in sys.path:
    sys.path.insert(0, _kaggle_path)

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.training.supervised import Replay
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
    """218K-param MLP — same as prior experiments for fair comparison."""
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


# ── Device ───────────────────────────────────────────────────────────────────

def _resolve_device():
    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 7:
        return "cuda"
    print("[Warning] GPU sm_" + str(major) + str(minor) + " < sm_70 — using CPU.")
    return "cpu"


# ── Base config ──────────────────────────────────────────────────────────────

BASE_CONFIG = {
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
}

MEM_SIZES = [50, 200, 500]


# ── Single run ────────────────────────────────────────────────────────────────

def run_one(strategy_name, mem_size, config):
    label = strategy_name + "_mem" + str(mem_size)
    print("\n" + "="*60)
    print("  " + strategy_name.upper() + " | mem_size=" + str(mem_size))
    print("="*60)

    scenario = PermutedMNIST(
        n_experiences=config["n_experiences"],
        seed=config["seed"],
    )

    model = GodelMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    log_path = "log_sweep_" + label + ".txt"
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger(), TextLogger(open(log_path, "w"))],
    )

    device = config["device"]

    if strategy_name == "replay_only":
        strategy = Replay(
            model=model, optimizer=optimizer, criterion=criterion,
            mem_size=mem_size,
            train_epochs=config["train_epochs"],
            train_mb_size=config["train_mb_size"],
            eval_mb_size=config["eval_mb_size"],
            device=device,
            plugins=[eval_plugin],
        )

    elif strategy_name == "godel_replay":
        strategy, _ = create_godel_replay_strategy(
            model=model, optimizer=optimizer, criterion=criterion,
            mem_size=mem_size,
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
        raise ValueError("Unknown strategy: " + strategy_name)

    for exp_id, experience in enumerate(scenario.train_stream):
        print("  Task " + str(exp_id + 1) + "/" + str(config["n_experiences"]))
        strategy.train(experience)
        strategy.eval(scenario.test_stream)

    metrics = eval_plugin.get_all_metrics()

    forgetting_key = [k for k in metrics if "StreamForgetting" in k]
    avg_forgetting = metrics[forgetting_key[0]][1][-1] if forgetting_key else None

    accuracy_key = [k for k in metrics if "Top1_Acc_Stream/eval_phase/test_stream" in k]
    final_accuracy = metrics[accuracy_key[0]][1][-1] if accuracy_key else None

    print("  [" + label + "] forgetting=" + str(avg_forgetting) + " | acc=" + str(final_accuracy))

    return {
        "strategy": strategy_name,
        "mem_size": mem_size,
        "avg_forgetting": avg_forgetting,
        "final_accuracy": final_accuracy,
        "log": log_path,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("GodelReplay Memory Buffer Sweep")
    print("PermutedMNIST | 10 tasks | 5 epochs | seed=42")
    print("Device: " + BASE_CONFIG["device"])
    print("Buffer sizes: " + str(MEM_SIZES))
    print("Strategies: replay_only, godel_replay")

    results = []
    for mem_size in MEM_SIZES:
        for strategy in ["replay_only", "godel_replay"]:
            r = run_one(strategy, mem_size, BASE_CONFIG)
            results.append(r)

    # Summary table
    print("\n" + "="*72)
    print("  MEMORY BUFFER SWEEP RESULTS")
    print("  PermutedMNIST (10 tasks) | Replay-only vs GodelReplay")
    print("="*72)
    print("  " + "mem_size".ljust(10) + "Strategy".ljust(16) +
          "Forgetting".rjust(12) + "Accuracy".rjust(10) + "Delta".rjust(10))
    print("  " + "-"*58)

    for mem_size in MEM_SIZES:
        r_replay = next((r for r in results if r["strategy"] == "replay_only" and r["mem_size"] == mem_size), None)
        r_godel  = next((r for r in results if r["strategy"] == "godel_replay" and r["mem_size"] == mem_size), None)

        if r_replay:
            print("  " + str(mem_size).ljust(10) + "replay_only".ljust(16) +
                  "{:.4f}".format(r_replay["avg_forgetting"]).rjust(12) +
                  "{:.4f}".format(r_replay["final_accuracy"]).rjust(10) + "  --")
        if r_godel and r_replay:
            delta = r_replay["avg_forgetting"] - r_godel["avg_forgetting"]
            pct   = delta / r_replay["avg_forgetting"] * 100 if r_replay["avg_forgetting"] else 0
            print("  " + str(mem_size).ljust(10) + "godel_replay".ljust(16) +
                  "{:.4f}".format(r_godel["avg_forgetting"]).rjust(12) +
                  "{:.4f}".format(r_godel["final_accuracy"]).rjust(10) +
                  ("  +{:.1f}%".format(pct)).rjust(10))
        print()

    print("="*72)
    print("Delta = Replay-only forgetting - GodelReplay forgetting")
    print("Positive delta = GodelPlugin helps. Larger delta at smaller mem_size = complementarity confirmed.")


if __name__ == "__main__":
    main()
