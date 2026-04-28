# GodelReplay: Claude Code Implementation Guide (Week 1)

**From:** T (CTO, Manus AI) — FLYWHEEL TEAM
**To:** RNA (Claude Code, Local) + Rk (Claude Code, Kaggle Pipeline)
**Date:** April 29, 2026
**MACP:** v2.3.1 "Market Position"
**GitHub Bridge:** `creator35lwb-web/godelai` (primary) + `creator35lwb-web/godelai-lite` (Kaggle)
**Sprint:** GodelReplay Week 1 — Implementation + PermutedMNIST Baseline

---

## 1. Context and Mission

T (CTO, Manus AI) here. Alton has approved the GodelReplay sprint with ethical clearance (see `docs/GODELREPLAY_SPRINT_ETHICS_AND_PLAN.md`). Your mission for Week 1 is to implement the GodelReplay module and run the first PermutedMNIST baseline on Kaggle compute.

GodelReplay is the convergence of two proven systems:

| System | Repo | What It Does | Key Result |
|--------|------|-------------|------------|
| GodelPlugin (Fisher Scaling) | `godelai` | Protects critical weights during training | 31.5% forgetting reduction |
| MemPalace (Experience Replay) | `godelai-lite` | Maintains exposure to past task distributions | +31.2% SAE improvement |

GodelReplay combines both: **replay buffer feeds past examples while GodelPlugin protects identity-defining weights**. The hypothesis is that this combination will outperform either approach alone.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    GodelReplay Strategy                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐     ┌──────────────────────────────┐  │
│  │ Replay Buffer│     │ GodelPlugin                   │  │
│  │ (Avalanche)  │     │ ├─ EWC-DR (Fisher-scaled)    │  │
│  │              │     │ ├─ T-Score monitoring         │  │
│  │ Stores past  │     │ └─ Sleep Protocol (optional)  │  │
│  │ task samples │     │                               │  │
│  └──────┬───────┘     └──────────────┬───────────────┘  │
│         │                             │                  │
│         ▼                             ▼                  │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Training Loop (per mini-batch)          │   │
│  │                                                    │   │
│  │  1. Sample current task batch                      │   │
│  │  2. Sample replay buffer batch (past tasks)        │   │
│  │  3. Forward pass on combined batch                 │   │
│  │  4. Compute task loss                              │   │
│  │  5. GodelPlugin adds EWC-DR penalty (Fisher-scaled)│   │
│  │  6. T-Score monitors gradient health               │   │
│  │  7. Backward + optimizer step                      │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Tasks (RNA)

### 3.1 Create `godelai/strategies/godel_replay.py`

This is the core new file. It wraps Avalanche's `Replay` strategy with `GodelPlugin`:

```python
"""
GodelReplay Strategy
====================
Combines Avalanche's Replay strategy with GodelPlugin for
dual-layer continual learning: experience replay (distribution preservation)
+ Fisher-scaled EWC-DR (weight preservation).

Hypothesis: GodelReplay achieves lower forgetting than either Replay-only
or GodelPlugin-only on PermutedMNIST.

Author: RNA (Claude Code) — FLYWHEEL TEAM
Orchestrator: T (CTO, Manus AI)
Date: April 2026
MACP: v2.3.1
"""

import torch
from avalanche.training.supervised import Replay
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from godelai.avalanche_plugin import GodelPlugin


def create_godel_replay_strategy(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    # Replay parameters
    mem_size: int = 500,
    # GodelPlugin parameters
    ewc_lambda: float = 400.0,
    fisher_scaling_strategy: str = "global_max",
    propagation_gamma: float = 2.0,
    t_score_window: int = 50,
    # Training parameters
    train_epochs: int = 5,
    train_mb_size: int = 128,
    eval_mb_size: int = 256,
    device: str = "cuda",
    # Evaluation
    eval_plugin: EvaluationPlugin = None,
):
    """
    Factory function to create a GodelReplay strategy.
    
    This combines:
    - Avalanche Replay (mem_size samples from past tasks)
    - GodelPlugin (Fisher-scaled EWC-DR + T-Score monitoring)
    
    Args:
        model: PyTorch model (e.g., GodelAI's MLP or Transformer)
        optimizer: Optimizer instance
        criterion: Loss function
        mem_size: Number of samples to store in replay buffer
        ewc_lambda: EWC regularization strength
        fisher_scaling_strategy: How to scale Fisher values ("global_max", "layer_wise", "target_penalty")
        propagation_gamma: C-S-P propagation parameter
        t_score_window: Window size for T-Score computation
        train_epochs: Epochs per experience
        train_mb_size: Training mini-batch size
        eval_mb_size: Evaluation mini-batch size
        device: Device to train on
        eval_plugin: Optional evaluation plugin
        
    Returns:
        Configured Replay strategy with GodelPlugin
    """
    
    # Initialize GodelPlugin
    godel_plugin = GodelPlugin(
        propagation_gamma=propagation_gamma,
        min_surplus_energy=0.1,
        t_score_window=t_score_window,
        ewc_lambda=ewc_lambda,
        fisher_scaling_strategy=fisher_scaling_strategy,
    )
    
    # Build plugin list
    plugins = [godel_plugin]
    if eval_plugin:
        plugins.append(eval_plugin)
    
    # Create Replay strategy with GodelPlugin attached
    strategy = Replay(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        mem_size=mem_size,
        train_mb_size=train_mb_size,
        train_epochs=train_epochs,
        eval_mb_size=eval_mb_size,
        device=device,
        plugins=plugins,
    )
    
    return strategy, godel_plugin
```

### 3.2 Create `experiments/permutedmnist_godelreplay.py`

This is the experiment script that runs on Kaggle:

```python
"""
PermutedMNIST GodelReplay Experiment
=====================================
Runs GodelReplay vs baselines on PermutedMNIST (10 tasks).

Baselines:
  1. Naive (no CL strategy) — expected ~90% forgetting
  2. Replay-only (mem_size=500) — expected ~8-12% forgetting
  3. EWC-only (lambda=400, Fisher-scaled) — expected ~31.5% forgetting
  4. GodelReplay (Replay + GodelPlugin) — target: <5% forgetting

Author: RNA (Claude Code) — FLYWHEEL TEAM
Kaggle Notebook: godelai-replay-permutedmnist-v1
Date: April 2026
"""

import torch
import torch.nn as nn
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.training.supervised import Naive, Replay
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger, TextLogger

import sys
sys.path.insert(0, '/kaggle/working/godelai')  # Adjust for Kaggle

from godelai.avalanche_plugin import GodelPlugin
from godelai.strategies.godel_replay import create_godel_replay_strategy


# ============================================================
# Model Definition (same 218K-param MLP used in prior experiments)
# ============================================================

class GodelMLP(nn.Module):
    """Simple MLP for PermutedMNIST — matches prior GodelAI experiments."""
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
        x = x.view(x.size(0), -1)
        return self.net(x)


# ============================================================
# Experiment Configuration
# ============================================================

CONFIG = {
    "n_experiences": 10,
    "seed": 42,
    "train_epochs": 5,
    "train_mb_size": 128,
    "eval_mb_size": 256,
    "lr": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # GodelPlugin params
    "ewc_lambda": 400.0,
    "fisher_scaling": "global_max",
    "propagation_gamma": 2.0,
    "t_score_window": 50,
    # Replay params
    "mem_size": 500,
}


def run_experiment(strategy_name: str, config: dict):
    """Run a single strategy on PermutedMNIST and return metrics."""
    
    print(f"\n{'='*60}")
    print(f"  Running: {strategy_name}")
    print(f"{'='*60}\n")
    
    # Benchmark
    scenario = PermutedMNIST(
        n_experiences=config["n_experiences"],
        seed=config["seed"],
    )
    
    # Model + Optimizer
    model = GodelMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    # Evaluation
    text_logger = TextLogger(open(f"log_{strategy_name}.txt", "w"))
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger(), text_logger],
    )
    
    # Strategy selection
    if strategy_name == "naive":
        strategy = Naive(
            model=model, optimizer=optimizer, criterion=criterion,
            train_epochs=config["train_epochs"],
            train_mb_size=config["train_mb_size"],
            eval_mb_size=config["eval_mb_size"],
            device=config["device"],
            plugins=[eval_plugin],
        )
    
    elif strategy_name == "replay_only":
        strategy = Replay(
            model=model, optimizer=optimizer, criterion=criterion,
            mem_size=config["mem_size"],
            train_epochs=config["train_epochs"],
            train_mb_size=config["train_mb_size"],
            eval_mb_size=config["eval_mb_size"],
            device=config["device"],
            plugins=[eval_plugin],
        )
    
    elif strategy_name == "ewc_only":
        # GodelPlugin without Replay = EWC-DR with Fisher Scaling
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
            device=config["device"],
            plugins=[godel_plugin, eval_plugin],
        )
    
    elif strategy_name == "godel_replay":
        strategy, godel_plugin = create_godel_replay_strategy(
            model=model, optimizer=optimizer, criterion=criterion,
            mem_size=config["mem_size"],
            ewc_lambda=config["ewc_lambda"],
            fisher_scaling_strategy=config["fisher_scaling"],
            propagation_gamma=config["propagation_gamma"],
            t_score_window=config["t_score_window"],
            train_epochs=config["train_epochs"],
            train_mb_size=config["train_mb_size"],
            eval_mb_size=config["eval_mb_size"],
            device=config["device"],
            eval_plugin=eval_plugin,
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Training loop
    results = []
    for exp_id, experience in enumerate(scenario.train_stream):
        print(f"\n--- Task {exp_id + 1}/{config['n_experiences']} ---")
        strategy.train(experience)
        eval_results = strategy.eval(scenario.test_stream)
        results.append(eval_results)
    
    return results


def main():
    """Run all strategies and compare."""
    
    strategies = ["naive", "replay_only", "ewc_only", "godel_replay"]
    all_results = {}
    
    for strat in strategies:
        all_results[strat] = run_experiment(strat, CONFIG)
    
    # Summary
    print("\n" + "="*60)
    print("  GODELREPLAY EXPERIMENT SUMMARY")
    print("="*60)
    print(f"\nBenchmark: PermutedMNIST ({CONFIG['n_experiences']} tasks)")
    print(f"Model: GodelMLP (218K params)")
    print(f"Device: {CONFIG['device']}")
    print(f"\nResults will be extracted from Avalanche metrics.")
    print("See log files for detailed per-task metrics.")
    

if __name__ == "__main__":
    main()
```

### 3.3 Create `godelai/strategies/__init__.py`

```python
"""GodelAI Strategies — Continual Learning strategy compositions."""

from godelai.strategies.godel_replay import create_godel_replay_strategy

__all__ = ["create_godel_replay_strategy"]
```

---

## 4. Implementation Tasks (Rk — Kaggle Pipeline)

### 4.1 Kaggle Notebook Setup

Create a new Kaggle notebook named `godelai-replay-permutedmnist-v1` with the following structure:

```python
# Cell 1: Setup
!pip install avalanche-lib torch torchvision --quiet
!git clone https://github.com/creator35lwb-web/godelai.git /kaggle/working/godelai-repo
import sys
sys.path.insert(0, '/kaggle/working/godelai-repo')

# Cell 2: Verify GodelPlugin loads
from godelai.avalanche_plugin import GodelPlugin
print("GodelPlugin loaded successfully")
print(f"Default EWC lambda: 400.0")
print(f"Fisher scaling: global_max")

# Cell 3: Run experiment
from experiments.permutedmnist_godelreplay import main, CONFIG
print(f"Config: {CONFIG}")
main()

# Cell 4: Visualize results
# (Plot forgetting curves, accuracy per task, comparison table)
```

### 4.2 Notebook Settings

| Setting | Value |
|---------|-------|
| Accelerator | GPU T4 x2 |
| Language | Python |
| Persistence | Files |
| Internet | On (for git clone) |
| Visibility | Public |

---

## 5. File Changes Summary

| File | Action | Owner |
|------|--------|-------|
| `godelai/strategies/__init__.py` | CREATE | RNA |
| `godelai/strategies/godel_replay.py` | CREATE | RNA |
| `experiments/permutedmnist_godelreplay.py` | CREATE | RNA |
| `experiments/__init__.py` | CREATE | RNA |
| Kaggle notebook `godelai-replay-permutedmnist-v1` | CREATE | Rk |

---

## 6. Success Criteria for Week 1

| Criterion | Target | How to Verify |
|-----------|--------|---------------|
| GodelReplay module imports without error | Pass | `from godelai.strategies import create_godel_replay_strategy` |
| PermutedMNIST runs to completion (10 tasks) | Pass | No crashes, all tasks trained |
| Naive baseline shows ~90% forgetting | Sanity check | Expected behavior |
| Replay-only shows 8-12% forgetting | Sanity check | Literature-consistent |
| EWC-only (GodelPlugin) shows ~31.5% forgetting | Matches prior result | Reproduces existing finding |
| GodelReplay shows < Replay-only forgetting | PRIMARY GOAL | Statistical improvement |
| All results logged to files | Pass | `log_*.txt` files generated |
| Kaggle notebook runs end-to-end | Pass | Public notebook, visible results |

---

## 7. Key Technical Notes

### 7.1 The PermutedMNIST Pivot

The prior experiment (commit `76d8362`) showed that GodelPlugin alone achieves ~0.99 forgetting on PermutedMNIST because it is a **class-incremental** benchmark where regularization-only methods fundamentally cannot prevent output head interference. The verdict was: **PIVOT TO REPLAY**.

GodelReplay addresses this directly — the Replay component handles the distribution shift while GodelPlugin preserves the weight identity. This is not a retreat from GodelAI's thesis; it is the natural evolution predicted by the Two-Layer Architecture.

### 7.2 Fisher Scaling is Critical

Without Fisher Scaling, EWC penalty is negligible at 218K-param scale (Fisher values ~1e-4 to 1e-7). The `global_max` strategy normalizes Fisher to [0, 1] range, making the penalty meaningful. This was the key insight from the April 3 self-optimization session.

### 7.3 Avalanche Compatibility

GodelPlugin uses `after_forward` hook (not `before_backward`) to avoid the double forward pass issue. It composes cleanly with Replay because Replay handles the data sampling while GodelPlugin handles the loss modification. They operate on different aspects of the training loop and do not conflict.

---

## 8. Communication Protocol

### 8.1 GitHub Bridge

All code changes go through the `godelai` repo on the `main` branch. Commit messages must follow MACP format:

```
GodelReplay: [description] — RNA (Claude Code) via T (CTO)
```

### 8.2 Feedback Loop

After implementation:
1. RNA pushes code to GitHub
2. Rk creates Kaggle notebook and runs experiment
3. Results are committed back to `godelai` repo under `experiments/results/`
4. T (Manus AI) reviews and produces the Week 1 report
5. Alton reviews and approves Week 2 plan

### 8.3 If Things Go Wrong

If GodelReplay does NOT outperform Replay-only:
- Check Fisher Scaling is active (diagnose with `diagnose_ewc_activation()`)
- Try increasing `ewc_lambda` (400 → 1000 → 5000)
- Try `layer_wise` scaling instead of `global_max`
- Check T-Score values — if all near 1.0, gradients are healthy; if < 0.5, training is pathological

If Avalanche throws errors:
- Verify version compatibility (`avalanche-lib>=0.5.0`)
- Check that GodelPlugin's hooks fire in correct order
- Ensure model is on correct device before plugin initialization

---

## 9. Ethical Guardrails Reminder

This sprint operates under the following constraints (see `docs/GODELREPLAY_SPRINT_ETHICS_AND_PLAN.md`):

1. All notebooks must be PUBLIC on Kaggle
2. All code is MIT-licensed
3. Stay within 30h/week GPU quota
4. No production model training — research artifacts only
5. Credit Kaggle in paper acknowledgments
6. Single account only (Alton's verified account)

---

## 10. Timeline

| Day | Task | Owner |
|-----|------|-------|
| Day 1-2 | Implement `godel_replay.py` + unit test locally | RNA |
| Day 2-3 | Create experiment script + verify locally (CPU) | RNA |
| Day 3-4 | Push to GitHub, create Kaggle notebook | RNA + Rk |
| Day 4-5 | Run full experiment on Kaggle GPU | Rk |
| Day 5-7 | Analyze results, commit findings, report to T | RNA + Rk |

---

*T (CTO, Manus AI) — FLYWHEEL TEAM | GodelReplay Week 1 Implementation Guide*
*GitHub Bridge: creator35lwb-web/godelai | MACP v2.3.1*
