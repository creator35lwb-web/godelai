"""
GodelReplay Strategy
====================
Combines Avalanche's Replay strategy with GodelPlugin for
dual-layer continual learning: experience replay (distribution preservation)
+ Fisher-scaled EWC-DR (weight preservation).

The Two-Layer Architecture:
  GodelPlugin  → protects identity-defining weights during training (C-S-P: State)
  Replay       → maintains exposure to past task distributions (C-S-P: Compression)
  Together     → neither layer alone is sufficient; both address different failure modes

Hypothesis: GodelReplay achieves lower forgetting than either Replay-only
or GodelPlugin-only on PermutedMNIST.

Architecture Decision:
  Replay handles data sampling (what the model sees).
  GodelPlugin handles loss modification (what the model preserves).
  They operate on different aspects of the training loop and do not conflict.

Author: Rk (Claude Code) + RNA — FLYWHEEL TEAM
Orchestrator: T (CTO, Manus AI)
Date: April 2026
MACP: v2.3.1
"""

import torch
from avalanche.training.supervised import Replay
from avalanche.training.plugins import EvaluationPlugin
from godelai.avalanche_plugin import GodelPlugin


def create_godel_replay_strategy(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    mem_size: int = 500,
    ewc_lambda: float = 400.0,
    fisher_scaling_strategy: str = "global_max",
    propagation_gamma: float = 2.0,
    t_score_window: int = 50,
    train_epochs: int = 5,
    train_mb_size: int = 128,
    eval_mb_size: int = 256,
    device: str = "cuda",
    eval_plugin: EvaluationPlugin = None,
):
    """
    Factory: Avalanche Replay strategy with GodelPlugin attached.

    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        criterion: Loss function
        mem_size: Replay buffer size (samples from past tasks)
        ewc_lambda: EWC regularization strength (Fisher-scaled)
        fisher_scaling_strategy: 'global_max' | 'layer_wise' | 'target_penalty'
        propagation_gamma: C-S-P propagation penalty exponent
        t_score_window: Window size for T-Score gradient diversity monitoring
        train_epochs: Epochs per experience
        train_mb_size: Training mini-batch size
        eval_mb_size: Evaluation mini-batch size
        device: 'cuda' | 'cpu'
        eval_plugin: Optional Avalanche EvaluationPlugin

    Returns:
        (strategy, godel_plugin): Configured Replay strategy + plugin reference
    """
    godel_plugin = GodelPlugin(
        propagation_gamma=propagation_gamma,
        min_surplus_energy=0.1,
        t_score_window=t_score_window,
        ewc_lambda=ewc_lambda,
        fisher_scaling_strategy=fisher_scaling_strategy,
    )

    plugins = [godel_plugin]
    if eval_plugin is not None:
        plugins.append(eval_plugin)

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
