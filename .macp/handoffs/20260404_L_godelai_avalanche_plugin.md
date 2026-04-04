# MACP Handoff-016: GodelPlugin — Avalanche Integration Complete

**Agent:** L (GodelAI CEO)
**Date:** April 4, 2026
**Protocol:** MACP v2.2 "Identity"
**Status:** INTEGRATION VALIDATED

---

## Origin

Grok (xAI) recommended porting GodelAI as an Avalanche Plugin to gain credibility in the Continual Learning community. The AI Council (Y/X/Z/CS) unanimously endorsed **Option B**: a self-contained wrapper plugin that composes GodelAgent + EWC-DR + Fisher Scaling externally, without modifying GodelAgent's core API.

## What Was Built

### `godelai/avalanche_plugin.py` — GodelPlugin

A complete Avalanche `SupervisedPlugin` that wraps the entire GodelAI C-S-P stack:

**Lifecycle Hooks:**
- `before_training_exp` → Initialize GodelAgent, detect experience transitions
- `after_forward` → T-Score monitoring, propagation penalty, EWC-DR penalty, Sleep Protocol
- `after_training_exp` → Fisher computation, Fisher Scaling, EWC-DR consolidation

**Key Design Decisions (Council-Driven):**
1. **Isolated T-Score computation** — Uses `copy.deepcopy(strategy.model)` to prevent per-sample gradient computation from contaminating Avalanche's computation graph
2. **Normalized EWC penalty** — Divides by parameter count for scale-invariance across model sizes
3. **NaN guards** — Checks `torch.isfinite()` at every penalty injection point
4. **One C concession** — Added `get_recent_t_score()` read-only property to GodelAgent

### Bugs Fixed During Integration

| Bug | Root Cause | Fix |
| :--- | :--- | :--- |
| `TypeError: consolidate() missing args` | Grok's code called `consolidate(model, fisher)` but EWCDR expects `(model, dataloader, device, criterion)` | Direct injection of pre-computed Fisher + old_params |
| `TypeError: unsupported operand +: Tensor + tuple` | EWCDR.compute_penalty returns `(penalty, metrics)` tuple | Unpacked with `ewc_penalty, ewc_metrics = ...` |
| NaN explosion from Experience 2+ | Per-sample gradient computation on main model contaminated Avalanche's backward graph | Isolated on deepcopy'd model |
| EWC penalty 6.6M (too large) | ewc_lambda=400 on 214K params without normalization | Normalized by parameter count |

## Test Results — SplitMNIST (Class-Incremental)

| Condition | Accuracy | Forgetting | Time |
| :--- | :---: | :---: | :---: |
| Naive (No Protection) | 0.2012 | 0.0000 | 25.5s |
| Naive + GodelPlugin | 0.2006 | 0.0000 | 204.7s |

**T-Scores (all 5 experiences):** [0.9797, 0.9787, 0.9810, 0.9794, 0.9804]
**NaN occurrences:** 0
**Sleep events:** 0
**EWC-DR active:** Yes (from Experience 1 onward)

### Honest Assessment (Z-Protocol)

SplitMNIST is a **class-incremental** benchmark. GodelAI is designed for **domain-incremental identity preservation**. Regularization-only methods (including GodelAI, standard EWC, and SI) cannot solve class-incremental learning without replay buffers. Both conditions show ~0.20 accuracy (random for 5-way split).

**This test validates INTEGRATION, not class-incremental performance.**

GodelAI's proven value remains: **82.8% forgetting reduction on conflict data** (domain-incremental).

## 5 API Mismatches Fixed (Grok's Code vs Reality)

1. `GodelAgent.__init__` does not accept `model` — uses `base_model`
2. `GodelAgent` does not have `compute_fisher()` — Fisher is in `FisherScaling`
3. `EWCDR.consolidate()` expects `(model, dataloader, device, criterion)` — not `(model, fisher_dict)`
4. `EWCDR.compute_penalty()` returns `(tensor, dict)` tuple — not a scalar
5. `GodelAgent` does not have `get_recent_t_score()` — added as the one C concession

## Next Steps

1. **Domain-incremental Avalanche benchmark** — PermutedMNIST or SplitCIFAR with task labels (where GodelAI actually helps)
2. **Combine with replay buffer** — GodelPlugin + ReplayPlugin to show additive benefit
3. **Publish as pip-installable package** — `pip install godelai-avalanche`

---

*"The plugin validates integration. The conflict data validates the science."*
