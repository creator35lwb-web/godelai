"""
GodelAI vs Avalanche Baselines — SplitMNIST Benchmark
=======================================================
Community-standard continual learning benchmark using Avalanche.
Compares three conditions:
  1. Naive (finetuning, no protection)
  2. Avalanche EWC (standard EWC baseline)
  3. GodelAI-EWC (T-Score + Sleep + Fisher Scaling + EWC)

Benchmark: SplitMNIST (5 experiences, 2 classes per experience)
Model: SimpleMLP (~10K params)
Metrics: Average Accuracy, Forgetting Measure

Recommended by: Grok (xAI) external analysis
Implemented by: L (GodelAI CEO) — MACP v2.2 "Identity"
Date: April 3, 2026
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

# Avalanche imports
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from avalanche.training.plugins import EWCPlugin, EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import TextLogger

# GodelAI imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from godelai.reg.fisher_scaling import scale_fisher

# ==================== CONFIG ====================
DEVICE = torch.device("cpu")
N_EXPERIENCES = 5
BATCH_SIZE = 64
EPOCHS = 2
LR = 0.01
EWC_LAMBDA = 0.4
SEED = 42

torch.manual_seed(SEED)

print("=" * 70)
print("GodelAI vs Avalanche Baselines — SplitMNIST Benchmark")
print(f"Author: L (GodelAI CEO) — MACP v2.2 'Identity'")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Config: {N_EXPERIENCES} experiences, {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")
print("=" * 70)

# ==================== BENCHMARK SETUP ====================
scenario = SplitMNIST(n_experiences=N_EXPERIENCES, seed=SEED)

# ==================== HELPER FUNCTIONS ====================

def compute_fisher_from_model(model, dataloader, criterion, n_samples=200):
    """Compute Fisher Information Matrix for EWC."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    count = 0
    for x, y, *_ in dataloader:
        if count >= n_samples:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data.pow(2)
        count += x.shape[0]
    return {k: v / max(count, 1) for k, v in fisher.items()}


def ewc_penalty(model, fisher, old_params, lam):
    """Compute EWC penalty."""
    pen = torch.tensor(0.0)
    for n, p in model.named_parameters():
        if n in fisher:
            pen += (lam * fisher[n] * (p - old_params[n]).pow(2)).sum()
    return pen


def compute_per_sample_tscore(model, x, y, criterion):
    """Compute T-Score using per-sample gradients (GodelAI's core metric)."""
    model.eval()
    grads = []
    bs = min(x.shape[0], 16)  # Limit for speed
    for i in range(bs):
        model.zero_grad()
        out = model(x[i:i+1])
        loss = criterion(out, y[i:i+1])
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        grads.append(g)
    if len(grads) < 2:
        return 0.5
    G = torch.stack(grads)
    sum_g = G.sum(dim=0)
    sum_sq = (G.norm(dim=1) ** 2).sum()
    N = len(grads)
    if sum_sq.item() < 1e-12:
        return 0.5
    return 1.0 - (sum_g.norm() ** 2 / sum_sq).item() / N


def sleep_protocol(model):
    """GodelAI Sleep Protocol: prune, decay, refresh."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                threshold = torch.std(param) * 0.1
                mask = torch.abs(param) > threshold
                param.data.mul_(mask.float())
                param.data.mul_(0.995)
                noise = torch.randn_like(param) * 0.001
                param.data.add_(noise)


def evaluate_all_experiences(model, test_stream, criterion):
    """Evaluate model on all test experiences, return per-experience accuracy."""
    model.eval()
    accs = {}
    with torch.no_grad():
        for exp in test_stream:
            dl = DataLoader(exp.dataset, batch_size=256)
            correct, total = 0, 0
            for x, y, *_ in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.shape[0]
            accs[exp.current_experience] = correct / max(total, 1)
    return accs


# ==================== CONDITION 1: NAIVE (No Protection) ====================
print(f"\n{'='*70}")
print("CONDITION 1: Naive Sequential Training (No Protection)")
print(f"{'='*70}")

torch.manual_seed(SEED)
model_naive = SimpleMLP(num_classes=10, input_size=28*28).to(DEVICE)
optimizer = SGD(model_naive.parameters(), lr=LR, momentum=0.9)
criterion = nn.CrossEntropyLoss()

naive_accs_after_training = {}
naive_accs_final = {}

for exp_idx, train_exp in enumerate(scenario.train_stream):
    dl = DataLoader(train_exp.dataset, batch_size=BATCH_SIZE, shuffle=True)
    model_naive.train()
    for ep in range(EPOCHS):
        for x, y, *_ in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model_naive(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Evaluate after training on this experience
    accs = evaluate_all_experiences(model_naive, scenario.test_stream, criterion)
    naive_accs_after_training[exp_idx] = accs
    print(f"  After Exp {exp_idx}: " + " | ".join([f"E{k}={v:.3f}" for k, v in sorted(accs.items())]))

naive_accs_final = naive_accs_after_training[N_EXPERIENCES - 1]
naive_avg_acc = sum(naive_accs_final.values()) / len(naive_accs_final)
print(f"  Final Average Accuracy: {naive_avg_acc:.4f}")

# Compute forgetting
naive_forgetting = {}
for exp_id in range(N_EXPERIENCES - 1):
    best_acc = max(naive_accs_after_training[t][exp_id] for t in range(exp_id, N_EXPERIENCES))
    final_acc = naive_accs_final[exp_id]
    naive_forgetting[exp_id] = best_acc - final_acc
naive_avg_forgetting = sum(naive_forgetting.values()) / max(len(naive_forgetting), 1)
print(f"  Average Forgetting: {naive_avg_forgetting:.4f}")


# ==================== CONDITION 2: AVALANCHE EWC (Standard Baseline) ====================
print(f"\n{'='*70}")
print("CONDITION 2: Avalanche EWC (Standard Baseline)")
print(f"{'='*70}")

torch.manual_seed(SEED)
model_ewc = SimpleMLP(num_classes=10, input_size=28*28).to(DEVICE)
optimizer_ewc = SGD(model_ewc.parameters(), lr=LR, momentum=0.9)

# Use Avalanche's built-in EWC strategy
log_file = open("results/avalanche_ewc_log.txt", "w")
eval_plugin = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[TextLogger(log_file)]
)

ewc_plugin = EWCPlugin(ewc_lambda=EWC_LAMBDA)
strategy_ewc = Naive(
    model_ewc, optimizer_ewc, criterion,
    train_mb_size=BATCH_SIZE,
    train_epochs=EPOCHS,
    eval_mb_size=256,
    device=DEVICE,
    plugins=[ewc_plugin, eval_plugin]
)

ewc_accs_after_training = {}
for exp_idx, train_exp in enumerate(scenario.train_stream):
    strategy_ewc.train(train_exp, num_workers=0)
    accs = evaluate_all_experiences(model_ewc, scenario.test_stream, criterion)
    ewc_accs_after_training[exp_idx] = accs
    print(f"  After Exp {exp_idx}: " + " | ".join([f"E{k}={v:.3f}" for k, v in sorted(accs.items())]))

log_file.close()

ewc_accs_final = ewc_accs_after_training[N_EXPERIENCES - 1]
ewc_avg_acc = sum(ewc_accs_final.values()) / len(ewc_accs_final)
print(f"  Final Average Accuracy: {ewc_avg_acc:.4f}")

ewc_forgetting = {}
for exp_id in range(N_EXPERIENCES - 1):
    best_acc = max(ewc_accs_after_training[t][exp_id] for t in range(exp_id, N_EXPERIENCES))
    final_acc = ewc_accs_final[exp_id]
    ewc_forgetting[exp_id] = best_acc - final_acc
ewc_avg_forgetting = sum(ewc_forgetting.values()) / max(len(ewc_forgetting), 1)
print(f"  Average Forgetting: {ewc_avg_forgetting:.4f}")


# ==================== CONDITION 3: GodelAI-EWC (T-Score + Sleep + Fisher Scaling) ====================
print(f"\n{'='*70}")
print("CONDITION 3: GodelAI-EWC (T-Score + Sleep + Fisher Scaling)")
print("  The C-S-P framework in action on a community-standard benchmark")
print(f"{'='*70}")

torch.manual_seed(SEED)
model_godel = SimpleMLP(num_classes=10, input_size=28*28).to(DEVICE)
optimizer_godel = SGD(model_godel.parameters(), lr=LR, momentum=0.9)

cumulative_fisher = {}
cumulative_old_params = {}
sleep_count = 0
t_score_history = []

godel_accs_after_training = {}
for exp_idx, train_exp in enumerate(scenario.train_stream):
    dl = DataLoader(train_exp.dataset, batch_size=BATCH_SIZE, shuffle=True)
    model_godel.train()

    exp_t_scores = []
    for ep in range(EPOCHS):
        for x, y, *_ in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # 1. Compute T-Score (GodelAI's gradient diversity metric)
            t_score = compute_per_sample_tscore(model_godel, x, y, criterion)
            exp_t_scores.append(t_score)

            # 2. Sleep Protocol: if T-Score drops below threshold
            if t_score < 0.1:
                sleep_protocol(model_godel)
                sleep_count += 1

            # 3. Standard forward pass + EWC penalty with Fisher Scaling
            optimizer_godel.zero_grad()
            out = model_godel(x)
            loss = criterion(out, y)

            if cumulative_fisher:
                pen = ewc_penalty(model_godel, cumulative_fisher, cumulative_old_params, EWC_LAMBDA)
                loss = loss + pen

            loss.backward()
            nn.utils.clip_grad_norm_(model_godel.parameters(), 1.0)
            optimizer_godel.step()

    # After training on this experience: compute & accumulate Fisher with scaling
    dl_fisher = DataLoader(train_exp.dataset, batch_size=BATCH_SIZE)
    fisher_raw = compute_fisher_from_model(model_godel, dl_fisher, criterion)
    fisher_scaled = scale_fisher(fisher_raw, strategy="global_max")

    for n, f in fisher_scaled.items():
        if n in cumulative_fisher:
            cumulative_fisher[n] = cumulative_fisher[n] + f
        else:
            cumulative_fisher[n] = f.clone()

    cumulative_old_params = {n: p.data.clone() for n, p in model_godel.named_parameters() if p.requires_grad}

    avg_t = sum(exp_t_scores) / max(len(exp_t_scores), 1)
    t_score_history.append({"exp": exp_idx, "avg_tscore": round(avg_t, 4), "n_steps": len(exp_t_scores)})

    accs = evaluate_all_experiences(model_godel, scenario.test_stream, criterion)
    godel_accs_after_training[exp_idx] = accs
    print(f"  After Exp {exp_idx}: " + " | ".join([f"E{k}={v:.3f}" for k, v in sorted(accs.items())]) + f"  T-Score={avg_t:.4f}")

godel_accs_final = godel_accs_after_training[N_EXPERIENCES - 1]
godel_avg_acc = sum(godel_accs_final.values()) / len(godel_accs_final)
print(f"  Final Average Accuracy: {godel_avg_acc:.4f}")
print(f"  Sleep Events: {sleep_count}")

godel_forgetting = {}
for exp_id in range(N_EXPERIENCES - 1):
    best_acc = max(godel_accs_after_training[t][exp_id] for t in range(exp_id, N_EXPERIENCES))
    final_acc = godel_accs_final[exp_id]
    godel_forgetting[exp_id] = best_acc - final_acc
godel_avg_forgetting = sum(godel_forgetting.values()) / max(len(godel_forgetting), 1)
print(f"  Average Forgetting: {godel_avg_forgetting:.4f}")


# ==================== RESULTS SUMMARY ====================
print(f"\n{'='*70}")
print("GODELAI vs AVALANCHE BASELINES — SPLITMNIST RESULTS")
print(f"{'='*70}")

print(f"\n  {'Method':<30} {'Avg Accuracy':>14} {'Avg Forgetting':>16} {'vs Naive':>10}")
print(f"  {'-'*70}")

naive_reduction = 0.0
ewc_reduction = ((naive_avg_forgetting - ewc_avg_forgetting) / max(abs(naive_avg_forgetting), 1e-8)) * 100
godel_reduction = ((naive_avg_forgetting - godel_avg_forgetting) / max(abs(naive_avg_forgetting), 1e-8)) * 100

print(f"  {'Naive (No Protection)':<30} {naive_avg_acc:>14.4f} {naive_avg_forgetting:>16.4f} {'baseline':>10}")
print(f"  {'Avalanche EWC':<30} {ewc_avg_acc:>14.4f} {ewc_avg_forgetting:>16.4f} {ewc_reduction:>+9.1f}%")
print(f"  {'GodelAI-EWC (C-S-P)':<30} {godel_avg_acc:>14.4f} {godel_avg_forgetting:>16.4f} {godel_reduction:>+9.1f}%")

print(f"\n  GodelAI vs Avalanche EWC:")
if ewc_avg_forgetting > 1e-8:
    godel_vs_ewc = ((ewc_avg_forgetting - godel_avg_forgetting) / ewc_avg_forgetting) * 100
    print(f"    Forgetting reduction vs Avalanche EWC: {godel_vs_ewc:+.1f}%")
else:
    print(f"    Avalanche EWC forgetting near zero — comparison not meaningful")

print(f"\n  T-Score History (GodelAI monitoring):")
for t in t_score_history:
    print(f"    Experience {t['exp']}: avg T-Score = {t['avg_tscore']:.4f} ({t['n_steps']} steps)")
print(f"    Total Sleep Events: {sleep_count}")

# ==================== SAVE RESULTS ====================
results = {
    "experiment": "GodelAI vs Avalanche — SplitMNIST",
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "author": "L (GodelAI CEO) — MACP v2.2 'Identity'",
    "recommended_by": "Grok (xAI) external analysis",
    "config": {
        "n_experiences": N_EXPERIENCES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "ewc_lambda": EWC_LAMBDA,
        "model": "SimpleMLP (Avalanche)",
    },
    "results": {
        "naive": {
            "avg_accuracy": round(naive_avg_acc, 6),
            "avg_forgetting": round(naive_avg_forgetting, 6),
            "per_exp_forgetting": {str(k): round(v, 6) for k, v in naive_forgetting.items()},
        },
        "avalanche_ewc": {
            "avg_accuracy": round(ewc_avg_acc, 6),
            "avg_forgetting": round(ewc_avg_forgetting, 6),
            "per_exp_forgetting": {str(k): round(v, 6) for k, v in ewc_forgetting.items()},
            "vs_naive_pct": round(ewc_reduction, 2),
        },
        "godelai_ewc": {
            "avg_accuracy": round(godel_avg_acc, 6),
            "avg_forgetting": round(godel_avg_forgetting, 6),
            "per_exp_forgetting": {str(k): round(v, 6) for k, v in godel_forgetting.items()},
            "vs_naive_pct": round(godel_reduction, 2),
            "sleep_count": sleep_count,
            "t_score_history": t_score_history,
        },
    },
}

out_path = Path("results") / f"avalanche_splitmnist_{results['timestamp']}.json"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(results, indent=2))
print(f"\n  Results saved: {out_path}")
print(f"\n{'='*70}")
print("BENCHMARK COMPLETE")
print(f"{'='*70}")
