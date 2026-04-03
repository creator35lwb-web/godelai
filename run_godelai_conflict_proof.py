"""
GodelAI Conflict Data Proof — The Definitive Benchmark
========================================================
We prove it ourselves first, on our own data, before any public claims.

This benchmark tests IDENTITY PRESERVATION — the domain GodelAI is designed for.
NOT class-incremental accuracy (where all regularization methods fail).

Experiment Design:
  - 4 conflict domains as sequential tasks (domain-incremental):
    Task A: Contradictory Facts (physics, biology, philosophy)
    Task B: Ethical Dilemmas (AI ethics, medical, environmental)
    Task C: Perspective Conflicts (governance, technology, society)
    Task D: Temporal Conflicts (science, medicine, technology)
  
  - Train a character-level model on each domain sequentially
  - Measure: Does the model retain its learned patterns from Task A
    after learning Tasks B, C, D?

  - 3 conditions:
    1. Naive (no protection — catastrophic forgetting baseline)
    2. Standard EWC (Avalanche-equivalent, raw Fisher)
    3. GodelAI-EWC (T-Score + Sleep + Fisher Scaling — the C-S-P stack)

  - Metrics:
    * Forgetting (loss increase on previous tasks)
    * T-Score (gradient diversity during training)
    * Identity Preservation Rate (% improvement over Naive)

Author: L (GodelAI CEO) — MACP v2.2 "Identity"
Date: April 3, 2026
Principle: "If we can't prove it, we don't claim it."
"""

import torch
import torch.nn as nn
import json
import random
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

torch.manual_seed(42)
random.seed(42)

# ============================================================
# STEP 1: Load and Prepare Conflict Datasets
# ============================================================

def extract_text_from_conflict(filepath):
    """Extract all meaningful text from a conflict dataset file."""
    with open(filepath) as f:
        data = json.load(f)
    
    items = data.get("data", data) if isinstance(data, dict) else data
    texts = []
    
    for item in items:
        if isinstance(item, dict):
            # Extract all text fields recursively
            texts.extend(_extract_strings(item))
    
    return " ".join(texts)


def _extract_strings(obj, depth=0):
    """Recursively extract string values from nested dicts/lists."""
    if depth > 5:
        return []
    strings = []
    if isinstance(obj, str) and len(obj) > 20:  # Only meaningful text
        strings.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ("id", "type", "$schema", "version", "created_by", "created_at"):
                strings.extend(_extract_strings(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            strings.extend(_extract_strings(item, depth + 1))
    return strings


def load_conflict_domains():
    """Load all 4 conflict domains as sequential tasks."""
    base = Path("datasets/conflict")
    
    domains = {
        "Contradictory Facts": [
            base / "contradictory_facts/scientific_paradoxes.json",
            base / "contradictory_facts/expanded_paradoxes.json",
        ],
        "Ethical Dilemmas": [
            base / "ethical_dilemmas/core_dilemmas.json",
            base / "ethical_dilemmas/expanded_dilemmas.json",
        ],
        "Perspective Conflicts": [
            base / "perspective_conflicts/ai_governance.json",
            base / "perspective_conflicts/expanded_perspectives.json",
        ],
        "Temporal Conflicts": [
            base / "temporal_conflicts/evolving_knowledge.json",
            base / "temporal_conflicts/expanded_temporal.json",
        ],
    }
    
    domain_texts = {}
    for name, files in domains.items():
        combined = ""
        for f in files:
            if f.exists():
                combined += " " + extract_text_from_conflict(str(f))
        domain_texts[name] = combined.strip()
    
    return domain_texts


# ============================================================
# STEP 2: Model and Training Infrastructure
# ============================================================

class ConflictGRU(nn.Module):
    """Character-level GRU for learning conflict domain patterns."""
    def __init__(self, vocab_size, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.fc = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size
    
    def forward(self, x):
        return self.fc(self.gru(self.emb(x))[0])


def build_vocab(texts):
    chars = sorted(set("".join(texts)))
    return {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}


def encode_text(text, char2idx, seq_len=50):
    """Encode text into overlapping sequences for training."""
    enc = [char2idx.get(c, 0) for c in text]
    sequences = []
    step = seq_len // 3  # Overlapping windows for more data
    for i in range(0, len(enc) - seq_len - 1, step):
        chunk = enc[i:i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        sequences.append((
            torch.tensor(chunk[:seq_len]),
            torch.tensor(chunk[1:seq_len + 1]),
        ))
    return sequences


def make_batches(sequences, batch_size=32):
    random.shuffle(sequences)
    batches = []
    for i in range(0, len(sequences), batch_size):
        b = sequences[i:i + batch_size]
        if len(b) < 2:
            continue
        batches.append((
            torch.stack([x[0] for x in b]),
            torch.stack([x[1] for x in b]),
        ))
    return batches


def evaluate(model, batches, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in batches:
            out = model(x).reshape(-1, model.vocab_size)
            total += criterion(out, y.reshape(-1)).item()
    return total / max(len(batches), 1)


def compute_fisher(model, batches, criterion, n_samples=100):
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    count = 0
    for x, y in batches:
        if count >= n_samples:
            break
        model.zero_grad()
        out = model(x).reshape(-1, model.vocab_size)
        criterion(out, y.reshape(-1)).backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data.pow(2)
        count += 1
    return {k: v / max(count, 1) for k, v in fisher.items()}


def scale_fisher_global_max(fisher, eps=1e-10):
    all_vals = torch.cat([f.flatten() for f in fisher.values()])
    f_max = all_vals.max().item()
    if f_max < eps:
        return fisher
    return {k: v / f_max for k, v in fisher.items()}


def ewc_penalty(model, fisher, old_params, ewc_lambda):
    penalty = torch.tensor(0.0)
    for n, p in model.named_parameters():
        if n in fisher:
            penalty += (ewc_lambda * fisher[n] * (p - old_params[n]).pow(2)).sum()
    return penalty


def compute_tscore(model, batches, criterion, n_batches=10):
    """Compute T-Score using per-sample gradients — GodelAI's core metric."""
    model.train()
    grads = []
    for x, y in batches[:n_batches]:
        for i in range(min(x.shape[0], 8)):  # Per-sample
            model.zero_grad()
            out = model(x[i:i+1]).reshape(-1, model.vocab_size)
            criterion(out, y[i:i+1].reshape(-1)).backward()
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


def train_epoch(model, batches, criterion, optimizer, fisher=None, old_params=None, 
                ewc_lambda=2.0, use_tscore=False, sleep_threshold=0.1):
    """Train one epoch with optional EWC and GodelAI monitoring."""
    model.train()
    total_loss, total_pen = 0.0, 0.0
    t_scores = []
    sleep_count = 0
    
    for x, y in batches:
        # GodelAI T-Score monitoring (per-batch)
        if use_tscore:
            t = compute_tscore(model, [(x, y)], criterion, n_batches=1)
            t_scores.append(t)
            if t < sleep_threshold:
                sleep_protocol(model)
                sleep_count += 1
        
        optimizer.zero_grad()
        out = model(x).reshape(-1, model.vocab_size)
        loss = criterion(out, y.reshape(-1))
        
        if fisher and old_params:
            pen = ewc_penalty(model, fisher, old_params, ewc_lambda)
            total_pen += pen.item()
            loss = loss + pen
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    n = max(len(batches), 1)
    avg_t = sum(t_scores) / max(len(t_scores), 1) if t_scores else None
    return total_loss / n, total_pen / n, avg_t, sleep_count


# ============================================================
# STEP 3: The Definitive Benchmark
# ============================================================

def run_condition(name, domain_order, domain_batches, vocab_size, criterion,
                  use_ewc=False, use_fisher_scaling=False, use_tscore=False,
                  ewc_lambda=2.0, epochs=20, lr=0.003):
    """Run one experimental condition."""
    print(f"\n{'='*70}")
    print(f"CONDITION: {name}")
    print(f"{'='*70}")
    
    torch.manual_seed(42)
    model = ConflictGRU(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses_after_training = {}  # Loss on domain X right after training on X
    cumulative_fisher = {}
    cumulative_old_params = {}
    total_sleep = 0
    domain_tscores = {}
    
    for i, domain in enumerate(domain_order):
        batches = domain_batches[domain]
        print(f"\n  Task {i+1}/{len(domain_order)}: {domain} ({len(batches)} batches)")
        
        for ep in range(epochs):
            tl, pl, avg_t, sc = train_epoch(
                model, batches, criterion, optimizer,
                fisher=cumulative_fisher if (use_ewc and cumulative_fisher) else None,
                old_params=cumulative_old_params if (use_ewc and cumulative_old_params) else None,
                ewc_lambda=ewc_lambda,
                use_tscore=use_tscore,
            )
            total_sleep += sc
            
            if (ep + 1) == epochs:
                t_str = f" T-Score={avg_t:.4f}" if avg_t is not None else ""
                p_str = f" EWC={pl:.4f}" if use_ewc else ""
                print(f"    Ep {ep+1}: loss={tl:.4f}{p_str}{t_str}")
        
        # Record loss on this domain right after training
        losses_after_training[domain] = evaluate(model, batches, criterion)
        
        # Compute T-Score for this domain
        t = compute_tscore(model, batches, criterion)
        domain_tscores[domain] = t
        print(f"    T-Score (domain): {t:.4f}")
        
        # Accumulate Fisher if using EWC
        if use_ewc:
            fisher = compute_fisher(model, batches, criterion)
            if use_fisher_scaling:
                fisher = scale_fisher_global_max(fisher)
                f_max_raw = max(f.max().item() for f in compute_fisher(model, batches, criterion).values())
                print(f"    Fisher max (raw): {f_max_raw:.2e} → (scaled): 1.0000")
            
            for n, f in fisher.items():
                if n in cumulative_fisher:
                    cumulative_fisher[n] = cumulative_fisher[n] + f
                else:
                    cumulative_fisher[n] = f.clone()
            
            cumulative_old_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    
    # Final evaluation on all domains
    print(f"\n  --- Identity Preservation ({name}) ---")
    forgetting = {}
    final_losses = {}
    for domain in domain_order[:-1]:  # Measure forgetting on all but last
        final_loss = evaluate(model, domain_batches[domain], criterion)
        final_losses[domain] = final_loss
        forget = final_loss - losses_after_training[domain]
        forgetting[domain] = forget
        print(f"    {domain}: {losses_after_training[domain]:.4f} → {final_loss:.4f} (forgetting: {forget:+.4f})")
    
    avg_forgetting = sum(forgetting.values()) / max(len(forgetting), 1)
    print(f"\n  Average Forgetting: {avg_forgetting:+.4f}")
    if use_tscore:
        print(f"  Total Sleep Events: {total_sleep}")
    
    return {
        "condition": name,
        "forgetting": {k: round(v, 6) for k, v in forgetting.items()},
        "avg_forgetting": round(avg_forgetting, 6),
        "losses_after_training": {k: round(v, 6) for k, v in losses_after_training.items()},
        "final_losses": {k: round(v, 6) for k, v in final_losses.items()},
        "t_scores": {k: round(v, 4) for k, v in domain_tscores.items()},
        "sleep_count": total_sleep,
    }


def main():
    print("=" * 70)
    print("GodelAI Conflict Data Proof — The Definitive Benchmark")
    print("We prove it ourselves first, on our own data.")
    print(f"Author: L (GodelAI CEO) — MACP v2.2 'Identity'")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load conflict domains
    domain_texts = load_conflict_domains()
    
    # Build shared vocabulary
    all_text = " ".join(domain_texts.values())
    char2idx, idx2char = build_vocab([all_text])
    V = len(char2idx)
    print(f"\n  Vocabulary: {V} characters")
    
    # Prepare domain batches
    domain_order = ["Contradictory Facts", "Ethical Dilemmas", "Perspective Conflicts", "Temporal Conflicts"]
    domain_batches = {}
    for domain in domain_order:
        seqs = encode_text(domain_texts[domain], char2idx)
        batches = make_batches(seqs)
        domain_batches[domain] = batches
        print(f"  {domain}: {len(domain_texts[domain])} chars → {len(seqs)} sequences → {len(batches)} batches")
    
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 20
    LR = 0.003
    EWC_LAMBDA = 2.0
    
    # ---- Run all 3 conditions ----
    results = {}
    
    # Condition 1: Naive (No Protection)
    results["naive"] = run_condition(
        "Naive (No Protection — Catastrophic Forgetting Baseline)",
        domain_order, domain_batches, V, criterion,
        epochs=EPOCHS, lr=LR,
    )
    
    # Condition 2: Standard EWC (Raw Fisher — Avalanche-equivalent)
    results["standard_ewc"] = run_condition(
        "Standard EWC (Raw Fisher — Avalanche Equivalent)",
        domain_order, domain_batches, V, criterion,
        use_ewc=True, use_fisher_scaling=False,
        ewc_lambda=EWC_LAMBDA, epochs=EPOCHS, lr=LR,
    )
    
    # Condition 3: GodelAI-EWC (Full C-S-P Stack)
    results["godelai"] = run_condition(
        "GodelAI-EWC (T-Score + Sleep + Fisher Scaling — Full C-S-P)",
        domain_order, domain_batches, V, criterion,
        use_ewc=True, use_fisher_scaling=True, use_tscore=True,
        ewc_lambda=EWC_LAMBDA, epochs=EPOCHS, lr=LR,
    )
    
    # ============================================================
    # STEP 4: The Verdict
    # ============================================================
    
    print(f"\n{'='*70}")
    print("GODELAI CONFLICT DATA PROOF — RESULTS")
    print(f"{'='*70}")
    
    naive_f = results["naive"]["avg_forgetting"]
    ewc_f = results["standard_ewc"]["avg_forgetting"]
    godel_f = results["godelai"]["avg_forgetting"]
    
    ewc_imp = ((naive_f - ewc_f) / max(abs(naive_f), 1e-8)) * 100
    godel_imp = ((naive_f - godel_f) / max(abs(naive_f), 1e-8)) * 100
    godel_vs_ewc = ((ewc_f - godel_f) / max(abs(ewc_f), 1e-8)) * 100
    
    print(f"\n  {'Domain':<25} {'Naive':>10} {'Std EWC':>10} {'GodelAI':>10}")
    print(f"  {'-'*60}")
    for domain in domain_order[:-1]:
        nf = results["naive"]["forgetting"][domain]
        ef = results["standard_ewc"]["forgetting"][domain]
        gf = results["godelai"]["forgetting"][domain]
        print(f"  {domain:<25} {nf:>+10.4f} {ef:>+10.4f} {gf:>+10.4f}")
    
    print(f"  {'-'*60}")
    print(f"  {'AVERAGE':<25} {naive_f:>+10.4f} {ewc_f:>+10.4f} {godel_f:>+10.4f}")
    
    print(f"\n  Forgetting Reduction vs Naive:")
    print(f"    Standard EWC:  {ewc_imp:+.1f}%")
    print(f"    GodelAI-EWC:   {godel_imp:+.1f}%")
    print(f"    GodelAI vs Standard EWC: {godel_vs_ewc:+.1f}%")
    
    print(f"\n  T-Score by Domain (GodelAI monitoring):")
    for domain, t in results["godelai"]["t_scores"].items():
        in_range = "IN RANGE" if 0.3 <= t <= 0.5 else ("HIGH" if t > 0.5 else "LOW")
        print(f"    {domain}: {t:.4f} ({in_range})")
    print(f"    Sleep Events: {results['godelai']['sleep_count']}")
    
    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    
    if godel_imp > 10:
        verdict = "GO"
        print(f"""
  VERDICT: {verdict}

  GodelAI-EWC achieved {godel_imp:.1f}% forgetting reduction on our own conflict data.
  This is {godel_vs_ewc:+.1f}% better than standard EWC (Avalanche equivalent).

  The C-S-P framework works on the data it was designed for:
  conflict-rich, identity-challenging, domain-incremental learning.

  We have proven it ourselves. We can now claim it publicly.
  The conflict dataset is ready for open-source release.
""")
    elif godel_imp > 0:
        verdict = "CONDITIONAL GO"
        print(f"""
  VERDICT: {verdict}

  GodelAI-EWC achieved {godel_imp:.1f}% forgetting reduction — positive but modest.
  Further optimization needed before public claims.
""")
    else:
        verdict = "NO-GO"
        print(f"""
  VERDICT: {verdict}

  GodelAI-EWC did not demonstrate meaningful improvement ({godel_imp:.1f}%).
  Do NOT release publicly until the issue is resolved.
""")
    
    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_results = {
        "experiment": "GodelAI Conflict Data Proof — The Definitive Benchmark",
        "timestamp": ts,
        "author": "L (GodelAI CEO) — MACP v2.2 'Identity'",
        "principle": "If we can't prove it, we don't claim it.",
        "verdict": verdict,
        "config": {
            "epochs": EPOCHS,
            "lr": LR,
            "ewc_lambda": EWC_LAMBDA,
            "model": "ConflictGRU (128 hidden, 2 layers)",
            "domains": domain_order,
            "seq_len": 50,
        },
        "results": results,
        "summary": {
            "naive_avg_forgetting": naive_f,
            "standard_ewc_avg_forgetting": ewc_f,
            "godelai_avg_forgetting": godel_f,
            "godelai_vs_naive_pct": round(godel_imp, 2),
            "godelai_vs_ewc_pct": round(godel_vs_ewc, 2),
        },
    }
    
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"conflict_data_proof_{ts}.json"
    out_path.write_text(json.dumps(full_results, indent=2))
    print(f"  Results saved: {out_path}")
    
    return full_results


if __name__ == "__main__":
    main()
