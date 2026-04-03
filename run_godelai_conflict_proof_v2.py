"""
GodelAI Conflict Data Proof v2 — Optimized
============================================
Same experiment as v1 but with optimized T-Score computation.
T-Score is computed every N batches (not every batch) to avoid CPU bottleneck.

Author: L (GodelAI CEO) — MACP v2.2 "Identity"
Date: April 3, 2026
"""

import torch
import torch.nn as nn
import json
import random
import os
import sys
from pathlib import Path
from datetime import datetime

torch.manual_seed(42)
random.seed(42)

# ============================================================
# Data Loading
# ============================================================

def _extract_strings(obj, depth=0):
    if depth > 5:
        return []
    strings = []
    if isinstance(obj, str) and len(obj) > 20:
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
                with open(f) as fh:
                    data = json.load(fh)
                items = data.get("data", data) if isinstance(data, dict) else data
                for item in items:
                    if isinstance(item, dict):
                        combined += " " + " ".join(_extract_strings(item))
        domain_texts[name] = combined.strip()
    return domain_texts

# ============================================================
# Model
# ============================================================

class ConflictGRU(nn.Module):
    def __init__(self, vocab_size, hidden=128, layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size
    
    def forward(self, x):
        return self.fc(self.gru(self.emb(x))[0])

# ============================================================
# Utilities
# ============================================================

def build_vocab(texts):
    chars = sorted(set("".join(texts)))
    return {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}

def encode_text(text, char2idx, seq_len=50):
    enc = [char2idx.get(c, 0) for c in text]
    sequences = []
    step = seq_len // 3
    for i in range(0, len(enc) - seq_len - 1, step):
        chunk = enc[i:i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        sequences.append((torch.tensor(chunk[:seq_len]), torch.tensor(chunk[1:seq_len + 1])))
    return sequences

def make_batches(sequences, batch_size=32):
    random.shuffle(sequences)
    batches = []
    for i in range(0, len(sequences), batch_size):
        b = sequences[i:i + batch_size]
        if len(b) < 2:
            continue
        batches.append((torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])))
    return batches

def evaluate(model, batches, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in batches:
            out = model(x).reshape(-1, model.vocab_size)
            total += criterion(out, y.reshape(-1)).item()
    return total / max(len(batches), 1)

def compute_fisher(model, batches, criterion, n_samples=50):
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

def compute_tscore_fast(model, x, y, criterion, n_samples=8):
    """Optimized T-Score: only use n_samples from the batch."""
    model.train()
    grads = []
    bs = min(x.shape[0], n_samples)
    for i in range(bs):
        model.zero_grad()
        out = model(x[i:i+1]).reshape(-1, model.vocab_size)
        criterion(out, y[i:i+1].reshape(-1)).backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        grads.append(g)
    if len(grads) < 2:
        return 0.5
    G = torch.stack(grads)
    N = len(grads)
    sum_sq = (G.norm(dim=1) ** 2).sum()
    if sum_sq.item() < 1e-12:
        return 0.5
    return 1.0 - (G.sum(dim=0).norm() ** 2 / sum_sq).item() / N

def sleep_protocol(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                threshold = torch.std(param) * 0.1
                mask = torch.abs(param) > threshold
                param.data.mul_(mask.float())
                param.data.mul_(0.995)
                param.data.add_(torch.randn_like(param) * 0.001)

# ============================================================
# Benchmark Runner
# ============================================================

def run_condition(name, domain_order, domain_batches, vocab_size, criterion,
                  use_ewc=False, use_fisher_scaling=False, use_tscore=False,
                  ewc_lambda=2.0, epochs=15, lr=0.003):
    
    print(f"\n{'='*70}")
    print(f"CONDITION: {name}")
    print(f"{'='*70}")
    
    torch.manual_seed(42)
    model = ConflictGRU(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: ConflictGRU ({param_count:,} parameters)")
    
    losses_after_training = {}
    cumulative_fisher = {}
    cumulative_old_params = {}
    total_sleep = 0
    domain_tscores = {}
    
    for i, domain in enumerate(domain_order):
        batches = domain_batches[domain]
        print(f"\n  Task {i+1}/{len(domain_order)}: {domain} ({len(batches)} batches)")
        
        exp_tscores = []
        for ep in range(epochs):
            model.train()
            ep_loss = 0.0
            ep_pen = 0.0
            
            for bi, (x, y) in enumerate(batches):
                # T-Score: compute every 10 batches (not every batch)
                if use_tscore and bi % 10 == 0:
                    t = compute_tscore_fast(model, x, y, criterion)
                    exp_tscores.append(t)
                    if t < 0.1:
                        sleep_protocol(model)
                        total_sleep += 1
                
                optimizer.zero_grad()
                out = model(x).reshape(-1, model.vocab_size)
                loss = criterion(out, y.reshape(-1))
                
                if use_ewc and cumulative_fisher:
                    pen = ewc_penalty(model, cumulative_fisher, cumulative_old_params, ewc_lambda)
                    ep_pen += pen.item()
                    loss = loss + pen
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
            
            n = max(len(batches), 1)
            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                t_str = ""
                if exp_tscores:
                    t_str = f" T={sum(exp_tscores[-5:])/min(len(exp_tscores),5):.4f}"
                p_str = f" EWC={ep_pen/n:.4f}" if use_ewc and cumulative_fisher else ""
                print(f"    Ep {ep+1:>2}: loss={ep_loss/n:.4f}{p_str}{t_str}")
        
        losses_after_training[domain] = evaluate(model, batches, criterion)
        
        # Domain T-Score
        t = compute_tscore_fast(model, batches[0][0], batches[0][1], criterion)
        domain_tscores[domain] = t
        
        # Accumulate Fisher
        if use_ewc:
            fisher = compute_fisher(model, batches, criterion)
            if use_fisher_scaling:
                f_max_raw = max(f.max().item() for f in fisher.values())
                fisher = scale_fisher_global_max(fisher)
                print(f"    Fisher: raw_max={f_max_raw:.2e} → scaled to [0,1]")
            
            for n, f in fisher.items():
                if n in cumulative_fisher:
                    cumulative_fisher[n] = cumulative_fisher[n] + f
                else:
                    cumulative_fisher[n] = f.clone()
            
            cumulative_old_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    
    # Final evaluation
    print(f"\n  --- Identity Preservation ({name}) ---")
    forgetting = {}
    final_losses = {}
    for domain in domain_order[:-1]:
        final_loss = evaluate(model, domain_batches[domain], criterion)
        final_losses[domain] = final_loss
        forget = final_loss - losses_after_training[domain]
        forgetting[domain] = forget
        pct = (forget / max(abs(losses_after_training[domain]), 1e-8)) * 100
        print(f"    {domain:<25}: {losses_after_training[domain]:.4f} → {final_loss:.4f} (forgetting: {forget:+.4f}, {pct:+.1f}%)")
    
    avg_forgetting = sum(forgetting.values()) / max(len(forgetting), 1)
    print(f"\n  AVERAGE FORGETTING: {avg_forgetting:+.4f}")
    if use_tscore:
        print(f"  Sleep Events: {total_sleep}")
    
    return {
        "condition": name,
        "forgetting": {k: round(v, 6) for k, v in forgetting.items()},
        "avg_forgetting": round(avg_forgetting, 6),
        "losses_after_training": {k: round(v, 6) for k, v in losses_after_training.items()},
        "final_losses": {k: round(v, 6) for k, v in final_losses.items()},
        "t_scores": {k: round(v, 4) for k, v in domain_tscores.items()},
        "sleep_count": total_sleep,
        "param_count": sum(p.numel() for p in model.parameters()),
    }


def main():
    print("=" * 70)
    print("GodelAI Conflict Data Proof v2 — The Definitive Benchmark")
    print("We prove it ourselves first, on our own data.")
    print(f"Author: L (GodelAI CEO) — MACP v2.2 'Identity'")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    domain_texts = load_conflict_domains()
    all_text = " ".join(domain_texts.values())
    char2idx, idx2char = build_vocab([all_text])
    V = len(char2idx)
    print(f"\n  Vocabulary: {V} characters")
    
    domain_order = ["Contradictory Facts", "Ethical Dilemmas", "Perspective Conflicts", "Temporal Conflicts"]
    domain_batches = {}
    for domain in domain_order:
        seqs = encode_text(domain_texts[domain], char2idx)
        batches = make_batches(seqs)
        domain_batches[domain] = batches
        print(f"  {domain}: {len(domain_texts[domain])} chars → {len(seqs)} seqs → {len(batches)} batches")
    
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 15
    LR = 0.003
    EWC_LAMBDA = 2.0
    
    # Condition 1: Naive
    r1 = run_condition(
        "Naive (No Protection)", domain_order, domain_batches, V, criterion,
        epochs=EPOCHS, lr=LR,
    )
    
    # Condition 2: Standard EWC (raw Fisher — Avalanche equivalent)
    r2 = run_condition(
        "Standard EWC (Raw Fisher)", domain_order, domain_batches, V, criterion,
        use_ewc=True, use_fisher_scaling=False,
        ewc_lambda=EWC_LAMBDA, epochs=EPOCHS, lr=LR,
    )
    
    # Condition 3: GodelAI-EWC (Full C-S-P Stack)
    r3 = run_condition(
        "GodelAI-EWC (Full C-S-P Stack)", domain_order, domain_batches, V, criterion,
        use_ewc=True, use_fisher_scaling=True, use_tscore=True,
        ewc_lambda=EWC_LAMBDA, epochs=EPOCHS, lr=LR,
    )
    
    # ============================================================
    # VERDICT
    # ============================================================
    
    print(f"\n{'='*70}")
    print("GODELAI CONFLICT DATA PROOF — FINAL RESULTS")
    print(f"{'='*70}")
    
    nf = r1["avg_forgetting"]
    ef = r2["avg_forgetting"]
    gf = r3["avg_forgetting"]
    
    ewc_imp = ((nf - ef) / max(abs(nf), 1e-8)) * 100
    godel_imp = ((nf - gf) / max(abs(nf), 1e-8)) * 100
    godel_vs_ewc = ((ef - gf) / max(abs(ef), 1e-8)) * 100
    
    print(f"\n  {'Domain':<25} {'Naive':>10} {'Std EWC':>10} {'GodelAI':>10}")
    print(f"  {'-'*60}")
    for domain in domain_order[:-1]:
        print(f"  {domain:<25} {r1['forgetting'][domain]:>+10.4f} {r2['forgetting'][domain]:>+10.4f} {r3['forgetting'][domain]:>+10.4f}")
    print(f"  {'-'*60}")
    print(f"  {'AVERAGE':<25} {nf:>+10.4f} {ef:>+10.4f} {gf:>+10.4f}")
    
    print(f"\n  Forgetting Reduction vs Naive:")
    print(f"    Standard EWC:        {ewc_imp:+.1f}%")
    print(f"    GodelAI-EWC (C-S-P): {godel_imp:+.1f}%")
    print(f"    GodelAI vs Std EWC:  {godel_vs_ewc:+.1f}%")
    
    print(f"\n  T-Score by Domain (GodelAI):")
    for domain, t in r3["t_scores"].items():
        print(f"    {domain}: {t:.4f}")
    print(f"    Sleep Events: {r3['sleep_count']}")
    
    # Verdict
    print(f"\n{'='*70}")
    if godel_imp > 20:
        verdict = "GO"
        print(f"  VERDICT: {verdict}")
        print(f"  GodelAI-EWC: {godel_imp:+.1f}% forgetting reduction vs Naive")
        print(f"  GodelAI vs Standard EWC: {godel_vs_ewc:+.1f}%")
        print(f"  PROVEN. Ready for public release.")
    elif godel_imp > 5:
        verdict = "CONDITIONAL GO"
        print(f"  VERDICT: {verdict}")
        print(f"  GodelAI-EWC: {godel_imp:+.1f}% — positive but needs more work.")
    else:
        verdict = "NO-GO"
        print(f"  VERDICT: {verdict}")
        print(f"  GodelAI-EWC: {godel_imp:+.1f}% — insufficient for public claims.")
    print(f"{'='*70}")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment": "GodelAI Conflict Data Proof v2",
        "timestamp": ts,
        "author": "L (GodelAI CEO) — MACP v2.2",
        "verdict": verdict,
        "config": {"epochs": EPOCHS, "lr": LR, "ewc_lambda": EWC_LAMBDA, "model": "ConflictGRU(128,2)"},
        "results": {"naive": r1, "standard_ewc": r2, "godelai": r3},
        "summary": {
            "naive_avg_forgetting": nf,
            "standard_ewc_avg_forgetting": ef,
            "godelai_avg_forgetting": gf,
            "godelai_vs_naive_pct": round(godel_imp, 2),
            "godelai_vs_ewc_pct": round(godel_vs_ewc, 2),
        },
    }
    out = Path("results") / f"conflict_data_proof_v2_{ts}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved: {out}")
    
    return results

if __name__ == "__main__":
    main()
