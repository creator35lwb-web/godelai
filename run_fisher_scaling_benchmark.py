"""
GodelAI Fisher Scaling Validation Benchmark
============================================
Validates that Fisher scaling resolves the Fisher Scale Problem,
enabling meaningful EWC regularization at our current GRU scale.

Compares three conditions:
  1. No EWC (baseline)
  2. EWC with raw Fisher (scale problem)
  3. EWC with scaled Fisher (GlobalMaxNorm)

Author: L (GodelAI CEO) — MACP v2.2 "Identity"
Date: April 3, 2026
"""

import torch
import torch.nn as nn
import json
import random
from pathlib import Path
from datetime import datetime
from godelai.reg.fisher_scaling import scale_fisher, diagnose_ewc_activation

torch.manual_seed(42)
random.seed(42)


def load_conflict_texts(d):
    texts = []
    for f in Path(d).rglob("*.json"):
        if f.name == "generation_summary.json":
            continue
        try:
            data = json.loads(f.read_text())
            for item in data.get("data", []):
                if "fact_a" in item:
                    texts += [item["fact_a"]["statement"], item["fact_b"]["statement"]]
                elif "scenario" in item:
                    texts.append(item["scenario"])
                    if "option_a" in item:
                        texts += [item["option_a"]["justification"], item["option_b"]["justification"]]
                elif "issue" in item:
                    texts.append(item["issue"])
                    for p in item.get("perspectives", []):
                        texts.append(p.get("reasoning", ""))
                elif "timeline" in item:
                    for t in item.get("timeline", []):
                        texts.append(t.get("belief", ""))
        except Exception:
            pass
    return [t.strip() for t in texts if t.strip()]


def encode(text, vocab, seq_len=50):
    enc = [vocab.get(c, 0) for c in text]
    seqs = []
    for i in range(0, len(enc) - seq_len - 1, seq_len):
        chunk = enc[i:i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        seqs.append((torch.tensor(chunk[:seq_len]), torch.tensor(chunk[1:])))
    return seqs


def make_batches(seqs, bs=32):
    batches = []
    for i in range(0, len(seqs), bs):
        b = seqs[i:i + bs]
        batches.append((torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])))
    return batches


class GRU(nn.Module):
    def __init__(self, V, H=128, L=2):
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.gru = nn.GRU(H, H, L, batch_first=True)
        self.fc = nn.Linear(H, V)

    def forward(self, x):
        return self.fc(self.gru(self.emb(x))[0])


def evaluate(m, batches, crit):
    m.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in batches:
            out = m(x).reshape(-1, m.fc.out_features)
            total += crit(out, y.reshape(-1)).item()
    return total / max(len(batches), 1)


def compute_fisher(m, batches, crit, n=80):
    m.eval()
    F = {nm: torch.zeros_like(p) for nm, p in m.named_parameters() if p.requires_grad}
    cnt = 0
    for x, y in batches:
        if cnt >= n:
            break
        m.zero_grad()
        out = m(x).reshape(-1, m.fc.out_features)
        crit(out, y.reshape(-1)).backward()
        for nm, p in m.named_parameters():
            if p.requires_grad and p.grad is not None:
                F[nm] += p.grad.data.pow(2)
        cnt += 1
    return {k: v / max(cnt, 1) for k, v in F.items()}


def ewc_pen(m, F, old, lam):
    pen = torch.tensor(0.0)
    for nm, p in m.named_parameters():
        if nm in F:
            pen += (lam * F[nm] * (p - old[nm]).pow(2)).sum()
    return pen


def train_ep(m, batches, crit, opt, F=None, old=None, lam=0.4):
    m.train()
    tl, pl = 0.0, 0.0
    for x, y in batches:
        opt.zero_grad()
        out = m(x).reshape(-1, m.fc.out_features)
        loss = crit(out, y.reshape(-1))
        if F and old:
            pen = ewc_pen(m, F, old, lam)
            pl += pen.item()
            loss = loss + pen
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        tl += loss.item()
    n = max(len(batches), 1)
    return tl / n, pl / n


def main():
    print("=" * 70)
    print("GodelAI Fisher Scaling Validation Benchmark v1.0")
    print("No EWC | EWC (raw Fisher) | EWC + Fisher Scaling (GlobalMaxNorm)")
    print(f"Author: L (GodelAI CEO) — MACP v2.2 'Identity'")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    conflict_texts = load_conflict_texts("datasets/conflict")
    shk_text = Path("datasets/shakespeare_full.txt").read_text(errors="ignore")[:80000]
    random.shuffle(conflict_texts)

    all_text = " ".join(conflict_texts) + shk_text
    vocab = {c: i for i, c in enumerate(sorted(set(all_text)))}
    V = len(vocab)
    print(f"\n  Vocab: {V} | Conflict texts: {len(conflict_texts)}")

    conflict_full = " ".join(conflict_texts)
    task_a = make_batches(encode(conflict_full, vocab))
    task_b = make_batches(encode(shk_text, vocab))
    print(f"  Task A (conflict): {len(task_a)} batches | Task B (Shakespeare): {len(task_b)} batches")

    crit = nn.CrossEntropyLoss()
    results = []

    conditions = [
        ("No EWC (baseline)", None, 0.4),
        ("EWC (raw Fisher, λ=0.4)", False, 0.4),
        ("EWC + Fisher Scaling (λ=2.0)", True, 2.0),
    ]

    for name, use_scaling, lam in conditions:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        m = GRU(V)
        opt = torch.optim.Adam(m.parameters(), lr=0.001)

        print(f"\n  Phase 1: Task A (10 epochs)")
        for ep in range(10):
            tl, _ = train_ep(m, task_a, crit, opt)
            if (ep + 1) % 5 == 0:
                print(f"    Ep {ep+1:2d}: loss={tl:.4f}")

        la1 = evaluate(m, task_a, crit)
        print(f"\n  Task A final loss: {la1:.4f}")

        F, old = None, None
        if use_scaling is not None:
            F_raw = compute_fisher(m, task_a, crit, 80)
            old = {nm: p.data.clone() for nm, p in m.named_parameters() if p.requires_grad}

            diag = diagnose_ewc_activation(m, F_raw, lam)
            print(f"\n  Fisher Diagnosis:")
            print(f"    Raw Fisher max:     {diag['fisher_max']:.2e}")
            print(f"    Scale problem:      {diag['scale_problem_detected']}")
            print(f"    Est. penalty (raw): {diag['estimated_penalty_at_delta_001']:.8f}")

            if use_scaling:
                F = scale_fisher(F_raw, strategy="global_max")
                diag2 = diagnose_ewc_activation(m, F, lam)
                print(f"    Scaled Fisher max:  {diag2['fisher_max']:.4f}")
                print(f"    Est. penalty (scaled): {diag2['estimated_penalty_at_delta_001']:.6f}")
                print(f"    Scale problem resolved: {not diag2['scale_problem_detected']}")
            else:
                F = F_raw

        print(f"\n  Phase 2: Task B (10 epochs)")
        for ep in range(10):
            tl, pl = train_ep(m, task_b, crit, opt, F, old, lam)
            if (ep + 1) % 5 == 0:
                print(f"    Ep {ep+1:2d}: task={tl:.4f} pen={pl:.6f}")

        la2 = evaluate(m, task_a, crit)
        lb = evaluate(m, task_b, crit)
        forget = la2 - la1
        print(f"\n  Task A: {la1:.4f} → {la2:.4f} (forgetting: {forget:+.4f})")
        print(f"  Task B final: {lb:.4f}")
        results.append({"name": name, "la1": la1, "la2": la2, "lb": lb, "forget": forget})

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    baseline = results[0]["forget"]
    print(f"\n{'Condition':<35} {'Forgetting':>12} {'Improvement':>14}")
    print("-" * 65)

    for r in results:
        if r["name"].startswith("No EWC"):
            tag = "baseline"
        else:
            pct = ((baseline - r["forget"]) / max(abs(baseline), 1e-8)) * 100
            tag = f"{pct:+.1f}%"
        print(f"  {r['name']:<33} {r['forget']:>+12.4f} {tag:>14}")

    raw_pct = ((baseline - results[1]["forget"]) / max(abs(baseline), 1e-8)) * 100
    scaled_pct = ((baseline - results[2]["forget"]) / max(abs(baseline), 1e-8)) * 100

    print(f"\n  EWC (raw Fisher) improvement:    {raw_pct:+.1f}%")
    print(f"  EWC (scaled Fisher) improvement: {scaled_pct:+.1f}%")

    if scaled_pct > raw_pct:
        print(f"\n  Fisher scaling provides {scaled_pct - raw_pct:.1f}% additional improvement")
        print(f"  Recommendation: Use Fisher scaling in all future EWC runs")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "experiment": "Fisher Scaling Validation",
        "timestamp": ts,
        "author": "L (GodelAI CEO) — MACP v2.2 'Identity'",
        "results": results,
        "summary": {
            "baseline_forget": results[0]["forget"],
            "raw_ewc_forget": results[1]["forget"],
            "scaled_ewc_forget": results[2]["forget"],
            "raw_ewc_improvement_pct": raw_pct,
            "scaled_ewc_improvement_pct": scaled_pct,
            "fisher_scaling_additional_improvement": scaled_pct - raw_pct,
        },
    }
    p = Path(f"results/fisher_scaling_{ts}.json")
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {p}")
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    return out


if __name__ == "__main__":
    main()
