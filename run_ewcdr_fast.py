"""
GodelAI EWC-DR Fast Benchmark
==============================
Replicates the original run_godel_ewc.py methodology exactly,
then adds EWC-DR as a third condition.

Uses the same Task A / Task B sequential learning setup that
previously validated 21.6% forgetting reduction.

Author: L (GodelAI CEO) — MACP v2.2 "Identity"
Date: April 3, 2026
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from datetime import datetime

torch.manual_seed(1337)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "seq_len": 50,
    "batch_size": 32,
    "hidden": 128,
    "layers": 2,
    "epochs_a": 10,
    "epochs_b": 10,
    "lr": 0.001,
    "ewc_lambda": 0.4,
    "dead_threshold": 1e-3,   # Fisher below this = "dead" parameter
    "reversal_strength": 0.05,
    "fisher_samples": 100,
}

# ── Model ─────────────────────────────────────────────────────────────────────
class GRU(nn.Module):
    def __init__(self, vocab_size, hidden, layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.fc  = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.fc(self.gru(self.emb(x))[0])

# ── Data ──────────────────────────────────────────────────────────────────────
def make_batches(text, vocab, seq_len, batch_size):
    enc = [vocab.get(c, 0) for c in text]
    seqs = []
    for i in range(0, len(enc) - seq_len - 1, seq_len):
        chunk = enc[i:i+seq_len+1]
        if len(chunk) < seq_len+1:
            break
        seqs.append((torch.tensor(chunk[:seq_len]), torch.tensor(chunk[1:])))
    batches = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        batches.append((xs, ys))
    return batches

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, batches, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in batches:
            out = model(x).reshape(-1, model.fc.out_features)
            total += criterion(out, y.reshape(-1)).item()
    return total / max(len(batches), 1)

# ── Fisher ────────────────────────────────────────────────────────────────────
def compute_fisher(model, batches, criterion, n=100):
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    count = 0
    for x, y in batches:
        if count >= n:
            break
        model.zero_grad()
        out = model(x).reshape(-1, model.fc.out_features)
        loss = criterion(out, y.reshape(-1))
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data.pow(2)
        count += 1
    return {k: v / max(count, 1) for k, v in fisher.items()}

# ── EWC Penalty ───────────────────────────────────────────────────────────────
def ewc_penalty(model, fisher, old_params, lam):
    penalty = torch.tensor(0.0)
    for name, param in model.named_parameters():
        if name in fisher:
            penalty += (lam * fisher[name] * (param - old_params[name]).pow(2)).sum()
    return penalty

# ── EWC-DR Penalty (Logits Reversal) ─────────────────────────────────────────
def ewcdr_penalty(model, fisher, old_params, lam, dead_thresh, reversal_str):
    """
    EWC-DR: Standard EWC for alive params, reversed penalty for dead params.
    Dead params (low Fisher) are ENCOURAGED to change freely.
    """
    alive_pen = torch.tensor(0.0)
    dead_rev  = torch.tensor(0.0)

    # Normalize Fisher for threshold comparison
    all_f = torch.cat([f.flatten() for f in fisher.values()])
    f_max = all_f.max().item()
    norm_thresh = dead_thresh * (f_max if f_max > 0 else 1.0)

    for name, param in model.named_parameters():
        if name not in fisher:
            continue
        f = fisher[name]
        delta_sq = (param - old_params[name]).pow(2)
        alive_mask = f >= norm_thresh
        dead_mask  = ~alive_mask

        if alive_mask.any():
            alive_pen = alive_pen + (lam * f * delta_sq)[alive_mask].sum()
        if dead_mask.any():
            dead_rev = dead_rev + (reversal_str * delta_sq)[dead_mask].sum()

    net = torch.clamp(alive_pen - dead_rev, min=0.0)
    return net, alive_pen.item(), dead_rev.item()

# ── Training Loop ─────────────────────────────────────────────────────────────
def train(model, batches, criterion, optimizer, fisher=None, old_params=None,
          mode="standard", lam=0.4, dead_thresh=1e-3, reversal_str=0.05):
    model.train()
    total_task = 0.0
    total_pen  = 0.0
    for x, y in batches:
        optimizer.zero_grad()
        out = model(x).reshape(-1, model.fc.out_features)
        task_loss = criterion(out, y.reshape(-1))

        if fisher is not None and old_params is not None:
            if mode == "ewc":
                pen = ewc_penalty(model, fisher, old_params, lam)
                total_pen += pen.item()
            elif mode == "ewcdr":
                pen, _, _ = ewcdr_penalty(model, fisher, old_params, lam,
                                          dead_thresh, reversal_str)
                total_pen += pen.item()
            else:
                pen = torch.tensor(0.0)
            loss = task_loss + pen
        else:
            loss = task_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_task += task_loss.item()

    n = max(len(batches), 1)
    return total_task / n, total_pen / n

# ── Main ──────────────────────────────────────────────────────────────────────
def run_condition(name, mode, task_a_batches, task_b_batches, vocab_size, criterion):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    torch.manual_seed(1337)
    model = GRU(vocab_size, CONFIG["hidden"], CONFIG["layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # Phase 1: Train on Task A
    print(f"\n  Phase 1: Task A ({CONFIG['epochs_a']} epochs)")
    for epoch in range(CONFIG["epochs_a"]):
        tl, _ = train(model, task_a_batches, criterion, optimizer)
        if (epoch+1) % 5 == 0:
            val = evaluate(model, task_a_batches, criterion)
            print(f"    Ep {epoch+1:2d}: loss={tl:.4f} val={val:.4f}")

    loss_a1 = evaluate(model, task_a_batches, criterion)
    print(f"\n  ✅ Task A loss after Phase 1: {loss_a1:.4f}")

    # Consolidate
    fisher, old_params = None, None
    if mode in ("ewc", "ewcdr"):
        print(f"  🔒 Computing Fisher ({CONFIG['fisher_samples']} samples)...")
        fisher = compute_fisher(model, task_a_batches, criterion, CONFIG["fisher_samples"])
        old_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

        # Report dead/alive split for EWC-DR
        if mode == "ewcdr":
            all_f = torch.cat([f.flatten() for f in fisher.values()])
            f_max = all_f.max().item()
            norm_thresh = CONFIG["dead_threshold"] * (f_max if f_max > 0 else 1.0)
            total_p = sum(f.numel() for f in fisher.values())
            dead_p  = sum((f < norm_thresh).sum().item() for f in fisher.values())
            print(f"     Fisher max: {f_max:.6f}")
            print(f"     Norm threshold: {norm_thresh:.6f}")
            print(f"     Dead params: {dead_p:,} / {total_p:,} ({dead_p/total_p*100:.1f}%)")
            print(f"     Alive params: {total_p-dead_p:,} / {total_p:,} ({(total_p-dead_p)/total_p*100:.1f}%)")

    # Phase 2: Train on Task B
    print(f"\n  Phase 2: Task B ({CONFIG['epochs_b']} epochs)")
    for epoch in range(CONFIG["epochs_b"]):
        tl, pen = train(model, task_b_batches, criterion, optimizer,
                        fisher, old_params, mode,
                        CONFIG["ewc_lambda"], CONFIG["dead_threshold"],
                        CONFIG["reversal_strength"])
        if (epoch+1) % 5 == 0:
            val_b = evaluate(model, task_b_batches, criterion)
            print(f"    Ep {epoch+1:2d}: task={tl:.4f} pen={pen:.4f} val_b={val_b:.4f}")

    loss_a2 = evaluate(model, task_a_batches, criterion)
    loss_b  = evaluate(model, task_b_batches, criterion)
    forgetting = loss_a2 - loss_a1

    print(f"\n  📊 Results:")
    print(f"     Task A after Phase 1: {loss_a1:.4f}")
    print(f"     Task A after Phase 2: {loss_a2:.4f}")
    print(f"     Task B final:         {loss_b:.4f}")
    print(f"     Forgetting:           {forgetting:+.4f}")

    return {"name": name, "mode": mode,
            "loss_a1": loss_a1, "loss_a2": loss_a2,
            "loss_b": loss_b, "forgetting": forgetting}


def main():
    print("=" * 70)
    print("GodelAI EWC-DR Fast Benchmark v1.0")
    print("Standard | Vanilla EWC | EWC-DR (Logits Reversal)")
    print(f"Author: L (GodelAI CEO) — MACP v2.2 'Identity'")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load Shakespeare
    shk_path = Path("datasets/shakespeare_full.txt")
    if shk_path.exists():
        text = shk_path.read_text(encoding="utf-8", errors="ignore")
        # 100KB each for speed on CPU
        task_a_text = text[:100_000]
        task_b_text = text[100_000:200_000]
        print(f"\n  Using Shakespeare: {len(task_a_text):,} + {len(task_b_text):,} chars")
    else:
        task_a_text = "The quick brown fox jumps over the lazy dog. " * 500
        task_b_text = "To be or not to be that is the question. " * 500
        print("\n  Using synthetic data")

    all_chars = sorted(set(task_a_text + task_b_text))
    vocab = {c: i for i, c in enumerate(all_chars)}
    vocab_size = len(vocab)
    print(f"  Vocabulary: {vocab_size} chars")

    task_a_batches = make_batches(task_a_text, vocab, CONFIG["seq_len"], CONFIG["batch_size"])
    task_b_batches = make_batches(task_b_text, vocab, CONFIG["seq_len"], CONFIG["batch_size"])
    print(f"  Task A batches: {len(task_a_batches)}, Task B batches: {len(task_b_batches)}")

    criterion = nn.CrossEntropyLoss()
    results = []

    results.append(run_condition("Standard (No EWC)", "standard",
                                  task_a_batches, task_b_batches, vocab_size, criterion))
    results.append(run_condition("Vanilla EWC", "ewc",
                                  task_a_batches, task_b_batches, vocab_size, criterion))
    results.append(run_condition("EWC-DR (Logits Reversal)", "ewcdr",
                                  task_a_batches, task_b_batches, vocab_size, criterion))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    baseline = results[0]["forgetting"]
    print(f"\n{'Condition':<30} {'Forgetting':>12} {'vs Baseline':>14} {'Task B Loss':>12}")
    print("-" * 72)

    for r in results:
        if r["mode"] == "standard":
            vs = "baseline"
        else:
            pct = ((baseline - r["forgetting"]) / max(abs(baseline), 1e-8)) * 100
            vs = f"{pct:+.1f}%"
        print(f"{r['name']:<30} {r['forgetting']:>+12.4f} {vs:>14} {r['loss_b']:>12.4f}")

    ewc_pct   = ((baseline - results[1]["forgetting"]) / max(abs(baseline), 1e-8)) * 100
    ewcdr_pct = ((baseline - results[2]["forgetting"]) / max(abs(baseline), 1e-8)) * 100

    print(f"\n  Vanilla EWC improvement:  {ewc_pct:+.1f}%")
    print(f"  EWC-DR improvement:       {ewcdr_pct:+.1f}%")

    if results[2]["forgetting"] < results[1]["forgetting"]:
        delta = results[1]["forgetting"] - results[2]["forgetting"]
        print(f"\n  ✅ EWC-DR outperforms Vanilla EWC by {delta:.4f} absolute forgetting")
    else:
        delta = results[2]["forgetting"] - results[1]["forgetting"]
        print(f"\n  ⚠️  Vanilla EWC outperforms EWC-DR by {delta:.4f}")
        print(f"     Recommendation: Tune dead_threshold (currently {CONFIG['dead_threshold']})")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "experiment": "EWC-DR Fast Benchmark",
        "timestamp": ts,
        "author": "L (GodelAI CEO) — MACP v2.2 'Identity'",
        "config": CONFIG,
        "results": results,
        "summary": {
            "baseline_forgetting": baseline,
            "vanilla_ewc_forgetting": results[1]["forgetting"],
            "ewc_dr_forgetting": results[2]["forgetting"],
            "vanilla_ewc_improvement_pct": ewc_pct,
            "ewc_dr_improvement_pct": ewcdr_pct,
            "ewc_dr_beats_vanilla": results[2]["forgetting"] < results[1]["forgetting"],
        }
    }
    p = Path(f"results/ewc_dr_fast_{ts}.json")
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\n  💾 Saved to: {p}")
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    return out


if __name__ == "__main__":
    main()
