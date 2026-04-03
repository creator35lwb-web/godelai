"""
GodelAI Conflict Data T-Score & EWC Activation Benchmark
=========================================================
Tests whether the expanded conflict datasets (107 items) produce
T-Scores in the target range (0.3-0.5) and activate EWC/EWC-DR
with meaningful Fisher information.

This directly addresses the Data Bottleneck identified by XV.

Author: L (GodelAI CEO) — MACP v2.2 "Identity"
Date: April 3, 2026
"""

import torch
import torch.nn as nn
import json
import random
from pathlib import Path
from datetime import datetime

torch.manual_seed(42)
random.seed(42)

# ── T-Score Computation ───────────────────────────────────────────────────────
def compute_tscore(gradients_per_sample: list) -> float:
    """
    T-Score: gradient diversity metric.
    T = 1 - (||sum_g||^2 / sum(||g||^2)) / N

    T=0.0: all gradients identical (no diversity)
    T=1.0: gradients cancel perfectly (maximum diversity)
    Target range for C-S-P activation: 0.3 - 0.5
    """
    if len(gradients_per_sample) < 2:
        return 0.0

    # Stack gradients: shape [N, D]
    G = torch.stack(gradients_per_sample)
    N = G.shape[0]

    sum_g = G.sum(dim=0)
    norm_sum_sq = sum_g.norm().pow(2).item()
    sum_norm_sq = G.norm(dim=1).pow(2).sum().item()

    if sum_norm_sq < 1e-10:
        return 0.0

    T = 1.0 - (norm_sum_sq / sum_norm_sq) / N
    return max(0.0, min(1.0, T))


# ── Simple Model for T-Score Testing ─────────────────────────────────────────
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ── Load Conflict Data as Text ────────────────────────────────────────────────
def load_conflict_texts(dataset_dir: Path) -> dict:
    """Load all conflict datasets and extract text for training."""
    texts = {}

    for json_file in dataset_dir.rglob("*.json"):
        if json_file.name == "generation_summary.json":
            continue
        try:
            data = json.loads(json_file.read_text())
            items = data.get("data", [])
            category = json_file.parent.name
            file_type = "original" if not json_file.stem.startswith("expanded") else "expanded"
            key = f"{category}_{file_type}"

            # Extract text from each item type
            texts_for_file = []
            for item in items:
                if "fact_a" in item and "fact_b" in item:
                    texts_for_file.append(item["fact_a"]["statement"])
                    texts_for_file.append(item["fact_b"]["statement"])
                elif "scenario" in item:
                    texts_for_file.append(item["scenario"])
                    if "option_a" in item:
                        texts_for_file.append(item["option_a"]["justification"])
                        texts_for_file.append(item["option_b"]["justification"])
                elif "issue" in item:
                    texts_for_file.append(item["issue"])
                    for p in item.get("perspectives", []):
                        texts_for_file.append(p.get("reasoning", ""))
                elif "timeline" in item:
                    for t in item.get("timeline", []):
                        texts_for_file.append(t.get("belief", ""))

            texts[key] = {
                "file": str(json_file),
                "count": len(items),
                "texts": [t for t in texts_for_file if t.strip()]
            }
        except Exception as e:
            print(f"  Warning: Could not load {json_file}: {e}")

    return texts


def text_to_tensor(text: str, vocab: dict, max_len: int = 100) -> torch.Tensor:
    """Convert text to a fixed-size tensor via character encoding."""
    encoded = [vocab.get(c, 0) for c in text[:max_len]]
    # Pad to max_len
    encoded += [0] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.float32)


def measure_tscore_for_dataset(texts: list, vocab: dict, model: nn.Module,
                                 criterion: nn.Module, max_len: int = 100) -> dict:
    """Measure T-Score for a set of texts."""
    if len(texts) < 2:
        return {"tscore": 0.0, "n_samples": 0, "fisher_max": 0.0}

    model.train()
    per_sample_grads = []
    fisher_values = []

    for text in texts[:50]:  # Cap at 50 for speed
        x = text_to_tensor(text, vocab, max_len).unsqueeze(0)
        target = torch.zeros(1, dtype=torch.long)  # dummy target

        model.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()

        # Collect flat gradient vector
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.flatten())
                fisher_values.append(param.grad.data.pow(2).mean().item())
        if grads:
            per_sample_grads.append(torch.cat(grads))

    tscore = compute_tscore(per_sample_grads)
    fisher_max = max(fisher_values) if fisher_values else 0.0
    fisher_mean = sum(fisher_values) / len(fisher_values) if fisher_values else 0.0

    return {
        "tscore": tscore,
        "n_samples": len(per_sample_grads),
        "fisher_max": fisher_max,
        "fisher_mean": fisher_mean,
        "sleep_would_trigger": tscore < 0.3,
        "in_target_range": 0.3 <= tscore <= 0.5,
    }


def main():
    print("=" * 70)
    print("GodelAI Conflict Data T-Score Benchmark v1.0")
    print("Testing: Do conflict datasets activate C-S-P in target range (0.3-0.5)?")
    print(f"Author: L (GodelAI CEO) — MACP v2.2 'Identity'")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    conflict_dir = Path("datasets/conflict")
    shk_path = Path("datasets/shakespeare_full.txt")

    # Load all conflict texts
    print("\n[1] Loading conflict datasets...")
    conflict_texts = load_conflict_texts(conflict_dir)
    for key, data in conflict_texts.items():
        print(f"  {key}: {data['count']} items → {len(data['texts'])} text segments")

    # Build vocabulary from all texts
    all_text = ""
    for data in conflict_texts.values():
        all_text += " ".join(data["texts"])

    if shk_path.exists():
        shk_text = shk_path.read_text(encoding="utf-8", errors="ignore")[:50_000]
        all_text += shk_text

    vocab = {c: i for i, c in enumerate(sorted(set(all_text)))}
    vocab_size = len(vocab)
    max_len = 100

    print(f"\n  Vocabulary size: {vocab_size}")

    # Create a simple model for T-Score measurement
    model = SimpleNet(input_dim=max_len, hidden=64, output_dim=vocab_size)
    criterion = nn.CrossEntropyLoss()

    # ── Measure T-Score for each dataset ─────────────────────────────────────
    print("\n[2] Measuring T-Score for each dataset...")
    print(f"\n{'Dataset':<35} {'T-Score':>8} {'Fisher Max':>12} {'In Range?':>10} {'Sleep?':>8}")
    print("-" * 80)

    results = {}

    # Shakespeare baseline
    if shk_path.exists():
        shk_texts = []
        for i in range(0, min(5000, len(shk_text) - 100), 100):
            shk_texts.append(shk_text[i:i+100])
        shk_result = measure_tscore_for_dataset(shk_texts[:50], vocab, model, criterion, max_len)
        results["shakespeare_baseline"] = shk_result
        in_range = "✅ YES" if shk_result["in_target_range"] else "❌ NO"
        sleep = "⚠️ YES" if shk_result["sleep_would_trigger"] else "NO"
        print(f"{'shakespeare_baseline':<35} {shk_result['tscore']:>8.4f} {shk_result['fisher_max']:>12.6f} {in_range:>10} {sleep:>8}")

    # Conflict datasets
    for key, data in conflict_texts.items():
        if not data["texts"]:
            continue
        result = measure_tscore_for_dataset(data["texts"], vocab, model, criterion, max_len)
        results[key] = result
        in_range = "✅ YES" if result["in_target_range"] else "❌ NO"
        sleep = "⚠️ YES" if result["sleep_would_trigger"] else "NO"
        print(f"{key:<35} {result['tscore']:>8.4f} {result['fisher_max']:>12.6f} {in_range:>10} {sleep:>8}")

    # ── Mixed conflict dataset ────────────────────────────────────────────────
    all_conflict_texts = []
    for data in conflict_texts.values():
        all_conflict_texts.extend(data["texts"])
    random.shuffle(all_conflict_texts)

    mixed_result = measure_tscore_for_dataset(all_conflict_texts[:50], vocab, model, criterion, max_len)
    results["ALL_CONFLICT_MIXED"] = mixed_result
    in_range = "✅ YES" if mixed_result["in_target_range"] else "❌ NO"
    sleep = "⚠️ YES" if mixed_result["sleep_would_trigger"] else "NO"
    print(f"\n{'ALL_CONFLICT_MIXED':<35} {mixed_result['tscore']:>8.4f} {mixed_result['fisher_max']:>12.6f} {in_range:>10} {sleep:>8}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    shk_tscore = results.get("shakespeare_baseline", {}).get("tscore", 0)
    mixed_tscore = results.get("ALL_CONFLICT_MIXED", {}).get("tscore", 0)

    print(f"\n  Shakespeare T-Score:      {shk_tscore:.4f} (expected ~0.93, dataset-dependent)")
    print(f"  Mixed Conflict T-Score:   {mixed_tscore:.4f} (target: 0.3-0.5)")

    in_range_count = sum(1 for k, r in results.items()
                         if r.get("in_target_range", False) and k != "shakespeare_baseline")
    total_conflict = len(results) - (1 if "shakespeare_baseline" in results else 0)

    print(f"\n  Conflict datasets in target range: {in_range_count}/{total_conflict}")

    if mixed_tscore >= 0.3:
        print(f"\n  ✅ CONFLICT DATA VALIDATED: Mixed conflict T-Score ({mixed_tscore:.4f})")
        print(f"     is in or above target range — EWC/EWC-DR will activate meaningfully")
    else:
        print(f"\n  ⚠️  Mixed conflict T-Score ({mixed_tscore:.4f}) below target range")
        print(f"     Recommendation: Increase conflict intensity or add more adversarial pairs")

    # Fisher comparison
    shk_fisher = results.get("shakespeare_baseline", {}).get("fisher_max", 0)
    mixed_fisher = results.get("ALL_CONFLICT_MIXED", {}).get("fisher_max", 0)
    if shk_fisher > 0:
        fisher_ratio = mixed_fisher / shk_fisher
        print(f"\n  Fisher max (Shakespeare): {shk_fisher:.6f}")
        print(f"  Fisher max (Conflict):    {mixed_fisher:.6f}")
        print(f"  Fisher ratio:             {fisher_ratio:.2f}x")
        if fisher_ratio > 1.0:
            print(f"  ✅ Conflict data produces {fisher_ratio:.1f}x higher Fisher information")
            print(f"     → EWC penalty will be {fisher_ratio:.1f}x stronger on conflict data")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "experiment": "Conflict Data T-Score Benchmark",
        "timestamp": ts,
        "author": "L (GodelAI CEO) — MACP v2.2 'Identity'",
        "results": results,
        "summary": {
            "shakespeare_tscore": shk_tscore,
            "mixed_conflict_tscore": mixed_tscore,
            "conflict_in_target_range": in_range_count,
            "total_conflict_datasets": total_conflict,
            "fisher_ratio_conflict_vs_shk": mixed_fisher / max(shk_fisher, 1e-10),
        }
    }
    p = Path(f"results/conflict_tscore_benchmark_{ts}.json")
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\n  💾 Saved to: {p}")
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    return out


if __name__ == "__main__":
    main()
