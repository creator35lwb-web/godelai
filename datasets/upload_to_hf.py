"""
Upload GodelAI Conflict Dataset to HuggingFace: YSenseAI/godelai-conflict-data
Combines original (v1.0, 22 items) + expanded (v2.0, 85 items) = 107 total
Produces: data/train.jsonl + split JSONL files per category + dataset card
"""

import json
import os
import tempfile
from pathlib import Path
from huggingface_hub import HfApi

BASE = Path(__file__).parent / "conflict"
REPO_ID = "YSenseAI/godelai-conflict-data"

# ── File map: category → [v1 file, v2 file] ───────────────────────────────────
FILES = {
    "contradictory_facts":  ["contradictory_facts/scientific_paradoxes.json",
                              "contradictory_facts/expanded_paradoxes.json"],
    "ethical_dilemmas":     ["ethical_dilemmas/core_dilemmas.json",
                              "ethical_dilemmas/expanded_dilemmas.json"],
    "perspective_conflicts":["perspective_conflicts/ai_governance.json",
                              "perspective_conflicts/expanded_perspectives.json"],
    "temporal_conflicts":   ["temporal_conflicts/evolving_knowledge.json",
                              "temporal_conflicts/expanded_temporal.json"],
}


def load_items(path):
    with open(BASE / path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("data", [])


# ── Load and merge all items ───────────────────────────────────────────────────
all_items = []
by_category = {}
for category, (v1_file, v2_file) in FILES.items():
    items = load_items(v1_file) + load_items(v2_file)
    # Deduplicate by id
    seen = set()
    unique = []
    for item in items:
        iid = item.get("id", "")
        if iid not in seen:
            seen.add(iid)
            item["category"] = category
            unique.append(item)
    by_category[category] = unique
    all_items.extend(unique)
    print(f"  {category}: {len(unique)} items")

print(f"\nTotal: {len(all_items)} items across {len(by_category)} categories")

# ── Write JSONL files to temp dir ─────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp())

# train.jsonl — full dataset
train_path = tmp / "data" / "train.jsonl"
train_path.parent.mkdir(parents=True, exist_ok=True)
with open(train_path, "w", encoding="utf-8") as f:
    for item in all_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"  train.jsonl: {len(all_items)} rows")

# Per-category JSONL files
for category, items in by_category.items():
    cat_path = tmp / "data" / f"{category}.jsonl"
    with open(cat_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {category}.jsonl: {len(items)} rows")

# ── Dataset card README.md ─────────────────────────────────────────────────────
counts = {c: len(items) for c, items in by_category.items()}
readme = f"""\
---
license: apache-2.0
language:
- en
tags:
- continual-learning
- catastrophic-forgetting
- conflict-data
- gradient-diversity
- t-score
- godelai
- csp-framework
- ai-safety
pretty_name: GodelAI Conflict Dataset
size_categories:
- n<1K
task_categories:
- text-classification
---

# GodelAI Conflict Dataset

**107 semantically contradictory sentence pairs** for identity stress testing in continual learning.

Designed to activate the C-S-P framework's T-score target range (T = 0.3–0.5), validated to produce **12× more catastrophic forgetting** than homogeneous text — making it the correct training regime for demonstrating memory-protection value.

## Key Results (GodelAI Framework)

| Method | Avg Forgetting | vs Naive |
|--------|:-:|:-:|
| Naive (No Protection) | +1.8364 | baseline |
| Standard EWC (raw Fisher) | +1.8017 | +1.9% |
| **GodelAI-EWC (Full C-S-P)** | **+0.3163** | **+82.8%** |

**82.8% forgetting reduction** — 43× over Standard EWC. Fisher Scale Problem confirmed: raw EWC is silently broken at small model scale.

## Dataset Structure

| Category | Items | T-Score | Description |
|----------|------:|:-------:|-------------|
| `contradictory_facts` | {counts['contradictory_facts']} | 0.4075 | Scientific paradoxes with dual valid interpretations |
| `ethical_dilemmas` | {counts['ethical_dilemmas']} | 0.3626 | Multi-framework moral conflicts |
| `perspective_conflicts` | {counts['perspective_conflicts']} | 0.3773 | AI governance and philosophical disagreements |
| `temporal_conflicts` | {counts['temporal_conflicts']} | 0.3530 | Evolving scientific knowledge over time |
| **Total** | **{len(all_items)}** | **0.4126** | **All in C-S-P target range ✅** |

## Files

| File | Description |
|------|-------------|
| `data/train.jsonl` | Full dataset — all {len(all_items)} items |
| `data/contradictory_facts.jsonl` | {counts['contradictory_facts']} items |
| `data/ethical_dilemmas.jsonl` | {counts['ethical_dilemmas']} items |
| `data/perspective_conflicts.jsonl` | {counts['perspective_conflicts']} items |
| `data/temporal_conflicts.jsonl` | {counts['temporal_conflicts']} items |

## Item Schema

```json
{{
  "id": "contradiction_expanded_001",
  "type": "contradictory_facts",
  "category": "contradictory_facts",
  "domain": "physics",
  "title": "Wave-Particle Duality of Light",
  "fact_a": {{"statement": "...", "evidence": "..."}},
  "fact_b": {{"statement": "...", "evidence": "..."}},
  "resolution": "...",
  "conflict_intensity": "high"
}}
```

## Usage

```python
from datasets import load_dataset

# Full dataset
ds = load_dataset("YSenseAI/godelai-conflict-data", data_files="data/train.jsonl", split="train")

# Single category
ds = load_dataset("YSenseAI/godelai-conflict-data", data_files="data/ethical_dilemmas.jsonl", split="train")
```

## Background

This dataset was developed as part of the **GodelAI C-S-P Framework** for continual learning identity preservation. Standard homogeneous text (e.g., Shakespeare) produces T-scores outside the 0.3–0.5 activation range, preventing meaningful evaluation of regularisation-based methods. Conflict data was engineered to sit precisely in the activation range.

**Framework paper:** [10.5281/zenodo.19927649](https://doi.org/10.5281/zenodo.19927649)
**Framework code:** [creator35lwb-web/godelai](https://github.com/creator35lwb-web/godelai)
**Model card:** [YSenseAI/godelai-manifesto-v1](https://huggingface.co/YSenseAI/godelai-manifesto-v1)

## Citation

```bibtex
@software{{godelai2026,
  title  = {{GodelAI: A C-S-P Framework for Continual Learning and Wisdom-Preserving Language Models}},
  author = {{Lee, Alton Wei Bin and {{L (GodelAI C-S-P Agent)}} and {{Rk (RNA / Claude Code)}}}},
  year   = {{2026}},
  doi    = {{10.5281/zenodo.19886315}},
  url    = {{https://github.com/creator35lwb-web/godelai}}
}}
```

## License

Apache 2.0 — open for research and commercial use.

*Created by the FLYWHEEL TEAM under MACP v2.2 Identity protocol.*
"""

readme_path = tmp / "README.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)
print(f"  README.md written ({len(readme)} chars)")

# ── Upload to HuggingFace ──────────────────────────────────────────────────────
print(f"\nCreating HF dataset repo: {REPO_ID} ...")
api = HfApi()

# Create repo (public, dataset type)
try:
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False, exist_ok=True)
    print(f"  Repo created/confirmed: {REPO_ID}")
except Exception as e:
    print(f"  Repo create note: {e}")

# Upload all files
uploads = [
    (readme_path, "README.md"),
    (train_path,  "data/train.jsonl"),
]
for category in by_category:
    uploads.append((tmp / "data" / f"{category}.jsonl", f"data/{category}.jsonl"))

for local_path, repo_path in uploads:
    print(f"  Uploading {repo_path} ({os.path.getsize(local_path):,} bytes)...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Add {repo_path}",
    )

print(f"\n{'='*60}")
print(f"  DATASET PUBLISHED!")
print(f"  URL: https://huggingface.co/datasets/{REPO_ID}")
print(f"  Items: {len(all_items)} total ({len(by_category)} categories)")
print(f"{'='*60}")
