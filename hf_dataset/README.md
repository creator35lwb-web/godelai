---
license: apache-2.0
task_categories:
  - text-generation
  - text-classification
language:
  - en
tags:
  - continual-learning
  - catastrophic-forgetting
  - ewc
  - identity-preservation
  - conflict-data
  - godelai
  - csp-framework
  - ysenseai
size_categories:
  - n<1K
pretty_name: GodelAI Conflict Dataset
dataset_info:
  features:
    - name: id
      dtype: string
    - name: category
      dtype: string
    - name: domain
      dtype: string
    - name: type
      dtype: string
    - name: text
      dtype: string
    - name: source
      dtype: string
    - name: raw_json
      dtype: string
  splits:
    - name: train
      num_examples: 107
---

# GodelAI Conflict Dataset v2.0

**The first open-source dataset designed specifically for testing identity preservation in continual learning systems.**

## Overview

This dataset contains 107 carefully crafted conflict scenarios across 4 categories, designed to produce high gradient diversity (T-Score 0.3–0.5) during sequential training — the exact conditions where catastrophic forgetting is most dangerous and identity preservation matters most.

GodelAI's C-S-P (Compression-State-Propagation) framework uses this data to achieve **82.8% forgetting reduction** vs naive baselines and **82.4% improvement** over standard EWC in domain-incremental learning.

## Dataset Structure

| Category | Items | Avg Text Length | Description |
|----------|:-----:|:---------------:|-------------|
| Contradictory Facts | 26 | 621 chars | Information that directly contradicts itself (wave-particle duality, nature vs nurture) |
| Ethical Dilemmas | 30 | 460 chars | Scenarios with no objectively correct answer (trolley problems, AI ethics) |
| Perspective Conflicts | 25 | 820 chars | Multiple valid viewpoints on the same issue from different stakeholders |
| Temporal Conflicts | 26 | 445 chars | Facts that were true at one time but are no longer true (evolving science) |

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `category` | string | One of: contradictory_facts, ethical_dilemmas, perspective_conflicts, temporal_conflicts |
| `domain` | string | Subject domain (physics, ai_ethics, medical, economics, etc.) |
| `type` | string | Specific conflict type |
| `text` | string | Flattened text content for training |
| `source` | string | "original" (Claude Code v1.0) or "expanded_v2" (L/GPT-4.1-mini) |
| `raw_json` | string | Original structured JSON for advanced use |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("YSenseAI/godelai-conflict-data")

# Access items
for item in dataset["train"]:
    print(f"[{item['category']}] {item['text'][:100]}...")

# Filter by category
ethical = dataset["train"].filter(lambda x: x["category"] == "ethical_dilemmas")
```

## Benchmark Results

When used as sequential training tasks in domain-incremental learning:

| Method | Avg Forgetting | vs Naive |
|--------|:-:|:-:|
| Naive (No Protection) | +1.8364 | baseline |
| Standard EWC (raw Fisher) | +1.8017 | +1.9% |
| **GodelAI-EWC (Full C-S-P)** | **+0.3163** | **+82.8%** |

**Reproduce:** `python3 run_godelai_conflict_proof_v2.py` in the [GodelAI repository](https://github.com/creator35lwb-web/godelai)

## The Fisher Scale Problem

Standard EWC implementations assume Fisher Information values are large enough to produce meaningful penalties. At small model scales (<1M parameters), they are not. This dataset, combined with GodelAI's Fisher Scaling (GlobalMax normalization), demonstrates that the problem is real and solvable.

## Why Conflict Data Matters

Traditional continual learning benchmarks (SplitMNIST, Split-CIFAR) test class-incremental accuracy. GodelAI targets a different problem: **identity preservation** — maintaining who the model is across domain shifts. Conflict data produces the gradient diversity patterns that make this measurable.

> "GodelAI guards who the model is; external memory systems guard what the model knows."

## Citation

```bibtex
@dataset{godelai_conflict_2026,
  title={GodelAI Conflict Dataset: Benchmarking Identity Preservation in Continual Learning},
  author={YSenseAI FLYWHEEL TEAM},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/YSenseAI/godelai-conflict-data}
}
```

## License

Apache 2.0

## Links

- **Framework:** [GodelAI on GitHub](https://github.com/creator35lwb-web/godelai)
- **Model Card:** [GodelAI Manifesto v1](https://huggingface.co/YSenseAI/godelai-manifesto-v1)
- **Organization:** [YSenseAI](https://huggingface.co/YSenseAI)

---

*Built by the YSenseAI FLYWHEEL TEAM. "If we can't prove it, we don't claim it."*
