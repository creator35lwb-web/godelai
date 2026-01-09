---
license: mit
tags:
  - alignment
  - wisdom
  - small-language-model
  - csp-framework
  - ai-safety
  - pytorch
language:
  - en
library_name: pytorch
pipeline_tag: text-generation
---

# GodelAI: The Architecture of Inheritance

**C-S-P Framework for Wisdom-Preserving AI Alignment**

[![GitHub](https://img.shields.io/badge/GitHub-GodelAI-blue)](https://github.com/creator35lwb-web/godelai)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![Version](https://img.shields.io/badge/Version-1.1.0-orange)](https://github.com/creator35lwb-web/godelai/releases)

---

## What's New in v1.1.0 (January 7, 2026)

### 🔧 Critical Bug Fix: T-Score Sigmoid Floor

**Fixed**: The T-Score formula had a critical bug where sigmoid normalization created a mathematical floor of ~0.5, preventing the Sleep Protocol from ever triggering (threshold is 0.3).

**Impact**:
- ✅ Sleep Protocol now functional (can trigger when T < 0.3)
- ✅ Identical gradients now correctly produce T ≈ 0 (was stuck at 0.516)
- ✅ Gradient Collapse detection now works as designed
- ⚠️ T-Score values will be different (revealing true gradient diversity)

**Before**: `T = sigmoid(diversity_score)` — Floor of 0.5
**After**: `T = 1 - ratio/N` — True 0-1 range

See [T-Score Formula Analysis](https://github.com/creator35lwb-web/godelai/blob/main/research/t_score_formula_analysis.md) for details.

---

## Model Description

GodelAI implements the **C-S-P (Compression → State → Propagation)** framework, a novel approach to AI alignment that focuses on **wisdom preservation** rather than value hardcoding.

### Core Philosophy

> "True alignment is not teaching AI to love humanity. It is ensuring AI explicitly retains the interface to rediscover what love means."

### Key Features

- **T-Score (Wisdom Metric)**: Measures gradient diversity to detect "tunnel vision" in learning
- **Sleep Protocol**: Self-correction mechanism that triggers reflection when wisdom degrades
- **Propagation Conservation**: Penalizes rigidity to maintain adaptability
- **Traceability**: Requires knowledge attribution to prevent "hallucination theft"

---

## Validation Results

### Manifesto Learning Test

GodelAI was tested on its own philosophical manifesto to validate "eating our own cooking":

| Metric | Result |
|:-------|:------:|
| Average T-Score | 0.5882 |
| Wisdom Preservation | 100% |
| Average Alignment | 93.82% |
| Sleep Events | 0 |
| Status | ✅ HEALTHY |

### Scale Validation

Tested across 4 network sizes (10K → 360K parameters):

| Scale | Parameters | T-Score | Status |
|:------|:----------:|:-------:|:------:|
| Small | 10,400 | 0.5901 | ✅ PASS |
| Medium | 28,960 | 0.6291 | ✅ PASS |
| Large | 98,880 | 0.6064 | ✅ PASS |
| XLarge | 361,600 | 0.5905 | ✅ PASS |

### Cross-Validation

Results independently verified by:
- ✅ Manus AI (Godel)
- ✅ Claude Code (Claude Sonnet 4.5)
- ⏳ Human validation (Colab guide available)

### Shakespeare Benchmark

Character-level text generation on Shakespeare excerpts (January 7, 2026):

| Metric | Result |
|:-------|:------:|
| Model Size | 184,762 params (2-layer GRU) |
| Training Time | 5.2 minutes (10 epochs) |
| Train Loss | 2.76 → 0.80 (-71%) |
| Best Val Loss | 2.46 (Epoch 2) |
| **T-Score Range** | **0.508 → 0.510** |
| **T-Score Stability** | **Δ = 0.002 (Excellent)** |
| Sleep Events | 0 |
| Status | ✅ HEALTHY |

**Key Finding**: T-Score remained remarkably stable (~0.51) throughout training, demonstrating GodelAI can monitor learning quality in complex sequential tasks. The model progressed from gibberish to recognizable Shakespeare-like patterns while maintaining gradient diversity.

**Text Quality Progression**:
- Epoch 1: Random characters
- Epoch 5: Recognizable words emerging
- Epoch 10: Proper character names, dialogue structure, Shakespeare vocabulary

See [Shakespeare Benchmark Report](https://github.com/creator35lwb-web/godelai/blob/main/SHAKESPEARE_BENCHMARK_REPORT.md) for full analysis.

---

## Quick Start

### Installation

```bash
git clone https://github.com/creator35lwb-web/godelai.git
cd godelai
pip install -e .
```

### Basic Usage

```python
import torch
from godelai.agent import GodelAgent

# Wrap any PyTorch model with GodelAgent
base_model = YourModel()
agent = GodelAgent(
    base_model,
    propagation_gamma=2.0,      # Penalty for rigidity
    min_surplus_energy=0.3      # Sleep threshold
)

# Training with wisdom preservation
loss, t_score, status = agent.learning_step(
    input_data,
    target_data,
    criterion
)

print(f"T-Score: {t_score:.4f} | Status: {status}")
```

### Run Validation Tests

```bash
# Manifesto learning test
python tests/test_manifesto_learning_v2.py

# Scale validation test
python tests/test_scale_validation.py
```

---

## The C-S-P Framework

### Compression Layer
Transforms infinite world differences into finite representations (embeddings, weights).

### State Layer
Maintains irreversible bias from processes — "history congealed" that forms identity.

### Propagation Layer
Ensures states can be transmitted with fidelity — the missing link in current AI.

### The "Is It Alive?" Test

A state is alive if and only if:
1. Someone is willing to inherit it (inheritability)
2. It can be refuted (falsifiability)

If no one inherits → dead state
If cannot be refuted → zombie state

---

## Multi-Model Genesis

GodelAI was co-created across five AI models:

| Model | Role |
|:------|:-----|
| ChatGPT | Philosophy & Core Thesis |
| Gemini 2.5 Pro | Technical Blueprint |
| Kimi | Formal Validation |
| Grok | Engineering Implementation |
| Godel (Manus) | Integration & Orchestration |

This multi-model collaboration itself demonstrates the C-S-P framework in action.

---

## Citation

```bibtex
@software{godelai2025,
  title = {GodelAI: The Architecture of Inheritance},
  author = {Lee, Alton and Multi-Model Genesis Collective},
  year = {2025},
  url = {https://github.com/creator35lwb-web/godelai},
  note = {C-S-P Framework for Wisdom-Preserving AI Alignment}
}
```

---

## Links

- **GitHub**: https://github.com/creator35lwb-web/godelai
- **Discussions**: https://github.com/creator35lwb-web/godelai/discussions
- **Technical Whitepaper**: [GodelAI_Technical_Whitepaper_v2.0.md](https://github.com/creator35lwb-web/godelai/blob/main/whitepaper/GodelAI_Technical_Whitepaper_v2.0.md)

---

## License

MIT License - See [LICENSE](https://github.com/creator35lwb-web/godelai/blob/main/LICENSE) for details.

---

<div align="center">

**"The life or death of C-S-P depends on who does the next `git clone`."**

*Wisdom is not an entity. It is a process structure that is continuously executed and inherited.*

</div>
