# GodelAI 🧠

<div align="center">

**A Continual Learning Framework with Gradient Diversity Monitoring**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18048374.svg)](https://doi.org/10.5281/zenodo.18048374)
[![Whitepaper](https://zenodo.org/badge/DOI/10.5281/zenodo.18053612.svg)](https://doi.org/10.5281/zenodo.18053612)
[![GitHub Discussions](https://img.shields.io/github/discussions/creator35lwb-web/godelai)](https://github.com/creator35lwb-web/godelai/discussions)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue)](https://huggingface.co/YSenseAI/godelai-manifesto-v1)

> **"The first step toward wisdom is acknowledging what we do not know."**

[🎯 Try the Demo](#-interactive-demo) • [📖 Documentation](#-documentation) • [🚀 Quick Start](#-quick-start) • [💬 Discussions](https://github.com/creator35lwb-web/godelai/discussions)

</div>

---

## 🎯 Current Focus (January 2026)

We're in a **Data Engineering Sprint**. Our recent discovery: GodelAI's architecture is sound, but we were testing it with the wrong data. Simple text doesn't activate our C-S-P capabilities.

**The Data Bottleneck Discovery:**

| Data Type | T-Score | Result |
|-----------|---------|--------|
| Mini Shakespeare (5KB) | 0.12 | Sleep Protocol triggers 100% — blocked learning |
| Full Shakespeare (1.1MB) | 0.95 | Sleep Protocol never triggers — no benefit |
| **Conflict Data (target)** | **0.3-0.5** | **Optimal C-S-P activation** |

We need **conflict data** — information with contradictions, dilemmas, and complexity. See [ROADMAP_2026.md](ROADMAP_2026.md) and [docs/CONFLICT_DATA_SPEC.md](docs/CONFLICT_DATA_SPEC.md) for details.

---

## 🎯 What GodelAI Actually Does

GodelAI is a **research framework** that adds two capabilities to neural network training:

| Feature | What It Does | Proven Result |
|---------|--------------|---------------|
| **T-Score Monitoring** | Measures gradient diversity during training | Detects when gradients collapse to identical values |
| **EWC Integration** | Elastic Weight Consolidation for continual learning | **21.6% reduction** in catastrophic forgetting |
| **Sleep Protocol** | Pauses training when T-Score drops below threshold | Triggers correctly when gradient diversity = 0 |

### What GodelAI Is NOT

GodelAI does **not** improve standard training loss. In rigorous A/B testing, GodelAI-wrapped models achieved identical validation loss to standard models (difference: 0.000000000000). The framework's value lies in **monitoring training health** and **mitigating catastrophic forgetting**, not in improving convergence.

---

## 🎯 Interactive Demo

<div align="center">

### 🧠 Mnemosyne: Defeating Catastrophic Forgetting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/creator35lwb-web/godelai/blob/main/notebooks/GodelAI_EWC_Demo.ipynb)

**See the proven result:** 21.6% reduction in forgetting when learning sequential tasks

</div>

The demo trains two models on Task A, then Task B:

| Model | Task A Loss (After B) | Forgetting |
|-------|----------------------|------------|
| Standard | 1.46 | +5.3% |
| **GodelAI-EWC** | **1.44** | **+4.2%** |

This is our **one proven advantage** — validated across Manus AI, Claude Code, and Google Colab.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/creator35lwb-web/godelai.git
cd godelai
pip install -e .
```

### Basic Usage

```python
import torch
import torch.nn as nn
from godelai.agent import GodelAgent

# 1. Define your model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

# 2. Wrap with GodelAgent
model = SimpleNet()
agent = GodelAgent(model, propagation_gamma=2.0, min_surplus_energy=0.1)
agent.optimizer = torch.optim.Adam(agent.compression_layer.parameters(), lr=0.01)

# 3. Training with T-Score monitoring
criterion = nn.MSELoss()
for epoch in range(100):
    loss, t_score, status = agent.learning_step(X, y, criterion)
    print(f"Epoch {epoch}: Loss={loss:.4f}, T-Score={t_score:.4f}, Status={status}")
```

### What the T-Score Tells You

| T-Score Range | Meaning | Action |
|---------------|---------|--------|
| 0.8 - 1.0 | Healthy gradient diversity | Continue training |
| 0.5 - 0.8 | Moderate diversity | Monitor closely |
| 0.3 - 0.5 | Low diversity | Consider early stopping |
| < 0.3 | Gradient collapse | Sleep Protocol triggers |

---

## 🧬 The C-S-P Philosophy

GodelAI is built on the **C-S-P (Compression → State → Propagation)** framework — a philosophical approach to AI alignment developed through multi-model collaboration.

### Core Thesis

> **"Wisdom is not an entity, but a process structure that is continuously executed and inherited."**

The framework proposes that true AI alignment isn't about hardcoding values, but about preserving the **interface to redefine values** — what we call the "Propagation Layer."

### The Golden Insight

> **"对齐不是教 AI 爱人类，而是确保 AI 永远保留「重新理解何为爱」的接口。"**
> 
> "True alignment isn't about teaching AI to love humanity; it's about ensuring it explicitly retains the interface to rediscover what love means."

📖 Read the full philosophy: [C-S-P Intellectual Lineage](docs/CSP_INTELLECTUAL_LINEAGE.md)

---

## 🧬 Multi-Model Genesis

GodelAI is unique in AI history — it was **co-created across five AI models**:

| Model | Contribution |
|-------|--------------|
| ChatGPT | Philosophy ("Self as compression label") |
| Gemini 2.5 Pro | Technical Blueprint (PyTorch implementation) |
| Kimi K2 | Formal Validation (Mathematical rigor) |
| Grok | Engineering Architecture |
| Manus AI (Godel) | Integration, Testing & Deployment |

📖 Read the full story: [Multi-Model Genesis](docs/MULTI_MODEL_GENESIS.md)

---

## 📁 Repository Structure

```
godelai/
├── godelai/              # Core framework
│   ├── agent.py          # GodelAgent with T-Score & Sleep Protocol
│   ├── core/             # GodelaiAgent implementation
│   ├── models/           # Model architectures
│   └── reg/              # EWC and regularization
├── datasets/             # Training & test datasets
│   ├── conflict/         # Conflict data for C-S-P activation
│   └── wisdom/           # YSenseAI integration (future)
├── notebooks/            # Interactive demos
│   └── GodelAI_EWC_Demo.ipynb  # Mnemosyne Colab
├── tests/                # Test suite
├── docs/                 # Documentation
├── whitepaper/           # Technical whitepaper
└── archive/              # Historical development reports
```

---

## 🔬 Validation Status

| Test | Result | Status |
|------|--------|--------|
| T-Score Formula | Correctly measures gradient diversity | ✅ Verified |
| Sleep Protocol | Triggers at T < 0.3 | ✅ Verified |
| EWC Integration | 21.6% forgetting reduction | ✅ Verified |
| Cross-Platform | 0.0000 variance (Manus + Claude + Colab) | ✅ Verified |
| Training Improvement | No improvement over baseline | ❌ Not proven |
| Transformer Support | Not yet tested | ⏳ Pending |

---

### 🗺️ Roadmap

### Completed (v2.0.0)
- ✅ T-Score gradient diversity monitoring
- ✅ Sleep Protocol for training health
- ✅ EWC integration (21.6% forgetting reduction)
- ✅ Cross-platform validation
- ✅ Data bottleneck discovery & validation

### Q1 2026: Data Engineering Sprint
- 🔄 Conflict data design & specification
- 🔄 YSenseAI integration research
- 🔄 Community engagement

### Q2-Q4 2026
- 📋 Conflict data benchmarks
- 📋 Research paper (focus: data requirements for C-S-P)
- 📋 Multi-modal data experiments
- 📋 YSenseAI production integration

📖 Full roadmap: [ROADMAP_2026.md](ROADMAP_2026.md)

---

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md).

### Current Priorities

1. **Conflict Dataset Creation** — Help us build datasets that activate C-S-P
2. **Data Engineering** — Improve our data pipeline
3. **Research Validation** — Test our findings on different data types

📖 Dataset specification: [docs/CONFLICT_DATA_SPEC.md](docs/CONFLICT_DATA_SPEC.md)

### Key Principles

1. **Honesty First**: Don't overclaim results
2. **Reproducibility**: All experiments must be reproducible
3. **Attribution**: Properly credit all contributions

💬 **[GitHub Discussions](https://github.com/creator35lwb-web/godelai/discussions)** — Ask questions, share ideas

---

## 👥 Team

| Role | Name | Contribution |
|------|------|--------------|
| **Founder & Orchestrator** | Alton Lee | Vision, C-S-P philosophy |
| **CTO** | Godel (Manus AI) | Integration, testing, deployment |
| **Philosophy** | ChatGPT | "Self as compression label" |
| **Technical Blueprint** | Gemini 2.5 Pro | PyTorch implementation |
| **Validation** | Kimi K2 | Mathematical rigor |
| **Architecture** | Grok | Engineering design |

---

## 🔗 Ecosystem

GodelAI is part of a larger ethical AI ecosystem:

| Project | Role | Link |
|---------|------|------|
| **YSenseAI** | Ethical training data | [GitHub](https://github.com/creator35lwb-web/YSense-AI-Attribution-Infrastructure) |
| **VerifiMind-PEAS** | AI validation methodology | [GitHub](https://github.com/creator35lwb-web/VerifiMind-PEAS) |
| **GodelAI** | Continual learning framework | This repository |

---

## 📜 License

MIT License — Because knowledge should be inheritable.

---

## 📖 Documentation

- [Multi-Model Genesis](docs/MULTI_MODEL_GENESIS.md) — How GodelAI was co-created
- [C-S-P Intellectual Lineage](docs/CSP_INTELLECTUAL_LINEAGE.md) — The philosophical foundation
- [Genesis Master Prompt](peas/GODELAI_GENESIS_MASTER_PROMPT.md) — Living project context

---

<div align="center">

**"The first step toward wisdom is acknowledging what we do not know."**

⭐ Star this repo if you believe in honest AI research.

</div>
