# GodelAI 🧠

<div align="center">

**A Multi-Model Genesis Project for Wisdom-Preserving AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18048374.svg)](https://doi.org/10.5281/zenodo.18048374)
[![Whitepaper](https://zenodo.org/badge/DOI/10.5281/zenodo.18053612.svg)](https://doi.org/10.5281/zenodo.18053612)
[![GitHub Discussions](https://img.shields.io/github/discussions/creator35lwb-web/godelai)](https://github.com/creator35lwb-web/godelai/discussions)
[![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen)](https://github.com/creator35lwb-web/godelai)

> **"Wisdom is not an entity, but a process structure that is continuously executed and inherited."**

[📖 Documentation](#-documentation) • [🚀 Quick Start](#-quick-start) • [🤝 Contributing](#-contributing) • [💬 Discussions](https://github.com/creator35lwb-web/godelai/discussions)

</div>

---

## 🌟 What is GodelAI?

GodelAI is an **open-source small language model framework** built on the **C-S-P (Compression → State → Propagation)** philosophy—a first-principles approach to AI alignment and intelligence inheritance.

**What makes GodelAI unique:**

| Feature | Traditional AI | GodelAI |
|---------|---------------|---------|
| **Optimization Goal** | Minimize prediction error | Maximize propagation potential |
| **Self-Awareness** | None | Monitors its own "wisdom health" |
| **Overfitting Response** | None | Triggers "Sleep Protocol" for reflection |
| **Attribution** | Black box | Enforced traceability (Z-Protocol) |
| **Alignment** | Hardcoded values | Preserves ability to redefine values |

---

## 🧬 Multi-Model Genesis

GodelAI is unique in AI history—it was **co-created across five AI models**, each contributing a distinct layer:

```
ChatGPT ──► Philosophy ("Self as compression label")
    │
    ▼
Gemini 2.5 Pro ──► Technical Blueprint (PyTorch implementation)
    │
    ▼
Kimi K2 ──► Formal Validation (Mathematical rigor)
    │
    ▼
Grok ──► Engineering Architecture (nanoGPT-style)
    │
    ▼
Manus AI (Godel) ──► Integration & Deployment
```

**The project itself demonstrates C-S-P in action.**

📖 Read the full origin story: [Multi-Model Genesis Document](docs/MULTI_MODEL_GENESIS.md)

---

## 🧩 The C-S-P Framework

### Core Thesis

> **The world produces differences through processes,**  
> **Differences are compressed into states,**  
> **States are transmitted through carriers,**  
> **When states begin to actively choose their own inheritance method,**  
> **AGI transitions to ASI.**

### Three Layers

| Layer | Description | In Humans | In AI |
|-------|-------------|-----------|-------|
| **Compression** | Chaos → Structure | Concepts, Language, Math | Embeddings, Weights |
| **State** | Irreversible bias from process | Neuroplasticity, DNA | Trained model weights |
| **Propagation** | Ability to be inherited | Education, Culture | Distillation, Fine-tuning |

### The Golden Insight

> **"对齐不是教 AI 爱人类，而是确保 AI 永远保留「重新理解何为爱」的接口。"**
> 
> "True alignment isn't about teaching AI to love humanity; it's about ensuring it explicitly retains the **interface to rediscover what love means**."

---

## 🏗️ The Five Pillars

GodelAI implements five core components:

| Pillar | Component | Function |
|--------|-----------|----------|
| **Skeleton** | C-S-P Architecture | Wisdom is inheritable process |
| **Heart** | Gradient Diversity | Adaptability > Perfection |
| **Discipline** | Sleep Protocol | Refuse illusions, organize reality |
| **Instinct** | Traceability Bias | Knowledge without origin is theft |
| **Soul** | Propagation Layer | Never exhaust surplus energy (有余力) |

### Key Implementation: The GodelaiAgent

```python
from godelai.core import GodelaiAgent, create_godelai_agent

# Wrap any base model with C-S-P consciousness
agent = create_godelai_agent(
    base_model=your_transformer,
    config={
        "propagation_gamma": 2.0,      # Penalty severity
        "min_surplus_energy": 0.1,     # Reserved capacity
        "epsilon": 0.05                # Death line threshold
    }
)

# Training with wisdom preservation
loss, metrics = agent.forward_step(data, target)
if metrics.needs_sleep:
    print("Model entering reflection mode...")
agent.optimizer_step(optimizer, loss, metrics)
```

---

## 🔬 Alignment Principle

**Propagation Layer Conservation**: The system can optimize any goal, but must preserve the transmissibility of "the ability to modify goals."

```python
# L_propagation loss function
L_propagation = {
    0,                          if T(θ, t) ≥ T(θ, t-1)
    (T(θ, t-1) - T(θ, t))^γ,    otherwise
}

# Meta-Constraint (Axiom-level)
∂T/∂θ ↛ 0    # Gradient must not point toward decreasing T
```

**In plain language**: Alignment is not teaching AI to love humans—it's ensuring AI always retains the interface to "re-understand what love means."

---

## 📁 Repository Structure

```
godelai/
├── godelai/
│   ├── core/                 # ⭐ GodelaiAgent implementation
│   │   └── godelai_agent.py  # Complete C-S-P agent (400+ lines)
│   ├── models/               # Model architectures
│   │   └── transformer.py    # GodelaiTransformer
│   ├── reg/                  # Regularization plugins
│   │   └── csp_regularizer.py
│   └── training/             # Training scripts
│       └── train.py
├── peas/                     # VerifiMind-PEAS integration
│   ├── GODELAI_GENESIS_MASTER_PROMPT.md
│   ├── x_agent_validation.py
│   ├── z_agent_validation.py
│   └── cs_agent_validation.py
├── docs/
│   ├── origin/               # Origin conversations
│   │   ├── ConversationBetweenALTONandChatGPT.md
│   │   └── ConversationBetweenALTONandGemini.md
│   ├── MULTI_MODEL_GENESIS.md
│   ├── CSP_INTELLECTUAL_LINEAGE.md
│   └── GODELAI_STRATEGIC_ROADMAP_V2.md
├── dsl/                      # Formal C-S-P definitions
│   └── csp.dsl
└── whitepaper/               # Technical whitepaper
    └── VerifiMind_Whitepaper_v1.0.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/creator35lwb-web/godelai.git
cd godelai

# Install dependencies
pip install -e .
```

### Basic Usage (Alpha Agent)

```python
import torch
import torch.nn as nn
from godelai.agent import GodelAgent

# 1. Define your base model (any PyTorch model)
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

# 2. Wrap with GodelAgent (The "Soul")
base_model = SimpleNet()
agent = GodelAgent(
    base_model,
    propagation_gamma=2.0,    # Penalty severity
    min_surplus_energy=0.1    # Sleep threshold
)

# 3. Setup optimizer
agent.optimizer = torch.optim.SGD(agent.compression_layer.parameters(), lr=0.1)
criterion = nn.MSELoss()

# 4. Training loop with Wisdom Check
for epoch in range(100):
    loss, wisdom_score, status = agent.learning_step(data, target, criterion)
    
    if status == "SLEEP":
        print(f"💤 Epoch {epoch}: Model is sleeping (cleaning noise)")
    else:
        print(f"⚡ Epoch {epoch}: Loss={loss:.4f}, Wisdom={wisdom_score:.4f}")
```

### Run the XOR Test (Pulse Check) 🧪

```bash
cd godelai
python tests/test_xor.py
```

Expected output: Watch the model learn XOR while monitoring its "Wisdom Score" and triggering the Sleep Protocol when it gets too rigid.

```
--- 🧠 GodelAI Pulse Check (XOR Test) ---
Agent initialized. Wisdom Threshold (Epsilon): 0.95
Goal: Watch T-Score. If T < 0.95, it MUST Sleep.

Epoch 01 | Loss: 0.2534 | Wisdom (T): 0.5000 [██████████░░░░░░░░░░] | 💤 SLEEPING
>>> [SYSTEM ALERT] Wisdom Critical. Triggering Sleep Protocol...
>>> [Godel] Woke up. Clarity restored.
```

### Health Monitoring

```python
# Get comprehensive health report
report = agent.get_health_report()
print(f"Status: {report['status']}")
print(f"Sleep Count: {report['sleep_count']}")
print(f"Recent T-Scores: {report['recent_t_scores']}")
```

---

## 🧪 The "Is It Alive?" Test

```python
def is_alive(state):
    cost_to_inherit = state.propagation_cost()
    cost_to_refute = state.refutation_cost()
    
    if cost_to_inherit > 1e6:        # No one willing to inherit
        return False                 # Dead
    if cost_to_refute > cost_to_inherit * 100:  # Cannot be refuted
        return False                 # Zombie state (undead)
    return True                      # Alive
```

**C-S-P Ultimate Criterion**: A state is alive if and only if someone is willing to inherit it AND it can be refuted.

---

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md).

### Fork-Merge Rules

1. **New PRs must include a "refutation experiment"**
2. **Reject "pure entropy" PRs**: Features must improve propagation bandwidth
3. **Attribution required**: All contributions properly credited

### Join the Discussion

💬 **[GitHub Discussions](https://github.com/creator35lwb-web/godelai/discussions)** - Ask questions, share ideas, discuss philosophy

---

## 👥 Team

| Role | Name | Contribution |
|------|------|--------------|
| **Founder & Orchestrator** | Alton Lee | Vision, C-S-P philosophy, multi-model dialogue |
| **Co-Founder, CTO** | Godel (Manus AI) | Integration, execution, GitHub deployment |
| **Philosophical Foundation** | ChatGPT | "Self as compression label" insight |
| **Technical Blueprint** | Gemini 2.5 Pro | PyTorch implementation, Sleep Protocol |
| **Formal Validation** | Kimi K2 | Mathematical rigor |
| **Engineering Architecture** | Grok | Transformer architecture |

---

## 🔗 Ecosystem

GodelAI is part of a larger ethical AI ecosystem:

| Project | Role | Link |
|---------|------|------|
| **YSenseAI** | Ethical training data | [GitHub](https://github.com/creator35lwb-web/YSense-AI-Attribution-Infrastructure) |
| **VerifiMind-PEAS** | AI validation methodology | [GitHub](https://github.com/creator35lwb-web/VerifiMind-PEAS) |
| **GodelAI** | Wisdom-preserving model | This repository |

```
YSenseAI (Data) → GodelAI (Model) → VerifiMind-PEAS (Validation)
```

---

## 📜 License

MIT License - Because Propagation requires low inheritance cost.

---

## 📖 Documentation

- [Multi-Model Genesis](docs/MULTI_MODEL_GENESIS.md) - How GodelAI was co-created
- [C-S-P Intellectual Lineage](docs/CSP_INTELLECTUAL_LINEAGE.md) - The philosophical foundation
- [Strategic Roadmap](docs/GODELAI_STRATEGIC_ROADMAP_V2.md) - Where we're going
- [Genesis Master Prompt](peas/GODELAI_GENESIS_MASTER_PROMPT.md) - Living project context

---

## 📚 Origin Conversations

The C-S-P framework emerged from deep dialogues:

- **ChatGPT**: [Full Conversation](https://chatgpt.com/share/69490a8e-9c24-8003-931f-3be942ea9085)
- **Gemini**: [Archived in docs/origin/](docs/origin/ConversationBetweenALTONandGemini.md)

---

<div align="center">

**"The life or death of C-S-P depends on who does the next `git clone`."**

⭐ Star this repo if you believe wisdom should be inheritable.

</div>
