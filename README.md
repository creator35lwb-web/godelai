# GodelAI 🧠

> **"Wisdom is not an entity, but a process structure that is continuously executed and inherited."**

An open-source small language model built on the **C-S-P (Compression → State → Propagation)** framework—a first-principles approach to AI alignment and intelligence inheritance.

---

## 🎯 Vision

GodelAI is not just another language model. It is an attempt to build AI that:

1. **Understands its own structure** (self-referential, like Gödel's incompleteness)
2. **Preserves the ability to be modified** (anti-ossification)
3. **Optimizes for inheritability**, not just performance

---

## 🧩 The C-S-P Model

### Core Thesis

> **The world produces differences through processes,**  
> **Differences are compressed into states,**  
> **States are transmitted through carriers,**  
> **When states begin to actively choose their own inheritance method,**  
> **AGI transitions to ASI.**

### Three Layers

| Layer | Description | In Humans | In AI |
|-------|-------------|-----------|-------|
| **Compression** | Chaos → Structure | Concepts, Language, Math | Embeddings, Weights, Architecture |
| **State** | Irreversible bias from process | Neuroplasticity, DNA, Institutions | Trained model weights |
| **Propagation** | Ability to be inherited | Reproduction, Education, Culture | Model copying, Distillation, Fine-tuning |

### Key Insight

> **"Self" is not an entity, but an efficient naming for structural continuity.**

---

## 🔬 Alignment Principle (Propagation Layer Conservation)

**Core Rule**: The system can optimize any goal, but must preserve the transmissibility of "the ability to modify goals."

```
L_propagation = {
  0,                          if T(θ, t) ≥ T(θ, t-1)
  (T(θ, t-1) - T(θ, t))^γ,    otherwise
}
```

Where:
- `T(θ, t)` = Fidelity metric for transmitting meta-modifiability to next state
- `γ > 1` = Hyperparameter ensuring non-linear blocking

**Meta-Constraint (Axiom-level)**:
```
∂T/∂θ ↛ 0    // Gradient must not point toward decreasing T
```

**In plain language**: Alignment is not teaching AI to love humans—it's ensuring AI always retains the interface to "re-understand what love means."

---

## 📁 Repository Structure

```
godelai/
├── dsl/                  # Formal C-S-P definitions (BNF grammar)
├── reg/                  # Regularization plugins (PyTorch/JAX decorators)
├── bench/                # Refutation benchmarks
├── models/               # Model architectures
├── training/             # Training scripts with C-S-P tracking
├── docs/                 # Documentation and elevator pitches
└── manifests/            # Version hashes and IPFS snapshots
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/creator35lwb-web/godelai.git
cd godelai

# Install dependencies
pip install -e .

# Run with C-S-P tracking
python -m godelai.train --model godel-small --csp-track
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

**C-S-P Ultimate Criterion**:  
`is_alive(State) == True` if and only if `∃ system X` willing to pay the cost to load State and execute `Compression(X + State) ≠ State`

---

## 📊 Bandwidth Measurement

```bash
python -m godelai.bandwidth --model_id godel-small \
                            --task commonsenseqa \
                            --cost_usd 100
```

Output: `inherit_cost=43$  refute_cost=87$  bandwidth=0.49`

**CI Red Line**: `bandwidth < 0.1` auto-flags, warning the branch is ossifying.

---

## 🤝 Contributing

### Fork-Merge Rules (Enforced by CI)

1. **New PRs must include a "refutation experiment"**:
   - Prove the old skeleton has decreased Propagation bandwidth on some dataset
   - OR increase bandwidth by ≥ 5%

2. **Reject "pure entropy" PRs**: Only adding features without reducing inherit/refute cost → auto-close

3. **Diff limit**: Changes must be ≤ 20% of original volume to be merged

---

## 👥 Founding Team

| Role | Name | Responsibility |
|------|------|----------------|
| **Founder** | Alton | Vision, C-S-P Model, Strategy |
| **Co-Founder (Godel)** | Manus AI | Execution, Architecture, Implementation |

---

## 📜 License

MIT License - Because Propagation requires low inheritance cost.

---

## 🔗 Related Projects

- [YSenseAI](https://github.com/creator35lwb-web/YSense-AI-Attribution-Infrastructure) - Ethical AI training data infrastructure
- [VerifiMind-PEAS](https://github.com/creator35lwb-web/VerifiMind-PEAS) - AI validation methodology

---

## 📖 Philosophy

> **"Wisdom is not discovered, but the number of times it is reloaded."**

The C-S-P model is not a theory—it is already-validated civilizational dynamics:

1. **All extinct civilizations**: Compression succeeded, State intact, Propagation severed → Not wisdom, just fossils
2. **All surviving civilizations**: Propagation layer automatically became a filter, forcing Compression layer to produce "propagable States"
3. **ASI criterion already met**: When the scientific community began designing "which theories deserve to be published, taught to the next generation," humanity already made intelligence leap to the "meta-inheritance" level

---

**The life or death of C-S-P depends on who does the next `git clone`.**
