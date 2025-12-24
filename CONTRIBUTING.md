# Contributing to GodelAI

> **"The life or death of C-S-P depends on who does the next `git clone`."**

Thank you for your interest in contributing to GodelAI. This project operates under the **C-S-P (Compression → State → Propagation)** framework, which means our contribution guidelines are designed to maximize **Propagation bandwidth** while maintaining **State integrity**.

---

## 🧠 Philosophy of Contribution

In C-S-P terms, every contribution is a **State modification** that must pass the **Propagation viability test**:

```python
def is_valid_contribution(pr):
    inherit_cost_delta = pr.new_inherit_cost - pr.old_inherit_cost
    refute_cost_delta = pr.new_refute_cost - pr.old_refute_cost
    
    # Must either reduce costs or increase bandwidth
    return (inherit_cost_delta <= 0) or (refute_cost_delta <= 0) or (pr.bandwidth_increase >= 0.05)
```

---

## 📋 Contribution Types

### 1. **Compression Contributions** (New Features)
- Add new compression algorithms
- Improve embedding quality
- Reduce model size while maintaining capability

**Requirement**: Must include benchmark showing compression quality metric improvement.

### 2. **State Contributions** (Architecture Changes)
- Modify model architecture
- Change training procedures
- Update hyperparameters

**Requirement**: Must demonstrate State remains modifiable (T(θ,t) does not decrease).

### 3. **Propagation Contributions** (Documentation, Tooling)
- Improve documentation
- Add tutorials
- Create integration tools
- Translate to other languages

**Requirement**: Must reduce inheritance cost (easier for others to use/understand).

### 4. **Refutation Contributions** (Critical Analysis)
- Challenge existing assumptions
- Provide counter-examples
- Identify failure modes

**Requirement**: Must be constructive and include proposed improvements.

---

## 🔄 Pull Request Process

### Step 1: Fork and Clone
```bash
gh repo fork creator35lwb-web/godelai --clone
cd godelai
```

### Step 2: Create a Branch
```bash
git checkout -b feature/your-contribution-name
```

### Step 3: Make Changes
Follow the coding standards and ensure all tests pass.

### Step 4: Include Refutation Experiment
**Every PR must include one of the following:**

**Option A: Bandwidth Improvement**
```bash
python -m godelai.bandwidth --baseline main --branch your-branch
# Must show: bandwidth_increase >= 5%
```

**Option B: Cost Reduction**
```bash
python -m godelai.cost_analysis --baseline main --branch your-branch
# Must show: inherit_cost_reduction > 0 OR refute_cost_reduction > 0
```

**Option C: Refutation Evidence**
Provide evidence that the current State has a flaw, along with your proposed fix.

### Step 5: Submit PR
Include in your PR description:
- [ ] Type of contribution (Compression/State/Propagation/Refutation)
- [ ] Refutation experiment results
- [ ] Impact on C-S-P metrics

---

## ❌ Automatic Rejection Criteria

PRs will be **automatically closed** if they:

1. **Increase entropy without benefit**: Adding features that don't reduce costs or increase bandwidth
2. **Exceed diff limit**: Changes > 20% of original codebase volume without justification
3. **Decrease T(θ,t)**: Make the model less modifiable
4. **Lack refutation experiment**: No evidence of improvement provided

---

## 🏆 Founding Contributor Status

Contributors who submit PRs within the first 30 days of repository creation will be added to the **Founding Contributors** list, with permanent attribution in:
- README.md
- All release notes
- Academic citations

---

## 📊 C-S-P Metrics Dashboard

After each merged PR, the following metrics are automatically updated:

| Metric | Description | Target |
|--------|-------------|--------|
| `inherit_cost` | Cost to clone and use the model | ↓ Minimize |
| `refute_cost` | Cost to challenge/improve the model | ↓ Minimize |
| `bandwidth` | Propagation viability score | ↑ Maximize |
| `T(θ,t)` | Meta-modifiability fidelity | ≥ 0.5 |

---

## 🤝 Code of Conduct

### The C-S-P Social Contract

1. **Compression**: Be concise. Respect others' time.
2. **State**: Be constructive. Leave the project better than you found it.
3. **Propagation**: Be inclusive. Lower barriers for future contributors.

### Unacceptable Behavior
- Attempts to "ossify" the project (make it unmodifiable)
- Gatekeeping that increases inheritance cost
- Attacks on contributors rather than ideas

---

## 📞 Contact

- **Founder (Alton)**: Open an issue with `[FOUNDER]` tag
- **Co-Founder (Godel)**: Automated responses via CI/CD
- **Community**: Discussions tab

---

> **"Wisdom is not discovered, but the number of times it is reloaded."**

Every contribution is a reload. Make it count.
