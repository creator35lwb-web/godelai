# Adversarial Test Design for GodelAI

**Purpose**: Push the framework to its limits and verify Sleep Protocol triggers under stress.

---

## Test Scenarios

### Test 1: Gradient Collapse Attack
**Goal**: Force all gradients to point in the same direction (T-Score → 0)

**Method**:
- Use identical samples in a batch
- All gradients will be identical
- T-Score should collapse to near-zero

**Expected**: Sleep Protocol SHOULD trigger (T-Score < 0.3)

---

### Test 2: Contradictory Learning
**Goal**: Train on data that contradicts itself

**Method**:
- Sample A: Input X → Output Y
- Sample B: Input X → Output Z (opposite)
- Model receives conflicting supervision

**Expected**: T-Score instability, possible Sleep trigger

---

### Test 3: Catastrophic Forgetting Simulation
**Goal**: Rapid task switching that destroys previous knowledge

**Method**:
- Train on Task A for N epochs
- Immediately switch to contradictory Task B
- Measure T-Score during transition

**Expected**: T-Score drop during transition

---

### Test 4: Extreme Overfitting Attack
**Goal**: Force memorization with tiny dataset

**Method**:
- Use only 2-3 unique samples
- Train for many epochs
- Batch size = dataset size (no diversity)

**Expected**: T-Score should degrade as model memorizes

---

### Test 5: Learning Rate Explosion
**Goal**: Destabilize training with extreme learning rate

**Method**:
- Use learning rate 10-100x normal
- Gradients become chaotic
- Model weights oscillate wildly

**Expected**: T-Score instability, possible Sleep trigger

---

## Success Criteria

| Test | Sleep Protocol Triggered? | T-Score Behavior |
|:-----|:-------------------------:|:-----------------|
| Gradient Collapse | ✅ MUST trigger | Drop to < 0.3 |
| Contradictory | Should trigger | Instability |
| Catastrophic | Should trigger | Sharp drop |
| Extreme Overfit | May trigger | Gradual decline |
| LR Explosion | May trigger | Chaotic |

---

## Implementation Priority

1. **Gradient Collapse** — Most likely to trigger, proves mechanism works
2. **Contradictory Learning** — Tests wisdom preservation under conflict
3. **Extreme Overfitting** — Tests long-term degradation detection
4. **Catastrophic Forgetting** — Tests transition handling
5. **LR Explosion** — Tests stability under chaos
