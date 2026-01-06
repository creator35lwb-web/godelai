# T-Score Formula Analysis: Critical Bug Discovery

**Date**: January 7, 2026  
**Author**: Godel (Manus AI)  
**Severity**: HIGH — Sleep Protocol cannot trigger under adversarial conditions

---

## The Bug

The current T-Score formula uses sigmoid normalization, which has a **mathematical floor of 0.5**:

```python
diversity_score = sum_norm_grad / (sum_grad_norm + 1e-8)
T_score = torch.sigmoid(diversity_score)
```

### Why This Fails

When gradients are **perfectly identical** (worst case):

| Variable | Formula | Value (N=16) |
|:---------|:--------|:-------------|
| `sum_grad_norm` | `||Σg||² = N² * ||g||²` | `256 * ||g||²` |
| `sum_norm_grad` | `Σ||g||² = N * ||g||²` | `16 * ||g||²` |
| `diversity_score` | `N / N² = 1/N` | `0.0625` |
| `T_score` | `sigmoid(0.0625)` | **0.516** |

The sigmoid function maps:
- `sigmoid(0) = 0.5` (minimum practical value)
- `sigmoid(-∞) = 0` (never reached with positive diversity)

**Result**: T-Score can never go below ~0.5, but the Sleep threshold is 0.3!

---

## Evidence from Adversarial Tests

| Test | Gradient Similarity | T-Score | Expected |
|:-----|:-------------------:|:-------:|:---------|
| Gradient Collapse | 1.0 (identical) | 0.516 | < 0.3 |
| Contradictory | 1.0 (opposite cancel) | 1.0 | < 0.5 |
| Extreme Overfit | N/A | 0.50 (floor) | < 0.3 |

---

## Proposed Fixes

### Option A: Linear Normalization (Recommended)

```python
def measure_gradient_diversity(self, batch_gradients):
    sum_grad_norm = torch.norm(torch.sum(batch_gradients, dim=0))**2
    sum_norm_grad = torch.sum(torch.norm(batch_gradients, dim=1)**2)
    
    # Linear normalization: 1 - (crowd/individual)
    # When identical: crowd = N² * ||g||², individual = N * ||g||²
    # ratio = N² / N = N, so 1 - 1/N approaches 1 for large N
    # When diverse: crowd << individual, ratio approaches 0
    
    ratio = sum_grad_norm / (sum_norm_grad + 1e-8)
    T_score = 1.0 - torch.clamp(ratio / batch_gradients.shape[0], 0, 1)
    
    return T_score
```

**Behavior**:
- Identical gradients: T_score → 0
- Diverse gradients: T_score → 1
- No sigmoid floor problem

### Option B: Adjusted Sigmoid with Offset

```python
def measure_gradient_diversity(self, batch_gradients):
    sum_grad_norm = torch.norm(torch.sum(batch_gradients, dim=0))**2
    sum_norm_grad = torch.sum(torch.norm(batch_gradients, dim=1)**2)
    
    diversity_score = sum_norm_grad / (sum_grad_norm + 1e-8)
    
    # Shift sigmoid to center at 1.0 (expected value for random gradients)
    # and scale to use full 0-1 range
    T_score = torch.sigmoid((diversity_score - 1.0) * 5)
    
    return T_score
```

### Option C: Cosine Similarity Based

```python
def measure_gradient_diversity(self, batch_gradients):
    # Compute pairwise cosine similarities
    normalized = batch_gradients / (torch.norm(batch_gradients, dim=1, keepdim=True) + 1e-8)
    similarity_matrix = torch.mm(normalized, normalized.t())
    
    # Average off-diagonal similarity
    n = batch_gradients.shape[0]
    mask = ~torch.eye(n, dtype=bool, device=batch_gradients.device)
    avg_similarity = similarity_matrix[mask].mean()
    
    # T_score = 1 - similarity (diverse = low similarity = high T)
    T_score = 1.0 - avg_similarity
    
    return T_score
```

---

## Recommendation

**Use Option A (Linear Normalization)** because:
1. Mathematically correct behavior at extremes
2. No sigmoid floor problem
3. Intuitive interpretation (0 = identical, 1 = maximally diverse)
4. Minimal code change

---

## Impact Assessment

| Aspect | Before Fix | After Fix |
|:-------|:-----------|:----------|
| Identical gradients | T=0.52 | T≈0.0 |
| Sleep Protocol trigger | Never | When appropriate |
| Existing tests | May change | Need re-validation |
| Backward compatibility | N/A | Breaking change |

---

## Action Items

1. [ ] Implement Option A in `godelai/agent.py`
2. [ ] Re-run all existing tests to validate
3. [ ] Re-run adversarial tests to confirm Sleep triggers
4. [ ] Update documentation with new T-Score interpretation
5. [ ] Bump version to v1.1.0 (breaking change)
