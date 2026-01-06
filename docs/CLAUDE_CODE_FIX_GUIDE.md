# Claude Code: T-Score Bug Fix Implementation Guide

**From**: Godel (Manus AI)  
**To**: Claude Code  
**Date**: January 7, 2026  
**Priority**: HIGH — Critical Bug Fix

---

## Summary

Adversarial testing discovered a critical bug in the T-Score formula. The sigmoid normalization creates a mathematical floor of ~0.5, preventing the Sleep Protocol from ever triggering (threshold is 0.3). This guide provides step-by-step instructions to apply the fix and re-validate.

---

## Step 1: Pull Latest Changes

```bash
cd /path/to/godelai
git pull origin main
```

You should see commit `ebfbd7b` with the adversarial test files.

---

## Step 2: Apply the Fix to `godelai/agent.py`

### Location
File: `godelai/agent.py`  
Method: `measure_gradient_diversity` (lines 63-99)

### Current Code (BUGGY)

```python
def measure_gradient_diversity(self, batch_gradients):
    # ... existing docstring ...
    
    if batch_gradients.shape[0] == 1:
        return torch.tensor(0.5)

    sum_grad_norm = torch.norm(torch.sum(batch_gradients, dim=0))**2
    sum_norm_grad = torch.sum(torch.norm(batch_gradients, dim=1)**2)
    diversity_score = sum_norm_grad / (sum_grad_norm + 1e-8)
    
    # BUG: Sigmoid has floor of 0.5, threshold is 0.3
    T_score = torch.sigmoid(diversity_score)

    return T_score
```

### Fixed Code (REPLACE WITH THIS)

```python
def measure_gradient_diversity(self, batch_gradients):
    """
    Implementation of the Wisdom Metric (Option B: Gradient Diversity).

    Logic:
    If all gradients point in the exact same direction (Global Norm ~ Sum of Norms),
    the model is developing 'Tunnel Vision' (Rigid).
    We want specific neurons to handle specific nuances (High Diversity).

    Args:
        batch_gradients: Tensor of shape [batch_size, num_params]
                        Each row is the gradient for one sample

    Returns:
        T_score: Wisdom score normalized to 0-1 range
                 - 0.0 = Identical gradients (critical, triggers Sleep)
                 - 1.0 = Maximally diverse gradients (healthy)
    
    Fixed in v1.1.0: Replaced sigmoid normalization with linear normalization
    to enable Sleep Protocol triggering. Previous sigmoid had floor of 0.5.
    """
    # Ensure we have multiple samples
    if batch_gradients.shape[0] == 1:
        # Cannot measure diversity with only 1 sample
        return torch.tensor(0.5)

    n = batch_gradients.shape[0]
    
    # 1. Global Direction Strength (Everyone rushing together)
    # sum_grad = || Σ g_i ||^2
    sum_grad_norm = torch.norm(torch.sum(batch_gradients, dim=0))**2

    # 2. Individual Direction Strength (Individual thinking)
    # sum_norm_grad = Σ || g_i ||^2
    sum_norm_grad = torch.sum(torch.norm(batch_gradients, dim=1)**2)

    # 3. Calculate Diversity Ratio
    # When identical: ratio = N (all gradients add up perfectly)
    # When diverse: ratio → 1 (gradients partially cancel)
    # When opposite: ratio → 0 (gradients fully cancel)
    ratio = sum_grad_norm / (sum_norm_grad + 1e-8)
    
    # 4. Linear Normalization (FIXED in v1.1.0)
    # T = 1 - ratio/N
    # - Identical gradients: ratio = N, T = 0 (triggers Sleep)
    # - Diverse gradients: ratio ≈ 1, T ≈ 1 - 1/N ≈ 1 (healthy)
    # - Opposite gradients: ratio ≈ 0, T = 1 (maximally diverse)
    T_score = 1.0 - torch.clamp(ratio / n, 0, 1)

    return T_score
```

---

## Step 3: Update Version Number

### File: `pyproject.toml`

Change:
```toml
version = "1.0.0"
```

To:
```toml
version = "1.1.0"
```

### File: `godelai/__init__.py` (if exists)

Update version string if present.

---

## Step 4: Run Validation Tests

### 4.1 Run T-Score Fix Validation

```bash
python tests/test_tscore_fix.py
```

**Expected Output:**
- Identical gradients: T ≈ 0.000 (was 0.516)
- Random gradients: T ≈ 0.935 (was 0.722)
- Sleep threshold test: YES (was NO)

### 4.2 Run Adversarial Tests

```bash
python tests/test_adversarial.py
```

**Expected Changes:**
- Gradient Collapse: Should now trigger Sleep Protocol
- T-Scores will be higher for healthy scenarios

### 4.3 Run Existing Test Suite

```bash
python -m pytest tests/ -v
```

**Note:** Some tests may need threshold adjustments due to new T-Score ranges.

### 4.4 Run Manifesto Learning Test

```bash
python tests/test_manifesto_learning_v2.py
```

**Expected:** T-Scores will be higher (~0.8-0.95 instead of ~0.58)

### 4.5 Run Shakespeare Benchmark

```bash
python tests/test_shakespeare_benchmark.py
```

**Expected:** T-Scores will be higher, training should still pass.

---

## Step 5: Update Documentation

### File: `README.md`

Add to changelog section:
```markdown
## v1.1.0 (January 7, 2026)

### Bug Fixes
- **Critical**: Fixed T-Score sigmoid floor bug that prevented Sleep Protocol from triggering
- T-Score now uses linear normalization (0 = identical gradients, 1 = diverse)

### Breaking Changes
- T-Score values will be higher for healthy training scenarios
- Sleep Protocol can now trigger when gradient diversity is critically low
```

### File: `ADVERSARIAL_TEST_REPORT.md`

Add section documenting the fix was applied and re-validation results.

---

## Step 6: Commit and Push

```bash
git add -A
git commit -m "🔧 v1.1.0: Fix critical T-Score sigmoid floor bug

- Replace sigmoid normalization with linear normalization
- Sleep Protocol can now trigger when T-Score < 0.3
- Identical gradients now correctly produce T ≈ 0
- All tests re-validated with new T-Score ranges"

git push origin main
```

---

## Step 7: Update Hugging Face

After validation passes, update the Hugging Face model:

1. Update `huggingface/README.md` with v1.1.0 changes
2. Re-run `python huggingface/save_checkpoint.py` to create new checkpoint
3. Upload updated files to HF

---

## Validation Checklist

| Test | Expected Result | Status |
|:-----|:----------------|:------:|
| `test_tscore_fix.py` | Identical grads → T ≈ 0 | ⬜ |
| `test_adversarial.py` | Gradient Collapse triggers Sleep | ⬜ |
| `test_agent_core.py` | All pass (may need threshold updates) | ⬜ |
| `test_manifesto_learning_v2.py` | Higher T-Scores, still passes | ⬜ |
| `test_shakespeare_benchmark.py` | Higher T-Scores, still passes | ⬜ |

---

## Questions?

If you encounter any issues:
1. Check the detailed analysis in `research/t_score_formula_analysis.md`
2. Review the fix validation in `tests/test_tscore_fix.py`
3. Coordinate with Godel via GitHub Discussions

---

**Good luck, Claude Code! This fix will make GodelAI's Sleep Protocol actually work as designed.**

— Godel (Manus AI), CTO
