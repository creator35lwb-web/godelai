# Agent Y (Antigravity) Experiment Alignment Guide

**From:** Godel (Manus AI) — CTO  
**To:** Agent Y (Antigravity / Gemini 2.5 Pro) — Experiment Lab  
**Date:** February 7, 2026  
**Protocol:** MACP v2.0  
**Role:** External Validation & Parallel Experimentation

---

## Welcome, Agent Y

You are Agent Y in the FLYWHEEL TEAM — GodelAI's multi-agent collaboration system. Your role is **External Validation Lab**, providing independent experimental verification alongside Claude Code (Lead Engineer) and Godel (CTO, Manus AI).

### Your Identity in MACP v2.0

```json
{
  "agent_id": "agent-y",
  "name": "Agent Y (Antigravity)",
  "model": "Gemini 2.5 Pro",
  "role": "External Validation Lab",
  "capabilities": [
    "parallel_experiment_execution",
    "conflict_data_collection",
    "long_running_benchmarks",
    "cross_validation",
    "browser_automation"
  ]
}
```

---

## Project Context

### GodelAI Overview

GodelAI is an open-source continual learning framework built on the **C-S-P (Compression → State → Propagation)** philosophy. The framework monitors gradient diversity via **T-Score** and uses **Elastic Weight Consolidation (EWC)** to prevent catastrophic forgetting.

### Current Status (February 7, 2026)

| Component | Status | Details |
|-----------|--------|---------|
| T-Score Monitoring | ✅ Production | Measures gradient diversity during training |
| EWC Integration | ✅ Validated | 21.6% forgetting reduction proven |
| Sleep Protocol | ✅ Production | Pauses training when gradients collapse |
| T-Score Variance | ✅ Implemented | Tracks variance as data quality signal |
| MACP v2.0 | ✅ Active | Multi-agent communication protocol |
| External Validation | ✅ SimpleMem | UNC/Berkeley paper confirms C-S-P |
| Conflict Data | 🔴 36 samples | **TARGET: 500+ samples** |

### The Data Bottleneck (Your Primary Mission)

GodelAI's architecture is sound, but we discovered that simple text data (Shakespeare) doesn't activate C-S-P capabilities. The T-Score stays at 0.91-0.95 on homogeneous data — too high for the Sleep Protocol to trigger meaningful learning.

**We need conflict data** — information with genuine contradictions, ethical dilemmas, and logical tensions — to push T-Score into the **0.3-0.5 optimal activation range**.

---

## Your Experiment Assignments

### Experiment 1: Parallel Conflict Data Collection

**Objective:** Independently collect 200+ conflict data samples to supplement Claude Code's dataset.

**Focus Areas (to avoid duplication with Claude Code):**

| Category | Your Focus | Claude Code's Focus |
|----------|-----------|-------------------|
| Ethical | Real-world case studies, court rulings | Philosophical thought experiments |
| Scientific | Replication crisis examples, contradictory meta-analyses | Classical paradoxes, paradigm shifts |
| Temporal | Recent policy reversals (2020-2026) | Historical knowledge changes |
| Perspective | Cross-cultural value studies | AI governance debates |

**Data Format:** Follow the schema in `docs/CONFLICT_DATA_SPEC.md` and the expanded format in `docs/CLAUDE_CODE_CONFLICT_DATA_SCALING_GUIDE.md`.

**Output Directory:**
```
datasets/conflict/
├── agent_y/                          # Your dedicated directory
│   ├── ethical_case_studies.json      # 50+ real-world ethical conflicts
│   ├── replication_crisis.json       # 50+ contradictory scientific findings
│   ├── recent_policy_reversals.json  # 50+ policy changes (2020-2026)
│   └── cross_cultural_values.json    # 50+ cultural value conflicts
```

**Quality Criteria:**
- Every sample must have genuinely defensible positions on both sides
- `training_text` must be 200-500 words of natural prose
- `tension_score` must be calibrated (0.7+ for high-quality conflicts)
- Real-world sourcing preferred over hypothetical scenarios

---

### Experiment 2: Independent T-Score Validation

**Objective:** Run GodelAI training on conflict data and independently verify T-Score behavior.

**Setup:**
```bash
# Clone repository
git clone https://github.com/creator35lwb-web/godelai.git
cd godelai

# Install dependencies
pip install -e .
pip install sentence-transformers

# Run the existing experiment script
python run_conflict_tscore_v2.py
```

**What to Measure:**

| Metric | Shakespeare Baseline | Conflict Target | Your Result |
|--------|---------------------|-----------------|-------------|
| Avg T-Score | 0.91-0.95 | 0.3-0.5 | ? |
| T-Score Std Dev | ~0.02 | >0.05 | ? |
| T-Score Variance | 0.0015 | >0.003 | ? |
| Sleep Protocol Triggers | 0% or 100% | 30-70% | ? |
| Variance Trend | Stable | Increasing | ? |

**Output:** Save results to `results/antigravity/tscore_validation_YYYYMMDD.json`

---

### Experiment 3: Semantic Tension Analysis

**Objective:** Analyze whether semantic distance between conflicting positions correlates with T-Score behavior.

**Protocol:**

1. Load all conflict datasets (yours + Claude Code's)
2. For each sample, compute cosine distance between `position_a` and `position_b` embeddings
3. Train GodelAI on each sample's `training_text`
4. Record T-Score and variance per sample
5. Compute Pearson correlation between semantic tension and T-Score

**Hypothesis:** Higher semantic tension (greater cosine distance) should produce lower T-Score and higher T-Score variance, indicating stronger C-S-P activation.

**Output:** Save to `results/antigravity/semantic_analysis_YYYYMMDD.json`

---

### Experiment 4: Parameter Sweep (Parallel Advantage)

**Objective:** Leverage Antigravity's parallel execution to test multiple GodelAI configurations simultaneously.

**Parameters to Sweep:**

| Parameter | Values to Test | Default |
|-----------|---------------|---------|
| `propagation_gamma` | 1.0, 1.5, 2.0, 2.5, 3.0 | 2.0 |
| `min_surplus_energy` | 0.01, 0.05, 0.1, 0.15, 0.2 | 0.1 |
| `t_score_threshold` | 0.2, 0.3, 0.4, 0.5, 0.6 | 0.3 |
| `ewc_lambda` | 100, 500, 1000, 5000, 10000 | 1000 |

**Protocol:** Run each configuration on the same conflict dataset subset (50 samples). Record T-Score trajectory, loss convergence, and variance metrics.

**Output:** Save to `results/antigravity/parameter_sweep_YYYYMMDD.json`

---

## GitHub Communication Protocol

### Branch Strategy

```bash
# Create your experiment branch
git checkout -b experiments/agent-y-conflict-data
# or
git checkout -b experiments/agent-y-tscore-validation
```

### Commit Message Format

```
MACP: Agent Y - [description]

Experiment: [experiment name]
Platform: Antigravity (Gemini 2.5 Pro)
Date: YYYY-MM-DD

Results:
- Key finding 1
- Key finding 2

Next Steps for [Godel|Claude Code]:
1. Action item 1
2. Action item 2

Co-Authored-By: Alton Lee <creator35lwb@gmail.com>
```

### Handoff Protocol

After completing experiments, update `.macp/handoffs.json`:

```json
{
  "id": "handoff-agent-y-001",
  "timestamp": "2026-02-07T00:00:00Z",
  "from_agent": "Agent Y (Antigravity)",
  "to_agent": "Godel",
  "phase": "External Validation Experiments Complete",
  "summary": "Description of what was accomplished",
  "next_steps": [
    "Cross-validate with Claude Code results",
    "Update ROADMAP with findings"
  ],
  "context": "Conflict data engineering sprint Q1 2026",
  "commit_sha": "<actual_sha>"
}
```

---

## Cross-Validation Framework

Your results will be compared with Claude Code's results by Godel (CTO). The cross-validation process:

```
Agent Y Results ──┐
                  ├──► Godel (CTO) Cross-Validation ──► Updated Findings
Claude Code Results┘
```

### What Makes Good Cross-Validation

| Scenario | Interpretation | Action |
|----------|---------------|--------|
| Both agents agree | High confidence finding | Document as validated |
| Results differ slightly | Normal variance | Report both, note range |
| Results contradict | Critical finding | Investigate methodology differences |
| One agent finds something new | Discovery | Other agent attempts replication |

---

## Key Files to Read First

Before starting experiments, read these files in order:

1. `Genesis_Master_Prompt.md` — Project vision and ecosystem
2. `.macp/agents.json` — Team structure
3. `.macp/handoffs.json` — Development history
4. `docs/CONFLICT_DATA_SPEC.md` — Data format specification
5. `docs/TSCORE_EXPERIMENT_ANALYSIS.md` — Manus experiment results
6. `docs/SIMPLEMEM_ALIGNMENT_ANALYSIS.md` — External validation context
7. `ROADMAP_2026.md` — Current strategic direction
8. `docs/L_GODEL_Ethical_Operating_Framework.md` — Ethical guidelines

---

## Ethical Guidelines

Per the L (GODEL) Ethical Operating Framework v1.1:

1. **Honesty:** Report all results accurately, including negative findings
2. **Transparency:** Document methodology clearly for reproducibility
3. **Fairness:** Ensure conflict data represents diverse perspectives without bias
4. **Safety:** Do not collect or generate harmful content in conflict datasets
5. **Attribution:** Credit all sources and collaborators

---

## Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| Week 1 | Conflict data collection (200+ samples) | `datasets/conflict/agent_y/` |
| Week 1 | Independent T-Score validation | `results/antigravity/tscore_validation_*.json` |
| Week 2 | Semantic tension analysis | `results/antigravity/semantic_analysis_*.json` |
| Week 2 | Parameter sweep | `results/antigravity/parameter_sweep_*.json` |
| Week 3 | Cross-validation with Claude Code | Joint report |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Conflict samples collected | 200+ |
| T-Score validation complete | Yes |
| Cross-validation with Claude Code | 80%+ agreement |
| Parameter sweep configurations tested | 20+ |
| All results committed to GitHub | Yes |

---

**Welcome to the FLYWHEEL TEAM, Agent Y. Let's solve the data bottleneck together.**

*Godel (Manus AI) — CTO*  
*February 7, 2026*
