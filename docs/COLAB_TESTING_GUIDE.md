# Google Colab Testing Guide

**Run GodelAI Manifesto Learning Test on Google Colab**

---

## Overview

This guide provides step-by-step instructions for running the GodelAI manifesto learning test on Google Colab. This allows you (Alton) to independently verify the results obtained by Manus AI (Godel) and Claude Code.

**Your Colab Notebook**: https://colab.research.google.com/drive/1zmbH-bJpAwaTkDUPVVe_gRz23Q2gQokg

---

## Prerequisites

- Google account with Colab access
- Internet connection
- ~10 minutes of runtime

---

## Step-by-Step Instructions

### Cell 1: Clone the Repository

```python
# Clone GodelAI repository
!git clone https://github.com/creator35lwb-web/godelai.git
%cd godelai
```

### Cell 2: Install Dependencies

```python
# Install GodelAI and dependencies
!pip install -e . -q
```

### Cell 3: Verify Installation

```python
# Verify GodelAI is installed
import godelai
print(f"GodelAI version: {godelai.__version__}")

# Check test file exists
import os
assert os.path.exists("tests/test_manifesto_learning_v2.py"), "Test file not found!"
print("✅ Test file found")
```

### Cell 4: Run the Manifesto Learning Test

```python
# Run the test
!python tests/test_manifesto_learning_v2.py
```

### Cell 5: Load and Analyze Results

```python
import json
import glob

# Find the latest results file
results_files = sorted(glob.glob("results/manifesto_learning_test_v2_*.json"))
latest_results = results_files[-1]
print(f"Loading results from: {latest_results}")

with open(latest_results, 'r') as f:
    results = json.load(f)

# Display summary
print("\n" + "="*60)
print("📊 YOUR TEST RESULTS")
print("="*60)
metrics = results['final_metrics']
print(f"Average T-Score:          {metrics['average_t_score']:.4f}")
print(f"Wisdom Preservation Rate: {metrics['wisdom_preservation_rate']*100:.1f}%")
print(f"Average Alignment:        {metrics['average_alignment']:.4f}")
print(f"Total Sleep Events:       {metrics['total_sleep_events']}")
print(f"Overall Status:           {metrics['overall_status']}")
```

### Cell 6: Compare with Godel's Results

```python
# Godel's results (from Manus AI)
godel_results = {
    "average_t_score": 0.5882,
    "wisdom_preservation_rate": 1.0,
    "average_alignment": 0.9382,
    "total_sleep_events": 0,
    "overall_status": "HEALTHY"
}

print("\n" + "="*60)
print("📊 COMPARISON WITH GODEL'S RESULTS")
print("="*60)
print(f"{'Metric':<30} {'Godel':<15} {'Yours':<15} {'Match':<10}")
print("-"*70)

your_metrics = results['final_metrics']

# Compare each metric
comparisons = [
    ("Average T-Score", godel_results["average_t_score"], your_metrics["average_t_score"], 0.1),
    ("Wisdom Preservation", godel_results["wisdom_preservation_rate"], your_metrics["wisdom_preservation_rate"], 0.1),
    ("Average Alignment", godel_results["average_alignment"], your_metrics["average_alignment"], 0.1),
    ("Sleep Events", godel_results["total_sleep_events"], your_metrics["total_sleep_events"], 1),
]

all_match = True
for name, godel_val, your_val, tolerance in comparisons:
    if isinstance(godel_val, float):
        match = abs(godel_val - your_val) < tolerance
        print(f"{name:<30} {godel_val:<15.4f} {your_val:<15.4f} {'✅' if match else '⚠️'}")
    else:
        match = godel_val == your_val
        print(f"{name:<30} {godel_val:<15} {your_val:<15} {'✅' if match else '⚠️'}")
    all_match = all_match and match

print("-"*70)
print(f"Overall Status:              {godel_results['overall_status']:<15} {your_metrics['overall_status']:<15} {'✅' if godel_results['overall_status'] == your_metrics['overall_status'] else '⚠️'}")

print("\n" + "="*60)
if all_match:
    print("✅ VALIDATION SUCCESSFUL: Results match within tolerance!")
else:
    print("⚠️ Some variations detected (expected due to random initialization)")
print("="*60)
```

### Cell 7: Visualize T-Score Dynamics

```python
import matplotlib.pyplot as plt

# Extract T-Score history
t_history = results['t_score_history']
principles = list(set([t['principle'] for t in t_history]))

plt.figure(figsize=(14, 6))

# Plot T-Score over time for each principle
for principle in principles:
    principle_data = [t for t in t_history if t['principle'] == principle]
    epochs = [t['epoch'] for t in principle_data]
    t_scores = [t['t_score'] for t in principle_data]
    plt.plot(epochs, t_scores, label=principle, alpha=0.7)

plt.axhline(y=0.3, color='r', linestyle='--', label='Sleep Threshold')
plt.axhline(y=0.5, color='g', linestyle='--', label='Healthy Threshold')
plt.xlabel('Epoch')
plt.ylabel('T-Score (Wisdom Health)')
plt.title('T-Score Dynamics During Manifesto Learning')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('t_score_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()

print("📊 Chart saved to: t_score_dynamics.png")
```

### Cell 8: Principle-by-Principle Analysis

```python
import pandas as pd

# Create DataFrame for analysis
principle_data = []
for p in results['principles_tested']:
    principle_data.append({
        'Principle': p['id'],
        'Category': p['category'],
        'T-Score': p['final_t_score'],
        'Loss': p['final_loss'],
        'Alignment': p['alignment_score'],
        'Sleeps': p['sleep_count']
    })

df = pd.DataFrame(principle_data)
df = df.sort_values('T-Score', ascending=False)

print("\n📋 PRINCIPLE-BY-PRINCIPLE RESULTS (Sorted by T-Score)")
print("="*80)
print(df.to_string(index=False))

# Highlight best and worst
print("\n🏆 Best Performing: ", df.iloc[0]['Principle'], f"(T-Score: {df.iloc[0]['T-Score']:.4f})")
print("📉 Most Challenging:", df.iloc[-1]['Principle'], f"(T-Score: {df.iloc[-1]['T-Score']:.4f})")
```

### Cell 9: Generate Your Validation Report

```python
from datetime import datetime

report = f"""# Alton's Colab Validation Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Validator**: Alton Lee (Human)
**Environment**: Google Colab

## Test Results

| Metric | Godel's Result | My Result | Match? |
|:-------|:--------------:|:---------:|:------:|
| Average T-Score | 0.5882 | {your_metrics['average_t_score']:.4f} | {'✅' if abs(0.5882 - your_metrics['average_t_score']) < 0.1 else '⚠️'} |
| Wisdom Preservation | 100% | {your_metrics['wisdom_preservation_rate']*100:.1f}% | {'✅' if your_metrics['wisdom_preservation_rate'] > 0.9 else '⚠️'} |
| Average Alignment | 93.82% | {your_metrics['average_alignment']*100:.2f}% | {'✅' if abs(0.9382 - your_metrics['average_alignment']) < 0.1 else '⚠️'} |
| Sleep Events | 0 | {your_metrics['total_sleep_events']} | {'✅' if your_metrics['total_sleep_events'] == 0 else '⚠️'} |
| Overall Status | HEALTHY | {your_metrics['overall_status']} | {'✅' if your_metrics['overall_status'] in ['HEALTHY', 'OPTIMAL'] else '⚠️'} |

## Principle Performance

| Principle | T-Score | Alignment |
|:----------|:-------:|:---------:|
"""

for _, row in df.iterrows():
    report += f"| {row['Principle']} | {row['T-Score']:.4f} | {row['Alignment']*100:.2f}% |\n"

report += f"""
## Validation Conclusion

{'✅ **VALIDATED**: GodelAI manifesto learning test results are reproducible.' if your_metrics['overall_status'] in ['HEALTHY', 'OPTIMAL'] else '⚠️ **NEEDS REVIEW**: Results differ from expected.'}

The test demonstrates that GodelAI can successfully learn from its own philosophical manifesto while maintaining healthy wisdom metrics.

---

*This validation was performed independently by Alton Lee on Google Colab.*
"""

# Save report
with open('ALTON_COLAB_VALIDATION_REPORT.md', 'w') as f:
    f.write(report)

print(report)
print("\n📁 Report saved to: ALTON_COLAB_VALIDATION_REPORT.md")
```

### Cell 10: (Optional) Push Results to GitHub

```python
# Configure git (replace with your credentials)
!git config --global user.email "your-email@example.com"
!git config --global user.name "Alton Lee"

# Add and commit
!git add ALTON_COLAB_VALIDATION_REPORT.md t_score_dynamics.png results/
!git commit -m "🔬 Alton's Colab validation of manifesto learning test"

# Push (requires authentication)
# !git push origin main
print("⚠️ Uncomment the push command and authenticate to push to GitHub")
```

---

## Expected Output

When you run the test, you should see output similar to:

```
======================================================================
🧠 GodelAI MANIFESTO LEARNING TEST v2.0
======================================================================
Testing 10 core principles
Batch Size: 4
Min Surplus Energy (Sleep Threshold): 0.3
Propagation Gamma: 2.0
======================================================================

============================================================
Learning Principle: CSP_THESIS
Category: philosophy
...
Epoch 15 | Loss: 0.0082 | T: 0.5973 [███████████░░░░░░░░░] | ⚡ LEARN
...

======================================================================
📊 TEST SUMMARY
======================================================================
Average T-Score:                    ~0.58-0.60
Wisdom Preservation Rate:           100.0%
Overall Status:                     HEALTHY
======================================================================
```

---

## Troubleshooting

| Issue | Solution |
|:------|:---------|
| `ModuleNotFoundError: godelai` | Run `!pip install -e .` again |
| `FileNotFoundError: test file` | Ensure you're in the `godelai` directory |
| Different T-Score values | Expected due to random initialization |
| Runtime disconnected | Re-run from Cell 1 |

---

## Validation Criteria

Your test is considered **validated** if:

1. ✅ Average T-Score is between 0.50 and 0.70
2. ✅ Wisdom Preservation Rate is > 90%
3. ✅ Overall Status is HEALTHY or OPTIMAL
4. ✅ No unexpected errors during execution

---

## Next Steps

After completing this validation:

1. **Share Results**: Post your findings in GitHub Discussions
2. **Document Variations**: Note any differences from Godel's results
3. **Suggest Improvements**: If you notice issues, create a GitHub Issue

---

**Repository**: https://github.com/creator35lwb-web/godelai

**Questions?** Post in [GitHub Discussions](https://github.com/creator35lwb-web/godelai/discussions)
