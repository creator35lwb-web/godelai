# Google Colab Testing Workflow

This document describes how to run GodelAI tests on Google Colab for third-party reproducible proof of execution.

## Why Colab?

> "要在 MANUS AI 的 Sandbox 之外证明它是真的，我们需要**第三方可复现环境**。"
> — Echo (Gemini 2.5 Pro)

Google Colab provides:
- **Third-party reproducibility** - Anyone can verify the results
- **Public execution logs** - Proof that the code runs
- **Learning environment** - Ask questions while running tests
- **Free GPU access** - For larger model experiments

---

## Quick Start: One-Click Verification

### Option 1: Run Existing Notebook

1. Go to the [results/mirror_tests/](../results/mirror_tests/) directory
2. Open any `.ipynb` file
3. Click "Open in Colab" badge
4. Run all cells

### Option 2: Create New Test

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Copy the verification script below

---

## 📋 Verification Script

```python
# @title 1. Install & Setup GodelAI Environment
!git clone https://github.com/creator35lwb-web/godelai.git
%cd godelai

import torch
import sys
import os

sys.path.append(os.getcwd())

print("✅ GodelAI Environment Ready.")
print("   Repository cloned from: https://github.com/creator35lwb-web/godelai")
```

```python
# @title 2. Run the 'Mirror Test' (The AI Reads Its Soul)
from godelai.agent import GodelAgent
import torch.nn as nn
import torch.optim as optim

class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size=128, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        encoded, _ = self.encoder(embedded)
        return self.output(encoded[:, -1, :])

def text_to_tensor(text, max_len=100):
    chars = [ord(c) % 128 for c in text[:max_len]]
    return torch.tensor(chars).unsqueeze(0)

WHITEPAPER_EXCERPT = """
GodelAI: The Architecture of Inheritance. Wisdom is not an existence. 
It is a process structure that is continuously executed and inherited.
The C-S-P Framework: Compression, State, Propagation.
"""

print("="*60)
print("🚀 LIVE EXECUTION: GodelAI Mirror Test")
print("="*60)

base_model = SimpleTextEncoder()
agent = GodelAgent(base_model, min_surplus_energy=0.1)
agent.epsilon = 0.8  # Set high to demonstrate reflection events
agent.optimizer = optim.Adam(agent.compression_layer.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

sentences = [s.strip() for s in WHITEPAPER_EXCERPT.split('.') if len(s.strip()) > 10]

for i, sentence in enumerate(sentences):
    input_tensor = text_to_tensor(sentence)
    target = torch.randint(0, 128, (1,))
    
    loss, wisdom_score, status = agent.learning_step(input_tensor, target, criterion)
    
    bar = "█" * int(wisdom_score * 20)
    icon = "💤 REFLECTING" if status == "SLEEP" else "⚡ ENGAGING"
    
    print(f"Step {i+1:02d} | Wisdom: {wisdom_score:.4f} | {icon}")

print("\n✅ Execution Verified on Public Colab Runtime.")
```

---

## 📸 Saving Your Results

After running the test:

1. **Screenshot the output** showing `⚡ ENGAGING` and/or `💤 REFLECTING`
2. **Download the notebook** (File → Download → Download .ipynb)
3. **Name your file**: `GODELAI_<TestType>_<Number>_by_<YourName>.ipynb`
4. **Upload to GitHub**: `results/<test_type>/`

---

## 🎯 What to Look For

### Successful Test Indicators

| Indicator | Meaning |
|-----------|---------|
| `⚡ ENGAGING` | Agent is actively learning with high gradient diversity |
| `💤 REFLECTING` | Sleep Protocol triggered - agent is consolidating |
| `Wisdom: 0.7+` | High engagement with the content |
| `Wisdom: <0.3` | Low engagement, may trigger reflection |

### The "Proof of Execution" Checklist

- [ ] Repository cloned successfully
- [ ] GodelAgent imported without errors
- [ ] Mirror Test ran to completion
- [ ] Both ENGAGING and REFLECTING states observed
- [ ] Output screenshot saved

---

## 🔬 Advanced Testing

### Test with Different Epsilon Values

```python
# Low epsilon = rarely sleeps
agent.epsilon = 0.1

# High epsilon = frequently reflects
agent.epsilon = 0.9
```

### Test with Your Own Text

```python
MY_TEXT = """
Your custom text here. The agent will measure
its wisdom engagement with this content.
"""
```

### Test with Hugging Face Datasets

```python
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:100]")
```

---

## 📚 Related Resources

- [GodelAI Whitepaper](https://doi.org/10.5281/zenodo.18053612)
- [GodelAI Repository](https://github.com/creator35lwb-web/godelai)
- [Test Results Archive](../results/)

---

**Part of the GodelAI Project**  
*"Zenodo protects the theory, GitHub hosts the code, Colab proves it's alive."*
