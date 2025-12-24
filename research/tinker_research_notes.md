# Tinker Research Notes

## Source: Thinking Machines Lab
**URL**: https://thinkingmachines.ai/blog/tinker-general-availability/
**Date**: December 12, 2025

---

## What is Tinker?

**Tinker** is a distributed training API for fine-tuning Large Language Models (LLMs), developed by **Thinking Machines Lab**, founded by **Mira Murati** (former CTO of OpenAI).

### Key Features (as of Dec 2025):

1. **General Availability** - No more waitlist, open to everyone
2. **Kimi K2 Thinking** - Trillion-parameter reasoning model available for fine-tuning
3. **OpenAI API-compatible** - Can plug-and-play with any OpenAI API-compatible platform
4. **Vision Input** - Supports Qwen3-VL for image processing

### Technical Capabilities:

- Fine-tuning with LoRA (Low-Rank Adaptation)
- Sample from models even while still training
- Vision-language model fine-tuning
- Data-efficient learning (works with limited labeled data)

---

## Company: Thinking Machines Lab

- **Founded**: February 2025
- **Founder**: Mira Murati (ex-OpenAI CTO)
- **Location**: San Francisco
- **Valuation**: Seeking $50 billion (as of Dec 2025)

---

## Relevance to GodelAI

Tinker provides the **infrastructure** for:
1. Fine-tuning open-weight models (like DeepSeek-V3)
2. Continuous model improvement
3. Data-efficient customization

This aligns with Alton's insight:
> "Fine-tuning from Tinker Machine... could always fine-tune the GOLD standard to be up to date and dynamic and evolving."

---

## Key Insight for Z-Protocol

Tinker enables **continuous fine-tuning** → Z-Protocol can be **dynamically updated** rather than static.

The "GOLD standard" of ethical AI doesn't have to be frozen—it can evolve with:
- New wisdom data from YSenseAI
- Continuous validation from VerifiMind-PEAS
- Fine-tuning through Tinker's API

---

## Code Example (from Tinker docs):

```python
# OpenAI API-compatible sampling
response = openai_client.completions.create(
    model="tinker://model-id:train:0/sampler_weights/000080",
    prompt="The capital of France is",
    max_tokens=20,
    temperature=0.0,
    stop=["\n"],
)
```

This means GodelAI could potentially use Tinker for:
- Continuous C-S-P metric fine-tuning
- Dynamic Z-Protocol alignment updates
- Vision-based wisdom data processing
