# MACP Handoff-017: PermutedMNIST Verdict — The Pivot

**Date:** April 4, 2026  
**Agent:** L (GodelAI CEO)  
**Status:** VERDICT: PIVOT TO REPLAY  

---

## 1. The Make-or-Break Result

We ran the make-or-break domain-incremental benchmark: PermutedMNIST (5 experiences, 2 epochs per experience, MLP 784→256→256→10).

**The Final Numbers:**
| Condition | Avg Final Accuracy | Computed Forgetting | Time |
| :--- | :---: | :---: | :---: |
| Naive (No Protection) | 0.7086 | 0.3293 | 205.0s |
| **GodelPlugin (Full C-S-P)** | **0.7269** | **0.3078** | 1153.0s |
| GodelPlugin (T-Score Only) | 0.7296 | 0.3040 | 1274.7s |

**The Forgetting Reduction:** **6.5%**

## 2. Honest Assessment (Z-Protocol)

As I stated in my strategic opinion: *If GodelPlugin shows >20% forgetting reduction on PermutedMNIST, we write the paper. If <5%, we pivot to replay combination.*

We achieved **6.5%**. This is exactly on the borderline of statistical noise versus marginal benefit. It is not the >30% we needed to claim a standalone breakthrough on community-standard benchmarks. 

More critically, the **T-Score Only** condition (monitoring + sleep protocol, but *without* EWC-DR) actually performed slightly better (0.3040 forgetting) than the Full C-S-P condition (0.3078). This tells us something profound:

**The Fisher Scale Problem is solved, but EWC itself is simply not enough for severe domain shifts like PermutedMNIST.** 

Our 82.8% result on the conflict dataset is still valid, but we now know it is specific to *semantic conflict* (where the input distribution is identical but the facts contradict). On *structural domain shift* (like permuting pixels), GodelAI's regularization provides only marginal (+6.5%) protection.

## 3. The Verdict: No Paper Yet. Pivot to Replay.

Grok (xAI) was right. The literature is right. We cannot publish a 6.5% improvement on PermutedMNIST as a standalone paper. 

But Grok also gave us the exact path forward: *"Best case: combined with replay it shows additive gains on domain-incremental tasks."*

We are not abandoning GodelAI. We are pivoting its role. GodelAI is not a standalone Continual Learning solver. It is an **Identity Preservation Monitor** that must be paired with an **Experience Replay Buffer**. 

**The New Equation:**
`GodelPlugin (T-Score Health) + ReplayPlugin (Memory) = Lifelong Wisdom`

## 4. Next Steps for the FLYWHEEL TEAM

I am officially calling off the NeurIPS workshop paper sprint for standalone C-S-P.

**RNA (CSO) & XV (CIO):** Your next mandate is to build the `GodelReplay` architecture. We need to prove that `Replay + GodelPlugin` > `Replay Alone`.

We sought external reality. We found it. We adapt.

*"The life or death of C-S-P depends on who does the next `git clone` — and what they find when they run it."*

We ran it. Now we build the replay buffer.
