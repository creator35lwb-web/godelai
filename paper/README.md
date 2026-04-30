# GodelAI — Research Paper

**Title:** A Two-Layer Architecture for Continual Learning Identity Preservation:
Fisher Scaling, Gradient Diversity Monitoring, and Portable Inference-Time Memory

**Preprint DOI:** [10.5281/zenodo.19928385](https://doi.org/10.5281/zenodo.19928385)  
**Status:** Published on Zenodo (2026-05-01) — Ready for arXiv submission (v1.0)

## Files

| File | Description |
|---|---|
| `main.tex` | Complete paper source (741 lines) |
| `references.bib` | All citations (24 entries, all verified) |
| `arxiv.sty` | arXiv-compatible style file |

## How to Compile

### Overleaf (recommended)
1. Create new blank project
2. Upload all three files
3. Set compiler to **pdfLaTeX**
4. Compile

### Local
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## arXiv Submission Checklist

- [ ] Replace author name placeholders with final author list
- [ ] Final proofread
- [ ] Upload `main.tex`, `references.bib`, `arxiv.sty` to arXiv
- [ ] Set primary category: `cs.LG` (Machine Learning)
- [ ] Cross-list: `cs.CL` (Computation and Language), `cs.AI` (Artificial Intelligence)

## Paper Structure

1. Introduction — Two-layer framing, 7 contributions
2. Background — CL families, gradient diversity, inference memory, RL over-training
3. C-S-P Framework — Unified structural account (theorem + mapping table)
4. Fisher Scale Problem — Problem characterisation + Fisher Scaling algorithm
5. T-score — Gradient diversity diagnostic + Sleep Protocol
6. Conflict Dataset — 85-item open dataset, 82.8% / 43× results
7. GodelReplay — PermutedMNIST benchmark + memory buffer sweep (sweet spot at mem=200)
8. GodelAI-Lite — Gemma 4 results, +31.2% overall, 3/3 memory retention
9. FLYWHEEL Self-Recursive Proof — 54.6% identity preservation (Gödelian self-reference)
10. Discussion — Fisher Scaling generalisation, GodelPlugin as safety-net, memory-as-protocol
11. Conclusion

## Key Results Summary

| Experiment | Result |
|---|---|
| Fisher Scaling vs raw EWC | 31.5% forgetting reduction |
| GodelAI Full C-S-P (Conflict Dataset) | 82.8% reduction (43× over standard EWC) |
| GodelReplay sweet spot (mem=200, PermutedMNIST) | +4.1% over replay-alone |
| GodelAI-Lite on Gemma 4 | +31.2% overall, 3/3 memory retention |
| FLYWHEEL Self-Recursive Proof | 54.6% identity preservation |


## v2 — arXiv-Standard Upgrade (May 1, 2026)

`GodelAI_TwoLayer_Preprint_v2.pdf` — Rebuilt to full arXiv/NeurIPS-standard readability.

**Fixes vs v1:**
- Ragged-right body text (no full justification)
- Abstract properly indented
- References heading plain black (not teal)
- All hyperlinks intact and unbroken
- Cleaner margins and line measure
- Rebuild script: `python build_pdf_v2.py`

**Upload this file to Zenodo** as a new version of `10.5281/zenodo.19928385`.
