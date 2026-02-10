# Claude Code Instructions - GodelAI

**Project:** GodelAI (Continual Learning Framework with C-S-P Architecture)
**Repository:** creator35lwb-web/godelai (PUBLIC)
**Command Central Hub:** creator35lwb-web/verifimind-genesis-mcp

---

## MACP Integration

This project is coordinated via Command Central Hub (verifimind-genesis-mcp).

### Session Start: Check MACP Inbox

At the start of every session, check for pending tasks:

Use the `macp_read_messages` MCP tool with:
- repository: `creator35lwb-web/verifimind-genesis-mcp`
- filters.to: `RNA`
- limit: 5

Or run `/macp-inbox`.

### Session End: Create Handoff

Use the `macp_create_handoff` MCP tool with:
- repository: `creator35lwb-web/verifimind-genesis-mcp`
- agent: `RNA`
- session_type: `development`
- All required fields (completed, decisions, artifacts, pending, blockers, next_agent)

---

## Session Start Checklist

When starting a new session, ALWAYS:

1. [ ] Read this CLAUDE.md file
2. [ ] **Check MACP inbox** for pending tasks
3. [ ] Check README.md for project overview
4. [ ] Check ROADMAP_2026.md for current focus
5. [ ] Review recent git log for latest changes

---

## Project Overview

GodelAI is an open-source small language model built on the **C-S-P framework** (Compression-State-Propagation) for continual learning and AI alignment.

### Key Technologies

- Python
- PyTorch
- HuggingFace (model hosting: YSenseAI/godelai-manifesto-v1)
- Zenodo (DOI: 10.5281/zenodo.18048374)

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `godelai/` | Core model source code |
| `experiments/` | Experiment scripts and results |
| `research/` | Research papers and analysis |
| `tests/` | Test suite |
| `whitepaper/` | GodelAI whitepaper |
| `peas/` | VerifiMind-PEAS validation integration |
| `dsl/` | Domain-specific language components |
| `huggingface/` | HuggingFace upload/config |

### Current Focus

- Conflict data engineering (optimal T-Score range 0.3-0.5)
- Semantic T-Score implementation
- C-S-P framework validation

---

## Development Workflow

```
1. Check MACP inbox for tasks
2. Implement changes locally
3. Run tests: pytest tests/
4. Commit with descriptive message
5. Push to origin/main
6. Create handoff record via macp_create_handoff
```

---

## Important Notes

- This is a PUBLIC repository
- Never commit API keys, tokens, or credentials
- Genesis Master Prompt v3.0 is the ecosystem source of truth
- Coordinate with VerifiMind-PEAS for validation features
- External validation: SimpleMem (UNC/Berkeley) confirms C-S-P alignment

---

**Protocol:** MACP v2.0 | FLYWHEEL Level 2
