# GodelAI Datasets

This directory contains datasets for testing and training GodelAI.

## Directory Structure

```
datasets/
├── conflict/           # Datasets designed to activate C-S-P capabilities
│   ├── ethical_dilemmas/
│   ├── contradictory_facts/
│   ├── temporal_conflicts/
│   └── perspective_conflicts/
└── wisdom/             # Placeholder for future YSenseAI integration
```

## Why Conflict Data?

GodelAI's C-S-P (Consciousness-Sleep-Perception) architecture is designed to handle complexity, contradiction, and change. Simple text data (like Shakespeare) doesn't activate these capabilities.

**The Data Bottleneck Discovery (January 16, 2026):**

| Data Type | T-Score | Sleep Protocol | Learning |
|-----------|---------|----------------|----------|
| Mini Shakespeare (5KB) | 0.12 | 100% activation | Blocked |
| Full Shakespeare (1.1MB) | 0.95 | 0% activation | Normal |
| **Conflict Data (target)** | **0.3-0.5** | **10-50% activation** | **Optimal** |

Conflict data forces the system to "think" by presenting:
- Ethical dilemmas with no clear answer
- Contradictory facts requiring uncertainty handling
- Temporal changes requiring belief updates
- Multiple perspectives requiring synthesis

## Specification

See [`docs/CONFLICT_DATA_SPEC.md`](../docs/CONFLICT_DATA_SPEC.md) for detailed specifications including:
- JSON schema for each conflict type
- Expected T-Score behavior
- Success criteria
- Quality checklist

## Contributing Datasets

We welcome contributions of conflict datasets! To contribute:

1. **Choose a conflict type** from the specification
2. **Create JSON file** following the schema
3. **Test with GodelAI** and document T-Score behavior
4. **Submit pull request** with your dataset and results

### Quality Requirements

- [ ] No single "correct" answer
- [ ] Multiple perspectives represented fairly
- [ ] Complexity score justified (0-1 scale)
- [ ] JSON schema validated
- [ ] T-Score behavior documented

## Future: YSenseAI Integration

The `wisdom/` directory is a placeholder for future integration with YSenseAI. When YSenseAI becomes a production platform collecting "wisdom data" with user consent, this directory will contain:

- Multi-modal wisdom inputs
- Real-time streaming conflicts
- Z-Protocol compliant data
- Consented user contributions

---

*Dataset structure created by Godel (Manus AI) — CTO, GodelAI*
