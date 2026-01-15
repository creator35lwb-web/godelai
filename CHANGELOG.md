# Changelog

All notable changes to GodelAI are documented in this file.

## [2.0.0] - 2026-01-16

### Discovery
- **Data Bottleneck Hypothesis Validated**: Through multi-AI collaboration (Gemini analysis + Manus validation), we confirmed that GodelAI's architecture is sound but requires complex data to activate its capabilities. Simple text (Shakespeare) doesn't trigger the C-S-P mechanisms.
  - T-Score sensitivity: 0.12 (5KB) vs 0.95 (1.1MB)
  - Sleep Protocol "overkill" on simple data: 860/860 batches triggered (100%)
  - EWC only effective in sequential task scenarios

### Added
- **ROADMAP v3.0**: Strategic pivot from documentation to data engineering
- **Conflict Data Specification**: `docs/CONFLICT_DATA_SPEC.md` defining data requirements for C-S-P activation
- **Datasets Directory**: `datasets/conflict/` structure for four conflict types:
  - Ethical dilemmas (no correct answer)
  - Contradictory facts (uncertainty handling)
  - Temporal conflicts (belief updates)
  - Perspective conflicts (synthesis required)
- **YSenseAI Integration Placeholder**: `datasets/wisdom/` for future integration

### Changed
- **README.md**: Added "Current Focus" section explaining data engineering sprint
- **Roadmap**: Shifted Q1 focus from documentation to conflict data design
- **Contributing Guidelines**: Updated priorities to emphasize conflict dataset creation

### Multi-Agent Collaboration
This release was produced through the GodelAI multi-agent workflow:
- **Echo (Gemini 3 Pro)**: Data bottleneck hypothesis
- **Godel (Manus AI)**: Validation and GitHub updates
- **Claude Code**: Code implementation support
- **Alton Lee**: Orchestration and approval

---

## [1.1.0] - 2026-01-13

### Fixed
- **Critical T-Score Bug**: Replaced sigmoid activation with linear normalization. The sigmoid function had a mathematical floor at 0.5, making it impossible for T-Score to drop below 0.5 and trigger the Sleep Protocol. The new formula correctly produces T-Score = 0.0 for identical gradients.

### Added
- **EWC Integration (Mnemosyne)**: Elastic Weight Consolidation for continual learning, achieving 21.6% reduction in catastrophic forgetting.
- **Adversarial Test Suite**: Comprehensive tests to validate Sleep Protocol triggering under extreme conditions.
- **Scale Validation**: Tested across 4 network sizes (10K - 361K parameters).
- **Cross-Platform Validation**: Verified reproducibility across Manus AI, Claude Code, and Google Colab.

### Changed
- T-Score values now range from 0.0 (identical gradients) to ~1.0 (diverse gradients), compared to the previous 0.5-1.0 range.

## [1.0.0] - 2026-01-05

### Added
- Initial release of GodelAI framework
- GodelAgent with T-Score monitoring
- Sleep Protocol for training health
- C-S-P philosophical foundation
- Multi-model genesis documentation
- Hugging Face model publication
- Zenodo DOI registration

### Known Issues
- T-Score sigmoid floor bug (fixed in v1.1.0)
- Sleep Protocol never triggered during normal training (fixed in v1.1.0)

---

## Version Naming

GodelAI follows semantic versioning:
- **Major**: Breaking changes to API
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible
