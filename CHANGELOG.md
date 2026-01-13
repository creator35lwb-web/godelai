# Changelog

All notable changes to GodelAI are documented in this file.

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
