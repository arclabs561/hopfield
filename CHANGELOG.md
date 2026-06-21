# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `retrieve_fy`: Hopfield-Fenchel-Young retrieval generalized over the
  separation map. `retrieve_lse` and `retrieve_sparsemax` are the softmax and
  sparsemax instances; any Fenchel-Young regularized-argmax (for example
  α-entmax from `fynch`) is a valid map, and sparse maps give exact single-step
  retrieval within the basin (Santos et al. 2024, arXiv:2411.08590).
- Dense and sparse one-step retrieval helpers: `softmax`, `sparsemax`,
  `lse_weights`, `sparsemax_weights`, `weighted_memory`, `retrieve_lse`, and
  `retrieve_sparsemax`.
- Examples for associative recall, sparse attention support, and LSE/LSR basin
  scanning.

## [0.1.1] - 2026-04-16

### Added
- Initial release: associative-memory energy functions (LSE, LSR, retrieval).
