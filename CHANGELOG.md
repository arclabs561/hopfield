# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Source history (unversioned)

These entries identify source changes without assigning a package release.

- Document existing energy-function panic conditions and Rust requirements:
  1.75 for default builds and 1.82 with `fynch`. Add compiler-floor CI checks;
  runtime behavior is unchanged.
  ([e2f951d](https://github.com/arclabs561/hopfield/commit/e2f951d28349a430a5b4663f632270eee75569ae))

## [Unreleased]

### Changed

- The optional `fynch` dependency now requires 0.3.2 (was 0.2.1). The built-in
  `sparsemax` stays local so the default build has no dependencies; a test
  under the `fynch` feature checks it against `fynch::sparsemax`.

## [0.3.0] - 2026-10-09

### Added
- `dot_product_weights` and `retrieve_dot_product`: the dot-product update of
  Ramsauer et al. (2020), Eq. 3.

### Changed
- Normalize LSE and LSR energies by inverse temperature, keeping their
  analytical gradients consistent with the documented definitions.
  `energy_lse`, `energy_lsr`, `energy_lse_grad` and `energy_lsr_grad` now
  return `1 / beta` times their previous values; results at `beta = 1` are
  unchanged.
- `energy_lse`, `energy_lsr`, `energy_lse_grad` and `energy_lsr_grad` panic
  when `beta` is not finite and positive; empty inputs still return early
  without checking `beta`.

### Fixed
- Docs no longer describe the distance-based LSE retrieval as Ramsauer's
  network; the two agree only for equal-norm memories.

## [0.2.2] - 2026-06-26

### Added
- `entmax_retrieval` example showing `fynch::Tsallis::entmax15()` as a
  Hopfield-Fenchel-Young retrieval map.
- `sparsemap_structured_retrieval` example showing SparseMAP over adjacent-pair
  structured memories.

## [0.2.1] - 2026-06-26

### Added
- Optional `fynch` feature with `regularized_weights` and
  `retrieve_regularized`, letting callers pass `fynch::Regularizer` values
  directly as Hopfield-Fenchel-Young separation maps.

## [0.2.0] - 2026-06-24

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
