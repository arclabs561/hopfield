# hopfield

Dense associative memory and modern Hopfield networks.

Energy-based content-addressable memory whose storage capacity grows
exponentially in the pattern dimension when the stored patterns are
well-separated (Ramsauer et al. 2021; capacity-as-spherical-codes analysis in
Hu et al. 2024). Two energy formulations: LSE (smooth, approximate retrieval)
and LSR (Epanechnikov kernel, exact single-step retrieval within the basin
radius, Hoover et al. 2025). Also includes dense and sparse one-step retrieval
maps (`retrieve_lse`, `retrieve_sparsemax`) for attention-style use.

## Usage

```toml
[dependencies]
hopfield = "0.1"
```

```rust
use hopfield::{retrieve_memory, energy_lse_grad};

// Store two memories as 2D points
let memories = vec![
    vec![0.0, 0.0],
    vec![10.0, 10.0],
];

// Query near the first memory
let query = vec![0.5, 0.5];
let (retrieved, iters) = retrieve_memory(
    query,
    &memories,
    |v, m| energy_lse_grad(v, m, 2.0),  // beta=2 inverse temperature
    0.1,   // learning rate
    100,   // max iterations
    1e-6,  // convergence tolerance
);

// Converges to [0, 0]
assert!(retrieved[0].abs() < 1.0);
assert!(retrieved[1].abs() < 1.0);
```

## Examples

```bash
cargo run --example associative_recall --release
cargo run --example sparse_attention --release
cargo run --example basin_scan --release
```

See [`examples/README.md`](examples/README.md) for what each example measures.

## Connections

- `rkhs` builds on this crate, re-exporting its energy and retrieval maps as the
  dense-associative-memory layer of its kernel surface.
- An optional `innr` dependency supplies SIMD-accelerated similarity for the
  energy kernels.
- The retrieval maps factor as *similarity then separation*: a dot-product score
  followed by a normalizing map (`softmax` for LSE, `sparsemax` for the sparse
  variant). Swapping the separation map for any Fenchel-Young regularizer is the
  Hopfield-Fenchel-Young generalization (Santos et al. 2024); the entmax and
  sparsemax primitives for it live in `fynch`.

## References

- Ramsauer et al., *Hopfield Networks is All You Need* (arXiv:2008.02217).
  Modern continuous Hopfield networks, LSE energy, one-step retrieval.
- Hu, Wu, and Liu, *Provably Optimal Memory Capacity for Modern Hopfield Models*
  (arXiv:2410.23126). Capacity as spherical-code packing; the precise regime
  behind "exponential capacity".
- Hoover et al., *Dense Associative Memory with Epanechnikov Energy*
  (arXiv:2506.10801). The log-sum-ReLU energy and exact single-step retrieval
  within the basin.
- Santos et al., *Hopfield-Fenchel-Young Networks* (arXiv:2411.08590).
  Generalizes the retrieval map to any Fenchel-Young regularizer; a natural
  extension direction for this crate.

Dual-licensed under MIT or Apache-2.0.
