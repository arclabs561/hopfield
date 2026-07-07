# hopfield

Associative memory algorithms.

`hopfield` stores vectors as memories and retrieves the nearest stored pattern
by descending an energy function. It includes LSE and LSR energies, dense and
sparse one-step retrieval maps, and optional Fenchel-Young retrieval through
`fynch`.

## Usage

```toml
[dependencies]
hopfield = "0.2.2"
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

## References

- Ramsauer et al., *Hopfield Networks is All You Need* (arXiv:2008.02217).
  Continuous Hopfield retrieval, LSE energy, one-step retrieval.
- Hu, Wu, and Liu, *Provably Optimal Memory Capacity for Modern Hopfield Models*
  (arXiv:2410.23126). Capacity as spherical-code packing; the precise regime
  behind "exponential capacity".
- Hoover et al., *Dense Associative Memory with Epanechnikov Energy*
  (arXiv:2506.10801). The log-sum-ReLU energy and exact single-step retrieval
  within the basin.
- Santos et al., *Hopfield-Fenchel-Young Networks* (arXiv:2411.08590).
  Generalizes the retrieval map to any Fenchel-Young regularizer; implemented
  here as `retrieve_fy` (with typed `fynch` regularizers under the `fynch` feature).

Dual-licensed under MIT or Apache-2.0.
