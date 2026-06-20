# hopfield

Dense associative memory and modern Hopfield networks.

Energy-based content-addressable memory with exponential storage capacity
(Ramsauer et al. 2021). Two energy formulations: LSE (smooth, approximate
retrieval) and LSR (Epanechnikov kernel, exact single-step retrieval within
basin radius, Hoover et al. 2025). Also includes dense and sparse one-step
retrieval maps (`retrieve_lse`, `retrieve_sparsemax`) for attention-style use.

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

Dual-licensed under MIT or Apache-2.0.
