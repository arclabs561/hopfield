# hopfield examples

Run examples with `--release`; the numeric outputs are easier to compare when
the optimizer is enabled.

| I want to... | Start with | Command |
|---|---|---|
| Recover a noisy query from a larger memory bank | `associative_recall` | `cargo run --example associative_recall --release` |
| Compare dense and sparse attention support | `sparse_attention` | `cargo run --example sparse_attention --release` |
| Inspect LSE vs LSR basin behavior | `basin_scan` | `cargo run --example basin_scan --release` |

`associative_recall` is the best first example. It stores 64 deterministic
patterns, corrupts one query, and reports nearest-memory rank, retrieval
distance, and top weights.

`sparse_attention` shows the practical difference between LSE/softmax weights
and sparsemax weights: dense attention spreads small mass across all memories,
while sparsemax returns a short explicit support.

`basin_scan` samples a one-dimensional line through memory space and reports
where LSR has compact support. It is useful when tuning `beta`.
