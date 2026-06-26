# hopfield examples

Run examples with `--release`; the numeric outputs are easier to compare when
the optimizer is enabled.

| Purpose | Example | Command |
|---|---|---|
| Recover a noisy query from a larger memory bank | `associative_recall` | `cargo run --example associative_recall --release` |
| Compare dense and sparse attention support | `sparse_attention` | `cargo run --example sparse_attention --release` |
| Compare softmax, entmax, and sparsemax retrieval | `entmax_retrieval` | `cargo run --example entmax_retrieval --features fynch --release` |
| Inspect LSE vs LSR basin behavior | `basin_scan` | `cargo run --example basin_scan --release` |

`associative_recall` is the best first example. It stores 64 deterministic
patterns, corrupts one query, and reports nearest-memory rank, retrieval
distance, and top weights.

Expected excerpt:

```text
memories: 64, dims: 16, beta: 18
target memory: #17
nearest raw memory: #17 at distance 0.4655
distance(query, target): 0.4655; distance(lse, target): 0.0209; distance(sparse, target): 0.0000
```

`sparse_attention` shows the practical difference between LSE/softmax weights
and sparsemax weights: dense attention spreads small mass across all memories,
while sparsemax returns a short explicit support.

Expected excerpt:

```text
memories on circle: 80, beta: 28
LSE support > 1e-6: 24; sparsemax support > 1e-12: 4
LSE entropy: 2.3066; sparsemax entropy: 1.3256
```

`entmax_retrieval` uses the optional `fynch` feature to pass
`fynch::Tsallis::entmax15()` as the Hopfield-Fenchel-Young retrieval map.
Entmax keeps a sparse local support, but does not collapse as aggressively as
sparsemax.

Expected excerpt:

```text
support > 1e-12: lse=38, entmax15=6, sparsemax=4
entropy: lse=2.3066, entmax15=1.3660, sparsemax=1.3256
```

`basin_scan` samples a one-dimensional line through memory space and reports
where LSR has compact support. It is useful when tuning `beta`.

Expected excerpt:

```text
memories: [0.0, 2.0, 5.0], beta: 2
LSR finite samples: 147; infinite samples: 54
LSR retrieval from x = 0.35: x = -0.000000 in 3 iterations
```
