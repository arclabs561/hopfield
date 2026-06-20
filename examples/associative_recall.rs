//! Associative recall from a deterministic memory bank.
//!
//! This demonstrates the modern-Hopfield-as-attention view from Ramsauer et al.:
//! a query scores all stored patterns, forms retrieval weights, and returns a
//! weighted memory. The example uses 64 memories in 16 dimensions so top-k and
//! retrieval distances are meaningful, not a two-point smoke test.

use hopfield::{lse_weights, retrieve_lse, retrieve_sparsemax, sparsemax_weights};

const N_MEMORIES: usize = 64;
const DIMS: usize = 16;
const TARGET: usize = 17;
const BETA: f64 = 18.0;

fn pattern(index: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..DIMS)
        .map(|dim| {
            let a = (index as f64 + 1.0) * (dim as f64 + 3.0) * 0.173;
            let b = (index as f64 * 7.0 + dim as f64 * 11.0) * 0.071;
            a.sin() + 0.5 * b.cos()
        })
        .collect();
    normalize(&mut v);
    v
}

fn normalize(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in v {
        *x /= norm;
    }
}

fn corrupt(v: &[f64]) -> Vec<f64> {
    let mut q: Vec<f64> = v
        .iter()
        .enumerate()
        .map(|(i, x)| x + 0.18 * ((i * 13 + 5) as f64).sin())
        .collect();
    normalize(&mut q);
    q
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn top_weights(weights: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut ranked: Vec<(usize, f64)> = weights.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(k);
    ranked
}

fn main() {
    let memories: Vec<Vec<f64>> = (0..N_MEMORIES).map(pattern).collect();
    let query = corrupt(&memories[TARGET]);

    let mut nearest: Vec<(usize, f64)> = memories
        .iter()
        .enumerate()
        .map(|(i, memory)| (i, l2(&query, memory)))
        .collect();
    nearest.sort_by(|a, b| a.1.total_cmp(&b.1));

    let lse = retrieve_lse(&query, &memories, BETA);
    let sparse = retrieve_sparsemax(&query, &memories, BETA);
    let lse_weights = lse_weights(&query, &memories, BETA);
    let sparse_weights = sparsemax_weights(&query, &memories, BETA);

    println!("memories: {N_MEMORIES}, dims: {DIMS}, beta: {BETA}");
    println!("target memory: #{TARGET}");
    println!(
        "nearest raw memory: #{} at distance {:.4}",
        nearest[0].0, nearest[0].1
    );
    println!(
        "distance(query, target): {:.4}; distance(lse, target): {:.4}; distance(sparse, target): {:.4}",
        l2(&query, &memories[TARGET]),
        l2(&lse, &memories[TARGET]),
        l2(&sparse, &memories[TARGET])
    );
    println!("top LSE weights: {:?}", top_weights(&lse_weights, 5));
    println!(
        "top sparsemax weights: {:?}",
        top_weights(&sparse_weights, 5)
    );
}
