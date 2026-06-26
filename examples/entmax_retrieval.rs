//! Hopfield-Fenchel-Young retrieval with entmax.
//!
//! This example uses `fynch::Tsallis::entmax15()` through hopfield's optional
//! `fynch` feature. Entmax is between softmax and sparsemax: it can zero out
//! far memories while still spreading mass across a small local support.

use hopfield::{
    lse_weights, regularized_weights, retrieve_lse, retrieve_regularized, retrieve_sparsemax,
    sparsemax_weights,
};

const N_MEMORIES: usize = 80;
const BETA: f64 = 28.0;

fn memory(index: usize) -> Vec<f64> {
    let theta = std::f64::consts::TAU * index as f64 / N_MEMORIES as f64;
    vec![theta.cos(), theta.sin()]
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn support(weights: &[f64], threshold: f64) -> usize {
    weights.iter().filter(|w| **w > threshold).count()
}

fn entropy(weights: &[f64]) -> f64 {
    weights
        .iter()
        .filter(|w| **w > 0.0)
        .map(|w| -w * w.ln())
        .sum()
}

fn top_weights(weights: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut ranked: Vec<(usize, f64)> = weights.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(k);
    ranked
}

fn main() {
    let memories: Vec<Vec<f64>> = (0..N_MEMORIES).map(memory).collect();
    let query = {
        let a = memory(10);
        let b = memory(11);
        vec![(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
    };

    let entmax = fynch::Tsallis::entmax15();
    let lse_weights = lse_weights(&query, &memories, BETA);
    let entmax_weights = regularized_weights(&query, &memories, BETA, &entmax);
    let sparse_weights = sparsemax_weights(&query, &memories, BETA);

    let lse = retrieve_lse(&query, &memories, BETA);
    let entmax_retrieval = retrieve_regularized(&query, &memories, BETA, &entmax);
    let sparse = retrieve_sparsemax(&query, &memories, BETA);

    println!("memories on circle: {N_MEMORIES}, beta: {BETA}");
    println!("query lies halfway between memories #10 and #11");
    println!(
        "support > 1e-12: lse={}, entmax15={}, sparsemax={}",
        support(&lse_weights, 1e-12),
        support(&entmax_weights, 1e-12),
        support(&sparse_weights, 1e-12)
    );
    println!(
        "entropy: lse={:.4}, entmax15={:.4}, sparsemax={:.4}",
        entropy(&lse_weights),
        entropy(&entmax_weights),
        entropy(&sparse_weights)
    );
    println!(
        "top entmax15 weights: {:?}",
        top_weights(&entmax_weights, 6)
    );
    println!(
        "distance(query, retrieval): lse={:.6}, entmax15={:.6}, sparsemax={:.6}",
        l2(&query, &lse),
        l2(&query, &entmax_retrieval),
        l2(&query, &sparse)
    );
}
