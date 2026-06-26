//! Bias Hopfield retrieval scores before the separation map.
//!
//! Modern Hopfield retrieval factors into similarity scores, a separation map,
//! and a weighted memory projection. This example adds two score-bias sources
//! before sparsemax: a small topical bias and a max-plus graph/path bias.

use hopfield::{sparsemax, weighted_memory};

const NEG_INF: f64 = -1.0e9;
const BETA: f64 = 2.0;

fn l2_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn similarity_logits(query: &[f64], memories: &[Vec<f64>]) -> Vec<f64> {
    memories
        .iter()
        .map(|memory| -0.5 * BETA * l2_sq(query, memory))
        .collect()
}

fn max_plus_square<const N: usize>(a: [[f64; N]; N]) -> [[f64; N]; N] {
    let mut out = [[NEG_INF; N]; N];
    for i in 0..N {
        for j in 0..N {
            let mut best = NEG_INF;
            for (k, &left) in a[i].iter().enumerate() {
                best = best.max(left + a[k][j]);
            }
            out[i][j] = best;
        }
    }
    out
}

fn path_bias<const N: usize>(two_hop: [[f64; N]; N], context: usize) -> [f64; N] {
    let mut out = [0.0; N];
    for (dst, score) in out.iter_mut().zip(two_hop[context]) {
        if score > NEG_INF / 2.0 {
            *dst = score;
        }
    }
    out
}

fn add_bias<const N: usize>(logits: &[f64], a: [f64; N], b: [f64; N]) -> [f64; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = logits[i] + a[i] + b[i];
    }
    out
}

fn argmax(xs: &[f64]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

fn support(weights: &[f64]) -> Vec<(usize, f64)> {
    weights
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, weight)| *weight > 1e-12)
        .collect()
}

fn main() {
    let memories = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![-1.0, 0.0],
        vec![0.0, -1.0],
    ];
    let query = memories[0].clone();

    let base_logits = similarity_logits(&query, &memories);
    let base_weights = sparsemax(&base_logits);
    let base_retrieval = weighted_memory(&memories, &base_weights);

    let topical_bias = [0.0, 0.2, 0.0, 0.0];
    let graph = [
        [0.0, 1.4, NEG_INF, NEG_INF],
        [NEG_INF, 0.0, 3.6, NEG_INF],
        [NEG_INF, NEG_INF, 0.0, 1.0],
        [0.5, NEG_INF, NEG_INF, 0.0],
    ];
    let graph_bias = path_bias(max_plus_square(graph), 0);
    let biased_logits = add_bias(&base_logits, topical_bias, graph_bias);
    let biased_weights = sparsemax(&biased_logits);
    let biased_retrieval = weighted_memory(&memories, &biased_weights);

    println!("base logits:   {base_logits:?}");
    println!("biased logits: {biased_logits:?}");
    println!("base support:   {:?}", support(&base_weights));
    println!("biased support: {:?}", support(&biased_weights));
    println!("base retrieval:   {base_retrieval:?}");
    println!("biased retrieval: {biased_retrieval:?}");

    assert_eq!(argmax(&base_logits), 0);
    assert_eq!(argmax(&biased_logits), 2);
    assert_eq!(support(&base_weights), vec![(0, 1.0)]);
    assert_eq!(support(&biased_weights), vec![(2, 1.0)]);
}
