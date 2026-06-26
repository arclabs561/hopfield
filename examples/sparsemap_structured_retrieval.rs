//! Structured retrieval with SparseMAP over adjacent-pair memories.
//!
//! This example keeps Hopfield's scoring step but changes the retrieval domain.
//! Instead of projecting logits onto the simplex over individual memories, it
//! projects onto the convex hull of adjacent-pair vertices. Each vertex means
//! "retrieve this neighboring pair together."

use hopfield::weighted_memory;

const N_MEMORIES: usize = 16;
const BETA: f64 = 40.0;
const TARGET: usize = 10;

fn memory(index: usize) -> Vec<f64> {
    let theta = std::f64::consts::TAU * index as f64 / N_MEMORIES as f64;
    vec![theta.cos(), theta.sin()]
}

fn l2_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    l2_sq(a, b).sqrt()
}

fn hopfield_logits(query: &[f64], memories: &[Vec<f64>]) -> Vec<f64> {
    memories
        .iter()
        .map(|memory| -0.5 * BETA * l2_sq(query, memory))
        .collect()
}

fn adjacent_pair_vertices(n: usize) -> Vec<Vec<f64>> {
    (0..n - 1)
        .map(|i| {
            let mut vertex = vec![0.0; n];
            vertex[i] = 0.5;
            vertex[i + 1] = 0.5;
            vertex
        })
        .collect()
}

fn nonzero_weights(weights: &[f64]) -> Vec<(usize, f64)> {
    weights
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, weight)| *weight > 1e-12)
        .collect()
}

fn main() {
    let memories: Vec<Vec<f64>> = (0..N_MEMORIES).map(memory).collect();
    let query = {
        let a = memory(TARGET);
        let b = memory(TARGET + 1);
        vec![(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
    };

    let logits = hopfield_logits(&query, &memories);
    let vertices = adjacent_pair_vertices(N_MEMORIES);
    let prediction = fynch::sparsemap_explicit(&logits, &vertices).unwrap();
    let retrieved = weighted_memory(&memories, &prediction.marginal);

    let active_pairs: Vec<(usize, usize, f64)> = prediction
        .active
        .iter()
        .map(|weight| (weight.vertex, weight.vertex + 1, weight.weight))
        .collect();

    println!("memories on circle: {N_MEMORIES}, beta: {BETA}");
    println!(
        "query lies halfway between memories #{TARGET} and #{}",
        TARGET + 1
    );
    println!("active adjacent pairs: {active_pairs:?}");
    println!(
        "retrieval weights > 1e-12: {:?}",
        nonzero_weights(&prediction.marginal)
    );
    println!(
        "distance(query, structured retrieval): {:.6}",
        l2(&query, &retrieved)
    );

    assert_eq!(active_pairs, vec![(TARGET, TARGET + 1, 1.0)]);
    assert!((prediction.marginal[TARGET] - 0.5).abs() < 1e-12);
    assert!((prediction.marginal[TARGET + 1] - 0.5).abs() < 1e-12);
    assert!(l2(&query, &retrieved) < 1e-12);
}
