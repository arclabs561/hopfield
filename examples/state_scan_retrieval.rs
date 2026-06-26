//! Gated state scan feeding Hopfield retrieval.
//!
//! This is a `statescan` proof sketch, not public API. A tiny recurrent scan
//! accumulates a task-relevant trace from a sequence, then Hopfield sparsemax
//! retrieval maps the final state back to a stored memory.

use hopfield::{retrieve_sparsemax, sparsemax_weights};

const BETA: f64 = 8.0;
const DECAY: f64 = 0.82;

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn add_scaled(state: &mut [f64], input: &[f64], scale: f64) {
    for (s, x) in state.iter_mut().zip(input) {
        *s = *s * DECAY + x * scale;
    }
}

fn scan<const D: usize>(events: &[(f64, [f64; D])]) -> Vec<f64> {
    let mut state = vec![0.0; D];
    for (gate, input) in events {
        add_scaled(&mut state, input, *gate);
    }
    state
}

fn argmax(xs: &[f64]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

fn main() {
    let memories = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![-1.0, 0.0, 0.0],
        vec![0.0, -1.0, 0.0],
    ];

    let events = [
        (0.15, [0.0, 1.0, 0.0]),
        (1.00, [-1.0, 0.0, 0.0]),
        (0.10, [1.0, 0.0, 0.0]),
        (0.05, [0.0, -1.0, 0.0]),
    ];

    let last_token = events.last().unwrap().1.to_vec();
    let scanned = scan(&events);
    let retrieved = retrieve_sparsemax(&scanned, &memories, BETA);
    let weights = sparsemax_weights(&scanned, &memories, BETA);

    println!(
        "last token nearest memory: {}",
        argmax(&sparsemax_weights(&last_token, &memories, BETA))
    );
    println!("scanned state: {scanned:?}");
    println!("retrieval weights: {weights:?}");
    println!("retrieved memory: {retrieved:?}");
    println!(
        "distance(scanned, target) {:.4}; distance(retrieved, target) {:.4}",
        l2(&scanned, &memories[2]),
        l2(&retrieved, &memories[2])
    );

    assert_ne!(argmax(&sparsemax_weights(&last_token, &memories, BETA)), 2);
    assert_eq!(argmax(&weights), 2);
    assert!(l2(&retrieved, &memories[2]) < l2(&scanned, &memories[2]));
}
