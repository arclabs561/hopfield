//! Trace energy descent during Hopfield retrieval.
//!
//! Uses the crate's LSE energy and gradient helpers to print scalar energy,
//! gradient norm, and optimizer steps during retrieval.
//!
//! Run: cargo run --example energy_descent_trace --release

use hopfield::{energy_lse, energy_lse_grad};

const BETA: f64 = 3.0;
const LEARNING_RATE: f64 = 0.12;
const STEPS: usize = 24;

#[derive(Debug)]
struct TraceStep {
    iter: usize,
    state: Vec<f64>,
    energy: f64,
    grad_norm: f64,
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn nearest(query: &[f64], memories: &[Vec<f64>]) -> usize {
    memories
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| l2(query, a).total_cmp(&l2(query, b)))
        .map(|(index, _)| index)
        .unwrap()
}

fn descent_trace(mut state: Vec<f64>, memories: &[Vec<f64>]) -> Vec<TraceStep> {
    let mut trace = Vec::with_capacity(STEPS + 1);

    for iter in 0..=STEPS {
        let energy = energy_lse(&state, memories, BETA);
        let grad = energy_lse_grad(&state, memories, BETA);
        let grad_norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();

        trace.push(TraceStep {
            iter,
            state: state.clone(),
            energy,
            grad_norm,
        });

        for (x, g) in state.iter_mut().zip(grad) {
            *x -= LEARNING_RATE * g;
        }
    }

    trace
}

fn main() {
    let memories = vec![vec![0.0, 0.0], vec![2.5, 0.0], vec![0.0, 2.5]];
    let query = vec![0.55, 0.22];
    let target = nearest(&query, &memories);
    let trace = descent_trace(query.clone(), &memories);

    println!("Hopfield energy descent trace");
    println!("==============================\n");
    println!("memories: {memories:?}");
    println!("query: {query:?}; nearest memory: #{target}");
    println!("beta: {BETA}; learning_rate: {LEARNING_RATE}; steps: {STEPS}\n");
    println!(
        "{:>4} | {:>11} | {:>11} | {:>20}",
        "iter", "energy", "grad_norm", "state"
    );
    println!("{}", "-".repeat(58));

    for step in trace.iter().step_by(4) {
        println!(
            "{:>4} | {:>11.6} | {:>11.6} | [{:>8.5}, {:>8.5}]",
            step.iter, step.energy, step.grad_norm, step.state[0], step.state[1]
        );
    }

    let first = trace.first().unwrap();
    let last = trace.last().unwrap();
    let first_distance = l2(&first.state, &memories[target]);
    let last_distance = l2(&last.state, &memories[target]);

    println!(
        "\nenergy: {:.6} -> {:.6}; distance to memory #{target}: {:.6} -> {:.6}",
        first.energy, last.energy, first_distance, last_distance
    );

    for window in trace.windows(2) {
        assert!(
            window[1].energy <= window[0].energy + 1e-12,
            "energy increased from iter {} to {}",
            window[0].iter,
            window[1].iter
        );
    }
    assert!(last.energy < first.energy - 0.1);
    assert!(last_distance < first_distance * 0.1);
}
