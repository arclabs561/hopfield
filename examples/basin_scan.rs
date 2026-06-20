//! Scan LSE and LSR behavior along a one-dimensional line.
//!
//! LSE has global support: every query has finite energy. LSR uses an
//! Epanechnikov kernel with compact support, so some points are outside every
//! memory basin and have infinite energy. This example samples 201 points and
//! reports the finite/infinite split.

use hopfield::{energy_lse, energy_lsr, energy_lsr_grad, retrieve_memory};

const BETA: f64 = 2.0;
const START: f64 = -2.0;
const END: f64 = 6.0;
const STEPS: usize = 201;

fn main() {
    let memories = vec![vec![0.0], vec![2.0], vec![5.0]];

    let mut finite_lsr = 0;
    let mut infinite_lsr = 0;
    let mut best_lse = (f64::INFINITY, 0.0);
    let mut best_lsr = (f64::INFINITY, 0.0);

    for i in 0..STEPS {
        let x = START + (END - START) * i as f64 / (STEPS - 1) as f64;
        let query = [x];
        let e_lse = energy_lse(&query, &memories, BETA);
        let e_lsr = energy_lsr(&query, &memories, BETA);

        if e_lse < best_lse.0 {
            best_lse = (e_lse, x);
        }
        if e_lsr.is_finite() {
            finite_lsr += 1;
            if e_lsr < best_lsr.0 {
                best_lsr = (e_lsr, x);
            }
        } else {
            infinite_lsr += 1;
        }
    }

    let query = vec![0.35];
    let (retrieved, iters) = retrieve_memory(
        query.clone(),
        &memories,
        |v, m| energy_lsr_grad(v, m, BETA),
        1.0 / BETA,
        10,
        1e-10,
    );

    println!("memories: [0.0, 2.0, 5.0], beta: {BETA}");
    println!("scan interval: [{START}, {END}], samples: {STEPS}");
    println!("LSR finite samples: {finite_lsr}; infinite samples: {infinite_lsr}");
    println!("lowest LSE energy at x = {:.3}", best_lse.1);
    println!("lowest LSR energy at x = {:.3}", best_lsr.1);
    println!(
        "LSR retrieval from x = {:.2}: x = {:.6} in {iters} iterations",
        query[0], retrieved[0]
    );
}
