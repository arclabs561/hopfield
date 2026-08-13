//! Pattern recall: the signature associative-memory property.
//!
//! Invariant: recall from a within-basin noisy probe returns the stored
//! pattern. Concretely, store several well-separated real-valued patterns,
//! perturb one of them with small deterministic noise so the probe stays
//! inside that pattern's basin of attraction, then run iterative energy
//! descent ([`hopfield::retrieve_memory`] driven by the real
//! [`hopfield::energy_lse_grad`] update). The recovered state must (a) be
//! nearest to the original target among all stored patterns and (b) be closer
//! to the target than the noisy probe was (the network denoises rather than
//! merely sitting still).
//!
//! These tests call the public retrieval API directly: the update rule lives
//! inside `retrieve_memory`/`energy_lse_grad`, so they exercise convergence
//! rather than re-implementing the dynamics.

use hopfield::{
    energy_lse, energy_lse_grad, energy_lsr, energy_lsr_grad, retrieve_memory, retrieve_sparsemax,
};

const BETA: f64 = 2.0;
const LR: f64 = 0.1;
const MAX_ITERS: usize = 1000;
const TOL: f64 = 1e-8;

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Index of the stored pattern nearest (in L2) to `v`.
fn nearest_index(v: &[f64], patterns: &[Vec<f64>]) -> usize {
    patterns
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| l2(v, a).total_cmp(&l2(v, b)))
        .map(|(i, _)| i)
        .unwrap()
}

/// Four well-separated patterns: scaled standard-basis vectors in 5-D, so every
/// pairwise distance is sqrt(18) ~ 4.24. Separation comfortably exceeds the
/// small probe noise, keeping each probe within its target basin.
fn well_separated_patterns() -> Vec<Vec<f64>> {
    let mut patterns = vec![vec![0.0; 5]; 4];
    for (i, p) in patterns.iter_mut().enumerate() {
        p[i] = 3.0;
    }
    patterns
}

/// Small deterministic perturbation (no RNG): a fixed offset per coordinate,
/// magnitude ~0.25, far below the inter-pattern distance.
fn perturb(pattern: &[f64]) -> Vec<f64> {
    pattern
        .iter()
        .enumerate()
        .map(|(i, x)| x + 0.25 * (((i * 7 + 3) as f64).sin()))
        .collect()
}

#[test]
fn recall_converges_to_stored_pattern_from_noisy_probe() {
    let patterns = well_separated_patterns();

    // Exercise every basin, not just one, so the property is not an accident of
    // the chosen target.
    for target in 0..patterns.len() {
        let probe = perturb(&patterns[target]);
        let probe_dist = l2(&probe, &patterns[target]);

        let (recovered, iters) = retrieve_memory(
            probe.clone(),
            &patterns,
            |v, m| energy_lse_grad(v, m, BETA),
            LR,
            MAX_ITERS,
            TOL,
        );

        // (a) The recovered state lands in the target's basin.
        assert_eq!(
            nearest_index(&recovered, &patterns),
            target,
            "target {target}: recovered state nearest to pattern {} not {target}",
            nearest_index(&recovered, &patterns),
        );

        // (b) Denoising: the network moved the probe closer to the stored
        // pattern, and landed tightly on it.
        let recovered_dist = l2(&recovered, &patterns[target]);
        assert!(
            recovered_dist < probe_dist,
            "target {target}: recall did not denoise (probe {probe_dist:.4} -> recovered {recovered_dist:.4})",
        );
        assert!(
            recovered_dist < 0.5,
            "target {target}: recovered state too far from stored pattern ({recovered_dist:.4})",
        );

        // Converged before exhausting the iteration budget.
        assert!(
            iters < MAX_ITERS,
            "target {target}: did not converge within {MAX_ITERS} iters",
        );
    }
}

#[test]
fn single_step_sparsemax_recall_selects_stored_pattern() {
    // Complementary single-step retrieval (Hopfield-as-attention): with strong
    // separation, sparsemax weighting drops the far patterns and returns the
    // exact stored pattern for a within-basin probe.
    let patterns = well_separated_patterns();
    let target = 2;
    let probe = perturb(&patterns[target]);

    let recovered = retrieve_sparsemax(&probe, &patterns, BETA);

    assert_eq!(
        nearest_index(&recovered, &patterns),
        target,
        "single-step recovered nearest to pattern {} not {target}",
        nearest_index(&recovered, &patterns),
    );
    assert!(
        l2(&recovered, &patterns[target]) < l2(&probe, &patterns[target]),
        "single-step retrieval did not denoise the probe",
    );
}

#[test]
fn energies_match_the_normalized_lse_and_lsr_definitions() {
    let memories = vec![vec![0.0]];
    let query = [0.5_f64];
    let beta = 4.0;

    let expected_lse = 0.5 * query[0] * query[0];
    let expected_lsr = -(1.0 / beta) * (1.0 - 0.5 * beta * query[0] * query[0]).ln();

    assert!((energy_lse(&query, &memories, beta) - expected_lse).abs() < 1e-12);
    assert!((energy_lsr(&query, &memories, beta) - expected_lsr).abs() < 1e-12);
}

#[test]
fn fenchel_young_retrieval_applies_beta_exactly_once() {
    let memories = vec![vec![0.0], vec![2.0]];
    let query = [0.5];

    let retrieved = hopfield::retrieve_fy(&query, &memories, 2.0, |logits| {
        assert_eq!(logits, &[-0.25, -2.25]);
        vec![1.0, 0.0]
    });

    assert_eq!(retrieved, memories[0]);
}

fn central_difference(energy: impl Fn(&[f64]) -> f64, point: &[f64], coordinate: usize) -> f64 {
    let epsilon = 1e-6;
    let mut above = point.to_vec();
    let mut below = point.to_vec();
    above[coordinate] += epsilon;
    below[coordinate] -= epsilon;
    (energy(&above) - energy(&below)) / (2.0 * epsilon)
}

#[test]
fn analytic_gradients_match_finite_differences() {
    let memories = vec![vec![-1.0, 0.5], vec![2.0, 1.0]];
    let point = [0.2, 0.7];
    let beta = 0.5;

    let lse_grad = energy_lse_grad(&point, &memories, beta);
    let lsr_grad = energy_lsr_grad(&point, &memories, beta);
    for coordinate in 0..point.len() {
        let lse_numeric =
            central_difference(|x| energy_lse(x, &memories, beta), &point, coordinate);
        let lsr_numeric =
            central_difference(|x| energy_lsr(x, &memories, beta), &point, coordinate);
        assert!((lse_grad[coordinate] - lse_numeric).abs() < 1e-8);
        assert!((lsr_grad[coordinate] - lsr_numeric).abs() < 1e-8);
    }
}

#[test]
fn lsr_preserves_exact_memories_and_basins_as_load_grows() {
    let beta = 2.0;
    for load in [1, 4, 16] {
        let memories: Vec<Vec<f64>> = (0..load).map(|i| vec![3.0 * i as f64]).collect();
        for target in 0..load {
            let exact = energy_lsr_grad(&memories[target], &memories, beta);
            assert_eq!(exact, vec![0.0]);

            let probe = vec![memories[target][0] + 0.1];
            let initial_distance = l2(&probe, &memories[target]);
            let (recovered, _) = retrieve_memory(
                probe,
                &memories,
                |v, m| energy_lsr_grad(v, m, beta),
                0.25,
                100,
                1e-12,
            );
            assert!(l2(&recovered, &memories[target]) < initial_distance);
            assert!(l2(&recovered, &memories[target]) < 1e-10);
        }
    }
}

#[test]
fn safe_gradient_steps_do_not_increase_energy() {
    let memories = vec![vec![-1.0], vec![1.0]];
    let beta = 1.0;
    let learning_rate = 0.1;

    for (energy, gradient) in [
        (
            energy_lse as fn(&[f64], &[Vec<f64>], f64) -> f64,
            energy_lse_grad as fn(&[f64], &[Vec<f64>], f64) -> Vec<f64>,
        ),
        (
            energy_lsr as fn(&[f64], &[Vec<f64>], f64) -> f64,
            energy_lsr_grad as fn(&[f64], &[Vec<f64>], f64) -> Vec<f64>,
        ),
    ] {
        let mut point = vec![0.25];
        let mut previous = energy(&point, &memories, beta);
        for _ in 0..20 {
            let grad = gradient(&point, &memories, beta);
            point[0] -= learning_rate * grad[0];
            let current = energy(&point, &memories, beta);
            assert!(current <= previous + 1e-12, "{current} > {previous}");
            previous = current;
        }
    }
}
