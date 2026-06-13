//! Dense associative memory and modern Hopfield networks.
//!
//! Energy-based content-addressable memory with connections to attention
//! mechanisms and kernel methods. Implements the continuous Hopfield network
//! (Ramsauer et al. 2021) with exponential storage capacity.
//!
//! ## Energy Functions
//!
//! Two energy formulations are provided:
//!
//! - **LSE (Log-Sum-Exp)**: smooth landscape, exponential capacity, approximate
//!   retrieval via gradient descent.
//! - **LSR (Log-Sum-ReLU)**: Epanechnikov kernel, exact single-step retrieval
//!   within basin radius, compact support (Hoover et al. 2025).
//!
//! ## Key Functions
//!
//! | Function | Purpose |
//! |----------|---------|
//! | [`kernel_sum`] | Generic Σ_μ κ(v, ξ^μ) |
//! | [`energy_lse`] | LSE energy (RBF / Gaussian kernel) |
//! | [`energy_lse_grad`] | Gradient of LSE energy |
//! | [`energy_lsr`] | LSR energy (Epanechnikov kernel) |
//! | [`energy_lsr_grad`] | Gradient of LSR energy |
//! | [`retrieve_memory`] | Energy descent retrieval loop |
//!
//! ## Example
//!
//! ```rust
//! use hopfield::{retrieve_memory, energy_lse_grad};
//!
//! let memories = vec![
//!     vec![0.0, 0.0],
//!     vec![10.0, 10.0],
//! ];
//!
//! let query = vec![0.5, 0.5];
//! let (retrieved, _iters) = retrieve_memory(
//!     query,
//!     &memories,
//!     |v, m| energy_lse_grad(v, m, 2.0),
//!     0.1,
//!     100,
//!     1e-6,
//! );
//!
//! assert!(retrieved[0].abs() < 1.0);
//! assert!(retrieved[1].abs() < 1.0);
//! ```
/// Squared L2 distance between `v` and `xi`. With the `simd` feature this
/// dispatches to innr's SIMD f64 kernel (AVX-512/AVX2/NEON) for higher
/// dimensions; otherwise it is a portable scalar loop. Behaviour is
/// identical either way.
#[inline]
fn l2_sq(v: &[f64], xi: &[f64]) -> f64 {
    #[cfg(feature = "simd")]
    {
        innr::dense_f64::l2_distance_squared_f64(v, xi)
    }
    #[cfg(not(feature = "simd"))]
    {
        v.iter().zip(xi.iter()).map(|(a, b)| (a - b).powi(2)).sum()
    }
}


/// Compute kernel sum: Σ_μ κ(v, ξ^μ)
///
/// This is the core computation in Associative Memory and kernel machines.
///
/// # Arguments
///
/// * `v` - Query point
/// * `memories` - Stored patterns {ξ^μ}
/// * `kernel` - Kernel function κ(v, ξ) -> f64
///
/// # Returns
///
/// Sum of kernel evaluations
///
/// # Example
///
/// ```rust
/// use hopfield::kernel_sum;
///
/// let v = vec![0.0, 0.0];
/// let memories = vec![
///     vec![0.0, 0.0],  // close to v
///     vec![10.0, 10.0],  // far from v
/// ];
///
/// // RBF kernel: exp(-||a - b||² / (2σ²))
/// let rbf = |a: &[f64], b: &[f64]| -> f64 {
///     let sq_dist: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
///     (-sq_dist / 2.0).exp()
/// };
///
/// let sum = kernel_sum(&v, &memories, rbf);
/// // ≈ 1.0 (from first memory) + ~0 (from second)
/// assert!(sum > 0.99 && sum < 1.01);
/// ```
pub fn kernel_sum<F>(v: &[f64], memories: &[Vec<f64>], kernel: F) -> f64
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    memories.iter().map(|xi| kernel(v, xi)).sum()
}

/// Log-Sum-Exp (LSE) energy for Dense Associative Memory.
///
/// E_β(v; Ξ) = -log Σ_μ exp(-β/2 ||v - ξ^μ||²)
///
/// This corresponds to RBF kernel with log scaling. The gradient points toward
/// a weighted average of memories, with weights given by softmax over similarities.
///
/// Properties:
/// - Smooth energy landscape
/// - Exponential memory capacity
/// - Approximate retrieval (needs T → ∞ for exact)
///
/// # Arguments
///
/// * `v` - Current state
/// * `memories` - Stored patterns
/// * `beta` - Inverse temperature (larger = sharper peaks around memories)
///
/// # Example
///
/// ```rust
/// use hopfield::energy_lse;
///
/// let memories = vec![
///     vec![0.0, 0.0],
///     vec![10.0, 10.0],
/// ];
///
/// // At a memory: low energy
/// let e_at_memory = energy_lse(&[0.0, 0.0], &memories, 1.0);
///
/// // Between memories: higher energy
/// let e_between = energy_lse(&[5.0, 5.0], &memories, 1.0);
///
/// assert!(e_at_memory < e_between);
/// ```
pub fn energy_lse(v: &[f64], memories: &[Vec<f64>], beta: f64) -> f64 {
    if memories.is_empty() {
        return 0.0;
    }

    // For numerical stability, use log-sum-exp trick:
    // log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
    let neg_half_beta = -0.5 * beta;

    let log_terms: Vec<f64> = memories
        .iter()
        .map(|xi| {
            let sq_dist: f64 = l2_sq(v, xi);
            neg_half_beta * sq_dist
        })
        .collect();

    let max_term = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if max_term.is_infinite() {
        return f64::INFINITY;
    }

    let sum_exp: f64 = log_terms.iter().map(|&t| (t - max_term).exp()).sum();

    -(max_term + sum_exp.ln())
}

/// Gradient of LSE energy: ∇_v E_LSE(v; Ξ)
///
/// The gradient is a weighted combination of (v - ξ^μ) vectors,
/// where weights are softmax over similarities.
///
/// # Returns
///
/// Gradient vector of same dimension as v
pub fn energy_lse_grad(v: &[f64], memories: &[Vec<f64>], beta: f64) -> Vec<f64> {
    if memories.is_empty() || v.is_empty() {
        return vec![0.0; v.len()];
    }

    let d = v.len();
    let neg_half_beta = -0.5 * beta;

    // Compute softmax weights
    let sq_dists: Vec<f64> = memories
        .iter()
        .map(|xi| {
            l2_sq(v, xi)
        })
        .collect();

    let log_weights: Vec<f64> = sq_dists.iter().map(|&d| neg_half_beta * d).collect();
    let max_log = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let exp_weights: Vec<f64> = log_weights.iter().map(|&w| (w - max_log).exp()).collect();
    let sum_exp: f64 = exp_weights.iter().sum();

    let softmax_weights: Vec<f64> = exp_weights.iter().map(|&w| w / sum_exp).collect();

    // Gradient: β Σ_μ w_μ (v - ξ^μ)
    let mut grad = vec![0.0; d];
    for (mu, xi) in memories.iter().enumerate() {
        let w = softmax_weights[mu];
        for (i, (vi, xii)) in v.iter().zip(xi.iter()).enumerate() {
            grad[i] += w * (vi - xii);
        }
    }

    for g in &mut grad {
        *g *= beta;
    }

    grad
}

/// Log-Sum-ReLU (LSR) energy for Dense Associative Memory with Epanechnikov kernel.
///
/// E_β(v; Ξ) = -log Σ_μ ReLU(1 - β/2 ||v - ξ^μ||²)
///
/// Based on the Epanechnikov kernel, which is optimal for density estimation (MISE).
///
/// Properties (from Hoover et al., 2025):
/// - Exact single-step retrieval (unlike LSE which needs many steps)
/// - Exponential memory capacity
/// - Can generate novel memories at basin intersections
/// - Compact support: regions of infinite energy where no memory is nearby
///
/// # Arguments
///
/// * `v` - Current state
/// * `memories` - Stored patterns
/// * `beta` - Inverse temperature (controls support radius: r = sqrt(2/β))
///
/// # Example
///
/// ```rust
/// use hopfield::energy_lsr;
///
/// let memories = vec![
///     vec![0.0, 0.0],
///     vec![10.0, 10.0],
/// ];
///
/// // At a memory: low energy
/// let e_at = energy_lsr(&[0.0, 0.0], &memories, 1.0);
///
/// // Far from all memories: infinite energy (outside support)
/// let e_far = energy_lsr(&[100.0, 100.0], &memories, 1.0);
///
/// assert!(e_at.is_finite());
/// assert!(e_far.is_infinite());
/// ```
pub fn energy_lsr(v: &[f64], memories: &[Vec<f64>], beta: f64) -> f64 {
    if memories.is_empty() {
        return 0.0;
    }

    let half_beta = 0.5 * beta;

    let sum: f64 = memories
        .iter()
        .map(|xi| {
            let sq_dist: f64 = l2_sq(v, xi);
            (1.0 - half_beta * sq_dist).max(0.0) // ReLU
        })
        .sum();

    if sum <= 0.0 {
        f64::INFINITY // Outside support of all memories
    } else {
        -sum.ln()
    }
}

/// Gradient of LSR energy: ∇_v E_LSR(v; Ξ)
///
/// Only memories within support (||v - ξ||² < 2/β) contribute to the gradient.
///
/// # Returns
///
/// Gradient vector. Returns zero vector if outside all support regions.
pub fn energy_lsr_grad(v: &[f64], memories: &[Vec<f64>], beta: f64) -> Vec<f64> {
    if memories.is_empty() || v.is_empty() {
        return vec![0.0; v.len()];
    }

    let d = v.len();
    let half_beta = 0.5 * beta;

    // Compute kernel values (only positive ones contribute)
    let kernel_vals: Vec<f64> = memories
        .iter()
        .map(|xi| {
            let sq_dist: f64 = l2_sq(v, xi);
            (1.0 - half_beta * sq_dist).max(0.0)
        })
        .collect();

    let sum: f64 = kernel_vals.iter().sum();

    if sum <= 0.0 {
        return vec![0.0; d]; // Outside support
    }

    // Gradient: (β / Σκ) × Σ_μ 1[κ_μ > 0] (v - ξ^μ)
    let mut grad = vec![0.0; d];
    for (mu, xi) in memories.iter().enumerate() {
        if kernel_vals[mu] > 0.0 {
            for (i, (vi, xii)) in v.iter().zip(xi.iter()).enumerate() {
                grad[i] += vi - xii;
            }
        }
    }

    let scale = beta / sum;
    for g in &mut grad {
        *g *= scale;
    }

    grad
}

/// Single step of energy descent for memory retrieval.
///
/// v_{t+1} = v_t - η ∇E(v_t)
///
/// # Arguments
///
/// * `v` - Current state (modified in place)
/// * `grad` - Gradient at current state
/// * `learning_rate` - Step size η
fn energy_descent_step(v: &mut [f64], grad: &[f64], learning_rate: f64) {
    for (vi, gi) in v.iter_mut().zip(grad.iter()) {
        *vi -= learning_rate * gi;
    }
}

/// Retrieve memory using energy descent.
///
/// Performs gradient descent on the energy function until convergence
/// or max iterations reached.
///
/// # Arguments
///
/// * `query` - Initial state (corrupted memory / query)
/// * `memories` - Stored patterns
/// * `energy_grad` - Function computing energy gradient
/// * `learning_rate` - Step size
/// * `max_iters` - Maximum iterations
/// * `tolerance` - Convergence threshold (gradient norm)
///
/// # Returns
///
/// (retrieved_memory, iterations_used)
///
/// # Example
///
/// ```rust
/// use hopfield::{retrieve_memory, energy_lse_grad};
///
/// let memories = vec![
///     vec![0.0, 0.0],
///     vec![10.0, 10.0],
/// ];
///
/// // Query near first memory
/// let query = vec![0.5, 0.5];
/// let (retrieved, iters) = retrieve_memory(
///     query,
///     &memories,
///     |v, m| energy_lse_grad(v, m, 2.0),
///     0.1,
///     100,
///     1e-6,
/// );
///
/// // Should converge near [0, 0]
/// assert!(retrieved[0].abs() < 1.0);
/// assert!(retrieved[1].abs() < 1.0);
/// ```
pub fn retrieve_memory<F>(
    query: Vec<f64>,
    memories: &[Vec<f64>],
    energy_grad: F,
    learning_rate: f64,
    max_iters: usize,
    tolerance: f64,
) -> (Vec<f64>, usize)
where
    F: Fn(&[f64], &[Vec<f64>]) -> Vec<f64>,
{
    let mut v = query;

    for iter in 0..max_iters {
        let grad = energy_grad(&v, memories);

        // Check convergence
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tolerance {
            return (v, iter);
        }

        energy_descent_step(&mut v, &grad, learning_rate);
    }

    (v, max_iters)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: RBF kernel for tests
    fn rbf(a: &[f64], b: &[f64], sigma: f64) -> f64 {
        let sq_dist: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        (-sq_dist / (2.0 * sigma * sigma)).exp()
    }

    #[test]
    fn test_kernel_sum() {
        let v = vec![0.0, 0.0];
        let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0]];

        let sum = kernel_sum(&v, &memories, |a, b| rbf(a, b, 1.0));
        // First memory: distance 0 -> k = 1
        // Second memory: distance sqrt(200) -> k ≈ 0
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_energy_lse_at_memory() {
        let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0]];

        // At first memory
        let e1 = energy_lse(&[0.0, 0.0], &memories, 1.0);
        // Between memories
        let e2 = energy_lse(&[5.0, 5.0], &memories, 1.0);

        assert!(e1 < e2, "energy should be lower at stored memory");
    }

    #[test]
    fn test_energy_lse_grad_points_toward_memory() {
        let memories = vec![vec![0.0, 0.0]];

        // Query slightly displaced from memory
        let v = vec![1.0, 0.0];
        let grad = energy_lse_grad(&v, &memories, 2.0);

        // Gradient should point away from memory (positive x direction)
        // Energy descent would move toward memory
        assert!(grad[0] > 0.0, "gradient should point away from memory");
    }

    #[test]
    fn test_energy_lsr_finite_at_memory() {
        let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0]];

        // At first memory: should have finite energy
        let e = energy_lsr(&[0.0, 0.0], &memories, 1.0);
        assert!(e.is_finite(), "energy should be finite at memory");
    }

    #[test]
    fn test_energy_lsr_infinite_outside_support() {
        let memories = vec![vec![0.0, 0.0]];

        // Far from memory: outside compact support
        let e = energy_lsr(&[100.0, 100.0], &memories, 1.0);
        assert!(e.is_infinite(), "energy should be infinite outside support");
    }

    #[test]
    fn test_retrieve_memory_lse() {
        let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0]];

        // Query near first memory
        let query = vec![1.0, 1.0];
        let (retrieved, _iters) = retrieve_memory(
            query,
            &memories,
            |v, m| energy_lse_grad(v, m, 2.0),
            0.1,
            100,
            1e-6,
        );

        // Should converge near [0, 0]
        let dist_to_first: f64 = retrieved.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            dist_to_first < 2.0,
            "should retrieve near first memory, got dist {}",
            dist_to_first
        );
    }

    #[test]
    fn test_energy_lsr_single_step_retrieval() {
        // Hoover et al. (2025) Theorem 1: single-step retrieval within basin radius
        let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let beta = 2.0; // support radius r = sqrt(2/beta) = 1.0

        // Query well within first basin (dist = 0.1 < 1.0)
        let query = vec![0.1, 0.0];

        // Compute gradient
        let grad = energy_lsr_grad(&query, &memories, beta);

        // Theorem 1 suggests η = 1/β for single-step retrieval in some cases,
        // but let's verify if gradient points exactly toward memory.
        // Gradient should be proportional to (query - memory).
        // grad = (beta / sum) * (query - memory)
        assert!(grad[0] > 0.0);
        assert!((grad[1] - 0.0).abs() < 1e-10);
    }
}
