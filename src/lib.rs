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
//! | [`lse_weights`] | Soft attention weights over memories |
//! | [`sparsemax_weights`] | Sparse attention weights over memories |
//! | [`retrieve_lse`] | One-step LSE memory retrieval |
//! | [`retrieve_sparsemax`] | One-step sparse memory retrieval |
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

/// Softmax projection onto the probability simplex.
///
/// This is the dense retrieval map used by LSE attention: every finite logit
/// receives positive mass, with larger logits receiving exponentially more.
///
/// # Example
///
/// ```rust
/// use hopfield::softmax;
///
/// let weights = softmax(&[1.0, 2.0, 3.0]);
/// assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// assert!(weights[2] > weights[1] && weights[1] > weights[0]);
/// ```
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f64 = exp.iter().sum();

    exp.into_iter().map(|x| x / sum).collect()
}

/// Sparsemax projection onto the probability simplex.
///
/// Sparsemax is the Euclidean projection of logits onto the simplex. Unlike
/// softmax, it can assign exact zero weight to low-scoring memories, making the
/// retrieval set explicit.
///
/// # Example
///
/// ```rust
/// use hopfield::sparsemax;
///
/// let weights = sparsemax(&[1.0, 1.0, 0.0]);
/// assert_eq!(weights, vec![0.5, 0.5, 0.0]);
/// ```
pub fn sparsemax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }

    let mut sorted = logits.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));

    let mut cumsum = 0.0;
    let mut support = 0;
    for (i, z) in sorted.iter().enumerate() {
        cumsum += z;
        let k = i + 1;
        if 1.0 + (k as f64) * z > cumsum {
            support = k;
        }
    }

    if support == 0 {
        return vec![0.0; logits.len()];
    }

    let tau = (sorted.iter().take(support).sum::<f64>() - 1.0) / support as f64;
    logits.iter().map(|&z| (z - tau).max(0.0)).collect()
}

fn similarity_logits(v: &[f64], memories: &[Vec<f64>], beta: f64) -> Vec<f64> {
    let neg_half_beta = -0.5 * beta;
    memories
        .iter()
        .map(|xi| neg_half_beta * l2_sq(v, xi))
        .collect()
}

/// LSE retrieval weights over memories.
///
/// The logits are `-beta / 2 * ||v - xi||^2`, then softmaxed. These are the
/// same weights used by [`energy_lse_grad`].
pub fn lse_weights(v: &[f64], memories: &[Vec<f64>], beta: f64) -> Vec<f64> {
    softmax(&similarity_logits(v, memories, beta))
}

/// Sparsemax retrieval weights over memories.
///
/// The logits match [`lse_weights`], but sparsemax projects them onto the
/// simplex with exact zeros for low-scoring memories.
pub fn sparsemax_weights(v: &[f64], memories: &[Vec<f64>], beta: f64) -> Vec<f64> {
    sparsemax(&similarity_logits(v, memories, beta))
}

/// Weighted average of stored memories.
///
/// Returns an empty vector if the memory bank is empty, the weight count does
/// not match the memory count, or stored memories have inconsistent dimensions.
pub fn weighted_memory(memories: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let Some(first) = memories.first() else {
        return Vec::new();
    };

    if weights.len() != memories.len() || memories.iter().any(|memory| memory.len() != first.len())
    {
        return Vec::new();
    }

    let mut out = vec![0.0; first.len()];
    for (weight, memory) in weights.iter().zip(memories.iter()) {
        for (dst, src) in out.iter_mut().zip(memory.iter()) {
            *dst += weight * src;
        }
    }
    out
}

/// One-step LSE retrieval as a dense weighted average of memories.
pub fn retrieve_lse(v: &[f64], memories: &[Vec<f64>], beta: f64) -> Vec<f64> {
    weighted_memory(memories, &lse_weights(v, memories, beta))
}

/// One-step sparsemax retrieval as a sparse weighted average of memories.
pub fn retrieve_sparsemax(v: &[f64], memories: &[Vec<f64>], beta: f64) -> Vec<f64> {
    weighted_memory(memories, &sparsemax_weights(v, memories, beta))
}

/// One-step Hopfield-Fenchel-Young retrieval, generalized over the separation map.
///
/// Every retrieval here factors as *similarity then separation then projection*:
/// `retrieve(v) = Xᵀ · separation(β · similarity_logits(v))`. The built-in
/// retrievers fix the separation map: [`retrieve_lse`] uses [`softmax`],
/// [`retrieve_sparsemax`] uses [`sparsemax`]. This function takes the separation
/// map as an argument, which is the Hopfield-Fenchel-Young generalization: any
/// Fenchel-Young regularized-argmax is a valid separation map. An α-entmax map
/// (α > 1, e.g. from the `fynch` crate) gives a sparse map with a positive
/// margin, hence exact single-step retrieval within the basin (Santos et al.
/// 2024, arXiv:2411.08590).
///
/// `separation` receives the similarity logits (one per stored memory) and
/// returns a same-length weight vector.
///
/// # Example
///
/// ```rust
/// use hopfield::{retrieve_fy, retrieve_lse, softmax};
///
/// let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
/// let v = [1.0, 1.0];
///
/// // Passing softmax as the separation map recovers retrieve_lse exactly.
/// let general = retrieve_fy(&v, &memories, 1.0, softmax);
/// let lse = retrieve_lse(&v, &memories, 1.0);
/// assert!(general.iter().zip(&lse).all(|(a, b)| (a - b).abs() < 1e-12));
/// ```
pub fn retrieve_fy<F>(v: &[f64], memories: &[Vec<f64>], beta: f64, separation: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let logits = similarity_logits(v, memories, beta);
    weighted_memory(memories, &separation(&logits))
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
    let weights = lse_weights(v, memories, beta);

    // Gradient: β Σ_μ w_μ (v - ξ^μ)
    let mut grad = vec![0.0; d];
    for (mu, xi) in memories.iter().enumerate() {
        let w = weights[mu];
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
    fn test_retrieve_fy_hardmax_returns_nearest_memory() {
        // A hardmax separation map (one-hot at the argmax logit) is a valid
        // Fenchel-Young map the built-in retrievers don't provide; it gives
        // exact nearest-memory retrieval, exercising the generalization beyond
        // the softmax/sparsemax instances.
        let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let v = [1.0, 1.0];

        let hardmax = |logits: &[f64]| {
            let argmax = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            (0..logits.len())
                .map(|i| if i == argmax { 1.0 } else { 0.0 })
                .collect::<Vec<f64>>()
        };

        let retrieved = retrieve_fy(&v, &memories, 1.0, hardmax);
        assert_eq!(
            retrieved,
            vec![0.0, 0.0],
            "hardmax should retrieve the nearest memory exactly"
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

    #[test]
    fn test_lsr_gradient_zero_at_stored_memory() {
        // Hoover et al. (2025), Theorem 1: for well-separated memories the LSR
        // energy gradient is EXACTLY zero at a stored pattern, which is why LSR
        // retrieves exactly in a single step (LSE is only zero as beta -> inf).
        // Here r = ||xi_1 - xi_2|| = sqrt(200) ~ 14.1; at beta = 2 the support
        // radius sqrt(2/beta) = 1 excludes the other memory, so only the
        // self-term is active and the gradient at xi_1 must vanish exactly.
        let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let grad = energy_lsr_grad(&memories[0], &memories, 2.0);
        assert!(
            grad.iter().all(|g| g.abs() < 1e-12),
            "LSR gradient at a stored memory must be exactly zero, got {grad:?}"
        );
    }

    #[test]
    fn test_softmax_is_dense_simplex_projection() {
        let weights = softmax(&[1.0, 2.0, 3.0]);

        assert_eq!(weights.len(), 3);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.iter().all(|w| *w > 0.0));
        assert!(weights[2] > weights[1] && weights[1] > weights[0]);
    }

    #[test]
    fn test_sparsemax_can_select_exact_support() {
        let weights = sparsemax(&[1.0, 1.0, 0.0]);

        assert_eq!(weights, vec![0.5, 0.5, 0.0]);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparsemax_retrieval_drops_far_memory() {
        let memories = vec![vec![0.0], vec![1.0], vec![10.0]];
        let query = [0.25];
        let weights = sparsemax_weights(&query, &memories, 2.0);
        let retrieved = weighted_memory(&memories, &weights);

        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(weights[2], 0.0);
        assert!((retrieved[0] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_weighted_memory_rejects_inconsistent_shapes() {
        assert!(weighted_memory(&[vec![0.0], vec![1.0]], &[1.0]).is_empty());
        assert!(weighted_memory(&[vec![0.0], vec![1.0, 2.0]], &[0.5, 0.5]).is_empty());
    }

    #[test]
    fn test_lse_and_sparsemax_retrieval_agree_on_clear_match() {
        let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0], vec![-10.0, 10.0]];
        let query = [0.1, -0.1];

        let dense = retrieve_lse(&query, &memories, 4.0);
        let sparse = retrieve_sparsemax(&query, &memories, 4.0);

        assert!(dense[0].abs() < 1e-6);
        assert!(dense[1].abs() < 1e-6);
        assert_eq!(sparse, vec![0.0, 0.0]);
    }
}
