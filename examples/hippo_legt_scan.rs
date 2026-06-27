//! HiPPO-LegT scan feeding Hopfield retrieval.
//!
//! This is a `statescan` / `hippo` proof sketch, not public API. It builds the
//! small HiPPO-LegT transition matrix locally, scans scalar signals into
//! Legendre coefficients, then retrieves the matching trend memory.
//!
//! Run: cargo run --example hippo_legt_scan --release

use hopfield::retrieve_sparsemax;

const N: usize = 4;
const DT: f64 = 0.02;
const STEPS: usize = 200;
const BETA: f64 = 140.0;

fn main() {
    let constant = run_signal(|_| 1.0);
    let rising = run_signal(|t| t);
    let falling = run_signal(|t| 1.0 - t);

    let memories = vec![
        constant.to_vec(),
        rising.to_vec(),
        falling.to_vec(),
        run_signal(|t| if t > 0.7 { 1.0 } else { 0.0 }).to_vec(),
    ];

    let query = run_signal(|t| 0.2 + 0.8 * t);
    let retrieved = retrieve_sparsemax(&query, &memories, BETA);

    println!("HiPPO-LegT coefficients, N={N}, dt={DT}");
    println!("constant: {constant:?}");
    println!("rising:   {rising:?}");
    println!("falling:  {falling:?}");
    println!("query:    {query:?}");
    println!("retrieved trend memory: {retrieved:?}");
    println!(
        "dist(query, rising) {:.6}; dist(query, falling) {:.6}",
        l2(&query, &rising),
        l2(&query, &falling)
    );

    assert!(constant[0] > 0.999);
    assert!(constant[1..].iter().all(|x| x.abs() < 1e-4));
    assert!(rising[1] > 0.05);
    assert!(falling[1] < -0.05);
    assert!(l2(&retrieved, &rising) < l2(&retrieved, &falling));
}

fn run_signal(signal: impl Fn(f64) -> f64) -> [f64; N] {
    let (a, b) = hippo_legt();
    let lhs = backward_euler_lhs(&a);
    let mut state = [0.0; N];

    for step in 0..STEPS {
        let t = step as f64 / STEPS as f64;
        let u = signal(t);
        let mut rhs = state;
        for i in 0..N {
            rhs[i] += DT * b[i] * u;
        }
        state = solve(lhs, rhs);
    }

    state
}

fn hippo_legt() -> ([[f64; N]; N], [f64; N]) {
    let mut a = [[0.0; N]; N];
    let mut b = [0.0; N];
    let mut scale = [0.0; N];

    for (q, value) in scale.iter_mut().enumerate() {
        *value = ((2 * q + 1) as f64).sqrt();
        b[q] = *value;
    }

    for row in 0..N {
        for col in 0..N {
            let sign = if row < col && (col - row) % 2 != 0 {
                -1.0
            } else {
                1.0
            };
            a[row][col] = -scale[row] * sign * scale[col];
        }
    }

    (a, b)
}

fn backward_euler_lhs(a: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut lhs = [[0.0; N]; N];
    for row in 0..N {
        for col in 0..N {
            lhs[row][col] = if row == col { 1.0 } else { 0.0 } - DT * a[row][col];
        }
    }
    lhs
}

fn solve(mut lhs: [[f64; N]; N], mut rhs: [f64; N]) -> [f64; N] {
    for col in 0..N {
        let pivot = (col..N)
            .max_by(|&i, &j| lhs[i][col].abs().total_cmp(&lhs[j][col].abs()))
            .unwrap();
        lhs.swap(col, pivot);
        rhs.swap(col, pivot);

        let denom = lhs[col][col];
        for value in lhs[col].iter_mut().skip(col) {
            *value /= denom;
        }
        rhs[col] /= denom;

        let pivot_row = lhs[col];
        for row in 0..N {
            if row == col {
                continue;
            }
            let factor = lhs[row][col];
            for (value, pivot_value) in lhs[row]
                .iter_mut()
                .skip(col)
                .zip(pivot_row.iter().skip(col))
            {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    rhs
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}
