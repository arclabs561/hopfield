//! MNIST pattern completion and capacity in a modern (continuous) Hopfield net.
//!
//! Store N images as memories, query with the bottom half occluded, retrieve.
//! Reports recall@1 (retrieved vector's nearest memory equals the original)
//! and reconstruction L2 as load N grows.
//!
//! ```sh
//! ./scripts/fetch_mnist.sh
//! cargo run --release --example mnist_pattern_completion
//! ```

use std::path::Path;
use std::process::ExitCode;

use hopfield::retrieve_lse;

const BETA: f64 = 25.0;
const SIDE: usize = 28;

fn be_u32(b: &[u8], o: usize) -> usize {
    u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize
}

/// Parse an IDX image file into unit-L2-normalized 784-dim vectors.
fn load_images(path: &Path, limit: usize) -> std::io::Result<Vec<Vec<f64>>> {
    let b = std::fs::read(path)?;
    let n = be_u32(&b, 4).min(limit);
    let d = be_u32(&b, 8) * be_u32(&b, 12);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = 16 + i * d;
        let mut v: Vec<f64> = b[start..start + d]
            .iter()
            .map(|&p| p as f64 / 255.0)
            .collect();
        normalize(&mut v);
        out.push(v);
    }
    Ok(out)
}

fn normalize(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Zero the bottom half of a 28x28 image and renormalize what remains.
fn occlude_bottom(img: &[f64]) -> Vec<f64> {
    let mut q = img.to_vec();
    for px in q.iter_mut().take(SIDE * SIDE).skip(SIDE * (SIDE / 2)) {
        *px = 0.0;
    }
    normalize(&mut q);
    q
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Index of the memory most similar (max dot product) to `v`.
fn nearest(v: &[f64], memories: &[Vec<f64>]) -> usize {
    (0..memories.len())
        .max_by(|&i, &j| dot(v, &memories[i]).total_cmp(&dot(v, &memories[j])))
        .unwrap()
}

fn main() -> ExitCode {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mnist");
    let images_path = dir.join("t10k-images-idx3-ubyte");
    if !images_path.exists() {
        eprintln!(
            "dataset not found at {}\nrun: ./scripts/fetch_mnist.sh",
            dir.display()
        );
        return ExitCode::SUCCESS;
    }

    let all = load_images(&images_path, 1000).unwrap();
    println!("loaded {} MNIST images (beta = {BETA})\n", all.len());
    println!("capacity curve (bottom-half occluded query):");
    println!("  {:>5}  {:>9}  {:>14}", "N", "recall@1", "recon L2");

    for &n in &[10usize, 25, 50, 100, 200, 500] {
        if n > all.len() {
            break;
        }
        let memories = &all[..n];
        let mut correct = 0;
        let mut recon = 0.0;
        for (i, mem) in memories.iter().enumerate() {
            let query = occlude_bottom(mem);
            let retrieved = retrieve_lse(&query, memories, BETA);
            if nearest(&retrieved, memories) == i {
                correct += 1;
            }
            recon += l2(&retrieved, mem);
        }
        println!(
            "  {n:>5}  {:>9.4}  {:>14.4}",
            correct as f64 / n as f64,
            recon / n as f64
        );
    }

    ExitCode::SUCCESS
}
