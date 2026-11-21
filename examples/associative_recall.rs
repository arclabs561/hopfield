//! Associative recall from a deterministic memory bank.
//!
//! A query scores all stored patterns, forms retrieval weights, and returns a
//! weighted memory. The example uses 64 memories in 16 dimensions and reports
//! nearest-memory rank, retrieval distance, and top weights.

use hopfield::{lse_weights, retrieve_lse, retrieve_sparsemax, sparsemax_weights};

const N_MEMORIES: usize = 64;
const DIMS: usize = 16;
const TARGET: usize = 17;
const BETA: f64 = 18.0;

fn pattern(index: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..DIMS)
        .map(|dim| {
            let a = (index as f64 + 1.0) * (dim as f64 + 3.0) * 0.173;
            let b = (index as f64 * 7.0 + dim as f64 * 11.0) * 0.071;
            a.sin() + 0.5 * b.cos()
        })
        .collect();
    normalize(&mut v);
    v
}

fn normalize(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in v {
        *x /= norm;
    }
}

fn corrupt(v: &[f64]) -> Vec<f64> {
    let mut q: Vec<f64> = v
        .iter()
        .enumerate()
        .map(|(i, x)| x + 0.18 * ((i * 13 + 5) as f64).sin())
        .collect();
    normalize(&mut q);
    q
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn top_weights(weights: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut ranked: Vec<(usize, f64)> = weights.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(k);
    ranked
}

fn unit_vector(text: &str) -> Result<Vec<f64>, String> {
    let mut values: Vec<f64> = text
        .split_whitespace()
        .map(|word| {
            word.parse::<f64>()
                .map_err(|_| format!("invalid coordinate {word:?}"))
        })
        .collect::<Result<_, _>>()?;
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return Err("expected finite coordinates".into());
    }
    let scale = values.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return Err("a zero vector has no direction".into());
    }
    for x in &mut values {
        *x /= scale;
    }
    normalize(&mut values);
    Ok(values)
}

fn memory_bank(text: &str, dims: usize) -> Result<Vec<Vec<f64>>, String> {
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let row = unit_vector(line).map_err(|e| format!("memory #{}: {e}", i))?;
        if row.len() != dims {
            return Err(format!(
                "memory #{i} has {} coordinates, query has {dims}",
                row.len()
            ));
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("memory file is empty".into());
    }
    Ok(rows)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let (memories, query, target) = match args.as_slice() {
        [] => {
            let memories: Vec<_> = (0..N_MEMORIES).map(pattern).collect();
            let query = corrupt(&memories[TARGET]);
            (memories, query, Some(TARGET))
        }
        [path, coordinates] => {
            let query = unit_vector(coordinates.to_str().ok_or("query must be text")?)?;
            let memories = memory_bank(&std::fs::read_to_string(path)?, query.len())?;
            (memories, query, None)
        }
        _ => return Err("usage: associative_recall [MEMORIES.txt 'QUERY COORDINATES']".into()),
    };

    let mut nearest: Vec<(usize, f64)> = memories
        .iter()
        .enumerate()
        .map(|(i, memory)| (i, l2(&query, memory)))
        .collect();
    nearest.sort_by(|a, b| a.1.total_cmp(&b.1));

    let lse = retrieve_lse(&query, &memories, BETA);
    let sparse = retrieve_sparsemax(&query, &memories, BETA);
    let lse_weights = lse_weights(&query, &memories, BETA);
    let sparse_weights = sparsemax_weights(&query, &memories, BETA);

    println!(
        "memories: {}, dims: {}, beta: {BETA}",
        memories.len(),
        query.len()
    );
    if let Some(target) = target {
        println!("target memory: #{target}");
    }
    println!(
        "nearest raw memory: #{} at distance {:.4}",
        nearest[0].0, nearest[0].1
    );
    if let Some(target) = target {
        println!(
        "distance(query, target): {:.4}; distance(lse, target): {:.4}; distance(sparse, target): {:.4}",
        l2(&query, &memories[target]),
        l2(&lse, &memories[target]),
        l2(&sparse, &memories[target])
    );
    } else {
        println!("LSE retrieval: {lse:?}");
        println!("sparsemax retrieval: {sparse:?}");
    }
    println!("top LSE weights: {:?}", top_weights(&lse_weights, 5));
    println!(
        "top sparsemax weights: {:?}",
        top_weights(&sparse_weights, 5)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_and_normalizes_rows_in_file_order() {
        assert_eq!(
            memory_bank("1e308 0\n0 1e-308\n", 2).unwrap(),
            vec![vec![1., 0.], vec![0., 1.]]
        );
    }

    #[test]
    fn rejects_rows_that_cannot_be_compared_to_the_query() {
        for bad in ["", "1", "1 0\n\n0 1", "0 0", "1 NaN", "inf 0", "x 0"] {
            assert!(memory_bank(bad, 2).is_err(), "{bad:?}");
        }
    }
}
