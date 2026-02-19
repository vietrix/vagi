use std::time::Instant;

use anyhow::Result;
use vagi_kernel::bit_lkan::BitLkanMatrix;

fn main() -> Result<()> {
    let rows = 512;
    let cols = 512;
    let mut data = vec![0.0_f32; rows * cols];
    for (idx, item) in data.iter_mut().enumerate() {
        let pattern = (idx % 7) as f32 - 3.0;
        *item = pattern / 3.0;
    }
    let matrix = BitLkanMatrix::from_f32(rows, cols, &data, 0.2)?;
    let x: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) / 6.0).collect();

    let warmup = matrix.matvec(&x)?;
    let _ = warmup.len();

    let loops = 200;
    let start = Instant::now();
    for _ in 0..loops {
        let _ = matrix.matvec(&x)?;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let per_iter_ms = elapsed * 1000.0 / loops as f64;

    println!(
        "Titanium benchmark: loops={} total={:.3}s per_iter={:.3}ms",
        loops, elapsed, per_iter_ms
    );
    println!("Target check: speedup vs Gen-1 > 140% should be validated with baseline run.");
    Ok(())
}
