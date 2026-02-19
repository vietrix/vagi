use std::path::Path;

use anyhow::{Result, bail};
use candle_core::Device;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vagi_kernel::titanium_kernels::{
    BitSlicedAccumulator, SophiaGConfig, SophiaGOptimizer, quantize_gradients_stochastic,
};
use vagi_kernel::trainer_engine::MmapTokenDataLoader;

const INPUT_PATH: &str = "data/input.txt";
const BATCH_SIZE: usize = 32;
const SEQ_LEN: usize = 64;
const STRIDE: usize = 32;
const PARAM_DIM: usize = 4096;
const MAX_STEPS: usize = 200;
const LOG_EVERY: usize = 20;

fn batch_to_gradients(input_ids: &[u32], targets: &[u32], out_dim: usize) -> Vec<f32> {
    let mut grads = vec![0.0_f32; out_dim];
    for (idx, (&x, &y)) in input_ids.iter().zip(targets.iter()).enumerate() {
        let slot = idx % out_dim;
        let diff = y as f32 - x as f32;
        grads[slot] += diff / 255.0;
    }
    let norm = (input_ids.len().max(1)) as f32;
    for g in &mut grads {
        *g /= norm;
    }
    grads
}

fn main() -> Result<()> {
    let input_path = Path::new(INPUT_PATH);
    if !input_path.exists() {
        bail!("missing training corpus at {}", input_path.display());
    }

    let device = Device::Cpu;
    let mut loader = MmapTokenDataLoader::open(input_path, BATCH_SIZE, SEQ_LEN, STRIDE)?;
    let mut params = vec![0.0_f32; PARAM_DIM];
    let mut sophia = SophiaGOptimizer::new(
        PARAM_DIM,
        SophiaGConfig {
            clip_threshold: 0.1,
            weight_decay: 1e-2,
            ..SophiaGConfig::default()
        },
    );
    let mut bit_acc = BitSlicedAccumulator::new(PARAM_DIM, 16)?;
    let mut rng = StdRng::seed_from_u64(42);

    let mut steps = 0usize;
    while steps < MAX_STEPS {
        let Some((input, target)) = loader.next_batch(&device)? else {
            break;
        };
        let input_flat = input.flatten_all()?.to_vec1::<u32>()?;
        let target_flat = target.flatten_all()?.to_vec1::<u32>()?;
        let grads = batch_to_gradients(&input_flat, &target_flat, PARAM_DIM);

        let ternary = quantize_gradients_stochastic(&grads, 0.25, &mut rng);
        let ternary_i8: Vec<i8> = ternary.iter().map(|q| q.as_i8()).collect();
        bit_acc.update_ternary(&ternary_i8)?;

        let stats = sophia.apply_step(&mut params, &grads, 8e-4)?;
        steps += 1;

        if steps % LOG_EVERY == 0 {
            let non_zero = ternary_i8.iter().filter(|v| **v != 0).count();
            println!(
                "step={} clipped={} max_update={:.5} ternary_nz={}/{}",
                stats.step, stats.clipped_updates, stats.max_abs_update, non_zero, PARAM_DIM
            );
        }
    }

    let signed_values = bit_acc.to_i32_vec();
    let checksum: i64 = signed_values.iter().map(|v| *v as i64).sum();
    println!(
        "Phase-2 kernel run complete: steps={} checksum={} hessian_dim={}",
        steps,
        checksum,
        sophia.hessian_diag().len()
    );
    Ok(())
}
