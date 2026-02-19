use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use vagi_kernel::hdc::RecursiveHdcMemory;
use vagi_kernel::homeostasis::HomeostasisEngine;
use vagi_kernel::model::gpt_kan::{LKanGPT, LKanGPTConfig};
use vagi_kernel::model::lkan::LiquidKanConfig;
use vagi_kernel::moe_gate::{ExpertProfile, MoeGate};
use vagi_kernel::trainer_engine::{
    CausalGradientSignal, CpuBatchPlanner, MmapTokenDataLoader, TitaniumTrainer, TitaniumTrainerConfig,
    VerifierLossConfig, VerifierLossConnector,
};
use vagi_kernel::world_model::WorldModel;

const INPUT_PATH: &str = "data/input.txt";
const MAX_BATCH_SIZE: usize = 96;
const SEQ_LEN: usize = 128;
const STRIDE: usize = 64;
const MAX_STEPS: usize = 400;
const LOG_EVERY: usize = 20;
const PROBE_SEQ_LEN: usize = 12;
const PREFETCH_LOOKAHEAD: usize = 3;

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, value) in values.iter().copied().enumerate() {
        if value > best_val {
            best_val = value;
            best_idx = idx;
        }
    }
    best_idx
}

fn probe_token_ids(model: &LKanGPT, input_ids: &Tensor, probe_len: usize) -> Result<Vec<u32>> {
    let (_batch, seq_len) = input_ids.dims2()?;
    let slice_len = seq_len.min(probe_len).max(1);
    let probe = input_ids.narrow(0, 0, 1)?.narrow(1, 0, slice_len)?;
    let logits = model.forward_logits(&probe)?;
    let logits_3d = logits.to_vec3::<f32>()?;
    let Some(seq_logits) = logits_3d.first() else {
        return Ok(vec![0_u32]);
    };
    let mut out = Vec::with_capacity(seq_logits.len());
    for token_logits in seq_logits {
        out.push(argmax(token_logits) as u32);
    }
    if out.is_empty() {
        out.push(0_u32);
    }
    Ok(out)
}

fn synthesize_logic_ir(step: usize, probe_ids: &[u32]) -> String {
    let mut lines = vec![format!("echo titanium_step_{step}")];
    for (idx, id) in probe_ids.iter().copied().enumerate() {
        match id % 6 {
            0 => lines.push(format!("set s{idx}={id}")),
            1 => lines.push(format!("append trace={id}")),
            2 => lines.push(format!("echo p{idx}_{id}")),
            3 => lines.push(format!("warn gate{idx}_{id}")),
            4 => lines.push("append mode=logic_probe".to_string()),
            _ => lines.push(format!("fail logic_mismatch_{idx}_{id}")),
        }
    }
    lines.join("\n")
}

fn bootstrap_experts(gate: &MoeGate) {
    let mut experts = Vec::new();
    for idx in 0..16usize {
        let mut centroid = Vec::with_capacity(64);
        for j in 0..64usize {
            let v = (((idx * 67 + j * 31) % 101) as f32 / 100.0) * 2.0 - 1.0;
            centroid.push(v);
        }
        experts.push(ExpertProfile {
            expert_id: format!("expert_{idx:02}"),
            centroid,
            path: None,
        });
    }
    gate.register_experts(experts);
}

fn build_causal_signal(model: &LKanGPT, probe_ids: &[u32], violations: &[String]) -> CausalGradientSignal {
    let token_id = probe_ids.last().copied().unwrap_or(0);
    let top = model.trace_causal_modules(token_id, violations, 8);
    let mut negative_gradient = vec![0.0_f32; model.bit_sliced_shadow_dim().max(1)];
    let mut top_modules = Vec::new();
    for (slot_idx, score, module_name) in top {
        let idx = slot_idx % negative_gradient.len();
        negative_gradient[idx] += score;
        top_modules.push(module_name);
    }
    CausalGradientSignal {
        negative_gradient,
        intensity: 0.35,
        top_modules,
    }
}

fn main() -> Result<()> {
    let input_path = Path::new(INPUT_PATH);
    if !input_path.exists() {
        fs::create_dir_all("data")?;
        fs::write(
            INPUT_PATH,
            "Titanium training corpus dummy data for CPU-first vAGI.".repeat(100),
        )?;
    }

    let device = Device::Cpu;
    let model_cfg = LKanGPTConfig {
        vocab_size: 256,
        hidden_dim: 256,
        num_layers: 6,
        num_heads: 8,
        kan_config: LiquidKanConfig {
            in_dim: 256,
            hidden_dim: 256,
            out_dim: 256,
            cheb_order: 5,
            dt: 0.1,
            tau_min: 1e-3,
            x_scale: 1.0,
        },
    };
    let batch_plan = CpuBatchPlanner::recommend(SEQ_LEN, model_cfg.hidden_dim, MAX_BATCH_SIZE);
    let batch_size = batch_plan.recommended_batch_size;
    let mut loader = MmapTokenDataLoader::open(input_path, batch_size, SEQ_LEN, STRIDE)
        .with_context(|| "failed to initialize mmap dataloader")?;

    let trainer_cfg = TitaniumTrainerConfig {
        optimizer_state_dim: model_cfg.hidden_dim * model_cfg.num_layers,
        ..TitaniumTrainerConfig::default()
    };
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let model = LKanGPT::new(vb.pp("lkan_gpt"), model_cfg).context("failed to initialize model")?;
    let mut trainer = TitaniumTrainer::new_lkan(model, &var_map, trainer_cfg)?;

    let mut verifier_connector = VerifierLossConnector::new(VerifierLossConfig::default())?;
    let homeostasis = HomeostasisEngine::new();
    let world_model = WorldModel::new();
    let moe_gate = Arc::new(MoeGate::new(8));
    bootstrap_experts(&moe_gate);
    let mut hdc_memory = RecursiveHdcMemory::new(0.85);
    let (prefetch_tx, prefetch_rx) = mpsc::channel::<Vec<String>>();
    let prefetch_gate = Arc::clone(&moe_gate);
    let prefetch_handle = thread::spawn(move || {
        while let Ok(experts) = prefetch_rx.recv() {
            prefetch_gate.prefetch_experts(&experts);
        }
    });

    println!(
        "Titanium Trainer started: data={} batch_size={} seq_len={} stride={} max_steps={} l3={}MB threads={}",
        loader.path().display(),
        batch_size,
        SEQ_LEN,
        STRIDE,
        MAX_STEPS,
        batch_plan.l3_bytes / (1024 * 1024),
        batch_plan.worker_threads
    );

    while trainer.step() < MAX_STEPS {
        let Some((input_ids, targets)) = loader.next_batch(&device)? else {
            break;
        };

        let probe_ids = probe_token_ids(trainer.model(), &input_ids, PROBE_SEQ_LEN)?;
        let logic_ir = synthesize_logic_ir(trainer.step() + 1, &probe_ids);
        let verifier_outcome = verifier_connector.evaluate_and_feedback(logic_ir.clone(), &homeostasis);

        let prefetch_ids = world_model.predict_expert_prefetch(&probe_ids, PREFETCH_LOOKAHEAD);
        let _ = prefetch_tx.send(prefetch_ids);

        hdc_memory.ingest_episode(&format!(
            "{}\nviolations={:?}\npenalty={:.4}",
            logic_ir, verifier_outcome.violations, verifier_outcome.weighted_penalty
        ));
        let hdc_rel = hdc_memory.bind_query_relevance("logic verifier causal");

        let lr_scale = ((1.0 - verifier_outcome.weighted_penalty as f64)
            * (0.85 + 0.15 * hdc_rel.max(0.0) as f64))
            .clamp(0.10, 1.0);
        trainer.set_lr_scale(lr_scale);

        let hormones = homeostasis.snapshot();
        let causal_signal = build_causal_signal(trainer.model(), &probe_ids, &verifier_outcome.violations);
        let metrics = trainer.train_batch_with_controls(
            &input_ids,
            &targets,
            verifier_outcome.weighted_penalty,
            Some(&hormones),
            Some(&causal_signal),
        )?;

        if metrics.step % LOG_EVERY == 0 {
            println!(
                "step={} loss={:.6} lr={:.6e} logic_penalty={:.4} verifier_pass={} vio={} streak={} cortisol={:.3} prefetch_loaded={} hdc_rel={:.3} causal_norm={:.3} epi_suppr={} ternary_nz={} sophia_clip={}",
                metrics.step,
                metrics.loss,
                metrics.applied_lr,
                verifier_outcome.weighted_penalty,
                verifier_outcome.pass,
                verifier_outcome.violation_count,
                verifier_outcome.failure_streak,
                hormones.cortisol,
                moe_gate.loaded_count(),
                hdc_rel,
                metrics.causal_injection_norm,
                metrics.epigenetic_suppressed,
                metrics.ternary_non_zero,
                metrics.sophia_clipped_updates
            );
        }
    }

    let final_hormone = homeostasis.snapshot();
    drop(prefetch_tx);
    let _ = prefetch_handle.join();
    println!(
        "Titanium Gen-1 Alpha run completed: step={} final_cortisol={:.3} bit_sliced_checksum={} hdc_master_popcount={} loaded_experts={}",
        trainer.step(),
        final_hormone.cortisol,
        trainer.bit_sliced_checksum(),
        hdc_memory.master_popcount(),
        moe_gate.loaded_count()
    );
    Ok(())
}
