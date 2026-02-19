//! CPU-first training engine (Titanium track).

use std::collections::VecDeque;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap, loss};
use memmap2::MmapOptions;

use crate::homeostasis::{HomeostasisEngine, HormoneEvent, NeuroTransmitters};
use crate::model::gpt_kan::LKanGPT;
use crate::models::VerifierRequest;
use crate::verifier::Verifier;
use crate::titanium_kernels::{
    BitSlicedAccumulator, SophiaGConfig, SophiaGOptimizer, SophiaGStepStats, apply_epigenetic_mask,
    quantize_gradients_stochastic_parallel,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingBackend {
    AdamW,
    SophiaG,
    TernaryBitSliced,
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizerKind {
    Lion,
    Sophia,
}

#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub optimizer: OptimizerKind,
    pub lr: f32,
    pub convergence_window: usize,
    pub local_convergence_epsilon: f32,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            optimizer: OptimizerKind::Lion,
            lr: 1e-3,
            convergence_window: 16,
            local_convergence_epsilon: 2e-4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainerStepResult {
    pub skipped: bool,
    pub loss: f32,
    pub logic_penalty: f32,
    pub applied_lr: f32,
}

/// Legacy controller preserved for existing dry-run binaries.
pub struct TrainerEngine {
    cfg: TrainerConfig,
    loss_history: VecDeque<f32>,
}

impl TrainerEngine {
    pub fn new(cfg: TrainerConfig) -> Self {
        Self {
            cfg,
            loss_history: VecDeque::new(),
        }
    }

    pub fn step(&mut self, loss: f32, logic_penalty: f32) -> TrainerStepResult {
        self.loss_history.push_back(loss);
        while self.loss_history.len() > self.cfg.convergence_window {
            self.loss_history.pop_front();
        }

        let converged = self.is_locally_converged();
        let skipped = converged && logic_penalty < 0.02;
        let opt_scale = match self.cfg.optimizer {
            OptimizerKind::Lion => 1.0,
            OptimizerKind::Sophia => 0.85,
        };
        let applied_lr = if skipped {
            0.0
        } else {
            self.cfg.lr * opt_scale / (1.0 + logic_penalty * 2.0)
        };

        TrainerStepResult {
            skipped,
            loss,
            logic_penalty,
            applied_lr,
        }
    }

    fn is_locally_converged(&self) -> bool {
        if self.loss_history.len() < self.cfg.convergence_window {
            return false;
        }
        let min_loss = self
            .loss_history
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max_loss = self
            .loss_history
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        (max_loss - min_loss) <= self.cfg.local_convergence_epsilon
    }
}

#[derive(Debug, Clone)]
pub struct TitaniumTrainerConfig {
    pub lr: f64,
    pub weight_decay: f64,
    pub backend: TrainingBackend,
    pub ternary_clip: f32,
    pub optimizer_state_dim: usize,
    pub bit_slice_bits: usize,
    pub sophia_cfg: SophiaGConfig,
}

impl Default for TitaniumTrainerConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            weight_decay: 1e-2,
            backend: TrainingBackend::SophiaG,
            ternary_clip: 0.25,
            optimizer_state_dim: 4096,
            bit_slice_bits: 16,
            sophia_cfg: SophiaGConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TitaniumStepMetrics {
    pub step: usize,
    pub loss: f32,
    pub batch_size: usize,
    pub seq_len: usize,
    pub applied_lr: f64,
    pub sophia_clipped_updates: usize,
    pub ternary_non_zero: usize,
    pub epigenetic_suppressed: usize,
    pub causal_injection_norm: f32,
}

#[derive(Debug, Clone)]
pub struct CausalGradientSignal {
    pub negative_gradient: Vec<f32>,
    pub intensity: f32,
    pub top_modules: Vec<String>,
}

/// Training interface for LKAN-GPT compatible models.
pub trait LKanGptTrainInterface {
    fn forward_train_logits(&self, input_ids: &Tensor) -> Result<Tensor>;
    fn vocab_size(&self) -> usize;
}

pub struct TitaniumTrainer {
    model: LKanGPT,
    optimizer: AdamW,
    sophia: SophiaGOptimizer,
    acc: BitSlicedAccumulator,
    cfg: TitaniumTrainerConfig,
    step: usize,
    lr_scale: f64,
    shadow_params: Vec<f32>,
    last_sophia_stats: Option<SophiaGStepStats>,
}

impl TitaniumTrainer {
    pub fn new_lkan(model: LKanGPT, var_map: &VarMap, cfg: TitaniumTrainerConfig) -> Result<Self> {
        let optimizer = AdamW::new(
            var_map.all_vars(),
            ParamsAdamW {
                lr: cfg.lr,
                weight_decay: cfg.weight_decay,
                ..Default::default()
            },
        )
        .context("failed to initialize AdamW for TitaniumTrainer")?;

        let param_dim = cfg
            .optimizer_state_dim
            .max(model.bit_sliced_shadow_dim())
            .max(1);
        let sophia = SophiaGOptimizer::new(
            param_dim,
            SophiaGConfig {
                weight_decay: cfg.sophia_cfg.weight_decay.max(cfg.weight_decay as f32),
                ..cfg.sophia_cfg.clone()
            },
        );
        let acc = BitSlicedAccumulator::new(param_dim, cfg.bit_slice_bits)?;

        Ok(Self {
            model,
            optimizer,
            sophia,
            acc,
            cfg,
            step: 0,
            lr_scale: 1.0,
            shadow_params: vec![0.0; param_dim],
            last_sophia_stats: None,
        })
    }

    pub fn model(&self) -> &LKanGPT {
        &self.model
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn set_lr_scale(&mut self, scale: f64) {
        self.lr_scale = scale.clamp(0.05, 1.0);
    }

    pub fn lr_scale(&self) -> f64 {
        self.lr_scale
    }

    pub fn sophia_stats(&self) -> SophiaGStepStats {
        self.last_sophia_stats.clone().unwrap_or(SophiaGStepStats {
            step: self.step,
            clipped_updates: 0,
            max_abs_update: 0.0,
        })
    }

    pub fn bit_sliced_checksum(&self) -> i64 {
        self.acc.to_i32_vec().iter().map(|v| *v as i64).sum()
    }

    fn synthesize_surrogate_gradients(
        &self,
        input_ids: &Tensor,
        targets: &Tensor,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        let input_flat = input_ids.flatten_all()?.to_vec1::<u32>()?;
        let target_flat = targets.flatten_all()?.to_vec1::<u32>()?;
        let mut grads = vec![0.0_f32; out_dim];
        for (idx, (&x, &y)) in input_flat.iter().zip(target_flat.iter()).enumerate() {
            let slot = idx % out_dim;
            grads[slot] += (y as f32 - x as f32) / 255.0;
        }
        let norm = input_flat.len().max(1) as f32;
        for g in &mut grads {
            *g /= norm;
        }
        Ok(grads)
    }

    pub fn train_batch(&mut self, input_ids: &Tensor, targets: &Tensor) -> Result<TitaniumStepMetrics> {
        self.train_batch_with_controls(input_ids, targets, 0.0, None, None)
    }

    pub fn train_batch_with_penalty(
        &mut self,
        input_ids: &Tensor,
        targets: &Tensor,
        logic_penalty: f32,
    ) -> Result<TitaniumStepMetrics> {
        self.train_batch_with_controls(input_ids, targets, logic_penalty, None, None)
    }

    pub fn train_batch_with_controls(
        &mut self,
        input_ids: &Tensor,
        targets: &Tensor,
        logic_penalty: f32,
        hormones: Option<&NeuroTransmitters>,
        causal_signal: Option<&CausalGradientSignal>,
    ) -> Result<TitaniumStepMetrics> {
        let (batch_size, seq_len) = input_ids
            .dims2()
            .context("expected input_ids shape [batch, seq_len]")?;
        
        let adaptive_lr = (self.cfg.lr * self.lr_scale) / (1.0 + logic_penalty as f64);
        self.optimizer.set_learning_rate(adaptive_lr);

        let logits = self.model.forward_train_logits(input_ids)?;
        let flat_logits = logits.reshape((batch_size * seq_len, self.model.vocab_size()))?;
        let flat_targets = targets.reshape((batch_size * seq_len,))?;
        let loss = loss::cross_entropy(&flat_logits, &flat_targets)?;
        let loss_value = loss.to_scalar::<f32>()?;
        
        let mut sophia_clipped_updates = 0usize;
        let mut ternary_non_zero = 0usize;
        let mut epigenetic_suppressed = 0usize;
        let mut causal_injection_norm = 0.0_f32;

        let backend_lr = match self.cfg.backend {
            TrainingBackend::AdamW => adaptive_lr,
            TrainingBackend::SophiaG | TrainingBackend::TernaryBitSliced => {
                let mut surrogate_grads = self.synthesize_surrogate_gradients(
                    input_ids,
                    targets,
                    self.shadow_params.len(),
                )?;
                if let Some(causal) = causal_signal {
                    let limit = surrogate_grads.len().min(causal.negative_gradient.len());
                    for idx in 0..limit {
                        let delta = causal.negative_gradient[idx] * causal.intensity;
                        surrogate_grads[idx] -= delta;
                        causal_injection_norm += delta.abs();
                    }
                }

                let ternary = quantize_gradients_stochastic_parallel(
                    &surrogate_grads,
                    self.cfg.ternary_clip,
                    (self.step as u64).wrapping_mul(0x9E3779B185EBCA87),
                );
                let mut ternary_i8: Vec<i8> = ternary.iter().map(|q| q.as_i8()).collect();
                if let Some(h) = hormones {
                    let epi = apply_epigenetic_mask(
                        &mut ternary_i8,
                        h.cortisol,
                        h.dopamine,
                        (self.step as u64).wrapping_mul(0xA0761D6478BD642F),
                    );
                    epigenetic_suppressed = epi.suppressed;
                }
                ternary_non_zero = ternary_i8.iter().filter(|v| **v != 0).count();
                self.acc.update_ternary(&ternary_i8)?;

                let stats =
                    self.sophia
                        .apply_step(&mut self.shadow_params, &surrogate_grads, adaptive_lr as f32)?;
                sophia_clipped_updates = stats.clipped_updates;
                self.last_sophia_stats = Some(stats.clone());
                let sophia_lr_scale = 1.0 / (1.0 + stats.max_abs_update as f64);
                adaptive_lr * sophia_lr_scale.clamp(0.2, 1.0)
            }
        };

        self.optimizer.set_learning_rate(backend_lr);
        self.optimizer.backward_step(&loss)?;

        if matches!(self.cfg.backend, TrainingBackend::AdamW) {
            self.last_sophia_stats = None;
        }

        self.step += 1;
        Ok(TitaniumStepMetrics {
            step: self.step,
            loss: loss_value + logic_penalty,
            batch_size,
            seq_len,
            applied_lr: backend_lr,
            sophia_clipped_updates,
            ternary_non_zero,
            epigenetic_suppressed,
            causal_injection_norm,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CpuBatchPlan {
    pub l3_bytes: usize,
    pub worker_threads: usize,
    pub recommended_batch_size: usize,
}

pub struct CpuBatchPlanner;

impl CpuBatchPlanner {
    pub fn recommend(seq_len: usize, hidden_dim: usize, max_batch: usize) -> CpuBatchPlan {
        let l3_bytes = detect_l3_cache_bytes().unwrap_or(32 * 1024 * 1024);
        let worker_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let bytes_per_sample = seq_len
            .saturating_mul(hidden_dim)
            .saturating_mul(std::mem::size_of::<f32>())
            .saturating_mul(6);
        let budget = ((l3_bytes as f64) * 0.70) as usize;
        let by_cache = (budget / bytes_per_sample.max(1)).max(1);
        let by_threads = worker_threads.saturating_mul(8).max(1);
        let recommended_batch_size = by_cache.min(by_threads).min(max_batch).max(1);
        CpuBatchPlan {
            l3_bytes,
            worker_threads,
            recommended_batch_size,
        }
    }
}

fn detect_l3_cache_bytes() -> Option<usize> {
    if let Ok(v) = std::env::var("VAGI_L3_BYTES") {
        if let Ok(parsed) = v.parse::<usize>() {
            return Some(parsed);
        }
    }
    #[cfg(target_os = "linux")]
    {
        let content =
            std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size").ok()?;
        let trimmed = content.trim().to_ascii_uppercase();
        if let Some(kb) = trimmed.strip_suffix('K').and_then(|s| s.parse::<usize>().ok()) {
            return Some(kb * 1024);
        }
        if let Some(mb) = trimmed.strip_suffix('M').and_then(|s| s.parse::<usize>().ok()) {
            return Some(mb * 1024 * 1024);
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct VerifierLossConfig {
    pub penalty_weight: f32,
    pub max_loop_iters: u32,
    pub side_effect_budget: u32,
    pub timeout_ms: u64,
    pub cortisol_escalation_threshold: u32,
}

impl Default for VerifierLossConfig {
    fn default() -> Self {
        Self {
            penalty_weight: 0.35,
            max_loop_iters: 64,
            side_effect_budget: 3,
            timeout_ms: 30,
            cortisol_escalation_threshold: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VerifierLossOutcome {
    pub pass: bool,
    pub raw_logic_penalty: f32,
    pub weighted_penalty: f32,
    pub violation_count: usize,
    pub failure_streak: u32,
    pub violations: Vec<String>,
}

pub struct VerifierLossConnector {
    verifier: Verifier,
    cfg: VerifierLossConfig,
    failure_streak: u32,
}

impl VerifierLossConnector {
    pub fn new(cfg: VerifierLossConfig) -> Result<Self> {
        Ok(Self {
            verifier: Verifier::new().context("failed to initialize verifier for loss connector")?,
            cfg,
            failure_streak: 0,
        })
    }

    pub fn evaluate_and_feedback(
        &mut self,
        patch_ir: String,
        homeostasis: &HomeostasisEngine,
    ) -> VerifierLossOutcome {
        let response = self.verifier.check(&VerifierRequest {
            patch_ir,
            max_loop_iters: Some(self.cfg.max_loop_iters),
            side_effect_budget: Some(self.cfg.side_effect_budget),
            timeout_ms: Some(self.cfg.timeout_ms),
        });

        if response.pass {
            self.failure_streak = 0;
            homeostasis.process_event(&HormoneEvent::RequestSuccess);
        } else {
            self.failure_streak = self.failure_streak.saturating_add(1);
            homeostasis.process_event(&HormoneEvent::RequestFailure);
            if self.failure_streak >= self.cfg.cortisol_escalation_threshold {
                let escalation_count = (self.failure_streak - self.cfg.cortisol_escalation_threshold + 1)
                    .min(3);
                for _ in 0..escalation_count {
                    homeostasis.process_event(&HormoneEvent::UserFrustration);
                }
            }
        }

        let weighted_penalty = (response.logic_penalty * self.cfg.penalty_weight).min(1.0);
        VerifierLossOutcome {
            pass: response.pass,
            raw_logic_penalty: response.logic_penalty,
            weighted_penalty,
            violation_count: response.violations.len(),
            failure_streak: self.failure_streak,
            violations: response.violations,
        }
    }
}

pub struct MmapTokenDataLoader {
    _file: File,
    mmap: memmap2::Mmap,
    path: PathBuf,
    batch_size: usize,
    seq_len: usize,
    stride: usize,
    cursor: usize,
}

impl MmapTokenDataLoader {
    pub fn open(path: impl AsRef<Path>, batch_size: usize, seq_len: usize, stride: usize) -> Result<Self> {
        if batch_size == 0 {
            bail!("batch_size must be > 0");
        }
        if seq_len == 0 {
            bail!("seq_len must be > 0");
        }
        if stride == 0 {
            bail!("stride must be > 0");
        }

        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)
            .with_context(|| format!("failed to open corpus for mmap: {}", path.display()))?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .with_context(|| format!("failed to mmap corpus: {}", path.display()))?
        };
        if mmap.len() <= seq_len + 1 {
            bail!(
                "corpus too small for seq_len={seq_len}: {} bytes available",
                mmap.len()
            );
        }

        Ok(Self {
            _file: file,
            mmap,
            path,
            batch_size,
            seq_len,
            stride,
            cursor: 0,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    pub fn is_exhausted(&self) -> bool {
        self.cursor + self.seq_len + 1 >= self.mmap.len()
    }

    pub fn next_batch(&mut self, device: &Device) -> Result<Option<(Tensor, Tensor)>> {
        let max_start = self.mmap.len() - self.seq_len - 1;
        if self.cursor > max_start {
            return Ok(None);
        }

        let mut x_buf = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut y_buf = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut start = self.cursor;

        for _ in 0..self.batch_size {
            if start > max_start {
                break;
            }
            let x_slice = &self.mmap[start..start + self.seq_len];
            let y_slice = &self.mmap[start + 1..start + self.seq_len + 1];
            x_buf.extend(x_slice.iter().map(|&b| b as u32));
            y_buf.extend(y_slice.iter().map(|&b| b as u32));
            start = start.saturating_add(self.stride);
        }

        if x_buf.is_empty() {
            return Ok(None);
        }

        let actual_batch = x_buf.len() / self.seq_len;
        self.cursor = start;
        let x = Tensor::from_slice(&x_buf, (actual_batch, self.seq_len), device)?;
        let y = Tensor::from_slice(&y_buf, (actual_batch, self.seq_len), device)?;
        Ok(Some((x, y)))
    }
}
