//! Titanium Gen-1 Alpha breakthrough kernels (Phase 2).
//!
//! This module provides:
//! - stochastic ternary gradient quantization
//! - Sophia-G style second-order optimizer core with update clipping
//! - bit-sliced accumulator for logic-only ternary updates

use anyhow::{Result, bail};
use rand::Rng;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernaryGrad {
    Neg,
    Zero,
    Pos,
}

impl TernaryGrad {
    #[inline]
    pub fn as_i8(self) -> i8 {
        match self {
            Self::Neg => -1,
            Self::Zero => 0,
            Self::Pos => 1,
        }
    }
}

/// FP32 -> ternary {-1, 0, +1} with stochastic rounding.
///
/// `clip` defines absolute clipping before quantization.
/// For |g| in [0, clip], probability of non-zero is |g| / clip.
pub fn quantize_gradients_stochastic<R: Rng>(
    gradients: &[f32],
    clip: f32,
    rng: &mut R,
) -> Vec<TernaryGrad> {
    let scale = clip.abs().max(1e-8);
    gradients
        .iter()
        .map(|g| {
            let clipped = g.clamp(-scale, scale);
            let p = (clipped.abs() / scale).min(1.0);
            let draw = rng.random::<f32>();
            if draw < p {
                if clipped.is_sign_negative() {
                    TernaryGrad::Neg
                } else {
                    TernaryGrad::Pos
                }
            } else {
                TernaryGrad::Zero
            }
        })
        .collect()
}

pub fn quantize_gradients_stochastic_parallel(
    gradients: &[f32],
    clip: f32,
    seed: u64,
) -> Vec<TernaryGrad> {
    let scale = clip.abs().max(1e-8);
    gradients
        .par_iter()
        .enumerate()
        .map(|(idx, g)| {
            let clipped = g.clamp(-scale, scale);
            let p = (clipped.abs() / scale).min(1.0);
            // Stateless pseudo-random draw per index for deterministic parallel quantization.
            let state = (idx as u64)
                .wrapping_mul(0x9E3779B185EBCA87)
                .wrapping_add(seed)
                .rotate_left(17);
            let draw = ((state >> 40) as f32) / ((1_u32 << 24) as f32);
            if draw < p {
                if clipped.is_sign_negative() {
                    TernaryGrad::Neg
                } else {
                    TernaryGrad::Pos
                }
            } else {
                TernaryGrad::Zero
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct EpigeneticMaskStats {
    pub suppressed: usize,
    pub exploratory_activated: usize,
}

/// Apply hormone-driven plasticity on ternary deltas.
///
/// - High cortisol suppresses risky updates (drive toward conservative behavior).
/// - High dopamine re-enables a controlled fraction of zeroed updates (drive exploration).
pub fn apply_epigenetic_mask(
    deltas: &mut [i8],
    cortisol: f32,
    dopamine: f32,
    seed: u64,
) -> EpigeneticMaskStats {
    let c = cortisol.clamp(0.0, 1.0);
    let d = dopamine.clamp(0.0, 1.0);
    let suppress_prob = (c * 0.65).clamp(0.0, 0.95);
    let explore_prob = (d * 0.30).clamp(0.0, 0.40);

    let mut suppressed = 0usize;
    let mut exploratory_activated = 0usize;
    for (idx, delta) in deltas.iter_mut().enumerate() {
        let h = (idx as u64)
            .wrapping_mul(0xD6E8FEB86659FD93)
            .wrapping_add(seed)
            .rotate_left(11);
        let draw1 = ((h >> 40) as f32) / ((1_u32 << 24) as f32);
        if *delta != 0 && draw1 < suppress_prob {
            *delta = 0;
            suppressed += 1;
            continue;
        }

        if *delta == 0 {
            let draw2 = (((h >> 16) as u32) as f32) / (u32::MAX as f32);
            if draw2 < explore_prob {
                *delta = if (h & 1) == 0 { 1 } else { -1 };
                exploratory_activated += 1;
            }
        }
    }

    EpigeneticMaskStats {
        suppressed,
        exploratory_activated,
    }
}

#[derive(Debug, Clone)]
pub struct SophiaGConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub rho: f32,
    pub clip_threshold: f32,
    pub weight_decay: f32,
}

impl Default for SophiaGConfig {
    fn default() -> Self {
        Self {
            beta1: 0.965,
            beta2: 0.99,
            epsilon: 1e-12,
            rho: 0.04,
            clip_threshold: 1.0,
            weight_decay: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SophiaGOptimizer {
    cfg: SophiaGConfig,
    exp_avg: Vec<f32>,
    hessian_diag_ema: Vec<f32>,
    step: usize,
}

#[derive(Debug, Clone)]
pub struct SophiaGStepStats {
    pub step: usize,
    pub clipped_updates: usize,
    pub max_abs_update: f32,
}

impl SophiaGOptimizer {
    pub fn new(param_len: usize, cfg: SophiaGConfig) -> Self {
        Self {
            cfg,
            exp_avg: vec![0.0; param_len],
            hessian_diag_ema: vec![0.0; param_len],
            step: 0,
        }
    }

    pub fn step_count(&self) -> usize {
        self.step
    }

    pub fn hessian_diag(&self) -> &[f32] {
        &self.hessian_diag_ema
    }

    /// Update diagonal Hessian estimate with a gradient sample.
    ///
    /// Gen-1 approximation: EMA(grad^2).
    pub fn update_hessian_diagonal(&mut self, grad_sample: &[f32]) -> Result<()> {
        if grad_sample.len() != self.hessian_diag_ema.len() {
            bail!(
                "hessian update mismatch: expected {}, got {}",
                self.hessian_diag_ema.len(),
                grad_sample.len()
            );
        }
        let b2 = self.cfg.beta2;
        let one_minus_b2 = 1.0 - b2;

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: runtime-gated by avx512f feature detection.
                unsafe {
                    update_hessian_diag_avx512(
                        &mut self.hessian_diag_ema,
                        grad_sample,
                        b2,
                        one_minus_b2,
                    );
                }
                return Ok(());
            }
        }

        if grad_sample.len() >= 16_384 && rayon::current_num_threads() > 1 {
            self.hessian_diag_ema
                .par_iter_mut()
                .zip(grad_sample.par_iter())
                .for_each(|(h, g)| {
                    *h = b2 * *h + one_minus_b2 * (g * g);
                });
        } else {
            for (h, g) in self.hessian_diag_ema.iter_mut().zip(grad_sample.iter()) {
                *h = b2 * *h + one_minus_b2 * (g * g);
            }
        }
        Ok(())
    }

    /// Sophia-G parameter update with clipping to prevent exploding updates.
    pub fn apply_step(&mut self, params: &mut [f32], gradients: &[f32], lr: f32) -> Result<SophiaGStepStats> {
        if params.len() != gradients.len() {
            bail!(
                "parameter/gradient length mismatch: params={} grads={}",
                params.len(),
                gradients.len()
            );
        }
        if params.len() != self.exp_avg.len() {
            bail!(
                "optimizer state mismatch: state={} params={}",
                self.exp_avg.len(),
                params.len()
            );
        }

        self.update_hessian_diagonal(gradients)?;

        let b1 = self.cfg.beta1;
        let eps = self.cfg.epsilon;
        let rho = self.cfg.rho.max(1e-8);
        let clip = self.cfg.clip_threshold.abs().max(1e-8);
        let wd = self.cfg.weight_decay;

        let mut clipped_updates = 0usize;
        let mut max_abs_update = 0.0_f32;

        if params.len() >= 16_384 && rayon::current_num_threads() > 1 {
            let (clip_count, max_abs) = params
                .par_iter_mut()
                .zip(self.exp_avg.par_iter_mut())
                .zip(self.hessian_diag_ema.par_iter())
                .zip(gradients.par_iter())
                .map(|(((p, m), h), g)| {
                    let g_wd = *g + wd * *p;
                    *m = b1 * *m + (1.0 - b1) * g_wd;
                    let denom = (*h + eps).max(eps);
                    let mut update = *m / (rho * denom);
                    let mut clipped = 0usize;
                    if update.abs() > clip {
                        clipped = 1;
                        update = update.signum() * clip;
                    }
                    *p -= lr * update;
                    (clipped, update.abs())
                })
                .reduce(|| (0usize, 0.0_f32), |a, b| (a.0 + b.0, a.1.max(b.1)));
            clipped_updates = clip_count;
            max_abs_update = max_abs;
        } else {
            for (((p, m), h), g) in params
                .iter_mut()
                .zip(self.exp_avg.iter_mut())
                .zip(self.hessian_diag_ema.iter())
                .zip(gradients.iter())
            {
                let g_wd = *g + wd * *p;
                *m = b1 * *m + (1.0 - b1) * g_wd;
                let denom = (*h + eps).max(eps);
                let mut update = *m / (rho * denom);
                let abs_before = update.abs();
                if abs_before > clip {
                    clipped_updates += 1;
                    update = update.signum() * clip;
                }
                max_abs_update = max_abs_update.max(update.abs());
                *p -= lr * update;
            }
        }

        self.step += 1;
        Ok(SophiaGStepStats {
            step: self.step,
            clipped_updates,
            max_abs_update,
        })
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn update_hessian_diag_avx512(
    hessian_diag_ema: &mut [f32],
    grad_sample: &[f32],
    beta2: f32,
    one_minus_beta2: f32,
) {
    let len = hessian_diag_ema.len();
    let mut i = 0usize;
    let vb2 = _mm512_set1_ps(beta2);
    let v1mb2 = _mm512_set1_ps(one_minus_beta2);
    while i + 16 <= len {
        // SAFETY: pointers are in-bounds due chunked loop condition.
        let h = unsafe { _mm512_loadu_ps(hessian_diag_ema.as_ptr().add(i)) };
        // SAFETY: pointers are in-bounds due chunked loop condition.
        let g = unsafe { _mm512_loadu_ps(grad_sample.as_ptr().add(i)) };
        let gg = _mm512_mul_ps(g, g);
        let next = _mm512_add_ps(_mm512_mul_ps(vb2, h), _mm512_mul_ps(v1mb2, gg));
        // SAFETY: pointers are in-bounds due chunked loop condition.
        unsafe { _mm512_storeu_ps(hessian_diag_ema.as_mut_ptr().add(i), next) };
        i += 16;
    }
    for j in i..len {
        hessian_diag_ema[j] = beta2 * hessian_diag_ema[j] + one_minus_beta2 * grad_sample[j] * grad_sample[j];
    }
}

/// Signed fixed-point accumulator represented by bit-planes (two's complement).
///
/// Each bit-plane stores one bit across up to 64 lanes per word.
/// Increment/decrement are performed via XOR/AND/NOT carry/borrow chains.
#[derive(Debug, Clone)]
pub struct BitSlicedAccumulator {
    bits: usize,
    lanes: usize,
    words: usize,
    planes: Vec<Vec<u64>>,
}

impl BitSlicedAccumulator {
    pub fn new(lanes: usize, bits: usize) -> Result<Self> {
        if lanes == 0 {
            bail!("lanes must be > 0");
        }
        if bits < 2 || bits > 31 {
            bail!("bits must be in [2, 31], got {bits}");
        }
        let words = lanes.div_ceil(64);
        let planes = (0..bits).map(|_| vec![0_u64; words]).collect();
        Ok(Self {
            bits,
            lanes,
            words,
            planes,
        })
    }

    pub fn bits(&self) -> usize {
        self.bits
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// Logic-only update from ternary deltas {-1, 0, +1}.
    pub fn update_ternary(&mut self, deltas: &[i8]) -> Result<()> {
        if deltas.len() != self.lanes {
            bail!(
                "delta size mismatch: expected {}, got {}",
                self.lanes,
                deltas.len()
            );
        }

        let mut pos_masks = vec![0_u64; self.words];
        let mut neg_masks = vec![0_u64; self.words];
        for (idx, d) in deltas.iter().copied().enumerate() {
            let word = idx / 64;
            let bit = idx % 64;
            let lane_mask = 1_u64 << bit;
            match d {
                1 => pos_masks[word] |= lane_mask,
                -1 => neg_masks[word] |= lane_mask,
                0 => {}
                other => bail!("invalid ternary delta at lane {idx}: {other}"),
            }
        }

        // Add +1 on selected lanes (carry chain).
        for (word_idx, mut carry) in pos_masks.into_iter().enumerate() {
            for plane_idx in 0..self.bits {
                let plane = self.planes[plane_idx][word_idx];
                let sum = plane ^ carry;
                carry &= plane;
                self.planes[plane_idx][word_idx] = sum;
                if carry == 0 {
                    break;
                }
            }
        }

        // Subtract 1 on selected lanes (borrow chain).
        for (word_idx, mut borrow) in neg_masks.into_iter().enumerate() {
            for plane_idx in 0..self.bits {
                let plane = self.planes[plane_idx][word_idx];
                let diff = plane ^ borrow;
                borrow &= !plane;
                self.planes[plane_idx][word_idx] = diff;
                if borrow == 0 {
                    break;
                }
            }
        }

        Ok(())
    }

    pub fn to_i32_vec(&self) -> Vec<i32> {
        let mut out = vec![0_i32; self.lanes];
        let sign_bit = self.bits - 1;
        for (lane, slot) in out.iter_mut().enumerate() {
            let word = lane / 64;
            let bit = lane % 64;
            let lane_mask = 1_u64 << bit;
            let mut raw = 0_i32;
            for plane in 0..self.bits {
                if (self.planes[plane][word] & lane_mask) != 0 {
                    raw |= 1_i32 << plane;
                }
            }
            // Sign extend from `bits` to i32.
            if ((raw >> sign_bit) & 1) != 0 {
                let mask = ((1_i64 << self.bits) - 1) as i32;
                raw |= !mask;
            }
            *slot = raw;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn ternary_quantizer_respects_signs() {
        let grads = [-2.0_f32, -0.3, 0.0, 0.3, 2.0];
        let mut rng = StdRng::seed_from_u64(7);
        let q = quantize_gradients_stochastic(&grads, 1.0, &mut rng);
        assert_eq!(q[0], TernaryGrad::Neg);
        assert_eq!(q[4], TernaryGrad::Pos);
    }

    #[test]
    fn sophia_updates_hessian_and_clips() -> Result<()> {
        let mut opt = SophiaGOptimizer::new(
            3,
            SophiaGConfig {
                clip_threshold: 0.05,
                ..SophiaGConfig::default()
            },
        );
        let mut params = vec![1.0_f32, -1.0, 0.5];
        let grads = vec![10.0_f32, -10.0, 5.0];
        let stats = opt.apply_step(&mut params, &grads, 1e-2)?;
        assert!(stats.clipped_updates > 0);
        assert_eq!(opt.step_count(), 1);
        assert!(opt.hessian_diag().iter().all(|v| *v > 0.0));
        Ok(())
    }

    #[test]
    fn bit_sliced_accumulator_tracks_ternary_updates() -> Result<()> {
        let mut acc = BitSlicedAccumulator::new(8, 8)?;
        acc.update_ternary(&[1, 1, 0, -1, 0, 0, 1, -1])?;
        acc.update_ternary(&[1, 0, 0, -1, 0, 0, -1, 1])?;
        let values = acc.to_i32_vec();
        assert_eq!(values[0], 2);
        assert_eq!(values[1], 1);
        assert_eq!(values[3], -2);
        assert_eq!(values[6], 0);
        assert_eq!(values[7], 0);
        Ok(())
    }

    #[test]
    fn epigenetic_mask_modulates_deltas() {
        let mut deltas = vec![1_i8; 256];
        let stats = apply_epigenetic_mask(&mut deltas, 0.95, 0.0, 7);
        assert!(stats.suppressed > 0);
    }
}
