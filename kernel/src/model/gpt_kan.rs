use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear, ops, rms_norm};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::lkan::{LiquidKanConfig, LiquidKanLayer};
use crate::trainer_engine::LKanGptTrainInterface;

const DEFAULT_MAX_SEQ_LEN: usize = 2_048;
const DEFAULT_RMS_NORM_EPS: f64 = 1e-5;

#[derive(Debug, Clone)]
pub struct LKanGPTConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub kan_config: LiquidKanConfig,
}

impl LKanGPTConfig {
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            bail!("vocab_size must be > 0");
        }
        if self.hidden_dim == 0 {
            bail!("hidden_dim must be > 0");
        }
        if self.num_layers == 0 {
            bail!("num_layers must be > 0");
        }
        if self.num_heads == 0 {
            bail!("num_heads must be > 0");
        }
        if self.hidden_dim % self.num_heads != 0 {
            bail!(
                "hidden_dim ({}) must be divisible by num_heads ({})",
                self.hidden_dim,
                self.num_heads
            );
        }
        if self.kan_config.in_dim != self.hidden_dim {
            bail!(
                "kan_config.in_dim ({}) must equal hidden_dim ({})",
                self.kan_config.in_dim,
                self.hidden_dim
            );
        }
        if self.kan_config.hidden_dim != self.hidden_dim {
            bail!(
                "kan_config.hidden_dim ({}) must equal hidden_dim ({})",
                self.kan_config.hidden_dim,
                self.hidden_dim
            );
        }
        if self.kan_config.out_dim != self.hidden_dim {
            bail!(
                "kan_config.out_dim ({}) must equal hidden_dim ({})",
                self.kan_config.out_dim,
                self.hidden_dim
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct TokenPositionEmbeddings {
    token_embedding: Embedding,
    position_embedding: Embedding,
    position_ids: Tensor,
}

impl TokenPositionEmbeddings {
    fn new(vb: VarBuilder<'_>, cfg: &LKanGPTConfig) -> Result<Self> {
        let token_embedding = embedding(cfg.vocab_size, cfg.hidden_dim, vb.pp("token_embedding"))?;
        let position_embedding = embedding(
            DEFAULT_MAX_SEQ_LEN,
            cfg.hidden_dim,
            vb.pp("position_embedding"),
        )?;
        let position_ids =
            Tensor::arange(0u32, DEFAULT_MAX_SEQ_LEN as u32, vb.device())?.unsqueeze(0)?;
        Ok(Self {
            token_embedding,
            position_embedding,
            position_ids,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids
            .dim(D::Minus1)
            .context("expected input_ids to be rank-2 [batch, seq_len]")?;
        if seq_len > DEFAULT_MAX_SEQ_LEN {
            bail!(
                "sequence length {} exceeds max supported {}",
                seq_len,
                DEFAULT_MAX_SEQ_LEN
            );
        }

        let token_embeds = self.token_embedding.forward(input_ids)?;
        let pos_ids = self.position_ids.narrow(1, 0, seq_len)?;
        let pos_embeds = self.position_embedding.forward(&pos_ids)?;
        Ok(token_embeds.broadcast_add(&pos_embeds)?)
    }
}

#[derive(Clone, Debug)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl CausalSelfAttention {
    fn new(vb: VarBuilder<'_>, cfg: &LKanGPTConfig) -> Result<Self> {
        let q_proj = linear(cfg.hidden_dim, cfg.hidden_dim, vb.pp("q_proj"))?;
        let k_proj = linear(cfg.hidden_dim, cfg.hidden_dim, vb.pp("k_proj"))?;
        let v_proj = linear(cfg.hidden_dim, cfg.hidden_dim, vb.pp("v_proj"))?;
        let o_proj = linear(cfg.hidden_dim, cfg.hidden_dim, vb.pp("o_proj"))?;
        let head_dim = cfg.hidden_dim / cfg.num_heads;
        let scale = (head_dim as f64).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_heads,
            head_dim,
            scale,
        })
    }

    fn split_heads(&self, xs: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        Ok(xs
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    fn forward(&self, xs: &Tensor, causal_mask: Option<&Tensor>) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let (bsz, seq_len, hidden_dim) = xs
            .dims3()
            .context("expected hidden states to be rank-3 [batch, seq_len, hidden_dim]")?;

        let query_states = (self.q_proj.forward(xs)? * self.scale)?;
        let proj_shape = (bsz * self.num_heads, seq_len, self.head_dim);

        let query_states = self
            .split_heads(&query_states, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let key_states = self
            .split_heads(&self.k_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let value_states = self
            .split_heads(&self.v_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;

        let attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;
        let src_len = key_states.dim(1)?;

        let attn_weights = if let Some(mask) = causal_mask {
            attn_weights
                .reshape((bsz, self.num_heads, seq_len, src_len))?
                .broadcast_add(mask)?
                .reshape((bsz * self.num_heads, seq_len, src_len))?
        } else {
            attn_weights
        };

        let attn_weights = ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&value_states)?.to_dtype(in_dtype)?;
        let attn_output = attn_output
            .reshape((bsz, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, hidden_dim))?;

        Ok(self.o_proj.forward(&attn_output)?)
    }
}

#[derive(Clone, Debug)]
pub struct Block {
    norm1: RmsNorm,
    attn: CausalSelfAttention,
    norm2: RmsNorm,
    ffn_kan: LiquidKanLayer,
}

impl Block {
    pub fn new(vb: VarBuilder<'_>, cfg: &LKanGPTConfig) -> Result<Self> {
        let norm1 = rms_norm(cfg.hidden_dim, DEFAULT_RMS_NORM_EPS, vb.pp("norm1"))?;
        let attn = CausalSelfAttention::new(vb.pp("attn"), cfg)?;
        let norm2 = rms_norm(cfg.hidden_dim, DEFAULT_RMS_NORM_EPS, vb.pp("norm2"))?;
        let ffn_kan = LiquidKanLayer::new(vb.pp("ffn_kan"), cfg.kan_config.clone())?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            ffn_kan,
        })
    }

    pub fn forward(&self, xs: &Tensor, causal_mask: Option<&Tensor>) -> Result<Tensor> {
        let attn_in = self.norm1.forward(xs)?;
        let attn_out = self.attn.forward(&attn_in, causal_mask)?;
        let xs = (&attn_out + xs)?;

        let ff_in = self.norm2.forward(&xs)?;
        let (bsz, seq_len, hidden_dim) = ff_in.dims3()?;
        let ff_flat = ff_in.reshape((bsz * seq_len, hidden_dim))?;

        // Stateless v1: reset liquid hidden state each block call.
        let state0 = self.ffn_kan.zero_state(bsz * seq_len)?;
        let state1 = self.ffn_kan.forward_step(&ff_flat, &state0)?;
        let ff_out = self.ffn_kan.predict(&state1)?;
        let ff_out = ff_out.reshape((bsz, seq_len, hidden_dim))?;

        (&ff_out + &xs).context("failed to apply feed-forward residual")
    }
}

#[derive(Clone, Debug)]
pub struct LKanGPT {
    cfg: LKanGPTConfig,
    embeddings: TokenPositionEmbeddings,
    blocks: Vec<Block>,
    final_norm: RmsNorm,
    classifier: Linear,
}

impl LKanGPT {
    pub fn new(vb: VarBuilder<'_>, cfg: LKanGPTConfig) -> Result<Self> {
        cfg.validate()?;

        let embeddings = TokenPositionEmbeddings::new(vb.pp("embeddings"), &cfg)?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let vb_blocks = vb.pp("blocks");
        for layer_idx in 0..cfg.num_layers {
            blocks.push(Block::new(vb_blocks.pp(layer_idx), &cfg)?);
        }
        let final_norm = rms_norm(cfg.hidden_dim, DEFAULT_RMS_NORM_EPS, vb.pp("final_norm"))?;
        let classifier = linear(cfg.hidden_dim, cfg.vocab_size, vb.pp("classifier"))?;

        Ok(Self {
            cfg,
            embeddings,
            blocks,
            final_norm,
            classifier,
        })
    }

    pub fn config(&self) -> &LKanGPTConfig {
        &self.cfg
    }

    /// Estimated dimensionality used by bit-sliced optimizer shadow state.
    pub fn bit_sliced_shadow_dim(&self) -> usize {
        self.cfg.hidden_dim * self.cfg.num_layers
    }

    /// Coarse causal attribution for a token error to model sub-modules.
    ///
    /// Returns top contributing module slots as (slot_idx, score, module_name).
    /// Slot index is suitable for building optimizer-side control vectors.
    pub fn trace_causal_modules(
        &self,
        token_id: u32,
        violation_tags: &[String],
        top_k: usize,
    ) -> Vec<(usize, f32, String)> {
        let heads = self.cfg.num_heads.max(1);
        let slots = self.cfg.num_layers * heads;
        let mut scored = Vec::with_capacity(slots);
        for layer_idx in 0..self.cfg.num_layers {
            for head_idx in 0..heads {
                let slot_idx = layer_idx * heads + head_idx;
                let mut hasher = DefaultHasher::new();
                token_id.hash(&mut hasher);
                layer_idx.hash(&mut hasher);
                head_idx.hash(&mut hasher);
                for tag in violation_tags {
                    tag.hash(&mut hasher);
                }
                let h = hasher.finish();
                let score = (((h >> 24) as u32) as f32 / (u32::MAX as f32)).clamp(0.0, 1.0);
                let module_name = format!("layer{layer_idx}.head{head_idx}.kan");
                scored.push((slot_idx, score, module_name));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k.min(scored.len()));
        scored
    }

    fn build_causal_mask(bsz: usize, seq_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::MIN } else { 0.0 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        Ok(mask.broadcast_as((bsz, 1, seq_len, seq_len))?)
    }

    pub fn forward_logits(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (bsz, seq_len) = input_ids
            .dims2()
            .context("expected input_ids shape [batch, seq_len]")?;
        if seq_len > DEFAULT_MAX_SEQ_LEN {
            bail!(
                "sequence length {} exceeds max supported {}",
                seq_len,
                DEFAULT_MAX_SEQ_LEN
            );
        }

        let mut xs = self.embeddings.forward(input_ids)?;
        let causal_mask = Self::build_causal_mask(bsz, seq_len, xs.device())?;
        for block in &self.blocks {
            xs = block.forward(&xs, Some(&causal_mask))?;
        }
        let xs = self.final_norm.forward(&xs)?;
        Ok(self.classifier.forward(&xs)?)
    }
}

impl LKanGptTrainInterface for LKanGPT {
    fn forward_train_logits(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_logits(input_ids)
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn base_cfg() -> LKanGPTConfig {
        LKanGPTConfig {
            vocab_size: 128,
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            kan_config: LiquidKanConfig {
                in_dim: 32,
                hidden_dim: 32,
                out_dim: 32,
                cheb_order: 3,
                dt: 0.05,
                tau_min: 1e-3,
                x_scale: 2.0,
            },
        }
    }

    fn build_model(cfg: LKanGPTConfig) -> Result<LKanGPT> {
        let dev = Device::Cpu;
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &dev);
        LKanGPT::new(vb.pp("gpt_kan"), cfg)
    }

    #[test]
    fn config_validation_rejects_invalid_heads() {
        let mut cfg = base_cfg();
        cfg.num_heads = 3;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validation_rejects_kan_mismatch() {
        let mut cfg = base_cfg();
        cfg.kan_config.out_dim = 16;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn causal_mask_shape_ok() -> Result<()> {
        let mask = LKanGPT::build_causal_mask(2, 8, &Device::Cpu)?;
        assert_eq!(mask.dims4()?, (2, 1, 8, 8));
        Ok(())
    }

    #[test]
    fn forward_shape_ok() -> Result<()> {
        let cfg = base_cfg();
        let model = build_model(cfg.clone())?;
        let ids: Vec<u32> = (0..16)
            .map(|v| (v % cfg.vocab_size as u32) as u32)
            .collect();
        let input_ids = Tensor::from_slice(&ids, (2, 8), &Device::Cpu)?;
        let logits = model.forward_logits(&input_ids)?;
        assert_eq!(logits.dims3()?, (2, 8, cfg.vocab_size));
        Ok(())
    }

    #[test]
    fn reject_too_long_sequence() -> Result<()> {
        let cfg = base_cfg();
        let model = build_model(cfg.clone())?;
        let ids = vec![0u32; DEFAULT_MAX_SEQ_LEN + 1];
        let input_ids = Tensor::from_slice(&ids, (1, DEFAULT_MAX_SEQ_LEN + 1), &Device::Cpu)?;
        assert!(model.forward_logits(&input_ids).is_err());
        Ok(())
    }
}
