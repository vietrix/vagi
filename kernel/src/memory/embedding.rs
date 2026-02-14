use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const MODEL_REVISION: &str = "refs/pr/21";
const EMBEDDING_DIM: usize = 384;

/// Rust-native embedding engine backed by candle + all-MiniLM-L6-v2.
///
/// Replaces the Python `sentence_transformers` dependency, making the
/// Kernel a fully self-contained inference unit.
pub struct EmbeddingEngine {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingEngine {
    /// Download (first run) and load the BERT model + tokenizer.
    ///
    /// Model files are cached under the HuggingFace Hub default cache
    /// directory (`~/.cache/huggingface/hub`).
    pub fn new() -> Result<Self> {
        let device = Device::Cpu;

        let repo = Repo::with_revision(
            MODEL_ID.to_string(),
            RepoType::Model,
            MODEL_REVISION.to_string(),
        );
        let api = Api::new().context("failed to create HuggingFace Hub API client")?;
        let api = api.repo(repo);

        let config_path = api
            .get("config.json")
            .context("failed to download config.json")?;
        let tokenizer_path = api
            .get("tokenizer.json")
            .context("failed to download tokenizer.json")?;
        let weights_path = api
            .get("model.safetensors")
            .context("failed to download model.safetensors")?;

        let config_str = std::fs::read_to_string(&config_path)
            .context("failed to read config.json")?;
        let config: Config = serde_json::from_str(&config_str)
            .context("failed to parse config.json")?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow::anyhow!("failed to load tokenizer: {err}"))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .context("failed to load model weights")?
        };
        let model =
            BertModel::load(vb, &config).context("failed to build BertModel from weights")?;

        tracing::info!(
            model_id = MODEL_ID,
            dim = EMBEDDING_DIM,
            "embedding engine loaded"
        );

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Load an embedding engine from a local directory containing
    /// `config.json`, `tokenizer.json`, and `model.safetensors`.
    pub fn from_dir(dir: &PathBuf) -> Result<Self> {
        let device = Device::Cpu;

        let config_path = dir.join("config.json");
        let tokenizer_path = dir.join("tokenizer.json");
        let weights_path = dir.join("model.safetensors");

        if !config_path.exists() {
            bail!("config.json not found in {}", dir.display());
        }
        if !tokenizer_path.exists() {
            bail!("tokenizer.json not found in {}", dir.display());
        }
        if !weights_path.exists() {
            bail!("model.safetensors not found in {}", dir.display());
        }

        let config_str = std::fs::read_to_string(&config_path)
            .context("failed to read config.json")?;
        let config: Config = serde_json::from_str(&config_str)
            .context("failed to parse config.json")?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow::anyhow!("failed to load tokenizer: {err}"))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .context("failed to load model weights")?
        };
        let model =
            BertModel::load(vb, &config).context("failed to build BertModel from weights")?;

        tracing::info!(
            dir = %dir.display(),
            dim = EMBEDDING_DIM,
            "embedding engine loaded from local directory"
        );

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Compute a 384-dimensional embedding for the given text.
    ///
    /// Pipeline: tokenize → BertModel forward → mean-pool (attention-mask
    /// aware) → L2-normalize.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            bail!("text must not be empty");
        }

        // --- Tokenize -------------------------------------------------
        let encoding = self
            .tokenizer
            .encode(trimmed, true)
            .map_err(|err| anyhow::anyhow!("tokenization failed: {err}"))?;

        let token_ids = encoding.get_ids().to_vec();
        let attention_mask_raw = encoding.get_attention_mask().to_vec();

        let token_ids_tensor = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids_tensor.zeros_like()?;
        let attention_mask_tensor =
            Tensor::new(&attention_mask_raw[..], &self.device)?.unsqueeze(0)?;

        // --- Forward ---------------------------------------------------
        let embeddings = self
            .model
            .forward(&token_ids_tensor, &token_type_ids, Some(&attention_mask_tensor))
            .context("BERT forward pass failed")?;

        // --- Mean-pool (attention-mask aware) --------------------------
        let mask_f32 = attention_mask_tensor
            .to_dtype(DType::F32)?
            .unsqueeze(2)?;
        let sum_mask = mask_f32.sum(1)?;
        let pooled = embeddings
            .broadcast_mul(&mask_f32)?
            .sum(1)?
            .broadcast_div(&sum_mask)?;

        // --- L2-normalize ----------------------------------------------
        let normalized = normalize_l2(&pooled)?;

        // --- Extract flat Vec<f32> -------------------------------------
        let vector: Vec<f32> = normalized.squeeze(0)?.to_vec1()?;
        if vector.len() != EMBEDDING_DIM {
            bail!(
                "embedding dimension mismatch: expected {EMBEDDING_DIM}, got {}",
                vector.len()
            );
        }

        Ok(vector)
    }

    /// Return the embedding dimension (always 384 for all-MiniLM-L6-v2).
    pub fn dim(&self) -> usize {
        EMBEDDING_DIM
    }
}

fn normalize_l2(tensor: &Tensor) -> Result<Tensor> {
    let l2 = tensor.sqr()?.sum_keepdim(1)?.sqrt()?;
    Ok(tensor.broadcast_div(&l2)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_l2_produces_unit_vector() {
        let device = Device::Cpu;
        let t = Tensor::new(&[[3.0_f32, 4.0]], &device).unwrap();
        let n = normalize_l2(&t).unwrap();
        let values: Vec<f32> = n.squeeze(0).unwrap().to_vec1().unwrap();
        let norm: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
