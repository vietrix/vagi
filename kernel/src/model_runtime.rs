use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::RwLock;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::models::{ModelInferResponse, ModelLoadResponse, ModelStatusResponse};

#[derive(Debug, Deserialize, Clone)]
pub struct ModelManifest {
    pub model_id: String,
    pub arch: String,
    pub version: String,
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    pub pad_id: usize,
    pub unk_id: usize,
    pub max_seq_len: usize,
    pub model_file: String,
    pub vocab_file: String,
    pub model_sha256: String,
    pub vocab_sha256: String,
}

#[derive(Debug, Deserialize)]
struct VocabFile {
    tokens: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GruLayerWeights {
    pub weight_ih: Vec<f32>,
    pub weight_hh: Vec<f32>,
    pub bias_ih: Vec<f32>,
    pub bias_hh: Vec<f32>,
    pub input_size: usize,
}

#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub manifest: ModelManifest,
    pub tokens: Vec<String>,
    pub token_to_id: HashMap<String, usize>,
    pub embedding: Vec<f32>,
    pub gru_layers: Vec<GruLayerWeights>,
    pub lm_head_weight: Vec<f32>,
    pub lm_head_bias: Vec<f32>,
}

/// Dispatches between different model architectures.
#[derive(Debug, Clone)]
pub enum LoadedArch {
    GruLm(LoadedModel),
    BitNet(crate::bitnet::BitNetModel),
}

impl LoadedArch {
    pub fn model_id(&self) -> &str {
        match self {
            Self::GruLm(m) => &m.manifest.model_id,
            Self::BitNet(m) => &m.config.model_id,
        }
    }

    pub fn arch_name(&self) -> &str {
        match self {
            Self::GruLm(_) => "tiny-gru-lm",
            Self::BitNet(_) => "bitnet-1.58b",
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Self::GruLm(m) => m.manifest.vocab_size,
            Self::BitNet(m) => m.config.vocab_size,
        }
    }
}

#[derive(Default)]
pub struct ModelRuntime {
    loaded: RwLock<Option<LoadedArch>>,
}

impl ModelRuntime {
    pub fn new() -> Self {
        Self {
            loaded: RwLock::new(None),
        }
    }

    pub fn load_from_dir(&self, model_dir: &str) -> Result<ModelLoadResponse> {
        let dir = Path::new(model_dir);
        let manifest_path = dir.join("manifest.json");
        let manifest: ModelManifest = serde_json::from_slice(
            &fs::read(&manifest_path).with_context(|| {
                format!("failed to read manifest at {}", manifest_path.display())
            })?,
        )?;

        if manifest.arch == "bitnet-1.58b" {
            return self.load_bitnet(dir, &manifest);
        }

        if manifest.arch != "tiny-gru-lm" {
            bail!("unsupported model arch `{}`", manifest.arch);
        }
        if manifest.vocab_size == 0 || manifest.embed_dim == 0 || manifest.hidden_dim == 0 {
            bail!("invalid manifest dimensions");
        }
        if manifest.num_layers == 0 {
            bail!("manifest.num_layers must be > 0");
        }
        let _ = &manifest.version;
        let _ = manifest.max_seq_len;

        let model_path = dir.join(&manifest.model_file);
        let vocab_path = dir.join(&manifest.vocab_file);
        let model_checksum = sha256_file(&model_path)?;
        let vocab_checksum = sha256_file(&vocab_path)?;
        if model_checksum != manifest.model_sha256 {
            bail!("model checksum mismatch");
        }
        if vocab_checksum != manifest.vocab_sha256 {
            bail!("vocab checksum mismatch");
        }

        let vocab: VocabFile = serde_json::from_slice(
            &fs::read(&vocab_path)
                .with_context(|| format!("failed to read vocab at {}", vocab_path.display()))?,
        )?;
        if vocab.tokens.len() != manifest.vocab_size {
            bail!(
                "vocab size mismatch: manifest={} file={}",
                manifest.vocab_size,
                vocab.tokens.len()
            );
        }

        let safetensor_bytes = fs::read(&model_path)
            .with_context(|| format!("failed to read model at {}", model_path.display()))?;
        let tensors = SafeTensors::deserialize(&safetensor_bytes)?;

        let embedding = load_matrix(
            &tensors,
            "embedding.weight",
            manifest.vocab_size,
            manifest.embed_dim,
        )?;

        let mut gru_layers = Vec::with_capacity(manifest.num_layers);
        for layer_idx in 0..manifest.num_layers {
            let input_size = if layer_idx == 0 {
                manifest.embed_dim
            } else {
                manifest.hidden_dim
            };
            let rows = manifest.hidden_dim * 3;
            let weight_ih = load_matrix(
                &tensors,
                &format!("gru.weight_ih_l{layer_idx}"),
                rows,
                input_size,
            )?;
            let weight_hh = load_matrix(
                &tensors,
                &format!("gru.weight_hh_l{layer_idx}"),
                rows,
                manifest.hidden_dim,
            )?;
            let bias_ih = load_vector(&tensors, &format!("gru.bias_ih_l{layer_idx}"), rows)?;
            let bias_hh = load_vector(&tensors, &format!("gru.bias_hh_l{layer_idx}"), rows)?;
            gru_layers.push(GruLayerWeights {
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                input_size,
            });
        }

        let lm_head_weight = load_matrix(
            &tensors,
            "lm_head.weight",
            manifest.vocab_size,
            manifest.hidden_dim,
        )?;
        let lm_head_bias = load_vector(&tensors, "lm_head.bias", manifest.vocab_size)?;

        let token_to_id = vocab
            .tokens
            .iter()
            .enumerate()
            .map(|(idx, token)| (token.clone(), idx))
            .collect();

        let loaded = LoadedModel {
            manifest: manifest.clone(),
            tokens: vocab.tokens,
            token_to_id,
            embedding,
            gru_layers,
            lm_head_weight,
            lm_head_bias,
        };

        let mut guard = self.loaded.write().expect("model runtime lock poisoned");
        *guard = Some(LoadedArch::GruLm(loaded));
        Ok(ModelLoadResponse {
            model_id: manifest.model_id,
            loaded: true,
            checksum_ok: true,
            arch: manifest.arch,
        })
    }

    /// Load a BitNet 1.58-bit ternary model.
    fn load_bitnet(&self, dir: &Path, manifest: &ModelManifest) -> Result<ModelLoadResponse> {
        let bitnet_model = crate::bitnet::BitNetModel::load(dir)?;
        let mut guard = self.loaded.write().expect("model runtime lock poisoned");
        *guard = Some(LoadedArch::BitNet(bitnet_model));
        Ok(ModelLoadResponse {
            model_id: manifest.model_id.clone(),
            loaded: true,
            checksum_ok: true,
            arch: manifest.arch.clone(),
        })
    }

    pub fn status(&self) -> ModelStatusResponse {
        let guard = self.loaded.read().expect("model runtime lock poisoned");
        if let Some(arch) = guard.as_ref() {
            ModelStatusResponse {
                loaded: true,
                model_id: Some(arch.model_id().to_string()),
                arch: Some(arch.arch_name().to_string()),
                vocab_size: Some(arch.vocab_size()),
            }
        } else {
            ModelStatusResponse {
                loaded: false,
                model_id: None,
                arch: None,
                vocab_size: None,
            }
        }
    }

    pub fn infer(&self, prompt: &str, max_new_tokens: usize) -> Result<ModelInferResponse> {
        let guard = self.loaded.read().expect("model runtime lock poisoned");
        let Some(arch) = guard.as_ref() else {
            bail!("model is not loaded");
        };
        match arch {
            LoadedArch::GruLm(model) => self.infer_gru(model, prompt, max_new_tokens),
            LoadedArch::BitNet(model) => self.infer_bitnet(model, prompt, max_new_tokens),
        }
    }

    fn infer_gru(&self, model: &LoadedModel, prompt: &str, max_new_tokens: usize) -> Result<ModelInferResponse> {
        let started = Instant::now();
        let max_tokens = max_new_tokens.clamp(1, 256);

        let mut hidden = vec![vec![0.0_f32; model.manifest.hidden_dim]; model.manifest.num_layers];
        let input_ids = model.encode_prompt(prompt);
        let mut logits = vec![0.0_f32; model.manifest.vocab_size];
        for token_id in input_ids {
            logits = model.forward_one_token(token_id, &mut hidden)?;
        }

        let mut generated_ids: Vec<usize> = Vec::new();
        let excluded = [
            model.manifest.bos_id,
            model.manifest.pad_id,
            model.manifest.eos_id,
        ];
        let mut next_id = argmax_with_exclusions(&logits, &excluded);
        for _ in 0..max_tokens {
            if next_id == model.manifest.eos_id {
                break;
            }
            generated_ids.push(next_id);
            logits = model.forward_one_token(next_id, &mut hidden)?;
            next_id = argmax_with_exclusions(&logits, &excluded);
        }

        let text = model.decode(&generated_ids);
        Ok(ModelInferResponse {
            model_id: model.manifest.model_id.clone(),
            text,
            tokens_generated: generated_ids.len(),
            latency_ms: started.elapsed().as_millis() as u64,
        })
    }

    fn infer_bitnet(
        &self,
        model: &crate::bitnet::BitNetModel,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<ModelInferResponse> {
        let started = Instant::now();
        let max_tokens = max_new_tokens.clamp(1, 256);

        let input_ids = model.encode_prompt(prompt);
        let mut logits = vec![0.0_f32; model.config.vocab_size];
        for &token_id in &input_ids {
            logits = model.forward_token(token_id)?;
        }

        let mut generated_ids: Vec<usize> = Vec::new();
        let excluded = [
            model.config.bos_id,
            model.config.pad_id,
            model.config.eos_id,
        ];
        let mut next_id = argmax_with_exclusions(&logits, &excluded);
        for _ in 0..max_tokens {
            if next_id == model.config.eos_id {
                break;
            }
            generated_ids.push(next_id);
            logits = model.forward_token(next_id)?;
            next_id = argmax_with_exclusions(&logits, &excluded);
        }

        let text = model.decode(&generated_ids);
        Ok(ModelInferResponse {
            model_id: model.config.model_id.clone(),
            text,
            tokens_generated: generated_ids.len(),
            latency_ms: started.elapsed().as_millis() as u64,
        })
    }

    pub fn infer_mcts(
        &self,
        prompt: &str,
        mcts_engine: &crate::mcts::MctsEngine,
        verifier: &crate::verifier::Verifier,
        world_model: &crate::world_model::WorldModel,
    ) -> Result<crate::models::MctsInferResponse> {
        let guard = self.loaded.read().expect("model runtime lock poisoned");
        let Some(arch) = guard.as_ref() else {
            bail!("model is not loaded");
        };

        // MCTS currently only supports GRU-LM (requires hidden state cloning).
        let model = match arch {
            LoadedArch::GruLm(m) => m,
            LoadedArch::BitNet(_) => bail!("MCTS is not yet supported for BitNet models"),
        };

        // Encode prompt through the model to get initial hidden state + logits.
        let mut hidden = vec![vec![0.0_f32; model.manifest.hidden_dim]; model.manifest.num_layers];
        let input_ids = model.encode_prompt(prompt);
        let mut logits = vec![0.0_f32; model.manifest.vocab_size];
        for token_id in input_ids {
            logits = model.forward_one_token(token_id, &mut hidden)?;
        }

        // Run MCTS search.
        let result = mcts_engine.search(model, &hidden, &logits, verifier, world_model)?;

        Ok(crate::models::MctsInferResponse {
            model_id: model.manifest.model_id.clone(),
            text: result.text,
            tokens_generated: result.token_ids.len(),
            branches_explored: result.branches_explored,
            best_branch_reward: result.best_reward,
            latency_ms: result.latency_ms,
        })
    }
}

impl LoadedModel {
    pub fn encode_prompt(&self, prompt: &str) -> Vec<usize> {
        let mut ids = Vec::with_capacity(prompt.chars().count() + 1);
        ids.push(self.manifest.bos_id);
        for ch in prompt.chars() {
            let token = ch.to_string();
            let id = self
                .token_to_id
                .get(&token)
                .copied()
                .unwrap_or(self.manifest.unk_id);
            ids.push(id);
        }
        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let special_ids: HashSet<usize> = [
            self.manifest.pad_id,
            self.manifest.bos_id,
            self.manifest.eos_id,
            self.manifest.unk_id,
        ]
        .into_iter()
        .collect();
        let mut out = String::new();
        for id in ids {
            if special_ids.contains(id) {
                continue;
            }
            if let Some(token) = self.tokens.get(*id) {
                out.push_str(token);
            }
        }
        out
    }

    pub fn forward_one_token(&self, token_id: usize, hidden: &mut [Vec<f32>]) -> Result<Vec<f32>> {
        if token_id >= self.manifest.vocab_size {
            bail!("token id out of range");
        }
        if hidden.len() != self.manifest.num_layers {
            bail!("hidden state layer mismatch");
        }
        let mut x = self.embedding_row(token_id);
        for (layer_idx, layer) in self.gru_layers.iter().enumerate() {
            let h_prev = hidden
                .get(layer_idx)
                .context("missing hidden layer while inferring")?;
            let h_next = gru_step(layer, &x, h_prev, self.manifest.hidden_dim);
            hidden[layer_idx] = h_next.clone();
            x = h_next;
        }
        Ok(self.lm_head(&x))
    }

    pub fn embedding_row(&self, token_id: usize) -> Vec<f32> {
        let start = token_id * self.manifest.embed_dim;
        let end = start + self.manifest.embed_dim;
        self.embedding[start..end].to_vec()
    }

    pub fn lm_head(&self, x: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0_f32; self.manifest.vocab_size];
        for (row, out) in logits.iter_mut().enumerate() {
            let base = row * self.manifest.hidden_dim;
            let mut sum = self.lm_head_bias[row];
            for (col, value) in x.iter().enumerate().take(self.manifest.hidden_dim) {
                sum += self.lm_head_weight[base + col] * *value;
            }
            *out = sum;
        }
        logits
    }
}

pub fn gru_step(layer: &GruLayerWeights, x: &[f32], h_prev: &[f32], hidden_dim: usize) -> Vec<f32> {
    let mut h_next = vec![0.0_f32; hidden_dim];
    for i in 0..hidden_dim {
        let r = sigmoid(
            linear_row(&layer.weight_ih, i, layer.input_size, x)
                + layer.bias_ih[i]
                + linear_row(&layer.weight_hh, i, hidden_dim, h_prev)
                + layer.bias_hh[i],
        );
        let z_idx = i + hidden_dim;
        let z = sigmoid(
            linear_row(&layer.weight_ih, z_idx, layer.input_size, x)
                + layer.bias_ih[z_idx]
                + linear_row(&layer.weight_hh, z_idx, hidden_dim, h_prev)
                + layer.bias_hh[z_idx],
        );
        let n_idx = i + hidden_dim * 2;
        let n_pre = linear_row(&layer.weight_ih, n_idx, layer.input_size, x) + layer.bias_ih[n_idx];
        let n_recur =
            linear_row(&layer.weight_hh, n_idx, hidden_dim, h_prev) + layer.bias_hh[n_idx];
        let n = (n_pre + r * n_recur).tanh();
        h_next[i] = (1.0 - z) * n + z * h_prev[i];
    }
    h_next
}

fn linear_row(weight: &[f32], row: usize, cols: usize, x: &[f32]) -> f32 {
    let offset = row * cols;
    let mut sum = 0.0_f32;
    for col in 0..cols {
        sum += weight[offset + col] * x[col];
    }
    sum
}

#[inline]
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn load_matrix(
    tensors: &SafeTensors<'_>,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor `{name}`"))?;
    if tensor.dtype() != Dtype::F32 {
        bail!("tensor `{name}` must be f32");
    }
    let shape = tensor.shape();
    if shape != [expected_rows, expected_cols] {
        bail!(
            "tensor `{name}` shape mismatch: expected [{expected_rows}, {expected_cols}] got {:?}",
            shape
        );
    }
    bytes_to_f32(tensor.data())
}

fn load_vector(tensors: &SafeTensors<'_>, name: &str, expected_len: usize) -> Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor `{name}`"))?;
    if tensor.dtype() != Dtype::F32 {
        bail!("tensor `{name}` must be f32");
    }
    let shape = tensor.shape();
    if shape != [expected_len] {
        bail!(
            "tensor `{name}` shape mismatch: expected [{expected_len}] got {:?}",
            shape
        );
    }
    bytes_to_f32(tensor.data())
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        bail!("invalid f32 byte length {}", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    hasher.update(bytes);
    Ok(hex::encode(hasher.finalize()))
}

pub fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, value) in values.iter().enumerate() {
        if *value > best_val {
            best_idx = idx;
            best_val = *value;
        }
    }
    best_idx
}

/// Return top-k tokens with their probabilities after softmax with temperature.
/// Used by MCTS for stochastic branch expansion.
pub fn softmax_top_k(logits: &[f32], k: usize, temperature: f32) -> Vec<(usize, f32)> {
    let temp = temperature.max(0.01);
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &v)| (idx, ((v - max_logit) / temp).exp()))
        .collect();
    let sum: f32 = exps.iter().map(|(_, e)| e).sum();
    for (_, p) in exps.iter_mut() {
        *p /= sum;
    }
    exps.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    exps.truncate(k);
    exps
}

pub fn argmax_with_exclusions(values: &[f32], excluded: &[usize]) -> usize {
    let excluded: HashSet<usize> = excluded.iter().copied().collect();
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, value) in values.iter().enumerate() {
        if excluded.contains(&idx) {
            continue;
        }
        if *value > best_val {
            best_idx = idx;
            best_val = *value;
        }
    }
    if best_val == f32::NEG_INFINITY {
        argmax(values)
    } else {
        best_idx
    }
}

#[cfg(test)]
mod tests {
    use super::{GruLayerWeights, gru_step};

    #[test]
    fn gru_step_keeps_hidden_size() {
        let hidden_dim = 4usize;
        let input_size = 3usize;
        let layer = GruLayerWeights {
            weight_ih: vec![0.01; hidden_dim * 3 * input_size],
            weight_hh: vec![0.02; hidden_dim * 3 * hidden_dim],
            bias_ih: vec![0.0; hidden_dim * 3],
            bias_hh: vec![0.0; hidden_dim * 3],
            input_size,
        };
        let x = vec![0.3, -0.2, 0.1];
        let h_prev = vec![0.0; hidden_dim];
        let h_next = gru_step(&layer, &x, &h_prev, hidden_dim);
        assert_eq!(h_next.len(), hidden_dim);
    }
}
