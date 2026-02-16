use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::model::gpt_kan::{LKanGPT, LKanGPTConfig};
use crate::model::lkan::LiquidKanConfig;
use crate::models::{MctsInferResponse, ModelInferResponse, ModelLoadResponse, ModelStatusResponse};

pub const VOCAB_CHARS: &str = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

const DEFAULT_MODEL_PATH: &str = "models/lkan-genesis.safetensors";
const DEFAULT_MODEL_FILENAME: &str = "lkan-genesis.safetensors";
const DEFAULT_MODEL_ID: &str = "lkan-genesis";
const MODEL_ARCH: &str = "lkan-gpt";
const MODEL_VERSION: &str = "v1";
const MAX_CONTEXT_LEN: usize = 64;
const MAX_NEW_TOKENS: usize = 256;

#[derive(Debug, Clone)]
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

impl ModelManifest {
    fn lkan_defaults(vocab_size: usize) -> Self {
        Self {
            model_id: DEFAULT_MODEL_ID.to_string(),
            arch: MODEL_ARCH.to_string(),
            version: MODEL_VERSION.to_string(),
            vocab_size,
            embed_dim: 128,
            hidden_dim: 128,
            num_layers: 8,
            bos_id: vocab_size,
            eos_id: vocab_size + 1,
            pad_id: vocab_size + 2,
            unk_id: vocab_size + 3,
            max_seq_len: MAX_CONTEXT_LEN,
            model_file: DEFAULT_MODEL_PATH.to_string(),
            vocab_file: String::new(),
            model_sha256: String::new(),
            vocab_sha256: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AvailableModel {
    pub model_id: String,
    pub checkpoint_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    id_to_char: Vec<char>,
    char_to_id: HashMap<char, u32>,
    fallback_id: u32,
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        let id_to_char: Vec<char> = VOCAB_CHARS.chars().collect();
        let char_to_id: HashMap<char, u32> = id_to_char
            .iter()
            .enumerate()
            .map(|(idx, ch)| (*ch, idx as u32))
            .collect();
        let fallback_id = *char_to_id.get(&' ').unwrap_or(&0);
        Self {
            id_to_char,
            char_to_id,
            fallback_id,
        }
    }
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }

    pub fn fallback_id(&self) -> u32 {
        self.fallback_id
    }

    pub fn encode_ids(&self, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = text
            .chars()
            .map(|ch| {
                self.char_to_id
                    .get(&ch)
                    .copied()
                    .unwrap_or(self.fallback_id)
            })
            .collect();
        if ids.is_empty() {
            ids.push(self.fallback_id);
        }
        ids
    }

    pub fn encode(&self, text: &str, device: &Device) -> Result<Tensor> {
        let ids = self.encode_ids(text);
        Tensor::from_slice(&ids, (1, ids.len()), device)
            .context("failed to build token id tensor in tokenizer")
    }

    pub fn decode(&self, token_id: u32) -> String {
        self.id_to_char
            .get(token_id as usize)
            .copied()
            .unwrap_or(' ')
            .to_string()
    }

    pub fn decode_ids(&self, token_ids: &[usize]) -> String {
        let mut out = String::with_capacity(token_ids.len());
        for &id in token_ids {
            out.push(self.id_to_char.get(id).copied().unwrap_or(' '));
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub manifest: ModelManifest,
    tokenizer: SimpleTokenizer,
    model: Arc<Mutex<Option<LKanGPT>>>,
    device: Device,
}

impl LoadedModel {
    fn from_runtime(runtime: &ModelRuntime) -> Self {
        let mut manifest = ModelManifest::lkan_defaults(runtime.tokenizer.vocab_size());
        if let Some(model_id) = runtime.current_model_id() {
            manifest.model_id = model_id;
        }
        if let Some(checkpoint_path) = runtime.current_checkpoint_path() {
            manifest.model_file = checkpoint_path.to_string_lossy().to_string();
        }

        Self {
            manifest,
            tokenizer: runtime.tokenizer.clone(),
            model: Arc::clone(&runtime.model),
            device: runtime.device.clone(),
        }
    }

    pub fn encode_prompt(&self, prompt: &str) -> Vec<usize> {
        self.tokenizer
            .encode_ids(prompt)
            .into_iter()
            .map(|id| id as usize)
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        self.tokenizer.decode_ids(ids)
    }

    pub fn forward_one_token(&self, token_id: usize, _hidden: &mut [Vec<f32>]) -> Result<Vec<f32>> {
        let guard = self.model.lock().expect("model runtime lock poisoned");
        let Some(model) = guard.as_ref() else {
            bail!("model is not loaded");
        };

        let bounded_token = if token_id < self.manifest.vocab_size {
            token_id as u32
        } else {
            self.tokenizer.fallback_id()
        };

        let input_ids = Tensor::from_slice(&[bounded_token], (1, 1), &self.device)
            .context("failed to create one-token input tensor")?;
        let logits = model
            .forward_logits(&input_ids)
            .context("LKanGPT forward failed in forward_one_token")?;
        let (_bsz, seq_len, vocab_size) = logits
            .dims3()
            .context("expected logits shape [1, seq_len, vocab_size]")?;
        if seq_len == 0 {
            bail!("forward_one_token received empty sequence logits");
        }
        logits
            .narrow(1, seq_len - 1, 1)?
            .reshape((vocab_size,))?
            .to_vec1::<f32>()
            .context("failed to convert logits to vector")
    }
}

pub struct ModelRuntime {
    pub model: Arc<Mutex<Option<LKanGPT>>>,
    pub device: Device,
    pub tokenizer: SimpleTokenizer,
    loaded_model_id: Arc<Mutex<Option<String>>>,
    loaded_checkpoint_path: Arc<Mutex<Option<PathBuf>>>,
}

impl Default for ModelRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRuntime {
    pub fn new() -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            device: Device::Cpu,
            tokenizer: SimpleTokenizer::new(),
            loaded_model_id: Arc::new(Mutex::new(None)),
            loaded_checkpoint_path: Arc::new(Mutex::new(None)),
        }
    }

    fn build_config(&self) -> Result<LKanGPTConfig> {
        let vocab_size = self.tokenizer.vocab_size();
        if vocab_size != 65 {
            bail!(
                "unexpected tokenizer vocab size: expected 65, got {}",
                vocab_size
            );
        }

        Ok(LKanGPTConfig {
            vocab_size,
            hidden_dim: 128,
            num_layers: 8,
            num_heads: 4,
            kan_config: LiquidKanConfig {
                in_dim: 128,
                hidden_dim: 128,
                out_dim: 128,
                cheb_order: 3,
                dt: 0.1,
                tau_min: 1e-2,
                x_scale: 1.0,
            },
        })
    }

    fn current_model_id(&self) -> Option<String> {
        self.loaded_model_id
            .lock()
            .expect("model runtime id lock poisoned")
            .clone()
    }

    fn current_checkpoint_path(&self) -> Option<PathBuf> {
        self.loaded_checkpoint_path
            .lock()
            .expect("model runtime path lock poisoned")
            .clone()
    }

    fn model_roots(&self) -> Vec<PathBuf> {
        let mut roots = Vec::new();
        let mut seen = HashSet::new();
        let mut push_unique = |path: PathBuf| {
            let key = path.to_string_lossy().to_string();
            if seen.insert(key) {
                roots.push(path);
            }
        };

        if let Ok(model_dir) = std::env::var("VAGI_MODEL_DIR") {
            let trimmed = model_dir.trim();
            if !trimmed.is_empty() {
                push_unique(PathBuf::from(trimmed));
            }
        }

        push_unique(PathBuf::from("models"));
        push_unique(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("models"));
        roots
    }

    fn discover_checkpoints(&self) -> Vec<AvailableModel> {
        let mut by_id: BTreeMap<String, PathBuf> = BTreeMap::new();
        for root in self.model_roots() {
            if !root.is_dir() {
                continue;
            }
            let Ok(entries) = fs::read_dir(&root) else {
                continue;
            };

            let mut candidates = Vec::new();
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
                    continue;
                };
                if !ext.eq_ignore_ascii_case("safetensors") {
                    continue;
                }
                if path.is_file() {
                    candidates.push(path);
                }
            }
            candidates.sort();

            for path in candidates {
                let model_id = model_id_from_path(&path);
                by_id.entry(model_id).or_insert(path);
            }
        }

        by_id
            .into_iter()
            .map(|(model_id, checkpoint_path)| AvailableModel {
                model_id,
                checkpoint_path,
            })
            .collect()
    }

    pub fn list_available_models(&self) -> Vec<AvailableModel> {
        let mut by_id: BTreeMap<String, PathBuf> = BTreeMap::new();
        for model in self.discover_checkpoints() {
            by_id.entry(model.model_id).or_insert(model.checkpoint_path);
        }
        if let (Some(model_id), Some(checkpoint_path)) =
            (self.current_model_id(), self.current_checkpoint_path())
        {
            by_id.entry(model_id).or_insert(checkpoint_path);
        }

        by_id
            .into_iter()
            .map(|(model_id, checkpoint_path)| AvailableModel {
                model_id,
                checkpoint_path,
            })
            .collect()
    }

    fn default_checkpoint_path(&self) -> Option<PathBuf> {
        let models = self.list_available_models();
        if let Some(preferred) = models.iter().find(|m| m.model_id == DEFAULT_MODEL_ID) {
            return Some(preferred.checkpoint_path.clone());
        }
        models.first().map(|m| m.checkpoint_path.clone())
    }

    fn find_checkpoint_by_model_id(&self, model_id: &str) -> Option<PathBuf> {
        let needle = model_id.to_ascii_lowercase();
        self.list_available_models()
            .into_iter()
            .find(|m| m.model_id.to_ascii_lowercase() == needle)
            .map(|m| m.checkpoint_path)
    }

    fn available_model_ids_display(&self) -> String {
        let ids: Vec<String> = self
            .list_available_models()
            .into_iter()
            .map(|m| m.model_id)
            .collect();
        if ids.is_empty() {
            "(none found in models directory)".to_string()
        } else {
            ids.join(", ")
        }
    }

    fn pick_checkpoint_in_dir(dir: &Path) -> Option<PathBuf> {
        let Ok(entries) = fs::read_dir(dir) else {
            return None;
        };
        let mut candidates = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
                continue;
            };
            if ext.eq_ignore_ascii_case("safetensors") && path.is_file() {
                candidates.push(path);
            }
        }
        candidates.sort();
        if let Some(preferred) = candidates.iter().find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.eq_ignore_ascii_case(DEFAULT_MODEL_FILENAME))
                .unwrap_or(false)
        }) {
            return Some(preferred.clone());
        }
        candidates.into_iter().next()
    }

    fn resolve_model_ref(&self, model_ref: &str) -> Result<PathBuf> {
        let trimmed = model_ref.trim();
        if trimmed.is_empty() {
            if let Some(path) = self.default_checkpoint_path() {
                return Ok(path);
            }
            bail!(
                "no checkpoint found. Put *.safetensors into one of: {}",
                self.model_roots()
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            );
        }

        let candidate = PathBuf::from(trimmed);
        if candidate.is_file() {
            return Ok(candidate);
        }
        if candidate.is_dir() {
            if let Some(path) = Self::pick_checkpoint_in_dir(&candidate) {
                return Ok(path);
            }
            bail!(
                "directory {} does not contain any .safetensors checkpoint",
                candidate.display()
            );
        }

        if let Some(path) = self.find_checkpoint_by_model_id(trimmed) {
            return Ok(path);
        }

        for root in self.model_roots() {
            let direct = root.join(trimmed);
            if direct.is_file() {
                return Ok(direct);
            }
            let with_ext = root.join(format!("{trimmed}.safetensors"));
            if with_ext.is_file() {
                return Ok(with_ext);
            }
        }

        bail!(
            "unknown model `{trimmed}`. Available models: {}",
            self.available_model_ids_display()
        )
    }

    fn load_from_checkpoint_path(&self, checkpoint_path: &Path) -> Result<ModelLoadResponse> {
        if !checkpoint_path.exists() {
            bail!(
                "missing checkpoint at {}. Train first using `cargo run -p vagi-kernel --bin train_lkan --release`.",
                checkpoint_path.display()
            );
        }

        let cfg = self.build_config()?;
        let weights = vec![checkpoint_path.to_path_buf()];
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weights, DType::F32, &self.device)
                .with_context(|| {
                    format!(
                        "failed to load safetensors checkpoint from {}",
                        checkpoint_path.display()
                    )
                })?
        };
        let model =
            LKanGPT::new(vb.pp("lkan_gpt"), cfg).context("failed to instantiate LKanGPT runtime")?;
        let model_id = model_id_from_path(checkpoint_path);

        {
            let mut guard = self.model.lock().expect("model runtime lock poisoned");
            *guard = Some(model);
        }
        {
            let mut model_id_guard = self
                .loaded_model_id
                .lock()
                .expect("model runtime id lock poisoned");
            *model_id_guard = Some(model_id.clone());
        }
        {
            let mut checkpoint_guard = self
                .loaded_checkpoint_path
                .lock()
                .expect("model runtime path lock poisoned");
            *checkpoint_guard = Some(checkpoint_path.to_path_buf());
        }

        Ok(ModelLoadResponse {
            model_id,
            loaded: true,
            checksum_ok: true,
            arch: MODEL_ARCH.to_string(),
        })
    }

    fn ensure_loaded_from_checkpoint(&self, checkpoint_path: &Path) -> Result<String> {
        let requested =
            fs::canonicalize(checkpoint_path).unwrap_or_else(|_| checkpoint_path.to_path_buf());
        let loaded = self.model.lock().expect("model runtime lock poisoned").is_some();
        if loaded {
            if let Some(current_path) = self.current_checkpoint_path() {
                let current = fs::canonicalize(&current_path).unwrap_or(current_path);
                if current == requested {
                    return Ok(self
                        .current_model_id()
                        .unwrap_or_else(|| model_id_from_path(checkpoint_path)));
                }
            }
        }
        let response = self.load_from_checkpoint_path(checkpoint_path)?;
        Ok(response.model_id)
    }

    pub fn ensure_loaded(&self, model_ref: Option<&str>) -> Result<String> {
        if let Some(model_ref) = model_ref {
            let checkpoint_path = self.resolve_model_ref(model_ref)?;
            return self.ensure_loaded_from_checkpoint(&checkpoint_path);
        }

        if self.model.lock().expect("model runtime lock poisoned").is_some() {
            return Ok(self
                .current_model_id()
                .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string()));
        }

        let Some(default_path) = self.default_checkpoint_path() else {
            bail!(
                "model is not loaded and no checkpoint found in models directory. \
                 Put *.safetensors into one of: {}",
                self.model_roots()
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            );
        };
        self.ensure_loaded_from_checkpoint(&default_path)
    }

    pub fn auto_load_from_models_dir(&self) -> Result<Option<ModelLoadResponse>> {
        if self.model.lock().expect("model runtime lock poisoned").is_some() {
            return Ok(None);
        }
        let Some(default_path) = self.default_checkpoint_path() else {
            return Ok(None);
        };
        Ok(Some(self.load_from_checkpoint_path(&default_path)?))
    }

    pub fn load(&self) -> Result<ModelLoadResponse> {
        let Some(default_path) = self.default_checkpoint_path() else {
            bail!(
                "missing checkpoint at {}. Train first using `cargo run -p vagi-kernel --bin train_lkan --release`.",
                DEFAULT_MODEL_PATH
            );
        };
        self.load_from_checkpoint_path(&default_path)
    }

    pub fn load_from_dir(&self, model_dir: &str) -> Result<ModelLoadResponse> {
        let checkpoint_path = self.resolve_model_ref(model_dir)?;
        self.load_from_checkpoint_path(&checkpoint_path)
    }

    pub fn status(&self) -> ModelStatusResponse {
        let loaded = self.model.lock().expect("model runtime lock poisoned").is_some();
        if loaded {
            ModelStatusResponse {
                loaded: true,
                model_id: self.current_model_id(),
                arch: Some(MODEL_ARCH.to_string()),
                vocab_size: Some(self.tokenizer.vocab_size()),
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

    fn next_token_id_with_model(&self, model: &LKanGPT, prompt: &str) -> Result<u32> {
        let clipped_prompt = last_chars(prompt, MAX_CONTEXT_LEN);
        let input_ids = self.tokenizer.encode(&clipped_prompt, &self.device)?;
        let logits = model
            .forward_logits(&input_ids)
            .context("forward_logits failed during inference")?;
        let (_bsz, seq_len, vocab_size) = logits
            .dims3()
            .context("expected logits shape [batch, seq_len, vocab_size]")?;
        if seq_len == 0 {
            bail!("received empty sequence logits while inferring");
        }

        let last_logits = logits
            .narrow(1, seq_len - 1, 1)?
            .reshape((vocab_size,))?
            .to_vec1::<f32>()?;
        Ok(argmax(&last_logits) as u32)
    }

    pub fn infer_next_char(&self, prompt: &str) -> Result<String> {
        let guard = self.model.lock().expect("model runtime lock poisoned");
        let Some(model) = guard.as_ref() else {
            bail!("model is not loaded");
        };
        let next_id = self.next_token_id_with_model(model, prompt)?;
        Ok(self.tokenizer.decode(next_id))
    }

    pub fn infer(&self, prompt: &str, max_new_tokens: usize) -> Result<ModelInferResponse> {
        let started = Instant::now();
        let max_tokens = max_new_tokens.clamp(1, MAX_NEW_TOKENS);

        let guard = self.model.lock().expect("model runtime lock poisoned");
        let Some(model) = guard.as_ref() else {
            bail!("model is not loaded");
        };

        let mut running_prompt = prompt.to_string();
        let mut generated_ids = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let next_id = self.next_token_id_with_model(model, &running_prompt)?;
            generated_ids.push(next_id as usize);
            running_prompt.push_str(&self.tokenizer.decode(next_id));
        }

        Ok(ModelInferResponse {
            model_id: self
                .current_model_id()
                .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string()),
            text: self.tokenizer.decode_ids(&generated_ids),
            tokens_generated: generated_ids.len(),
            latency_ms: started.elapsed().as_millis() as u64,
            think_trace: None,
        })
    }

    pub fn infer_mcts(
        &self,
        prompt: &str,
        mcts_engine: &crate::mcts::MctsEngine,
        _verifier: &crate::verifier::Verifier,
        _world_model: &crate::world_model::WorldModel,
    ) -> Result<MctsInferResponse> {
        // v1 compatibility mode: fallback to deterministic greedy decode.
        let max_tokens = mcts_engine.config.max_depth.clamp(1, MAX_NEW_TOKENS);
        let output = self.infer(prompt, max_tokens)?;
        Ok(MctsInferResponse {
            model_id: output.model_id,
            text: output.text,
            tokens_generated: output.tokens_generated,
            branches_explored: 1,
            best_branch_reward: 0.0,
            latency_ms: output.latency_ms,
        })
    }

    pub fn as_loaded_model(&self) -> LoadedModel {
        LoadedModel::from_runtime(self)
    }
}

fn model_id_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(|stem| stem.to_string())
        .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string())
}

fn last_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars()
        .rev()
        .take(max_chars)
        .collect::<Vec<char>>()
        .into_iter()
        .rev()
        .collect()
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

pub fn softmax_top_k(logits: &[f32], k: usize, temperature: f32) -> Vec<(usize, f32)> {
    let temp = temperature.max(0.01);
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &logit)| (idx, ((logit - max_logit) / temp).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, value)| *value).sum::<f32>().max(1e-12);
    for (_, value) in &mut probs {
        *value /= sum;
    }
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    probs.truncate(k);
    probs
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
    use std::path::Path;

    use super::{SimpleTokenizer, argmax_with_exclusions, last_chars, model_id_from_path};

    #[test]
    fn tokenizer_matches_expected_vocab_size() {
        let tok = SimpleTokenizer::new();
        assert_eq!(tok.vocab_size(), 65);
    }

    #[test]
    fn last_chars_clips_unicode_safely() {
        let clipped = last_chars("abcdef", 3);
        assert_eq!(clipped, "def");
    }

    #[test]
    fn argmax_exclusions_fallback_when_all_blocked() {
        let values = [0.1_f32, 0.2, 0.3];
        let idx = argmax_with_exclusions(&values, &[0, 1, 2]);
        assert_eq!(idx, 2);
    }

    #[test]
    fn model_id_derived_from_checkpoint_stem() {
        let path = Path::new("models/lkan-gen2.safetensors");
        assert_eq!(model_id_from_path(path), "lkan-gen2");
    }
}
