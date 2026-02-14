use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenState {
    pub session_id: String,
    pub vector: Vec<f32>,
    pub step: u64,
    pub checksum: String,
    pub updated_at: String,
}

impl HiddenState {
    pub fn new(session_id: String, hidden_size: usize) -> Self {
        let vector = vec![0.0; hidden_size];
        Self {
            session_id,
            checksum: checksum_for_vector(&vector),
            vector,
            step: 0,
            updated_at: Utc::now().to_rfc3339(),
        }
    }

    pub fn refresh_metadata(&mut self) {
        self.updated_at = Utc::now().to_rfc3339();
        self.checksum = checksum_for_vector(&self.vector);
    }
}

pub fn checksum_for_vector(vector: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for v in vector {
        hasher.update(v.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

#[derive(Debug, Deserialize)]
pub struct InitStateRequest {
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateStateRequest {
    pub session_id: String,
    pub input: String,
}

#[derive(Debug, Deserialize)]
pub struct SnapshotRequest {
    pub session_id: String,
    pub epoch: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct SnapshotResponse {
    pub key: String,
    pub step: u64,
    pub checksum: String,
}

#[derive(Debug, Deserialize)]
pub struct WorldSimulateRequest {
    pub session_id: Option<String>,
    pub action: String,
}

#[derive(Debug, Serialize)]
pub struct WorldSimulateResponse {
    pub risk_score: f32,
    pub confidence: f32,
    pub predicted_effects: Vec<String>,
    pub causal_path: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct VerifierRequest {
    pub patch_ir: String,
    pub max_loop_iters: Option<u32>,
    pub side_effect_budget: Option<u32>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct VerifierResponse {
    pub pass: bool,
    pub violations: Vec<String>,
    pub cost: u32,
    pub timeout_hit: bool,
    pub wasi_ok: bool,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub hidden_size: usize,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelLoadRequest {
    pub model_dir: String,
}

#[derive(Debug, Serialize)]
pub struct ModelLoadResponse {
    pub model_id: String,
    pub loaded: bool,
    pub checksum_ok: bool,
    pub arch: String,
}

#[derive(Debug, Serialize)]
pub struct ModelStatusResponse {
    pub loaded: bool,
    pub model_id: Option<String>,
    pub arch: Option<String>,
    pub vocab_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ModelInferRequest {
    pub prompt: String,
    pub max_new_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ModelInferResponse {
    pub model_id: String,
    pub text: String,
    pub tokens_generated: usize,
    pub latency_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct MctsInferRequest {
    pub prompt: String,
    pub max_new_tokens: Option<usize>,
    pub num_branches: Option<usize>,
    pub exploration_c: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct MctsInferResponse {
    pub model_id: String,
    pub text: String,
    pub tokens_generated: usize,
    pub branches_explored: usize,
    pub best_branch_reward: f32,
    pub latency_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct MemoryAddRequest {
    pub text: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct MemoryAddResponse {
    pub id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct MemorySearchRequest {
    pub vector: Vec<f32>,
    pub top_k: usize,
}

#[derive(Debug, Serialize)]
pub struct MemorySearchItem {
    pub text: String,
    pub score: f32,
}

#[derive(Debug, Serialize)]
pub struct MemorySearchResponse {
    pub results: Vec<MemorySearchItem>,
}

#[derive(Debug, Deserialize)]
pub struct MemoryAddTextRequest {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct MemoryEmbedRequest {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct MemoryEmbedResponse {
    pub vector: Vec<f32>,
    pub dim: usize,
}
