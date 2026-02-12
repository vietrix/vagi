use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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

