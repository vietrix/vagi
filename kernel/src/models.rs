use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

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

#[derive(Debug, Deserialize)]
pub struct JitExecuteRequest {
    pub logic: String,
    pub input: i64,
}

#[derive(Debug, Serialize)]
pub struct JitExecuteResponse {
    pub output: i64,
    pub backend: &'static str,
    pub op_count: usize,
    pub compile_micros: u64,
    pub execute_micros: u64,
    pub normalized_logic: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct HdcTemplateUpsertRequest {
    pub template_id: String,
    pub logic_template: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct HdcTemplateUpsertResponse {
    pub template_id: String,
    pub token_count: usize,
    pub dimension_bits: usize,
    pub stored_at: String,
}

#[derive(Debug, Deserialize)]
pub struct HdcTemplateQueryRequest {
    pub query: String,
    pub top_k: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct HdcTemplateMatch {
    pub template_id: String,
    pub similarity: f32,
    pub tags: Vec<String>,
    pub logic_template: String,
}

#[derive(Debug, Serialize)]
pub struct HdcTemplateQueryResponse {
    pub hits: Vec<HdcTemplateMatch>,
    pub dimension_bits: usize,
}

#[derive(Debug, Deserialize)]
pub struct HdcTemplateBindRequest {
    pub template_id: String,
    #[serde(default)]
    pub bindings: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct HdcTemplateBindResponse {
    pub template_id: String,
    pub bound_logic: String,
    pub placeholders_resolved: usize,
    pub normalized_logic: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct HdcWeaveExecuteRequest {
    pub query: String,
    pub input: i64,
    pub top_k: Option<usize>,
    #[serde(default)]
    pub bindings: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct HdcWeaveExecuteResponse {
    pub template_id: String,
    pub similarity: f32,
    pub bound_logic: String,
    pub output: i64,
    pub backend: &'static str,
    pub compile_micros: u64,
    pub execute_micros: u64,
}

#[derive(Debug, Deserialize)]
pub struct HdcWeavePlanRequest {
    pub query: String,
    pub input: i64,
    pub top_k: Option<usize>,
    pub risk_threshold: Option<f32>,
    pub verifier_required: Option<bool>,
    pub session_id: Option<String>,
    #[serde(default)]
    pub bindings: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct HdcWeavePlanCandidate {
    pub template_id: String,
    pub similarity: f32,
    pub bound_logic: String,
    pub output: i64,
    pub verifier_pass: bool,
    pub verifier_violations: Vec<String>,
    pub risk_score: f32,
    pub confidence: f32,
    pub compile_micros: u64,
    pub execute_micros: u64,
    pub accepted: bool,
    pub rejection_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct HdcWeavePlanResponse {
    pub selected_template_id: Option<String>,
    pub selected_index: Option<usize>,
    pub candidates: Vec<HdcWeavePlanCandidate>,
    pub backend: &'static str,
}

#[derive(Debug, Deserialize)]
pub struct HdcEvolutionMutateRequest {
    pub template_id: Option<String>,
    pub query: Option<String>,
    pub generations: Option<usize>,
    pub population_size: Option<usize>,
    pub survivors: Option<usize>,
    pub risk_threshold: Option<f32>,
    pub seed_input: Option<i64>,
    pub promote: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HdcEvolutionCandidate {
    pub candidate_id: String,
    pub generation: usize,
    pub score: f32,
    pub output: i64,
    pub verifier_pass: bool,
    pub verifier_violations: Vec<String>,
    pub risk_score: f32,
    pub confidence: f32,
    pub compile_micros: u64,
    pub execute_micros: u64,
    pub logic: String,
}

#[derive(Debug, Serialize)]
pub struct HdcEvolutionGenerationReport {
    pub generation: usize,
    pub best_candidate_id: String,
    pub best_score: f32,
    pub best_risk_score: f32,
    pub best_verifier_pass: bool,
}

#[derive(Debug, Serialize)]
pub struct HdcEvolutionMutateResponse {
    pub base_template_id: String,
    pub promoted_template_id: Option<String>,
    pub generations_run: usize,
    pub population_size: usize,
    pub survivors: usize,
    pub total_candidates_evaluated: usize,
    pub history: Vec<HdcEvolutionGenerationReport>,
    pub final_candidates: Vec<HdcEvolutionCandidate>,
    pub backend: &'static str,
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
