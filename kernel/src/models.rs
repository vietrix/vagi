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
    pub logic_penalty: f32,
}

#[derive(Debug, Deserialize)]
pub struct VerifierActRequest {
    pub patch_ir: String,
    pub max_steps: Option<u32>,
    pub max_output_bytes: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct VerifierActResponse {
    pub pass: bool,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub state_changes: Vec<String>,
    pub steps_executed: u32,
    pub runtime: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub hidden_size: usize,
}

#[derive(Debug, Deserialize)]
pub struct MicroOodaRequest {
    pub session_id: Option<String>,
    pub input: String,
    pub risk_threshold: Option<f32>,
    pub max_decide_iters: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct MicroOodaResponse {
    pub handled: bool,
    pub reason: String,
    pub draft: String,
    pub risk_score: f32,
    pub confidence: f32,
    pub verifier_pass: bool,
    pub violations: Vec<String>,
    pub iterations: u32,
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
    /// Enable System 2 "deep thought" reasoning mode.
    pub use_reasoning: Option<bool>,
    /// Number of thinking iterations (default: 50).
    pub thinking_iterations: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ModelInferResponse {
    pub model_id: String,
    pub text: String,
    pub tokens_generated: usize,
    pub latency_ms: u64,
    /// Present when `use_reasoning = true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think_trace: Option<ThinkTrace>,
}

/// Trace metadata from System 2 reasoning.
#[derive(Debug, Clone, Serialize)]
pub struct ThinkTrace {
    pub nodes_explored: usize,
    pub best_score: f32,
    pub branches_pruned: usize,
    pub think_latency_ms: u64,
    pub verifier_pass: bool,
    pub iterations_completed: usize,
    pub best_depth: u32,
    /// Chronological log of the thinking process (e.g. "Selected node 5 -> Expanded 3 children").
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub process_log: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", content = "content")]
pub enum ThinkingEvent {
    #[serde(rename = "thought")]
    Thought(String),
    #[serde(rename = "token")]
    Token(String),
    #[serde(rename = "summary")]
    Summary(ThinkTrace),
    #[serde(rename = "done")]
    Done,
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

// ─── Homeostasis ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct HomeostasisStatusResponse {
    pub dopamine: f32,
    pub cortisol: f32,
    pub oxytocin: f32,
    pub energy: f32,
    pub mood: String,
    pub mood_description: String,
}

// ─── Knowledge Graph ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct KnowledgeAddRequest {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub subject_type: Option<String>,
    pub object_type: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct KnowledgeAddResponse {
    pub nodes: usize,
    pub edges: usize,
}

#[derive(Debug, Deserialize)]
pub struct KnowledgeQueryRequest {
    pub node: String,
    pub direction: Option<String>, // "causes", "effects", "path"
    pub target: Option<String>,    // for path queries
    pub max_depth: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct KnowledgeQueryResponse {
    pub query_node: String,
    pub chains: Vec<Vec<CausalStepDto>>,
    pub pagerank_top: Vec<(String, f32)>,
}

#[derive(Debug, Serialize)]
pub struct CausalStepDto {
    pub from: String,
    pub relation: String,
    pub to: String,
    pub strength: f32,
}

// ─── Affect ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct AffectDetectRequest {
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AffectDetectResponse {
    pub valence: f32,
    pub arousal: f32,
    pub dominance: f32,
    pub trust: f32,
    pub label: String,
}

#[derive(Debug, Serialize)]
pub struct AffectModulateResponse {
    pub target_valence: f32,
    pub target_arousal: f32,
    pub target_dominance: f32,
    pub target_trust: f32,
    pub warmth: f32,
    pub formality: f32,
    pub encouragement: f32,
    pub humor: f32,
    pub suggested_framing: String,
}

// ─── MoE Gate ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct MoeSelectRequest {
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct MoeSelectionItem {
    pub expert_id: String,
    pub score: f32,
}

#[derive(Debug, Serialize)]
pub struct MoeSelectResponse {
    pub selected: Vec<MoeSelectionItem>,
    pub loaded_count: usize,
}
