use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::KernelContext;
use crate::models::{
    ErrorResponse, HealthResponse, InitStateRequest, MctsInferRequest, MctsInferResponse,
    MemoryAddRequest, MemoryAddResponse, MemoryAddTextRequest, MemoryEmbedRequest,
    MemoryEmbedResponse, MemorySearchItem, MemorySearchRequest, MemorySearchResponse,
    ModelInferRequest, ModelInferResponse, ModelLoadRequest, ModelLoadResponse,
    ModelStatusResponse, SnapshotRequest, SnapshotResponse, UpdateStateRequest, VerifierRequest,
    VerifierResponse, WorldSimulateRequest, WorldSimulateResponse,
};

pub fn build_router(ctx: Arc<KernelContext>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/internal/state/init", post(init_state))
        .route("/internal/state/update", post(update_state))
        .route("/internal/state/{session_id}", get(get_state))
        .route("/internal/state/snapshot", post(snapshot_state))
        .route("/internal/world/simulate", post(simulate_world))
        .route("/internal/verifier/check", post(verify_patch))
        .route("/internal/memory/add", post(memory_add))
        .route("/internal/memory/add_text", post(memory_add_text))
        .route("/internal/memory/embed", post(memory_embed))
        .route("/internal/memory/search", post(memory_search))
        .route("/internal/model/load", post(load_model))
        .route("/internal/model/status", get(model_status))
        .route("/internal/infer", post(model_infer))
        .route("/internal/infer/mcts", post(model_infer_mcts))
        .with_state(ctx)
}

async fn healthz(State(ctx): State<Arc<KernelContext>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        hidden_size: ctx.state_manager.hidden_size(),
    })
}

async fn init_state(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<InitStateRequest>,
) -> Json<crate::models::HiddenState> {
    Json(ctx.state_manager.init_session(request.session_id))
}

async fn update_state(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<UpdateStateRequest>,
) -> Result<Json<crate::models::HiddenState>, ApiError> {
    let state = ctx
        .state_manager
        .update_state(&request.session_id, &request.input)
        .map_err(ApiError::bad_request)?;
    Ok(Json(state))
}

async fn get_state(
    State(ctx): State<Arc<KernelContext>>,
    Path(session_id): Path<String>,
) -> Result<Json<crate::models::HiddenState>, ApiError> {
    let Some(state) = ctx.state_manager.get_state(&session_id) else {
        return Err(ApiError::not_found(format!(
            "session_id `{session_id}` does not exist"
        )));
    };
    Ok(Json(state))
}

async fn snapshot_state(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<SnapshotRequest>,
) -> Result<Json<SnapshotResponse>, ApiError> {
    let Some(state) = ctx.state_manager.get_state(&request.session_id) else {
        return Err(ApiError::not_found(format!(
            "session_id `{}` does not exist",
            request.session_id
        )));
    };
    let epoch = request.epoch.unwrap_or(state.step);
    let key = ctx
        .snapshot_store
        .save_state(&state, epoch)
        .map_err(ApiError::internal)?;
    Ok(Json(SnapshotResponse {
        key,
        step: state.step,
        checksum: state.checksum,
    }))
}

async fn simulate_world(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<WorldSimulateRequest>,
) -> Json<WorldSimulateResponse> {
    let _sid = request.session_id;
    Json(ctx.world_model.simulate(&request.action))
}

async fn verify_patch(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<VerifierRequest>,
) -> Json<VerifierResponse> {
    Json(ctx.verifier.check(&request))
}

async fn memory_add(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<MemoryAddRequest>,
) -> Result<Json<MemoryAddResponse>, ApiError> {
    let store = Arc::clone(&ctx.memory_store);
    let id = tokio::task::spawn_blocking(move || store.add(request.text, request.vector))
        .await
        .map_err(|err| ApiError::internal(format!("memory add task failed: {err}")))?
        .map_err(ApiError::bad_request)?;
    Ok(Json(MemoryAddResponse { id }))
}

async fn memory_add_text(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<MemoryAddTextRequest>,
) -> Result<Json<MemoryAddResponse>, ApiError> {
    let text = request.text.trim().to_string();
    if text.is_empty() {
        return Err(ApiError::bad_request("text must not be empty"));
    }
    let engine = Arc::clone(&ctx.embedding_engine);
    let store = Arc::clone(&ctx.memory_store);
    let embed_text = text.clone();
    let (vector, id) = tokio::task::spawn_blocking(move || -> anyhow::Result<(Vec<f32>, uuid::Uuid)> {
        let vector = engine.embed(&embed_text)?;
        let id = store.add(text, vector.clone())?;
        Ok((vector, id))
    })
    .await
    .map_err(|err| ApiError::internal(format!("task failed: {err}")))?
    .map_err(ApiError::bad_request)?;
    let _ = vector;
    Ok(Json(MemoryAddResponse { id }))
}

async fn memory_embed(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<MemoryEmbedRequest>,
) -> Result<Json<MemoryEmbedResponse>, ApiError> {
    let text = request.text.trim().to_string();
    if text.is_empty() {
        return Err(ApiError::bad_request("text must not be empty"));
    }
    let engine = Arc::clone(&ctx.embedding_engine);
    let vector = tokio::task::spawn_blocking(move || engine.embed(&text))
        .await
        .map_err(|err| ApiError::internal(format!("task failed: {err}")))?
        .map_err(ApiError::bad_request)?;
    let dim = vector.len();
    Ok(Json(MemoryEmbedResponse { vector, dim }))
}

async fn memory_search(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<MemorySearchRequest>,
) -> Result<Json<MemorySearchResponse>, ApiError> {
    let store = Arc::clone(&ctx.memory_store);
    let query_vector = request.vector;
    let top_k = request.top_k;
    let rows = tokio::task::spawn_blocking(move || store.search(&query_vector, top_k))
        .await
        .map_err(|err| ApiError::internal(format!("memory search task failed: {err}")))?
        .map_err(ApiError::bad_request)?;

    let results = rows
        .into_iter()
        .map(|(text, score)| MemorySearchItem { text, score })
        .collect();
    Ok(Json(MemorySearchResponse { results }))
}

async fn load_model(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<ModelLoadRequest>,
) -> Result<Json<ModelLoadResponse>, ApiError> {
    let response = ctx
        .model_runtime
        .load_from_dir(&request.model_dir)
        .map_err(ApiError::bad_request)?;
    Ok(Json(response))
}

async fn model_status(State(ctx): State<Arc<KernelContext>>) -> Json<ModelStatusResponse> {
    Json(ctx.model_runtime.status())
}

async fn model_infer(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<ModelInferRequest>,
) -> Result<Json<ModelInferResponse>, ApiError> {
    let response = ctx
        .model_runtime
        .infer(&request.prompt, request.max_new_tokens.unwrap_or(64))
        .map_err(ApiError::bad_request)?;
    Ok(Json(response))
}

async fn model_infer_mcts(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<MctsInferRequest>,
) -> Result<Json<MctsInferResponse>, ApiError> {
    let runtime = Arc::clone(&ctx.model_runtime);
    let mcts = Arc::clone(&ctx.mcts_engine);
    let verifier = Arc::clone(&ctx.verifier);
    let world = Arc::clone(&ctx.world_model);
    let prompt = request.prompt;
    let max_tokens = request.max_new_tokens.unwrap_or(64);

    let response = tokio::task::spawn_blocking(move || {
        let mut engine = (*mcts).clone();
        if let Some(branches) = request.num_branches {
            engine.config.num_branches = branches.clamp(2, 8);
        }
        if let Some(c) = request.exploration_c {
            engine.config.exploration_c = c.clamp(0.1, 4.0);
        }
        engine.config.max_depth = max_tokens.clamp(1, 256);

        runtime.infer_mcts(&prompt, &engine, &verifier, &world)
    })
    .await
    .map_err(|err| ApiError::internal(format!("mcts task failed: {err}")))?
    .map_err(ApiError::bad_request)?;

    Ok(Json(response))
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request<E: ToString>(err: E) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: err.to_string(),
        }
    }

    fn not_found(message: String) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message,
        }
    }

    fn internal<E: ToString>(err: E) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: err.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let payload = Json(ErrorResponse {
            error: self.message,
        });
        (self.status, payload).into_response()
    }
}
