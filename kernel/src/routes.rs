use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::KernelContext;
use crate::models::{
    ErrorResponse, HealthResponse, InitStateRequest, SnapshotRequest, SnapshotResponse,
    UpdateStateRequest, VerifierRequest, VerifierResponse, WorldSimulateRequest,
    WorldSimulateResponse,
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

