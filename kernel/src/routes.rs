use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::KernelContext;
use crate::models::{
    ErrorResponse, HealthResponse, InitStateRequest, JitExecuteRequest, JitExecuteResponse,
    HdcTemplateBindRequest, HdcTemplateBindResponse, HdcTemplateQueryRequest,
    HdcTemplateQueryResponse, HdcTemplateUpsertRequest, HdcTemplateUpsertResponse,
    HdcEvolutionMutateRequest, HdcEvolutionMutateResponse,
    HdcWeaveExecuteRequest, HdcWeaveExecuteResponse, HdcWeavePlanCandidate,
    HdcWeavePlanRequest, HdcWeavePlanResponse, SnapshotRequest, SnapshotResponse,
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
        .route("/internal/jit/execute", post(execute_jit))
        .route("/internal/hdc/templates/upsert", post(hdc_upsert_template))
        .route("/internal/hdc/templates/query", post(hdc_query_templates))
        .route("/internal/hdc/templates/bind", post(hdc_bind_template))
        .route("/internal/hdc/evolution/mutate", post(hdc_evolution_mutate))
        .route("/internal/hdc/weave/execute", post(hdc_weave_execute))
        .route("/internal/hdc/weave/plan", post(hdc_weave_plan))
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

async fn execute_jit(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<JitExecuteRequest>,
) -> Result<Json<JitExecuteResponse>, ApiError> {
    let response = ctx
        .jit_engine
        .compile_and_execute(&request)
        .map_err(ApiError::bad_request)?;
    Ok(Json(response))
}

async fn hdc_upsert_template(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<HdcTemplateUpsertRequest>,
) -> Result<Json<HdcTemplateUpsertResponse>, ApiError> {
    let response = ctx
        .hdc_memory
        .upsert_template(&request)
        .map_err(ApiError::bad_request)?;
    Ok(Json(response))
}

async fn hdc_query_templates(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<HdcTemplateQueryRequest>,
) -> Result<Json<HdcTemplateQueryResponse>, ApiError> {
    let response = ctx
        .hdc_memory
        .query_templates(&request)
        .map_err(ApiError::bad_request)?;
    Ok(Json(response))
}

async fn hdc_bind_template(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<HdcTemplateBindRequest>,
) -> Result<Json<HdcTemplateBindResponse>, ApiError> {
    let response = ctx
        .hdc_memory
        .bind_template(&request)
        .map_err(ApiError::bad_request)?;
    Ok(Json(response))
}

async fn hdc_evolution_mutate(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<HdcEvolutionMutateRequest>,
) -> Result<Json<HdcEvolutionMutateResponse>, ApiError> {
    let response = ctx
        .mutation_engine
        .evolve_templates(&request)
        .map_err(ApiError::bad_request)?;
    Ok(Json(response))
}

async fn hdc_weave_execute(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<HdcWeaveExecuteRequest>,
) -> Result<Json<HdcWeaveExecuteResponse>, ApiError> {
    let query_response = ctx
        .hdc_memory
        .query_templates(&HdcTemplateQueryRequest {
            query: request.query.clone(),
            top_k: Some(request.top_k.unwrap_or(1).clamp(1, 5)),
        })
        .map_err(ApiError::bad_request)?;

    let best = query_response
        .hits
        .first()
        .ok_or_else(|| ApiError::bad_request("no template found for query"))?;

    let bind_response = ctx
        .hdc_memory
        .bind_template(&HdcTemplateBindRequest {
            template_id: best.template_id.clone(),
            bindings: request.bindings.clone(),
        })
        .map_err(ApiError::bad_request)?;

    let jit_response = ctx
        .jit_engine
        .compile_and_execute(&JitExecuteRequest {
            logic: bind_response.bound_logic.clone(),
            input: request.input,
        })
        .map_err(ApiError::bad_request)?;

    Ok(Json(HdcWeaveExecuteResponse {
        template_id: best.template_id.clone(),
        similarity: best.similarity,
        bound_logic: bind_response.bound_logic,
        output: jit_response.output,
        backend: jit_response.backend,
        compile_micros: jit_response.compile_micros,
        execute_micros: jit_response.execute_micros,
    }))
}

async fn hdc_weave_plan(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<HdcWeavePlanRequest>,
) -> Result<Json<HdcWeavePlanResponse>, ApiError> {
    let query_response = ctx
        .hdc_memory
        .query_templates(&HdcTemplateQueryRequest {
            query: request.query.clone(),
            top_k: Some(request.top_k.unwrap_or(3).clamp(1, 10)),
        })
        .map_err(ApiError::bad_request)?;

    let verifier_required = request.verifier_required.unwrap_or(true);
    let risk_threshold = request.risk_threshold.unwrap_or(0.65).clamp(0.01, 0.99);

    let mut candidates = Vec::new();
    for hit in &query_response.hits {
        let mut candidate = HdcWeavePlanCandidate {
            template_id: hit.template_id.clone(),
            similarity: hit.similarity,
            bound_logic: String::new(),
            output: 0,
            verifier_pass: false,
            verifier_violations: Vec::new(),
            risk_score: 1.0,
            confidence: 0.0,
            compile_micros: 0,
            execute_micros: 0,
            accepted: false,
            rejection_reason: None,
        };

        match ctx.hdc_memory.bind_template(&HdcTemplateBindRequest {
            template_id: hit.template_id.clone(),
            bindings: request.bindings.clone(),
        }) {
            Ok(bind_response) => {
                candidate.bound_logic = bind_response.bound_logic.clone();

                match ctx.jit_engine.compile_and_execute(&JitExecuteRequest {
                    logic: bind_response.bound_logic.clone(),
                    input: request.input,
                }) {
                    Ok(jit_response) => {
                        candidate.output = jit_response.output;
                        candidate.compile_micros = jit_response.compile_micros;
                        candidate.execute_micros = jit_response.execute_micros;

                        let sim = ctx.world_model.simulate(&bind_response.bound_logic);
                        candidate.risk_score = sim.risk_score;
                        candidate.confidence = sim.confidence;

                        let verify = ctx.verifier.check(&VerifierRequest {
                            patch_ir: bind_response.bound_logic,
                            max_loop_iters: Some(2_048),
                            side_effect_budget: Some(3),
                            timeout_ms: Some(80),
                        });
                        candidate.verifier_pass = verify.pass;
                        candidate.verifier_violations = verify.violations;

                        let pass_gate =
                            (!verifier_required || candidate.verifier_pass)
                                && candidate.risk_score <= risk_threshold;
                        candidate.accepted = pass_gate;
                        if !pass_gate {
                            candidate.rejection_reason = Some(format!(
                                "verifier_pass={} risk_score={:.2} threshold={:.2}",
                                candidate.verifier_pass, candidate.risk_score, risk_threshold
                            ));
                        }
                    }
                    Err(err) => {
                        candidate.rejection_reason =
                            Some(format!("jit_compile_or_execute_failed:{err}"));
                    }
                }
            }
            Err(err) => {
                candidate.rejection_reason = Some(format!("template_bind_failed:{err}"));
            }
        }

        candidates.push(candidate);
    }

    candidates.sort_by(|a, b| {
        b.accepted
            .cmp(&a.accepted)
            .then_with(|| a.risk_score.total_cmp(&b.risk_score))
            .then_with(|| b.similarity.total_cmp(&a.similarity))
            .then_with(|| a.execute_micros.cmp(&b.execute_micros))
    });

    let selected_index = candidates.iter().position(|candidate| candidate.accepted);
    let selected_template_id = selected_index.map(|idx| candidates[idx].template_id.clone());

    Ok(Json(HdcWeavePlanResponse {
        selected_template_id,
        selected_index,
        candidates,
        backend: "wasmtime-cranelift-jit",
    }))
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
