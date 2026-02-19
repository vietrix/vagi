use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{BoxError, Json, Router};
use serde::Deserialize;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::KernelContext;
use crate::homeostasis::HormoneEvent;
use crate::knowledge_graph::NodeType;
use crate::models::{
    AffectDetectRequest, AffectDetectResponse, AffectModulateResponse, CausalStepDto,
    ErrorResponse, HealthResponse, HomeostasisStatusResponse, InitStateRequest,
    KnowledgeAddRequest, KnowledgeAddResponse, KnowledgeQueryRequest, KnowledgeQueryResponse,
    MctsInferRequest, MctsInferResponse, MemoryAddRequest, MemoryAddResponse,
    MemoryAddTextRequest, MemoryEmbedRequest, MemoryEmbedResponse, MemorySearchItem,
    MemorySearchRequest, MemorySearchResponse, MicroOodaRequest, MicroOodaResponse,
    MoeSelectRequest, MoeSelectResponse, MoeSelectionItem, ModelInferRequest, ModelInferResponse, ModelLoadRequest, ModelLoadResponse,
    ModelStatusResponse, SnapshotRequest, SnapshotResponse, UpdateStateRequest,
    VerifierActRequest, VerifierActResponse, VerifierRequest, VerifierResponse,
    WorldSimulateRequest, WorldSimulateResponse,
};

pub fn build_router(ctx: Arc<KernelContext>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/models", get(openai_models))
        .route("/v1/models", get(openai_models))
        .route("/chat/completions", post(openai_chat_completions))
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/internal/state/init", post(init_state))
        .route("/internal/state/update", post(update_state))
        .route("/internal/state/{session_id}", get(get_state))
        .route("/internal/state/snapshot", post(snapshot_state))
        .route("/internal/world/simulate", post(simulate_world))
        .route("/internal/verifier/check", post(verify_patch))
        .route("/internal/verifier/act", post(verify_act))
        .route("/internal/ooda/micro_run", post(micro_ooda_run))
        .route("/internal/memory/add", post(memory_add))
        .route("/internal/memory/add_text", post(memory_add_text))
        .route("/internal/memory/embed", post(memory_embed))
        .route("/internal/memory/search", post(memory_search))
        .route("/internal/model/load", post(load_model))
        .route("/internal/model/status", get(model_status))
        .route("/internal/infer", post(model_infer))
        .route("/internal/infer/stream", post(model_infer_stream))
        .route("/internal/infer/mcts", post(model_infer_mcts))
        // ── Cognitive Architecture ──
        .route("/internal/homeostasis/status", get(homeostasis_status))
        .route("/internal/homeostasis/event", post(homeostasis_event))
        .route("/internal/knowledge/add", post(knowledge_add))
        .route("/internal/knowledge/query", post(knowledge_query))
        .route("/internal/affect/detect", post(affect_detect))
        .route("/internal/affect/modulate", post(affect_modulate))
        .route("/internal/moe/select", post(moe_select))
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

async fn verify_act(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<VerifierActRequest>,
) -> Json<VerifierActResponse> {
    Json(ctx.verifier.execute_act(&request))
}

fn is_micro_ooda_candidate(input: &str) -> bool {
    let lower = input.to_lowercase();
    let blocked_keywords = [
        "production",
        "deploy",
        "drop table",
        "rm -rf",
        "credential",
        "secret",
        "sudo",
        "privilege",
        "root",
        "migration",
    ];
    if blocked_keywords.iter().any(|k| lower.contains(k)) {
        return false;
    }
    let word_count = lower.split_whitespace().count();
    word_count <= 80 && input.len() <= 1_200
}

async fn micro_ooda_run(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<MicroOodaRequest>,
) -> Json<MicroOodaResponse> {
    if !is_micro_ooda_candidate(&request.input) {
        return Json(MicroOodaResponse {
            handled: false,
            reason: "high_risk_or_complex_request".to_string(),
            draft: String::new(),
            risk_score: 1.0,
            confidence: 0.0,
            verifier_pass: false,
            violations: vec!["micro_ooda_not_applicable".to_string()],
            iterations: 0,
        });
    }

    let risk_threshold = request.risk_threshold.unwrap_or(0.45).clamp(0.05, 0.95);
    let max_iters = request.max_decide_iters.unwrap_or(2).clamp(1, 4);
    let session_id = request
        .session_id
        .unwrap_or_else(|| "micro-ooda-session".to_string());

    let mut draft = format!(
        "echo micro_ooda_start\nset objective=fast_safe_response\necho input:{}",
        request.input.replace('\n', " ").trim()
    );
    let mut last_risk = 1.0_f32;
    let mut last_confidence = 0.0_f32;
    let mut last_violations: Vec<String> = Vec::new();
    let mut last_verifier_pass = false;

    for iter in 1..=max_iters {
        let sim = ctx.world_model.simulate(&draft);
        let verifier = ctx.verifier.check(&VerifierRequest {
            patch_ir: draft.clone(),
            max_loop_iters: Some(256),
            side_effect_budget: Some(2),
            timeout_ms: Some(40),
        });
        last_risk = sim.risk_score;
        last_confidence = sim.confidence;
        last_violations = verifier.violations.clone();
        last_verifier_pass = verifier.pass;

        if verifier.pass && sim.risk_score <= risk_threshold {
            return Json(MicroOodaResponse {
                handled: true,
                reason: "micro_ooda_success".to_string(),
                draft,
                risk_score: sim.risk_score,
                confidence: sim.confidence,
                verifier_pass: true,
                violations: verifier.violations,
                iterations: iter,
            });
        }

        draft = format!(
            "{draft}\nwarn refinement_iter={iter}\nset guardrail=strict_validation\nappend trace=session:{session_id}"
        );
    }

    Json(MicroOodaResponse {
        handled: false,
        reason: "micro_ooda_safety_gate_not_satisfied".to_string(),
        draft,
        risk_score: last_risk,
        confidence: last_confidence,
        verifier_pass: last_verifier_pass,
        violations: last_violations,
        iterations: max_iters,
    })
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
    let (vector, id) =
        tokio::task::spawn_blocking(move || -> anyhow::Result<(Vec<f32>, uuid::Uuid)> {
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
    let use_reasoning = request.use_reasoning.unwrap_or(false);

    if use_reasoning {
        // ── System 2: Deep Thought via TreeSearchAgent ────────────────
        let runtime = Arc::clone(&ctx.model_runtime);
        let verifier = Arc::clone(&ctx.verifier);
        let world = Arc::clone(&ctx.world_model);
        let prompt = request.prompt;
        let iterations = request.thinking_iterations.unwrap_or(50).clamp(5, 200);
        let max_tokens = request.max_new_tokens.unwrap_or(128).clamp(1, 256);

        let response = tokio::task::spawn_blocking(move || {
            use crate::reasoning::{ThinkConfig, TreeSearchAgent};

            let config = ThinkConfig {
                iterations,
                max_tokens_per_draft: max_tokens,
                ..ThinkConfig::default()
            };
            let mut agent = TreeSearchAgent::new(&prompt, config);
            let result = agent.think(&runtime, &verifier, &world)?;

            // Get model_id for response.
            let model_id = runtime
                .status()
                .model_id
                .unwrap_or_else(|| "unknown".into());

            Ok::<_, anyhow::Error>(ModelInferResponse {
                model_id,
                text: result.best_draft,
                tokens_generated: 0, // N/A for draft-level reasoning
                latency_ms: result.latency_ms,
                think_trace: Some(crate::models::ThinkTrace {
                    nodes_explored: result.nodes_explored,
                    best_score: result.best_score,
                    branches_pruned: result.branches_pruned,
                    think_latency_ms: result.latency_ms,
                    verifier_pass: result.verifier_pass,
                    iterations_completed: result.iterations_completed,
                    best_depth: result.best_depth,
                    process_log: result.process_log,
                }),
            })
        })
        .await
        .map_err(|err| ApiError::internal(format!("think task failed: {err}")))?
        .map_err(ApiError::bad_request)?;

        Ok(Json(response))
    } else {
        // ── System 1: Fast reflex (greedy decode) ────────────────────
        let response = ctx
            .model_runtime
            .infer(&request.prompt, request.max_new_tokens.unwrap_or(64))
            .map_err(ApiError::bad_request)?;
        Ok(Json(response))
    }
}

async fn openai_models(State(ctx): State<Arc<KernelContext>>) -> Json<serde_json::Value> {
    let data: Vec<serde_json::Value> = ctx
        .model_runtime
        .list_available_models()
        .into_iter()
        .map(|model| {
            serde_json::json!({
                "id": model.model_id,
                "object": "model",
                "created": unix_now(),
                "owned_by": "vagi-kernel"
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": data
    }))
}

async fn openai_chat_completions(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<OpenAiChatRequest>,
) -> Result<impl IntoResponse, ApiError> {
    if request.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }

    let model_id = ctx
        .model_runtime
        .ensure_loaded(request.model.as_deref())
        .map_err(ApiError::bad_request)?;
    let max_new_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(128)
        .clamp(1, 256);
    let prompt = openai_messages_to_prompt(&request.messages);
    let created = unix_now();
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4().simple());

    if request.stream.unwrap_or(false) {
        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let runtime = Arc::clone(&ctx.model_runtime);
        let model_id = model_id.clone();
        let completion_id = completion_id.clone();
        tokio::task::spawn_blocking(move || {
            let role_chunk = serde_json::json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": { "role": "assistant" },
                    "finish_reason": serde_json::Value::Null
                }]
            });
            let _ = tx.blocking_send(Ok(role_chunk.to_string()));

            let mut running_prompt = prompt;
            for _ in 0..max_new_tokens {
                match runtime.infer_next_char(&running_prompt) {
                    Ok(token) => {
                        running_prompt.push_str(&token);
                        let chunk = serde_json::json!({
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "delta": { "content": token },
                                "finish_reason": serde_json::Value::Null
                            }]
                        });
                        if tx.blocking_send(Ok(chunk.to_string())).is_err() {
                            return;
                        }
                    }
                    Err(err) => {
                        let _ = tx.blocking_send(Err(format!("stream error: {err}")));
                        return;
                    }
                }
            }

            let final_chunk = serde_json::json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            });
            let _ = tx.blocking_send(Ok(final_chunk.to_string()));
            let _ = tx.blocking_send(Ok("[DONE]".to_string()));
        });

        let stream = ReceiverStream::new(rx).map(|event| match event {
            Ok(data) => Ok::<_, BoxError>(Event::default().data(data)),
            Err(err) => Ok::<_, BoxError>(Event::default().event("error").data(err)),
        });
        return Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response());
    }

    let infer = ctx
        .model_runtime
        .infer(&prompt, max_new_tokens)
        .map_err(ApiError::bad_request)?;
    let completion_tokens = infer.tokens_generated;
    let prompt_tokens = prompt.chars().count().max(1);
    let text = infer.text;

    Ok(Json(serde_json::json!({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }))
    .into_response())
}

async fn model_infer_stream(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<ModelInferRequest>,
) -> impl IntoResponse {
    let (tx, rx) = tokio::sync::mpsc::channel(100);

    let runtime = Arc::clone(&ctx.model_runtime);
    let verifier = Arc::clone(&ctx.verifier);
    let world = Arc::clone(&ctx.world_model);
    let prompt = request.prompt;
    let iterations = request.thinking_iterations.unwrap_or(50).clamp(5, 200);
    let max_tokens = request.max_new_tokens.unwrap_or(128).clamp(1, 256);

    tokio::task::spawn_blocking(move || {
        use crate::reasoning::{ThinkConfig, TreeSearchAgent};

        let config = ThinkConfig {
            iterations,
            max_tokens_per_draft: max_tokens,
            ..ThinkConfig::default()
        };
        let mut agent = TreeSearchAgent::new(&prompt, config);
        // We ignore the actual result as events are streamed.
        // Log errors if necessary but don't panic.
        if let Err(e) = agent.think_stream(&runtime, &verifier, &world, tx) {
            eprintln!("Streaming think failed: {}", e);
        }
    });

    let stream = ReceiverStream::new(rx).map(|result| match result {
        Ok(event) => Ok::<_, BoxError>(Event::default().json_data(event).unwrap()),
        Err(e) => Ok::<_, BoxError>(Event::default().event("error").data(e.to_string())),
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
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

// ─── Homeostasis ─────────────────────────────────────────────────────────────

async fn homeostasis_status(
    State(ctx): State<Arc<KernelContext>>,
) -> Json<HomeostasisStatusResponse> {
    let snap = ctx.homeostasis.snapshot();
    Json(HomeostasisStatusResponse {
        dopamine: snap.dopamine,
        cortisol: snap.cortisol,
        oxytocin: snap.oxytocin,
        energy: snap.energy,
        mood: format!("{:?}", snap.mood()),
        mood_description: snap.mood_description().to_string(),
    })
}

async fn homeostasis_event(
    State(ctx): State<Arc<KernelContext>>,
    Json(event): Json<HormoneEvent>,
) -> Json<HomeostasisStatusResponse> {
    ctx.homeostasis.process_event(&event);
    let snap = ctx.homeostasis.snapshot();
    Json(HomeostasisStatusResponse {
        dopamine: snap.dopamine,
        cortisol: snap.cortisol,
        oxytocin: snap.oxytocin,
        energy: snap.energy,
        mood: format!("{:?}", snap.mood()),
        mood_description: snap.mood_description().to_string(),
    })
}

// ─── Knowledge Graph ─────────────────────────────────────────────────────────

fn parse_node_type(s: &Option<String>) -> NodeType {
    match s.as_deref() {
        Some("entity") => NodeType::Entity,
        Some("event") => NodeType::Event,
        Some("preference") => NodeType::Preference,
        Some("behavior") => NodeType::Behavior,
        _ => NodeType::Concept,
    }
}

async fn knowledge_add(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<KnowledgeAddRequest>,
) -> Json<KnowledgeAddResponse> {
    let subj_type = parse_node_type(&request.subject_type);
    let obj_type = parse_node_type(&request.object_type);
    ctx.knowledge_graph.add_fact(
        &request.subject,
        &request.relation,
        &request.object,
        subj_type,
        obj_type,
    );
    let (nodes, edges) = ctx.knowledge_graph.stats();
    Json(KnowledgeAddResponse { nodes, edges })
}

async fn knowledge_query(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<KnowledgeQueryRequest>,
) -> Json<KnowledgeQueryResponse> {
    let max_depth = request.max_depth.unwrap_or(3);
    let direction = request.direction.as_deref().unwrap_or("effects");

    let chains = match direction {
        "causes" => ctx.knowledge_graph.query_causes(&request.node, max_depth),
        "path" => {
            if let Some(ref target) = request.target {
                ctx.knowledge_graph
                    .find_path(&request.node, target)
                    .map(|p| vec![p])
                    .unwrap_or_default()
            } else {
                vec![]
            }
        }
        _ => ctx.knowledge_graph.query_effects(&request.node, max_depth),
    };

    let dto_chains: Vec<Vec<CausalStepDto>> = chains
        .into_iter()
        .map(|chain| {
            chain
                .into_iter()
                .map(|s| CausalStepDto {
                    from: s.from,
                    relation: s.relation,
                    to: s.to,
                    strength: s.strength,
                })
                .collect()
        })
        .collect();

    let pagerank_top = ctx
        .knowledge_graph
        .pagerank(20, 0.85)
        .into_iter()
        .take(10)
        .collect();

    Json(KnowledgeQueryResponse {
        query_node: request.node,
        chains: dto_chains,
        pagerank_top,
    })
}

// ─── Affect ──────────────────────────────────────────────────────────────────

async fn affect_detect(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<AffectDetectRequest>,
) -> Json<AffectDetectResponse> {
    let em = ctx.affect_engine.detect_emotion(&request.text);
    Json(AffectDetectResponse {
        valence: em.valence,
        arousal: em.arousal,
        dominance: em.dominance,
        trust: em.trust,
        label: em.label().to_string(),
    })
}

async fn affect_modulate(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<AffectDetectRequest>,
) -> Json<AffectModulateResponse> {
    let user_emotion = ctx.affect_engine.detect_emotion(&request.text);
    let (target, modulation) = ctx.affect_engine.plan_response_tone(&user_emotion);
    Json(AffectModulateResponse {
        target_valence: target.valence,
        target_arousal: target.arousal,
        target_dominance: target.dominance,
        target_trust: target.trust,
        warmth: modulation.warmth,
        formality: modulation.formality,
        encouragement: modulation.encouragement,
        humor: modulation.humor,
        suggested_framing: modulation.suggested_framing,
    })
}

async fn moe_select(
    State(ctx): State<Arc<KernelContext>>,
    Json(request): Json<MoeSelectRequest>,
) -> Json<MoeSelectResponse> {
    let selected = ctx.moe_gate.top2_gate(&request.embedding);
    for item in &selected {
        ctx.moe_gate.ensure_loaded(&item.expert_id);
    }
    Json(MoeSelectResponse {
        selected: selected
            .into_iter()
            .map(|item| MoeSelectionItem {
                expert_id: item.expert_id,
                score: item.score,
            })
            .collect(),
        loaded_count: ctx.moe_gate.loaded_count(),
    })
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

#[derive(Debug, Deserialize)]
struct OpenAiChatRequest {
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<OpenAiMessage>,
    stream: Option<bool>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct OpenAiMessage {
    role: String,
    content: OpenAiMessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAiMessageContent {
    Text(String),
    Parts(Vec<OpenAiMessagePart>),
}

#[derive(Debug, Deserialize)]
struct OpenAiMessagePart {
    #[serde(rename = "type")]
    part_type: Option<String>,
    text: Option<String>,
}

fn openai_messages_to_prompt(messages: &[OpenAiMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        let role = match message.role.as_str() {
            "system" => "System",
            "assistant" => "Assistant",
            _ => "User",
        };
        prompt.push_str(role);
        prompt.push_str(": ");
        prompt.push_str(&openai_message_content_to_text(&message.content));
        prompt.push('\n');
    }
    prompt.push_str("Assistant:");
    prompt
}

fn openai_message_content_to_text(content: &OpenAiMessageContent) -> String {
    match content {
        OpenAiMessageContent::Text(text) => text.clone(),
        OpenAiMessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| {
                if part.part_type.as_deref().unwrap_or("text") == "text" {
                    part.text.clone()
                } else {
                    None
                }
            })
            .collect::<Vec<String>>()
            .join("\n"),
    }
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod openai_tests {
    use super::{OpenAiMessage, OpenAiMessageContent, OpenAiMessagePart, openai_messages_to_prompt};

    #[test]
    fn openai_prompt_from_text_messages() {
        let messages = vec![
            OpenAiMessage {
                role: "system".into(),
                content: OpenAiMessageContent::Text("You are concise".into()),
            },
            OpenAiMessage {
                role: "user".into(),
                content: OpenAiMessageContent::Text("Hello".into()),
            },
        ];
        let prompt = openai_messages_to_prompt(&messages);
        assert!(prompt.contains("System: You are concise"));
        assert!(prompt.contains("User: Hello"));
        assert!(prompt.ends_with("Assistant:"));
    }

    #[test]
    fn openai_prompt_from_parts() {
        let messages = vec![OpenAiMessage {
            role: "user".into(),
            content: OpenAiMessageContent::Parts(vec![
                OpenAiMessagePart {
                    part_type: Some("text".into()),
                    text: Some("a".into()),
                },
                OpenAiMessagePart {
                    part_type: Some("input_image".into()),
                    text: Some("ignored".into()),
                },
                OpenAiMessagePart {
                    part_type: Some("text".into()),
                    text: Some("b".into()),
                },
            ]),
        }];
        let prompt = openai_messages_to_prompt(&messages);
        assert!(prompt.contains("User: a\nb"));
    }
}
