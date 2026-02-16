use std::convert::Infallible;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use axum::extract::{Path as AxumPath, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::{Json, Router};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::KernelContext;
use crate::models::{ErrorResponse, ModelInferResponse, ModelLoadResponse, ModelStatusResponse};

const INDEX_HTML: &str = include_str!("../webui/index.html");
const APP_JS: &str = include_str!("../webui/app.js");
const STYLES_CSS: &str = include_str!("../webui/styles.css");

const DEFAULT_CHAT_TITLE: &str = "New Chat";
const DEFAULT_MAX_NEW_TOKENS: usize = 128;
const MAX_NEW_TOKENS: usize = 256;
const PROMPT_HISTORY_WINDOW: usize = 12;

#[derive(Clone)]
pub struct WebUiState {
    ctx: Arc<KernelContext>,
    store: Arc<Mutex<ChatStore>>,
}

pub fn build_web_ui_router(ctx: Arc<KernelContext>, data_root: PathBuf) -> Result<Router> {
    let store = ChatStore::new(data_root).context("failed to initialize web ui chat store")?;
    let state = WebUiState {
        ctx,
        store: Arc::new(Mutex::new(store)),
    };

    Ok(Router::new()
        .route("/", get(index))
        .route("/app.js", get(app_js))
        .route("/styles.css", get(styles_css))
        .route("/api/status", get(api_status))
        .route("/api/model/load", axum::routing::post(api_model_load))
        .route("/api/chat", axum::routing::post(api_chat))
        .route("/api/chat/stream", axum::routing::post(api_chat_stream))
        .route(
            "/api/chats",
            get(api_list_chats).post(api_create_chat),
        )
        .route(
            "/api/chats/{chat_id}",
            get(api_get_chat).delete(api_delete_chat),
        )
        .route(
            "/api/chats/{chat_id}/messages",
            axum::routing::post(api_append_message),
        )
        .route(
            "/api/chats/{chat_id}/messages/stream",
            axum::routing::post(api_append_message_stream),
        )
        .with_state(state))
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn app_js() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        APP_JS,
    )
}

async fn styles_css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], STYLES_CSS)
}

async fn api_status(State(state): State<WebUiState>) -> Json<ModelStatusResponse> {
    Json(state.ctx.model_runtime.status())
}

async fn api_model_load(
    State(state): State<WebUiState>,
    Json(request): Json<LoadModelRequest>,
) -> Result<Json<ModelLoadResponse>, WebUiError> {
    let response = if let Some(model_dir) = request.model_dir {
        state
            .ctx
            .model_runtime
            .load_from_dir(&model_dir)
            .map_err(WebUiError::bad_request)?
    } else {
        state
            .ctx
            .model_runtime
            .load()
            .map_err(WebUiError::bad_request)?
    };
    Ok(Json(response))
}

async fn api_chat(
    State(state): State<WebUiState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<ModelInferResponse>, WebUiError> {
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err(WebUiError::bad_request("prompt must not be empty"));
    }
    let max_new_tokens = normalize_max_tokens(request.max_new_tokens);
    let response = state
        .ctx
        .model_runtime
        .infer(prompt, max_new_tokens)
        .map_err(WebUiError::bad_request)?;
    Ok(Json(response))
}

async fn api_chat_stream(
    State(state): State<WebUiState>,
    Json(request): Json<ChatRequest>,
) -> Result<impl IntoResponse, WebUiError> {
    ensure_model_loaded(&state)?;

    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err(WebUiError::bad_request("prompt must not be empty"));
    }
    let max_new_tokens = normalize_max_tokens(request.max_new_tokens);

    let (tx, rx) = mpsc::channel(128);
    let runtime = Arc::clone(&state.ctx.model_runtime);
    let prompt = prompt.to_string();

    tokio::task::spawn_blocking(move || {
        let result = generate_streamed_text(&runtime, &prompt, max_new_tokens, &tx);
        match result {
            Ok(generated) => {
                let _ = tx.blocking_send(StreamPayload::done(generated));
            }
            Err(err) => {
                let _ = tx.blocking_send(StreamPayload::error(err.to_string()));
            }
        }
    });

    Ok(sse_from_receiver(rx))
}

async fn api_list_chats(
    State(state): State<WebUiState>,
) -> Result<Json<ChatListResponse>, WebUiError> {
    let store = state
        .store
        .lock()
        .map_err(|_| WebUiError::internal("chat store lock poisoned"))?;
    let chats = store.list_chats().map_err(WebUiError::internal)?;
    Ok(Json(ChatListResponse { chats }))
}

async fn api_create_chat(
    State(state): State<WebUiState>,
    Json(request): Json<CreateChatRequest>,
) -> Result<Json<ChatSession>, WebUiError> {
    let store = state
        .store
        .lock()
        .map_err(|_| WebUiError::internal("chat store lock poisoned"))?;
    let chat = store
        .create_chat(request.title)
        .map_err(WebUiError::internal)?;
    Ok(Json(chat))
}

async fn api_get_chat(
    State(state): State<WebUiState>,
    AxumPath(chat_id): AxumPath<String>,
) -> Result<Json<ChatSession>, WebUiError> {
    validate_chat_id(&chat_id).map_err(WebUiError::bad_request)?;

    let store = state
        .store
        .lock()
        .map_err(|_| WebUiError::internal("chat store lock poisoned"))?;
    let Some(chat) = store.get_chat(&chat_id).map_err(WebUiError::internal)? else {
        return Err(WebUiError::not_found(format!(
            "chat_id `{chat_id}` does not exist"
        )));
    };
    Ok(Json(chat))
}

async fn api_delete_chat(
    State(state): State<WebUiState>,
    AxumPath(chat_id): AxumPath<String>,
) -> Result<impl IntoResponse, WebUiError> {
    validate_chat_id(&chat_id).map_err(WebUiError::bad_request)?;

    let store = state
        .store
        .lock()
        .map_err(|_| WebUiError::internal("chat store lock poisoned"))?;
    let deleted = store.delete_chat(&chat_id).map_err(WebUiError::internal)?;
    if !deleted {
        return Err(WebUiError::not_found(format!(
            "chat_id `{chat_id}` does not exist"
        )));
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn api_append_message(
    State(state): State<WebUiState>,
    AxumPath(chat_id): AxumPath<String>,
    Json(request): Json<AppendMessageRequest>,
) -> Result<Json<ChatSession>, WebUiError> {
    validate_chat_id(&chat_id).map_err(WebUiError::bad_request)?;
    ensure_model_loaded(&state)?;

    let content = request.content.trim().to_string();
    if content.is_empty() {
        return Err(WebUiError::bad_request("content must not be empty"));
    }
    let max_new_tokens = normalize_max_tokens(request.max_new_tokens);

    let chat_after_user = {
        let store = state
            .store
            .lock()
            .map_err(|_| WebUiError::internal("chat store lock poisoned"))?;
        let Some(chat) = store
            .append_message(&chat_id, "user", content)
            .map_err(WebUiError::internal)?
        else {
            return Err(WebUiError::not_found(format!(
                "chat_id `{chat_id}` does not exist"
            )));
        };
        chat
    };

    let prompt = build_chat_prompt(&chat_after_user);
    let infer = state
        .ctx
        .model_runtime
        .infer(&prompt, max_new_tokens)
        .map_err(WebUiError::bad_request)?;

    let store = state
        .store
        .lock()
        .map_err(|_| WebUiError::internal("chat store lock poisoned"))?;
    let Some(chat) = store
        .append_message(&chat_id, "assistant", infer.text)
        .map_err(WebUiError::internal)?
    else {
        return Err(WebUiError::not_found(format!(
            "chat_id `{chat_id}` does not exist"
        )));
    };
    Ok(Json(chat))
}

async fn api_append_message_stream(
    State(state): State<WebUiState>,
    AxumPath(chat_id): AxumPath<String>,
    Json(request): Json<AppendMessageRequest>,
) -> Result<impl IntoResponse, WebUiError> {
    validate_chat_id(&chat_id).map_err(WebUiError::bad_request)?;
    ensure_model_loaded(&state)?;

    let content = request.content.trim().to_string();
    if content.is_empty() {
        return Err(WebUiError::bad_request("content must not be empty"));
    }
    let max_new_tokens = normalize_max_tokens(request.max_new_tokens);

    let chat_after_user = {
        let store = state
            .store
            .lock()
            .map_err(|_| WebUiError::internal("chat store lock poisoned"))?;
        let Some(chat) = store
            .append_message(&chat_id, "user", content)
            .map_err(WebUiError::internal)?
        else {
            return Err(WebUiError::not_found(format!(
                "chat_id `{chat_id}` does not exist"
            )));
        };
        chat
    };
    let prompt = build_chat_prompt(&chat_after_user);

    let runtime = Arc::clone(&state.ctx.model_runtime);
    let store = Arc::clone(&state.store);
    let chat_id_for_save = chat_id.clone();

    let (tx, rx) = mpsc::channel(128);
    tokio::task::spawn_blocking(move || {
        let result = generate_streamed_text(&runtime, &prompt, max_new_tokens, &tx);
        match result {
            Ok(generated) => {
                if let Ok(chat_store) = store.lock() {
                    let _ = chat_store.append_message(&chat_id_for_save, "assistant", generated.clone());
                }
                let _ = tx.blocking_send(StreamPayload::done(generated));
            }
            Err(err) => {
                let _ = tx.blocking_send(StreamPayload::error(err.to_string()));
            }
        }
    });

    Ok(sse_from_receiver(rx))
}

fn sse_from_receiver(
    rx: mpsc::Receiver<StreamPayload>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let stream = ReceiverStream::new(rx).map(|payload| {
        let event = Event::default().json_data(payload).unwrap_or_else(|_| {
            Event::default().data("{\"type\":\"error\",\"content\":\"serialization error\"}")
        });
        Ok::<Event, Infallible>(event)
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn generate_streamed_text(
    runtime: &crate::model_runtime::ModelRuntime,
    prompt: &str,
    max_new_tokens: usize,
    tx: &mpsc::Sender<StreamPayload>,
) -> Result<String> {
    let mut running_prompt = prompt.to_string();
    let mut generated = String::new();
    for _ in 0..max_new_tokens {
        let token = runtime
            .infer_next_char(&running_prompt)
            .context("failed while streaming next token")?;
        running_prompt.push_str(&token);
        generated.push_str(&token);
        if tx.blocking_send(StreamPayload::token(token)).is_err() {
            break;
        }
    }
    Ok(generated)
}

fn normalize_max_tokens(value: Option<usize>) -> usize {
    value.unwrap_or(DEFAULT_MAX_NEW_TOKENS).clamp(1, MAX_NEW_TOKENS)
}

fn ensure_model_loaded(state: &WebUiState) -> Result<(), WebUiError> {
    if state.ctx.model_runtime.status().loaded {
        Ok(())
    } else {
        Err(WebUiError::bad_request(
            "model is not loaded, call /api/model/load first",
        ))
    }
}

fn validate_chat_id(chat_id: &str) -> Result<()> {
    Uuid::parse_str(chat_id)
        .with_context(|| format!("invalid chat_id `{chat_id}`"))?;
    Ok(())
}

fn derive_title(content: &str) -> String {
    let compact = content.replace('\n', " ").trim().to_string();
    if compact.is_empty() {
        return DEFAULT_CHAT_TITLE.to_string();
    }
    let mut title: String = compact.chars().take(42).collect();
    if compact.chars().count() > 42 {
        title.push_str("...");
    }
    title
}

fn build_chat_prompt(chat: &ChatSession) -> String {
    let mut prompt = String::from(
        "You are an assistant in a chat UI. Continue the conversation naturally.\n\n",
    );
    let start = chat.messages.len().saturating_sub(PROMPT_HISTORY_WINDOW);
    for message in &chat.messages[start..] {
        match message.role.as_str() {
            "user" => prompt.push_str("User: "),
            "assistant" => prompt.push_str("Assistant: "),
            _ => prompt.push_str("System: "),
        }
        prompt.push_str(&message.content);
        prompt.push('\n');
    }
    prompt.push_str("Assistant:");
    prompt
}

fn read_json_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let raw = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_json_file_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let raw = serde_json::to_vec_pretty(value).context("failed to serialize json")?;
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, raw).with_context(|| format!("failed to write {}", tmp.display()))?;
    if path.exists() {
        fs::remove_file(path).with_context(|| format!("failed to replace {}", path.display()))?;
    }
    fs::rename(&tmp, path)
        .with_context(|| format!("failed to move {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

#[derive(Debug)]
struct ChatStore {
    chats_dir: PathBuf,
}

impl ChatStore {
    fn new(root_dir: PathBuf) -> Result<Self> {
        let chats_dir = root_dir.join("chats");
        fs::create_dir_all(&chats_dir)
            .with_context(|| format!("failed to create {}", chats_dir.display()))?;
        Ok(Self { chats_dir })
    }

    fn chat_path(&self, chat_id: &str) -> PathBuf {
        self.chats_dir.join(format!("{chat_id}.json"))
    }

    fn list_chats(&self) -> Result<Vec<ChatSummary>> {
        let mut chats = Vec::new();
        for entry in fs::read_dir(&self.chats_dir)
            .with_context(|| format!("failed to scan {}", self.chats_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|v| v.to_str()) != Some("json") {
                continue;
            }
            let chat: ChatSession = read_json_file(&path)?;
            chats.push(ChatSummary {
                id: chat.id,
                title: chat.title,
                updated_at: chat.updated_at,
                message_count: chat.messages.len(),
            });
        }
        chats.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(chats)
    }

    fn create_chat(&self, title: Option<String>) -> Result<ChatSession> {
        let now = Utc::now().to_rfc3339();
        let chat = ChatSession {
            id: Uuid::new_v4().to_string(),
            title: title
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| DEFAULT_CHAT_TITLE.to_string()),
            created_at: now.clone(),
            updated_at: now,
            messages: Vec::new(),
        };
        self.save_chat(&chat)?;
        Ok(chat)
    }

    fn get_chat(&self, chat_id: &str) -> Result<Option<ChatSession>> {
        let path = self.chat_path(chat_id);
        if !path.exists() {
            return Ok(None);
        }
        let chat: ChatSession = read_json_file(&path)?;
        Ok(Some(chat))
    }

    fn save_chat(&self, chat: &ChatSession) -> Result<()> {
        let path = self.chat_path(&chat.id);
        write_json_file_atomic(&path, chat)?;
        Ok(())
    }

    fn delete_chat(&self, chat_id: &str) -> Result<bool> {
        let path = self.chat_path(chat_id);
        if !path.exists() {
            return Ok(false);
        }
        fs::remove_file(&path).with_context(|| format!("failed to delete {}", path.display()))?;
        Ok(true)
    }

    fn append_message(
        &self,
        chat_id: &str,
        role: &str,
        content: String,
    ) -> Result<Option<ChatSession>> {
        let Some(mut chat) = self.get_chat(chat_id)? else {
            return Ok(None);
        };
        if role != "user" && role != "assistant" && role != "system" {
            bail!("unsupported role `{role}`");
        }

        let now = Utc::now().to_rfc3339();
        if role == "user" && chat.title == DEFAULT_CHAT_TITLE {
            chat.title = derive_title(&content);
        }

        chat.messages.push(ChatMessage {
            role: role.to_string(),
            content,
            created_at: now.clone(),
        });
        chat.updated_at = now;
        self.save_chat(&chat)?;
        Ok(Some(chat))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatSummary {
    pub id: String,
    pub title: String,
    pub updated_at: String,
    pub message_count: usize,
}

#[derive(Debug, Deserialize)]
struct LoadModelRequest {
    model_dir: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    prompt: String,
    max_new_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct CreateChatRequest {
    title: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AppendMessageRequest {
    content: String,
    max_new_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
struct ChatListResponse {
    chats: Vec<ChatSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct StreamPayload {
    #[serde(rename = "type")]
    payload_type: String,
    content: String,
}

impl StreamPayload {
    fn token(content: String) -> Self {
        Self {
            payload_type: "token".to_string(),
            content,
        }
    }

    fn done(content: String) -> Self {
        Self {
            payload_type: "done".to_string(),
            content,
        }
    }

    fn error(content: String) -> Self {
        Self {
            payload_type: "error".to_string(),
            content,
        }
    }
}

#[derive(Debug)]
pub struct WebUiError {
    status: StatusCode,
    message: String,
}

impl WebUiError {
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

impl IntoResponse for WebUiError {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            Json(ErrorResponse {
                error: self.message,
            }),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatStore, DEFAULT_CHAT_TITLE, build_chat_prompt};

    #[test]
    fn chat_store_create_and_append_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ChatStore::new(tmp.path().to_path_buf()).unwrap();

        let chat = store.create_chat(None).unwrap();
        assert_eq!(chat.title, DEFAULT_CHAT_TITLE);
        assert_eq!(chat.messages.len(), 0);

        let chat = store
            .append_message(&chat.id, "user", "hello kernel".to_string())
            .unwrap()
            .unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert_ne!(chat.title, DEFAULT_CHAT_TITLE);

        let chat = store
            .append_message(&chat.id, "assistant", "hello user".to_string())
            .unwrap()
            .unwrap();
        assert_eq!(chat.messages.len(), 2);

        let list = store.list_chats().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].message_count, 2);
    }

    #[test]
    fn prompt_contains_roles_and_assistant_tail() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ChatStore::new(tmp.path().to_path_buf()).unwrap();
        let chat = store.create_chat(Some("demo".to_string())).unwrap();
        let chat = store
            .append_message(&chat.id, "user", "first".to_string())
            .unwrap()
            .unwrap();
        let chat = store
            .append_message(&chat.id, "assistant", "second".to_string())
            .unwrap()
            .unwrap();

        let prompt = build_chat_prompt(&chat);
        assert!(prompt.contains("User: first"));
        assert!(prompt.contains("Assistant: second"));
        assert!(prompt.ends_with("Assistant:"));
    }
}
