pub mod bitnet;
pub mod hdc;
pub mod memory;
pub mod mcts;
pub mod model_runtime;
pub mod models;
pub mod routes;
pub mod snapshot;
pub mod state_space;
pub mod verifier;
pub mod world_model;

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use memory::embedding::EmbeddingEngine;
use memory::vector_store::VectorStore;
use mcts::MctsEngine;
use model_runtime::ModelRuntime;
use snapshot::SnapshotStore;
use state_space::StateManager;
use verifier::Verifier;
use world_model::WorldModel;

pub const HIDDEN_SIZE: usize = 2_048;

pub struct KernelContext {
    pub state_manager: Arc<StateManager>,
    pub snapshot_store: Arc<SnapshotStore>,
    pub memory_store: Arc<VectorStore>,
    pub embedding_engine: Arc<EmbeddingEngine>,
    pub world_model: Arc<WorldModel>,
    pub verifier: Arc<Verifier>,
    pub model_runtime: Arc<ModelRuntime>,
    pub mcts_engine: Arc<MctsEngine>,
}

impl KernelContext {
    pub fn new(snapshot_path: &Path, memory_path: &Path) -> Result<Self> {
        let embedding_engine = EmbeddingEngine::new()
            .context("failed to initialise embedding engine")?;
        Ok(Self {
            state_manager: Arc::new(StateManager::new(HIDDEN_SIZE)),
            snapshot_store: Arc::new(SnapshotStore::new(snapshot_path)?),
            memory_store: Arc::new(VectorStore::new(memory_path)?),
            embedding_engine: Arc::new(embedding_engine),
            world_model: Arc::new(WorldModel::new()),
            verifier: Arc::new(Verifier::new()?),
            model_runtime: Arc::new(ModelRuntime::new()),
            mcts_engine: Arc::new(MctsEngine::new(mcts::MctsConfig::default())),
        })
    }
}
