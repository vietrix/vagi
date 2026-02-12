pub mod hdc;
pub mod jit;
pub mod models;
pub mod mutation;
pub mod routes;
pub mod snapshot;
pub mod state_space;
pub mod verifier;
pub mod world_model;

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use hdc::HolographicMemory;
use jit::JitEngine;
use mutation::MutationEngine;
use snapshot::SnapshotStore;
use state_space::StateManager;
use verifier::Verifier;
use world_model::WorldModel;

pub const HIDDEN_SIZE: usize = 2_048;

pub struct KernelContext {
    pub state_manager: Arc<StateManager>,
    pub snapshot_store: Arc<SnapshotStore>,
    pub world_model: Arc<WorldModel>,
    pub verifier: Arc<Verifier>,
    pub jit_engine: Arc<JitEngine>,
    pub hdc_memory: Arc<HolographicMemory>,
    pub mutation_engine: Arc<MutationEngine>,
}

impl KernelContext {
    pub fn new(snapshot_path: &Path) -> Result<Self> {
        let hdc_memory = Arc::new(HolographicMemory::new());
        let jit_engine = Arc::new(JitEngine::new()?);
        let world_model = Arc::new(WorldModel::new());
        let verifier = Arc::new(Verifier::new()?);

        Ok(Self {
            state_manager: Arc::new(StateManager::new(HIDDEN_SIZE)),
            snapshot_store: Arc::new(SnapshotStore::new(snapshot_path)?),
            world_model: world_model.clone(),
            verifier: verifier.clone(),
            jit_engine: jit_engine.clone(),
            hdc_memory: hdc_memory.clone(),
            mutation_engine: Arc::new(MutationEngine::new(
                hdc_memory,
                jit_engine,
                world_model,
                verifier,
            )),
        })
    }
}
