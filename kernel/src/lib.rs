pub mod models;
pub mod routes;
pub mod snapshot;
pub mod state_space;
pub mod verifier;
pub mod world_model;

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
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
}

impl KernelContext {
    pub fn new(snapshot_path: &Path) -> Result<Self> {
        Ok(Self {
            state_manager: Arc::new(StateManager::new(HIDDEN_SIZE)),
            snapshot_store: Arc::new(SnapshotStore::new(snapshot_path)?),
            world_model: Arc::new(WorldModel::new()),
            verifier: Arc::new(Verifier::new()?),
        })
    }
}

