use std::fs;
use std::path::Path;

use anyhow::Result;
use redb::{Database, ReadableDatabase, TableDefinition};

use crate::models::HiddenState;

const SNAPSHOT_TABLE: TableDefinition<&str, &str> = TableDefinition::new("snapshots");

pub struct SnapshotStore {
    db: Database,
}

impl SnapshotStore {
    pub fn new(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let db = Database::create(path)?;
        Ok(Self { db })
    }

    pub fn save_state(&self, state: &HiddenState, epoch: u64) -> Result<String> {
        let key = format!("{}:{}", state.session_id, epoch);
        let value = serde_json::to_string(state)?;
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(SNAPSHOT_TABLE)?;
            table.insert(key.as_str(), value.as_str())?;
        }
        write_txn.commit()?;
        Ok(key)
    }

    pub fn load_state(&self, key: &str) -> Result<Option<HiddenState>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SNAPSHOT_TABLE)?;
        let Some(value) = table.get(key)? else {
            return Ok(None);
        };
        let json: &str = value.value();
        let state: HiddenState = serde_json::from_str(json)?;
        Ok(Some(state))
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use crate::models::HiddenState;

    use super::SnapshotStore;

    #[test]
    fn saves_and_reads_snapshot() {
        let temp = tempdir().expect("temp dir");
        let db_path = temp.path().join("snapshots.redb");
        let store = SnapshotStore::new(&db_path).expect("create store");
        let state = HiddenState::new("sid-1".to_string(), 8);
        let key = store.save_state(&state, 10).expect("save snapshot");
        let loaded = store
            .load_state(&key)
            .expect("load")
            .expect("state must exist");
        assert_eq!(loaded.session_id, "sid-1");
        assert_eq!(loaded.vector.len(), 8);
    }
}
