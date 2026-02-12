use std::collections::HashMap;
use std::sync::RwLock;

use anyhow::{Result, bail};

use crate::models::HiddenState;

pub struct StateManager {
    hidden_size: usize,
    states: RwLock<HashMap<String, HiddenState>>,
}

impl StateManager {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            states: RwLock::new(HashMap::new()),
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn init_session(&self, session_id: Option<String>) -> HiddenState {
        let sid = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let mut guard = self.states.write().expect("state lock poisoned");
        let state = guard
            .entry(sid.clone())
            .or_insert_with(|| HiddenState::new(sid.clone(), self.hidden_size));
        state.clone()
    }

    pub fn get_state(&self, session_id: &str) -> Option<HiddenState> {
        let guard = self.states.read().ok()?;
        guard.get(session_id).cloned()
    }

    pub fn update_state(&self, session_id: &str, input: &str) -> Result<HiddenState> {
        if input.is_empty() {
            bail!("input must not be empty");
        }

        let mut guard = self.states.write().expect("state lock poisoned");
        let state = guard
            .entry(session_id.to_string())
            .or_insert_with(|| HiddenState::new(session_id.to_string(), self.hidden_size));

        // Linear stream: process input in fixed chunks, no global re-attention.
        for chunk in input.as_bytes().chunks(256) {
            let encoded = encode_chunk(chunk, self.hidden_size);
            for (idx, h) in state.vector.iter_mut().enumerate() {
                let x = encoded[idx];
                let gate = sigmoid(0.7 * x + 0.3 * *h);
                *h = gate * *h + (1.0 - gate) * x.tanh();
            }
            state.step += 1;
        }

        state.refresh_metadata();
        Ok(state.clone())
    }
}

fn encode_chunk(chunk: &[u8], hidden_size: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; hidden_size];
    if chunk.is_empty() {
        return out;
    }

    for (offset, byte) in chunk.iter().enumerate() {
        let idx = (offset.wrapping_mul(131) + (*byte as usize)) % hidden_size;
        let normalized = (*byte as f32 / 255.0) * 2.0 - 1.0;
        out[idx] += normalized * 0.35;
    }

    // Keep bounded values for numerical stability.
    out.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
    out
}

#[inline]
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

#[cfg(test)]
mod tests {
    use super::StateManager;

    #[test]
    fn hidden_vector_size_stays_constant() {
        let manager = StateManager::new(2048);
        let sid = "session-a";
        let updated = manager
            .update_state(sid, "xin chao vagi")
            .expect("must update state");
        assert_eq!(updated.vector.len(), 2048);
        assert!(updated.step >= 1);
    }

    #[test]
    fn checksum_changes_when_state_changes() {
        let manager = StateManager::new(128);
        let sid = "session-b";
        let first = manager.update_state(sid, "alpha").expect("first update");
        let second = manager.update_state(sid, "beta").expect("second update");
        assert_ne!(first.checksum, second.checksum);
    }
}

