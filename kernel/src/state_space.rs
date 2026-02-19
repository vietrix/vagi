use std::collections::HashMap;
use std::sync::RwLock;

use anyhow::{Result, bail};

use crate::models::HiddenState;

const L2_BLOCK: usize = 128;

#[derive(Debug, Clone)]
struct SsrRecurrence {
    // h_t = a*h_{t-1} + b*x_t + c*mean(h_{t-1})
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
}

impl SsrRecurrence {
    fn new(hidden_size: usize) -> Self {
        let mut a = vec![0.0; hidden_size];
        let mut b = vec![0.0; hidden_size];
        let mut c = vec![0.0; hidden_size];
        for idx in 0..hidden_size {
            // Keep dynamics stable (<1.0) and deterministic.
            a[idx] = 0.86 + (idx % 11) as f32 * 0.005;
            b[idx] = 0.12 + (idx % 7) as f32 * 0.003;
            c[idx] = 0.01 + (idx % 5) as f32 * 0.001;
        }
        Self { a, b, c }
    }

    fn step(&self, h: &mut [f32], x: &[f32]) {
        let mean_h = h.iter().copied().sum::<f32>() / h.len().max(1) as f32;
        for start in (0..h.len()).step_by(L2_BLOCK) {
            let end = (start + L2_BLOCK).min(h.len());
            for idx in start..end {
                let next = self.a[idx] * h[idx] + self.b[idx] * x[idx] + self.c[idx] * mean_h;
                h[idx] = next.clamp(-1.0, 1.0);
            }
        }
    }
}

pub struct StateManager {
    hidden_size: usize,
    states: RwLock<HashMap<String, HiddenState>>,
    recurrence: SsrRecurrence,
}

impl StateManager {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            states: RwLock::new(HashMap::new()),
            recurrence: SsrRecurrence::new(hidden_size),
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

        for chunk in input.as_bytes().chunks(256) {
            let encoded = encode_chunk(chunk, self.hidden_size);
            self.recurrence.step(&mut state.vector, &encoded);
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

    out.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
    out
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
