//! Mixture-of-Experts gate with HDC-space routing and dynamic loading.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct ExpertProfile {
    pub expert_id: String,
    pub centroid: Vec<f32>,
    pub path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ExpertSelection {
    pub expert_id: String,
    pub score: f32,
}

#[derive(Debug)]
pub struct MoeGate {
    experts: RwLock<Vec<ExpertProfile>>,
    loaded: RwLock<HashMap<String, Vec<f32>>>,
    lru: RwLock<VecDeque<String>>,
    max_loaded_experts: usize,
}

impl MoeGate {
    pub fn new(max_loaded_experts: usize) -> Self {
        Self {
            experts: RwLock::new(Vec::new()),
            loaded: RwLock::new(HashMap::new()),
            lru: RwLock::new(VecDeque::new()),
            max_loaded_experts: max_loaded_experts.max(2),
        }
    }

    pub fn register_experts(&self, experts: Vec<ExpertProfile>) {
        let mut guard = self.experts.write().expect("moe experts lock poisoned");
        *guard = experts;
    }

    pub fn top2_gate(&self, hdc_embedding: &[f32]) -> Vec<ExpertSelection> {
        let guard = self.experts.read().expect("moe experts lock poisoned");
        let mut scored: Vec<ExpertSelection> = guard
            .iter()
            .map(|expert| ExpertSelection {
                expert_id: expert.expert_id.clone(),
                score: cosine_similarity(&expert.centroid, hdc_embedding),
            })
            .collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(2).collect()
    }

    pub fn ensure_loaded(&self, expert_id: &str) {
        {
            let loaded = self.loaded.read().expect("moe loaded lock poisoned");
            if loaded.contains_key(expert_id) {
                drop(loaded);
                self.touch_lru(expert_id);
                return;
            }
        }

        let maybe_profile = {
            let experts = self.experts.read().expect("moe experts lock poisoned");
            experts.iter().find(|e| e.expert_id == expert_id).cloned()
        };

        let Some(profile) = maybe_profile else {
            return;
        };

        let payload = load_expert_payload(&profile).unwrap_or_default();
        {
            let mut loaded = self.loaded.write().expect("moe loaded lock poisoned");
            loaded.insert(expert_id.to_string(), payload);
            while loaded.len() > self.max_loaded_experts {
                if let Some(evicted) = self.pop_lru() {
                    loaded.remove(&evicted);
                } else {
                    break;
                }
            }
        }
        self.touch_lru(expert_id);
    }

    pub fn loaded_count(&self) -> usize {
        self.loaded
            .read()
            .expect("moe loaded lock poisoned")
            .len()
    }

    pub fn prefetch_experts(&self, expert_ids: &[String]) {
        for expert_id in expert_ids {
            self.ensure_loaded(expert_id);
        }
    }

    fn touch_lru(&self, expert_id: &str) {
        let mut lru = self.lru.write().expect("moe lru lock poisoned");
        if let Some(idx) = lru.iter().position(|entry| entry == expert_id) {
            lru.remove(idx);
        }
        lru.push_back(expert_id.to_string());
    }

    fn pop_lru(&self) -> Option<String> {
        let mut lru = self.lru.write().expect("moe lru lock poisoned");
        lru.pop_front()
    }
}

fn load_expert_payload(profile: &ExpertProfile) -> Result<Vec<f32>> {
    let Some(path) = &profile.path else {
        return Ok(profile.centroid.clone());
    };
    let bytes = fs::read(path)
        .with_context(|| format!("failed to load expert payload at {}", path.display()))?;
    if bytes.is_empty() {
        return Ok(profile.centroid.clone());
    }
    let mut out = Vec::with_capacity(bytes.len());
    for chunk in bytes.chunks(4) {
        let mut buf = [0_u8; 4];
        for (idx, b) in chunk.iter().enumerate() {
            buf[idx] = *b;
        }
        out.push(f32::from_le_bytes(buf));
    }
    Ok(out)
}

pub fn discover_expert_profiles(dir: &Path) -> Result<Vec<ExpertProfile>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut profiles = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("expert") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        profiles.push(ExpertProfile {
            expert_id: stem.to_string(),
            centroid: vec![0.0; 64],
            path: Some(path),
        });
    }
    Ok(profiles)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for idx in 0..n {
        dot += a[idx] * b[idx];
        na += a[idx] * a[idx];
        nb += b[idx] * b[idx];
    }
    if na <= 1e-8 || nb <= 1e-8 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top2_gating_returns_two() {
        let gate = MoeGate::new(3);
        gate.register_experts(vec![
            ExpertProfile {
                expert_id: "logic".to_string(),
                centroid: vec![1.0, 0.0, 0.0],
                path: None,
            },
            ExpertProfile {
                expert_id: "code".to_string(),
                centroid: vec![0.0, 1.0, 0.0],
                path: None,
            },
            ExpertProfile {
                expert_id: "math".to_string(),
                centroid: vec![0.7, 0.7, 0.0],
                path: None,
            },
        ]);
        let selected = gate.top2_gate(&[0.9, 0.8, 0.0]);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn prefetch_loads_experts() {
        let gate = MoeGate::new(4);
        gate.register_experts(vec![
            ExpertProfile {
                expert_id: "expert_00".to_string(),
                centroid: vec![1.0, 0.0, 0.0],
                path: None,
            },
            ExpertProfile {
                expert_id: "expert_01".to_string(),
                centroid: vec![0.0, 1.0, 0.0],
                path: None,
            },
        ]);
        gate.prefetch_experts(&["expert_00".to_string(), "expert_01".to_string()]);
        assert_eq!(gate.loaded_count(), 2);
    }
}
