use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use anyhow::{Result, bail};
use chrono::Utc;
use regex::Regex;
use sha2::{Digest, Sha256};

use crate::models::{
    HdcTemplateBindRequest, HdcTemplateBindResponse, HdcTemplateMatch, HdcTemplateQueryRequest,
    HdcTemplateQueryResponse, HdcTemplateUpsertRequest, HdcTemplateUpsertResponse,
};

const HDC_DIM_BITS: usize = 10_240;
const HDC_DIM_WORDS: usize = HDC_DIM_BITS / 64;
const TOKEN_BIND_BITS: usize = 8;

#[derive(Clone)]
struct HyperVector {
    bits: [u64; HDC_DIM_WORDS],
}

impl HyperVector {
    fn zero() -> Self {
        Self {
            bits: [0; HDC_DIM_WORDS],
        }
    }

    fn encode_tokens(tokens: &[String]) -> Self {
        let mut out = Self::zero();
        for token in tokens {
            let token_vec = Self::token_vector(token);
            out.xor_inplace(&token_vec);
        }
        out
    }

    fn token_vector(token: &str) -> Self {
        let digest = Sha256::digest(token.as_bytes());
        let mut out = Self::zero();
        for i in 0..TOKEN_BIND_BITS {
            let byte_ix = (i * 2) % digest.len();
            let idx = u16::from_le_bytes([digest[byte_ix], digest[byte_ix + 1]]) as usize;
            out.set_bit(idx % HDC_DIM_BITS);
        }
        out
    }

    fn set_bit(&mut self, idx: usize) {
        let word_ix = idx / 64;
        let bit_ix = idx % 64;
        self.bits[word_ix] |= 1u64 << bit_ix;
    }

    fn xor_inplace(&mut self, other: &Self) {
        for (lhs, rhs) in self.bits.iter_mut().zip(other.bits.iter()) {
            *lhs ^= *rhs;
        }
    }

    fn similarity(&self, other: &Self) -> f32 {
        let distance: u32 = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(lhs, rhs)| (lhs ^ rhs).count_ones())
            .sum();
        1.0 - (distance as f32 / HDC_DIM_BITS as f32)
    }
}

#[derive(Clone)]
struct TemplateEntry {
    template_id: String,
    tags: Vec<String>,
    logic_template: String,
    vector: HyperVector,
}

pub struct HolographicMemory {
    templates: RwLock<HashMap<String, TemplateEntry>>,
}

impl HolographicMemory {
    pub fn new() -> Self {
        let memory = Self {
            templates: RwLock::new(HashMap::new()),
        };
        memory.bootstrap_defaults();
        memory
    }

    pub fn upsert_template(
        &self,
        request: &HdcTemplateUpsertRequest,
    ) -> Result<HdcTemplateUpsertResponse> {
        let template_id = request.template_id.trim();
        if template_id.is_empty() {
            bail!("template_id must not be empty");
        }
        if request.logic_template.trim().is_empty() {
            bail!("logic_template must not be empty");
        }

        let mut tokens = tokenize(&request.logic_template);
        let tag_tokens = request
            .tags
            .iter()
            .flat_map(|tag| tokenize(tag))
            .map(|token| format!("tag_{token}"))
            .collect::<Vec<_>>();
        tokens.extend(tag_tokens);
        if tokens.is_empty() {
            bail!("logic_template does not contain indexable tokens");
        }

        let entry = TemplateEntry {
            template_id: template_id.to_string(),
            tags: normalize_tags(&request.tags),
            logic_template: request.logic_template.clone(),
            vector: HyperVector::encode_tokens(&tokens),
        };

        let mut guard = self
            .templates
            .write()
            .map_err(|_| anyhow::anyhow!("failed to lock HDC memory for write"))?;
        guard.insert(entry.template_id.clone(), entry);

        Ok(HdcTemplateUpsertResponse {
            template_id: template_id.to_string(),
            token_count: tokens.len(),
            dimension_bits: HDC_DIM_BITS,
            stored_at: Utc::now().to_rfc3339(),
        })
    }

    pub fn query_templates(
        &self,
        request: &HdcTemplateQueryRequest,
    ) -> Result<HdcTemplateQueryResponse> {
        let query = request.query.trim();
        if query.is_empty() {
            bail!("query must not be empty");
        }
        let tokens = tokenize(query);
        if tokens.is_empty() {
            bail!("query does not contain indexable tokens");
        }
        let query_vec = HyperVector::encode_tokens(&tokens);
        let top_k = request.top_k.unwrap_or(3).clamp(1, 20);

        let guard = self
            .templates
            .read()
            .map_err(|_| anyhow::anyhow!("failed to lock HDC memory for read"))?;

        let mut hits = guard
            .values()
            .map(|entry| HdcTemplateMatch {
                template_id: entry.template_id.clone(),
                similarity: query_vec.similarity(&entry.vector),
                tags: entry.tags.clone(),
                logic_template: entry.logic_template.clone(),
            })
            .collect::<Vec<_>>();

        hits.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });
        hits.truncate(top_k);

        Ok(HdcTemplateQueryResponse {
            hits,
            dimension_bits: HDC_DIM_BITS,
        })
    }

    pub fn bind_template(&self, request: &HdcTemplateBindRequest) -> Result<HdcTemplateBindResponse> {
        let template_id = request.template_id.trim();
        if template_id.is_empty() {
            bail!("template_id must not be empty");
        }

        let guard = self
            .templates
            .read()
            .map_err(|_| anyhow::anyhow!("failed to lock HDC memory for read"))?;
        let entry = guard
            .get(template_id)
            .ok_or_else(|| anyhow::anyhow!("template `{template_id}` not found"))?;

        let placeholder_regex = Regex::new(r"\{\{\s*([a-zA-Z0-9_-]+)\s*\}\}")?;
        let mut missing = HashSet::new();
        for caps in placeholder_regex.captures_iter(&entry.logic_template) {
            let key = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
            if !request.bindings.contains_key(key) {
                missing.insert(key.to_string());
            }
        }
        if !missing.is_empty() {
            let mut missing_list = missing.into_iter().collect::<Vec<_>>();
            missing_list.sort();
            bail!("missing bindings for placeholders: {}", missing_list.join(", "));
        }

        let mut placeholders_resolved = 0usize;
        let bound_logic = placeholder_regex
            .replace_all(&entry.logic_template, |caps: &regex::Captures| {
                let key = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
                if let Some(value) = request.bindings.get(key) {
                    placeholders_resolved += 1;
                    value.clone()
                } else {
                    caps.get(0)
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_default()
                }
            })
            .to_string();

        let normalized_logic = bound_logic
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToString::to_string)
            .collect::<Vec<_>>();

        Ok(HdcTemplateBindResponse {
            template_id: template_id.to_string(),
            bound_logic,
            placeholders_resolved,
            normalized_logic,
        })
    }

    pub fn template_snapshot(&self, template_id: &str) -> Result<Option<(String, Vec<String>)>> {
        let key = template_id.trim();
        if key.is_empty() {
            bail!("template_id must not be empty");
        }
        let guard = self
            .templates
            .read()
            .map_err(|_| anyhow::anyhow!("failed to lock HDC memory for read"))?;
        let snapshot = guard
            .get(key)
            .map(|entry| (entry.logic_template.clone(), entry.tags.clone()));
        Ok(snapshot)
    }

    fn bootstrap_defaults(&self) {
        let defaults = [
            (
                "python_secure_v1",
                "add 5\nmul 2\nxor 3\n# python secure validate hash timeout",
                vec!["python".to_string(), "secure".to_string(), "auth".to_string()],
            ),
            (
                "rust_perf_v1",
                "mul 2\nshl 1\nand 4095\n# rust performance low allocation",
                vec!["rust".to_string(), "perf".to_string(), "latency".to_string()],
            ),
            (
                "api_guard_v1",
                "add 1\nor 8\nand 1023\n# api validation rate-limit audit",
                vec!["api".to_string(), "guard".to_string(), "safety".to_string()],
            ),
        ];
        for (template_id, logic_template, tags) in defaults {
            let _ = self.upsert_template(&HdcTemplateUpsertRequest {
                template_id: template_id.to_string(),
                logic_template: logic_template.to_string(),
                tags,
            });
        }
    }
}

fn normalize_tags(tags: &[String]) -> Vec<String> {
    tags.iter()
        .map(|tag| tag.trim().to_ascii_lowercase())
        .filter(|tag| !tag.is_empty())
        .collect()
}

fn tokenize(input: &str) -> Vec<String> {
    input
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-')
        .map(|token| token.trim().to_ascii_lowercase())
        .filter(|token| token.len() >= 2)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::models::{HdcTemplateBindRequest, HdcTemplateQueryRequest, HdcTemplateUpsertRequest};

    use super::HolographicMemory;

    #[test]
    fn upsert_and_query_returns_best_match() {
        let memory = HolographicMemory::new();
        memory
            .upsert_template(&HdcTemplateUpsertRequest {
                template_id: "custom_python_expert".to_string(),
                logic_template: "add 2\nmul 3\n# python secure patch zxqv_token".to_string(),
                tags: vec!["python".to_string(), "secure".to_string(), "zxqv".to_string()],
            })
            .expect("upsert python template");
        memory
            .upsert_template(&HdcTemplateUpsertRequest {
                template_id: "rust_perf".to_string(),
                logic_template: "mul 2\nshl 1\n# rust perf".to_string(),
                tags: vec!["rust".to_string(), "perf".to_string()],
            })
            .expect("upsert rust template");

        let response = memory
            .query_templates(&HdcTemplateQueryRequest {
                query: "need secure python patch zxqv".to_string(),
                top_k: Some(1),
            })
            .expect("query templates");

        assert_eq!(response.hits.len(), 1);
        assert_eq!(response.hits[0].template_id, "custom_python_expert");
    }

    #[test]
    fn bind_replaces_placeholders() {
        let memory = HolographicMemory::new();
        memory
            .upsert_template(&HdcTemplateUpsertRequest {
                template_id: "templated".to_string(),
                logic_template: "add {{delta}}\nxor {{mask}}".to_string(),
                tags: vec!["generic".to_string()],
            })
            .expect("upsert template");

        let mut bindings = HashMap::new();
        bindings.insert("delta".to_string(), "7".to_string());
        bindings.insert("mask".to_string(), "3".to_string());

        let response = memory
            .bind_template(&HdcTemplateBindRequest {
                template_id: "templated".to_string(),
                bindings,
            })
            .expect("bind template");

        assert_eq!(response.placeholders_resolved, 2);
        assert_eq!(response.bound_logic, "add 7\nxor 3");
    }
}
