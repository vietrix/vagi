use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use chrono::Utc;
use sha2::{Digest, Sha256};

use crate::hdc::HolographicMemory;
use crate::jit::JitEngine;
use crate::models::{
    HdcEvolutionCandidate, HdcEvolutionGenerationReport, HdcEvolutionMutateRequest,
    HdcEvolutionMutateResponse, HdcTemplateQueryRequest, HdcTemplateUpsertRequest, VerifierRequest,
};
use crate::verifier::Verifier;
use crate::world_model::WorldModel;

pub struct MutationEngine {
    hdc_memory: Arc<HolographicMemory>,
    jit_engine: Arc<JitEngine>,
    world_model: Arc<WorldModel>,
    verifier: Arc<Verifier>,
}

impl MutationEngine {
    pub fn new(
        hdc_memory: Arc<HolographicMemory>,
        jit_engine: Arc<JitEngine>,
        world_model: Arc<WorldModel>,
        verifier: Arc<Verifier>,
    ) -> Self {
        Self {
            hdc_memory,
            jit_engine,
            world_model,
            verifier,
        }
    }

    pub fn evolve_templates(
        &self,
        request: &HdcEvolutionMutateRequest,
    ) -> Result<HdcEvolutionMutateResponse> {
        let base_template_id = self.resolve_base_template_id(request)?;
        let (base_logic, base_tags) = self
            .hdc_memory
            .template_snapshot(&base_template_id)?
            .ok_or_else(|| anyhow!("template `{base_template_id}` not found"))?;

        let generations = request.generations.unwrap_or(3).clamp(1, 20);
        let population_size = request.population_size.unwrap_or(8).clamp(2, 64);
        let survivors = request.survivors.unwrap_or(2).clamp(1, population_size);
        let risk_threshold = request.risk_threshold.unwrap_or(0.65).clamp(0.01, 0.99);
        let seed_input = request.seed_input.unwrap_or(13);
        let promote = request.promote.unwrap_or(true);

        let mut parent_logic = vec![base_logic.clone()];
        let mut history = Vec::new();
        let mut final_candidates = Vec::new();
        let mut total_candidates_evaluated = 0usize;
        let mut global_best_candidate: Option<HdcEvolutionCandidate> = None;

        for generation in 1..=generations {
            let mut generation_candidates = Vec::new();
            for variant in 0..population_size {
                let parent = &parent_logic[variant % parent_logic.len()];
                let candidate_logic = if generation == 1 && variant == 0 {
                    parent.clone()
                } else {
                    mutate_logic(
                        parent,
                        &base_template_id,
                        generation,
                        variant,
                        seed_input,
                        risk_threshold,
                    )
                };

                let candidate = match self.evaluate_candidate(
                    generation,
                    variant,
                    &base_template_id,
                    &candidate_logic,
                    seed_input,
                    risk_threshold,
                ) {
                    Ok(candidate) => candidate,
                    Err(err) => HdcEvolutionCandidate {
                        candidate_id: format!(
                            "{}:g{}-v{}-failed",
                            base_template_id, generation, variant
                        ),
                        generation,
                        score: -5.0,
                        output: 0,
                        verifier_pass: false,
                        verifier_violations: vec![format!("candidate_eval_failed:{err}")],
                        risk_score: 0.99,
                        confidence: 0.01,
                        compile_micros: 0,
                        execute_micros: 0,
                        logic: candidate_logic.clone(),
                    },
                };
                update_global_best(&mut global_best_candidate, &candidate);
                generation_candidates.push(candidate);
                total_candidates_evaluated += 1;
            }

            generation_candidates.sort_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| a.risk_score.total_cmp(&b.risk_score))
                    .then_with(|| b.verifier_pass.cmp(&a.verifier_pass))
                    .then_with(|| a.execute_micros.cmp(&b.execute_micros))
            });

            let best = generation_candidates
                .first()
                .ok_or_else(|| anyhow!("mutation generation produced no candidates"))?;

            history.push(HdcEvolutionGenerationReport {
                generation,
                best_candidate_id: best.candidate_id.clone(),
                best_score: best.score,
                best_risk_score: best.risk_score,
                best_verifier_pass: best.verifier_pass,
            });

            parent_logic = generation_candidates
                .iter()
                .take(survivors)
                .map(|candidate| candidate.logic.clone())
                .collect();

            final_candidates = generation_candidates;
        }

        let mut promoted_template_id = None;
        if promote {
            if let Some(best) = global_best_candidate.as_ref() {
                if best.verifier_pass && best.risk_score <= risk_threshold {
                    let promoted_id = build_promoted_template_id(&base_template_id, &best.logic);
                    let mut tags = base_tags;
                    tags.push("mutated".to_string());
                    tags.push("evolution".to_string());
                    tags.push(format!("parent_{base_template_id}"));
                    self.hdc_memory.upsert_template(&HdcTemplateUpsertRequest {
                        template_id: promoted_id.clone(),
                        logic_template: best.logic.clone(),
                        tags,
                    })?;
                    promoted_template_id = Some(promoted_id);
                }
            }
        }

        Ok(HdcEvolutionMutateResponse {
            base_template_id,
            promoted_template_id,
            generations_run: generations,
            population_size,
            survivors,
            total_candidates_evaluated,
            history,
            final_candidates,
            backend: "wasmtime-cranelift-jit",
        })
    }

    fn resolve_base_template_id(&self, request: &HdcEvolutionMutateRequest) -> Result<String> {
        if let Some(template_id) = request.template_id.as_ref() {
            let trimmed = template_id.trim();
            if trimmed.is_empty() {
                bail!("template_id must not be empty");
            }
            return Ok(trimmed.to_string());
        }

        let query = request
            .query
            .as_ref()
            .map(|q| q.trim().to_string())
            .filter(|q| !q.is_empty())
            .ok_or_else(|| anyhow!("either template_id or query is required"))?;

        let query_res = self.hdc_memory.query_templates(&HdcTemplateQueryRequest {
            query,
            top_k: Some(1),
        })?;
        let hit = query_res
            .hits
            .first()
            .ok_or_else(|| anyhow!("no template matched query"))?;
        Ok(hit.template_id.clone())
    }

    fn evaluate_candidate(
        &self,
        generation: usize,
        variant: usize,
        base_template_id: &str,
        logic: &str,
        seed_input: i64,
        risk_threshold: f32,
    ) -> Result<HdcEvolutionCandidate> {
        let candidate_id = format!("g{generation}-v{variant}-{}", short_hash(logic));
        let jit = self.jit_engine.compile_and_execute(&crate::models::JitExecuteRequest {
            logic: logic.to_string(),
            input: seed_input,
        })?;
        let sim = self.world_model.simulate(logic);
        let verify = self.verifier.check(&VerifierRequest {
            patch_ir: logic.to_string(),
            max_loop_iters: Some(2_048),
            side_effect_budget: Some(3),
            timeout_ms: Some(80),
        });

        let score = compute_candidate_score(
            verify.pass,
            sim.risk_score,
            sim.confidence,
            jit.compile_micros,
            jit.execute_micros,
            risk_threshold,
        );

        Ok(HdcEvolutionCandidate {
            candidate_id: format!("{base_template_id}:{candidate_id}"),
            generation,
            score,
            output: jit.output,
            verifier_pass: verify.pass,
            verifier_violations: verify.violations,
            risk_score: sim.risk_score,
            confidence: sim.confidence,
            compile_micros: jit.compile_micros,
            execute_micros: jit.execute_micros,
            logic: logic.to_string(),
        })
    }
}

fn compute_candidate_score(
    verifier_pass: bool,
    risk_score: f32,
    confidence: f32,
    compile_micros: u64,
    execute_micros: u64,
    risk_threshold: f32,
) -> f32 {
    let mut score = 0.0_f32;
    score += if verifier_pass { 1.3 } else { -1.2 };
    score += (1.0 - risk_score).clamp(0.0, 1.0) * 1.1;
    score += confidence.clamp(0.0, 1.0) * 0.5;
    score += if risk_score <= risk_threshold { 0.4 } else { -0.6 };
    let latency = compile_micros.saturating_add(execute_micros).max(1);
    score += (1500.0 / latency as f32).clamp(0.0, 0.5);
    score
}

fn mutate_logic(
    parent_logic: &str,
    base_template_id: &str,
    generation: usize,
    variant: usize,
    seed_input: i64,
    risk_threshold: f32,
) -> String {
    let mut out_lines = Vec::new();
    let mut seen_op = false;

    for (line_ix, line) in parent_logic.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            out_lines.push(trimmed.to_string());
            continue;
        }

        let mut parts = trimmed.split_whitespace();
        let raw_op = parts.next().unwrap_or_default().to_ascii_lowercase();
        let raw_val = parts.next().unwrap_or_default();
        if parts.next().is_some() {
            out_lines.push(trimmed.to_string());
            continue;
        }
        let Ok(value) = raw_val.parse::<i64>() else {
            out_lines.push(trimmed.to_string());
            continue;
        };

        let hash = mix_hash(&format!(
            "{base_template_id}|{generation}|{variant}|{line_ix}|{raw_op}|{seed_input}"
        ));
        let delta = ((hash % 5) + 1) as i64;
        let mode = (hash % 6) as u8;
        let mut op = maybe_swap_op(&raw_op, hash);
        let mut new_val = match mode {
            0 => value.wrapping_add(delta),
            1 => value.wrapping_sub(delta),
            2 => value ^ delta,
            3 => value.wrapping_mul((delta % 2) + 1),
            4 => value.wrapping_add((risk_threshold * 10.0) as i64),
            _ => value.wrapping_sub((generation as i64).min(3)),
        };

        if op == "shl" || op == "shr" {
            if new_val < 0 {
                new_val = -new_val;
            }
            new_val = new_val.clamp(0, 31);
        }
        if op == "and" && new_val <= 0 {
            new_val = 1023;
        }

        out_lines.push(format!("{op} {new_val}"));
        seen_op = true;
    }

    if !seen_op {
        out_lines.push("add 1".to_string());
    }
    let patch_hash = mix_hash(&format!("{base_template_id}|{generation}|{variant}|guard"));
    if patch_hash % 3 == 0 {
        out_lines.push("and 4095".to_string());
    } else if patch_hash % 5 == 0 {
        out_lines.push("xor 7".to_string());
    }
    out_lines.push(format!(
        "# evolved generation={generation} variant={variant} ts={}",
        Utc::now().timestamp()
    ));

    out_lines.join("\n")
}

fn maybe_swap_op(op: &str, hash: u64) -> String {
    if hash % 7 != 0 {
        return op.to_string();
    }
    match op {
        "add" => "sub".to_string(),
        "sub" => "add".to_string(),
        "xor" => "or".to_string(),
        "or" => "xor".to_string(),
        "shl" => "shr".to_string(),
        "shr" => "shl".to_string(),
        _ => op.to_string(),
    }
}

fn mix_hash(input: &str) -> u64 {
    let digest = Sha256::digest(input.as_bytes());
    u64::from_le_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ])
}

fn short_hash(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    format!(
        "{:02x}{:02x}{:02x}{:02x}",
        digest[0], digest[1], digest[2], digest[3]
    )
}

fn build_promoted_template_id(base_template_id: &str, logic: &str) -> String {
    format!("{}_mut_{}", base_template_id, short_hash(logic))
}

fn update_global_best(
    global_best: &mut Option<HdcEvolutionCandidate>,
    candidate: &HdcEvolutionCandidate,
) {
    if is_better_candidate(candidate, global_best.as_ref()) {
        *global_best = Some(candidate.clone());
    }
}

fn is_better_candidate(
    candidate: &HdcEvolutionCandidate,
    current_best: Option<&HdcEvolutionCandidate>,
) -> bool {
    let Some(best) = current_best else {
        return true;
    };

    match candidate.score.total_cmp(&best.score) {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }
    match best.risk_score.total_cmp(&candidate.risk_score) {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }
    if candidate.verifier_pass != best.verifier_pass {
        return candidate.verifier_pass;
    }
    candidate.execute_micros < best.execute_micros
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::hdc::HolographicMemory;
    use crate::jit::JitEngine;
    use crate::models::HdcEvolutionMutateRequest;
    use crate::verifier::Verifier;
    use crate::world_model::WorldModel;

    use super::{MutationEngine, update_global_best};

    fn candidate(id: &str, generation: usize, score: f32) -> crate::models::HdcEvolutionCandidate {
        crate::models::HdcEvolutionCandidate {
            candidate_id: id.to_string(),
            generation,
            score,
            output: 1,
            verifier_pass: true,
            verifier_violations: Vec::new(),
            risk_score: 0.2,
            confidence: 0.8,
            compile_micros: 20,
            execute_micros: 10,
            logic: "add 1".to_string(),
        }
    }

    #[test]
    fn mutation_engine_evolves_and_returns_candidates() {
        let engine = MutationEngine::new(
            Arc::new(HolographicMemory::new()),
            Arc::new(JitEngine::new().expect("jit")),
            Arc::new(WorldModel::new()),
            Arc::new(Verifier::new().expect("verifier")),
        );
        let response = engine
            .evolve_templates(&HdcEvolutionMutateRequest {
                template_id: Some("python_secure_v1".to_string()),
                query: None,
                generations: Some(2),
                population_size: Some(4),
                survivors: Some(2),
                risk_threshold: Some(0.8),
                seed_input: Some(11),
                promote: Some(true),
            })
            .expect("evolve templates");
        assert_eq!(response.generations_run, 2);
        assert_eq!(response.population_size, 4);
        assert!(!response.final_candidates.is_empty());
    }

    #[test]
    fn evolution_elitism_keeps_global_best_from_early_generation() {
        let g1_best = candidate("g1-best", 1, 3.2);
        let g2_worse = candidate("g2-worse", 2, 1.8);
        let g3_worse = candidate("g3-worse", 3, 1.1);

        let mut global_best = None;
        update_global_best(&mut global_best, &g1_best);
        update_global_best(&mut global_best, &g2_worse);
        update_global_best(&mut global_best, &g3_worse);

        let elite = global_best.expect("global best present");
        assert_eq!(elite.candidate_id, "g1-best");
        assert_eq!(elite.generation, 1);
    }
}
