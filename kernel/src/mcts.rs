//! Monte Carlo Tree Search engine for multi-branch token generation.
//!
//! Replaces the single-path greedy `argmax` decode with a deliberate tree search
//! that explores **N branches**, evaluates each via the world model + verifier,
//! and returns the highest-reward path.

use std::collections::HashSet;
use std::time::Instant;

use anyhow::Result;

use crate::model_runtime::{LoadedModel, argmax_with_exclusions, softmax_top_k};
use crate::verifier::Verifier;
use crate::world_model::WorldModel;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Tunable parameters for the MCTS search.
#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// Number of top-k tokens to expand at each node (branching factor).
    pub num_branches: usize,
    /// Maximum depth (tokens) per simulation rollout.
    pub max_depth: usize,
    /// UCB1 exploration constant (√2 ≈ 1.414 is the default).
    pub exploration_c: f32,
    /// Maximum number of rollout simulations before returning.
    pub simulation_budget: usize,
    /// Temperature for softmax sampling during expansion.
    pub temperature: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_branches: 3,
            max_depth: 64,
            exploration_c: 1.414,
            simulation_budget: 30,
            temperature: 0.8,
        }
    }
}

// ─── Tree Node ───────────────────────────────────────────────────────────────

/// A node in the MCTS search tree.
#[derive(Debug, Clone)]
struct MctsNode {
    /// Token ID that led to this node (None for the root).
    token_id: Option<usize>,
    /// Accumulated token sequence from root to this node.
    token_sequence: Vec<usize>,
    /// Snapshot of hidden states at this node.
    hidden_snapshot: Vec<Vec<f32>>,
    /// Total accumulated reward from all rollouts through this node.
    total_reward: f64,
    /// Number of times this node has been visited.
    visit_count: u32,
    /// Child indices in the arena.
    children: Vec<usize>,
    /// Whether this node has been expanded.
    expanded: bool,
}

impl MctsNode {
    fn new(
        token_id: Option<usize>,
        token_sequence: Vec<usize>,
        hidden_snapshot: Vec<Vec<f32>>,
    ) -> Self {
        Self {
            token_id,
            token_sequence,
            hidden_snapshot,
            total_reward: 0.0,
            visit_count: 0,
            children: Vec::new(),
            expanded: false,
        }
    }

    /// UCB1 score for node selection. Higher = more attractive to explore.
    fn ucb1(&self, parent_visits: u32, c: f32) -> f64 {
        if self.visit_count == 0 {
            return f64::INFINITY;
        }
        let exploitation = self.total_reward / self.visit_count as f64;
        let exploration =
            c as f64 * ((parent_visits as f64).ln() / self.visit_count as f64).sqrt();
        exploitation + exploration
    }
}

// ─── Search Result ───────────────────────────────────────────────────────────

/// The result of an MCTS search.
#[derive(Debug, Clone)]
pub struct MctsResult {
    /// The decoded text from the best branch.
    pub text: String,
    /// Token IDs of the best branch.
    pub token_ids: Vec<usize>,
    /// Total branches explored.
    pub branches_explored: usize,
    /// Reward of the best branch (0.0 – 1.3).
    pub best_reward: f32,
    /// Time taken for the search.
    pub latency_ms: u64,
}

// ─── Engine ──────────────────────────────────────────────────────────────────

/// The MCTS engine. Stateless — all state lives in the tree per search call.
#[derive(Clone)]
pub struct MctsEngine {
    pub config: MctsConfig,
}

impl MctsEngine {
    pub fn new(config: MctsConfig) -> Self {
        Self { config }
    }

    /// Run MCTS search starting from a prompt's hidden state.
    ///
    /// # Arguments
    /// * `model` — the loaded language model
    /// * `initial_hidden` — hidden state after encoding the prompt
    /// * `initial_logits` — logits from the last prompt token
    /// * `verifier` — for reward scoring
    /// * `world_model` — for reward scoring
    pub fn search(
        &self,
        model: &LoadedModel,
        initial_hidden: &[Vec<f32>],
        initial_logits: &[f32],
        verifier: &Verifier,
        world_model: &WorldModel,
    ) -> Result<MctsResult> {
        let started = Instant::now();

        // Arena-based tree storage.
        let mut arena: Vec<MctsNode> = Vec::with_capacity(256);
        let root_idx = 0;
        arena.push(MctsNode::new(None, Vec::new(), initial_hidden.to_vec()));

        let excluded: HashSet<usize> = [
            model.manifest.bos_id,
            model.manifest.pad_id,
        ]
        .into_iter()
        .collect();

        // Initial expansion of root.
        self.expand(
            root_idx,
            initial_logits,
            model,
            &excluded,
            &mut arena,
        )?;

        // Run simulation budget.
        for _ in 0..self.config.simulation_budget {
            // SELECT: walk tree from root using UCB1.
            let leaf_idx = self.select(root_idx, &arena);

            // EXPAND: if leaf has been visited, expand it.
            if arena[leaf_idx].visit_count > 0 && !arena[leaf_idx].expanded {
                let mut hidden = arena[leaf_idx].hidden_snapshot.clone();
                if let Some(tid) = arena[leaf_idx].token_id {
                    let logits = model.forward_one_token(tid, &mut hidden)?;
                    self.expand(leaf_idx, &logits, model, &excluded, &mut arena)?;
                    arena[leaf_idx].hidden_snapshot = hidden;
                }
            }

            // SIMULATE: greedy rollout from leaf.
            let rollout_text =
                self.simulate_rollout(leaf_idx, model, &arena)?;

            // SCORE: reward = confidence * (1 - risk) + verifier bonus.
            let reward = self.compute_reward(&rollout_text, verifier, world_model);

            // BACKPROPAGATE: update all ancestors.
            self.backpropagate(leaf_idx, reward, &mut arena);
        }

        // Pick the child of root with highest average reward.
        let best = self.best_child(root_idx, &arena);
        let best_sequence = arena[best].token_sequence.clone();
        let best_text = model.decode(&best_sequence);
        let best_reward = if arena[best].visit_count > 0 {
            (arena[best].total_reward / arena[best].visit_count as f64) as f32
        } else {
            0.0
        };

        Ok(MctsResult {
            text: best_text,
            token_ids: best_sequence,
            branches_explored: arena.len(),
            best_reward,
            latency_ms: started.elapsed().as_millis() as u64,
        })
    }

    // ── Select ───────────────────────────────────────────────────────────

    /// Walk from `node_idx` to a leaf using UCB1 selection.
    fn select(&self, node_idx: usize, arena: &[MctsNode]) -> usize {
        let node = &arena[node_idx];
        if node.children.is_empty() {
            return node_idx;
        }
        let parent_visits = node.visit_count.max(1);
        let best_child = node
            .children
            .iter()
            .max_by(|&&a, &&b| {
                let ua = arena[a].ucb1(parent_visits, self.config.exploration_c);
                let ub = arena[b].ucb1(parent_visits, self.config.exploration_c);
                ua.partial_cmp(&ub).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(node_idx);
        self.select(best_child, arena)
    }

    // ── Expand ───────────────────────────────────────────────────────────

    /// Expand a node: create children for top-k tokens from the given logits.
    fn expand(
        &self,
        node_idx: usize,
        logits: &[f32],
        model: &LoadedModel,
        excluded: &HashSet<usize>,
        arena: &mut Vec<MctsNode>,
    ) -> Result<()> {
        if arena[node_idx].expanded {
            return Ok(());
        }

        let candidates =
            softmax_top_k(logits, self.config.num_branches + excluded.len(), self.config.temperature);

        let parent_seq = arena[node_idx].token_sequence.clone();
        let parent_hidden = arena[node_idx].hidden_snapshot.clone();

        let mut child_indices = Vec::new();
        for (token_id, _prob) in candidates {
            if excluded.contains(&token_id) {
                continue;
            }
            if child_indices.len() >= self.config.num_branches {
                break;
            }

            let mut child_seq = parent_seq.clone();
            child_seq.push(token_id);

            // Forward pass to get child's hidden state.
            let mut child_hidden = parent_hidden.clone();
            model.forward_one_token(token_id, &mut child_hidden)?;

            let child_node = MctsNode::new(Some(token_id), child_seq, child_hidden);
            let child_idx = arena.len();
            arena.push(child_node);
            child_indices.push(child_idx);
        }

        arena[node_idx].children = child_indices;
        arena[node_idx].expanded = true;
        Ok(())
    }

    // ── Simulate ─────────────────────────────────────────────────────────

    /// Greedy rollout from a leaf node to generate text for scoring.
    fn simulate_rollout(
        &self,
        node_idx: usize,
        model: &LoadedModel,
        arena: &[MctsNode],
    ) -> Result<String> {
        let node = &arena[node_idx];
        let mut hidden = node.hidden_snapshot.clone();
        let mut generated = node.token_sequence.clone();
        let excluded = [
            model.manifest.bos_id,
            model.manifest.pad_id,
            model.manifest.eos_id,
        ];

        // If node has a token, get logits from it; otherwise use last generated.
        let mut logits = if let Some(tid) = node.token_id {
            model.forward_one_token(tid, &mut hidden)?
        } else if let Some(&last) = generated.last() {
            model.forward_one_token(last, &mut hidden)?
        } else {
            // Root node with no tokens yet — return empty text.
            return Ok(String::new());
        };

        let remaining = self.config.max_depth.saturating_sub(generated.len());
        for _ in 0..remaining {
            let next = argmax_with_exclusions(&logits, &excluded);
            if next == model.manifest.eos_id {
                break;
            }
            generated.push(next);
            logits = model.forward_one_token(next, &mut hidden)?;
        }

        Ok(model.decode(&generated))
    }

    // ── Reward ───────────────────────────────────────────────────────────

    /// Compute reward for a rollout text using the verifier and world model.
    fn compute_reward(
        &self,
        text: &str,
        verifier: &Verifier,
        world_model: &WorldModel,
    ) -> f64 {
        if text.trim().is_empty() {
            return 0.0;
        }

        // World model evaluation.
        let sim = world_model.simulate(text);
        let confidence = sim.confidence as f64;
        let risk = sim.risk_score as f64;

        // Verifier evaluation.
        let verifier_result = verifier.check(&crate::models::VerifierRequest {
            patch_ir: text.to_string(),
            max_loop_iters: Some(128),
            side_effect_budget: Some(3),
            timeout_ms: Some(30),
        });

        let base_reward = confidence * (1.0 - risk);
        let verifier_bonus = if verifier_result.pass { 0.3 } else { -0.1 };

        (base_reward + verifier_bonus).clamp(0.0, 1.3)
    }

    // ── Backpropagate ────────────────────────────────────────────────────

    /// Update visit counts and rewards from leaf back to root.
    fn backpropagate(&self, node_idx: usize, reward: f64, arena: &mut [MctsNode]) {
        arena[node_idx].visit_count += 1;
        arena[node_idx].total_reward += reward;

        // Walk up via sequence length to find parent (arena-based approach).
        // Since children always have longer sequences, we find the parent
        // by searching for the node whose children include our index.
        // For efficiency, we do a simple linear scan since trees are small.
        for parent_idx in (0..node_idx).rev() {
            if arena[parent_idx].children.contains(&node_idx) {
                self.backpropagate(parent_idx, reward, arena);
                break;
            }
        }
    }

    // ── Best Child ───────────────────────────────────────────────────────

    /// Pick the child with the highest average reward (exploitation only).
    fn best_child(&self, node_idx: usize, arena: &[MctsNode]) -> usize {
        let node = &arena[node_idx];
        if node.children.is_empty() {
            return node_idx;
        }
        node.children
            .iter()
            .max_by(|&&a, &&b| {
                let avg_a = if arena[a].visit_count > 0 {
                    arena[a].total_reward / arena[a].visit_count as f64
                } else {
                    0.0
                };
                let avg_b = if arena[b].visit_count > 0 {
                    arena[b].total_reward / arena[b].visit_count as f64
                } else {
                    0.0
                };
                avg_a.partial_cmp(&avg_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(node_idx)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ucb1_unvisited_is_infinity() {
        let node = MctsNode::new(Some(0), vec![], vec![]);
        assert!(node.ucb1(10, 1.414).is_infinite());
    }

    #[test]
    fn ucb1_increases_with_fewer_visits() {
        let mut node_a = MctsNode::new(Some(0), vec![], vec![]);
        node_a.visit_count = 10;
        node_a.total_reward = 5.0;

        let mut node_b = MctsNode::new(Some(1), vec![], vec![]);
        node_b.visit_count = 1;
        node_b.total_reward = 0.5;

        // With equal average reward, the less-visited node should have higher UCB1.
        let ucb_a = node_a.ucb1(100, 1.414);
        let ucb_b = node_b.ucb1(100, 1.414);
        // node_b has exploration advantage.
        assert!(ucb_b > ucb_a);
    }

    #[test]
    fn mcts_config_defaults_are_reasonable() {
        let config = MctsConfig::default();
        assert_eq!(config.num_branches, 3);
        assert_eq!(config.max_depth, 64);
        assert!(config.exploration_c > 1.0 && config.exploration_c < 2.0);
        assert!(config.simulation_budget > 0);
    }
}
