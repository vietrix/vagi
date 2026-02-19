use std::collections::HashMap;

use petgraph::graph::{DiGraph, NodeIndex};

use crate::models::WorldSimulateResponse;

pub struct WorldModel {
    graph: DiGraph<&'static str, f32>,
    nodes: HashMap<&'static str, NodeIndex>,
}

impl WorldModel {
    pub fn new() -> Self {
        let mut graph = DiGraph::<&'static str, f32>::new();
        let mut nodes = HashMap::new();

        for name in [
            "ReceiveInput",
            "ValidateInput",
            "HashSecret",
            "QueryDatabase",
            "IssueToken",
            "PersistAudit",
        ] {
            let idx = graph.add_node(name);
            nodes.insert(name, idx);
        }

        let edges = [
            ("ReceiveInput", "ValidateInput", 0.9),
            ("ValidateInput", "HashSecret", 0.8),
            ("HashSecret", "QueryDatabase", 0.85),
            ("QueryDatabase", "IssueToken", 0.8),
            ("IssueToken", "PersistAudit", 0.7),
        ];

        for (from, to, weight) in edges {
            graph.add_edge(nodes[from], nodes[to], weight);
        }

        Self { graph, nodes }
    }

    pub fn simulate(&self, action: &str) -> WorldSimulateResponse {
        let lower = action.to_lowercase();
        let mut risk = 0.18_f32;
        let mut effects = vec!["Bắt đầu mô phỏng nhân quả theo đồ thị khái niệm.".to_string()];
        let path = self.estimate_causal_path(&lower);

        if contains_any(&lower, &["drop", "delete", "rm -rf", "truncate"]) {
            risk += 0.52;
            effects.push("Phát hiện hành vi xóa dữ liệu, rủi ro toàn vẹn tăng mạnh.".to_string());
        }

        if contains_any(&lower, &["unsafe", "eval(", "exec("]) {
            risk += 0.28;
            effects.push("Thực thi không an toàn có thể phá vỡ sandbox boundary.".to_string());
        }

        if contains_any(&lower, &["hash", "sanitize", "validate", "timeout", "rate limit"]) {
            risk -= 0.12;
            effects.push("Có cơ chế phòng thủ làm giảm xác suất lỗi hệ thống.".to_string());
        }

        let risk_score = risk.clamp(0.01, 0.99);
        let confidence = (1.0 - risk_score).clamp(0.01, 0.99);
        effects.push(format!(
            "Risk score={risk_score:.2}, confidence={confidence:.2}"
        ));

        WorldSimulateResponse {
            risk_score,
            confidence,
            predicted_effects: effects,
            causal_path: path,
        }
    }

    fn estimate_causal_path(&self, action: &str) -> Vec<String> {
        let mut path = vec!["ReceiveInput".to_string(), "ValidateInput".to_string()];
        if action.contains("hash") || action.contains("password") {
            path.push("HashSecret".to_string());
        }
        if action.contains("db") || action.contains("query") || action.contains("login") {
            path.push("QueryDatabase".to_string());
        }
        if action.contains("token") || action.contains("session") || action.contains("login") {
            path.push("IssueToken".to_string());
        }
        path.push("PersistAudit".to_string());
        path
    }

    pub fn graph_node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn has_anchor_nodes(&self) -> bool {
        self.nodes.contains_key("ReceiveInput") && self.nodes.contains_key("PersistAudit")
    }

    /// Predict experts likely needed in future decode steps.
    ///
    /// Returns deterministic expert IDs for lookahead prefetch.
    pub fn predict_expert_prefetch(&self, token_ids: &[u32], lookahead: usize) -> Vec<String> {
        let horizon = lookahead.clamp(1, 8);
        let mut out = Vec::with_capacity(horizon);
        let mut seed = token_ids.iter().fold(0_u64, |acc, t| {
            acc.wrapping_mul(0x9E3779B185EBCA87).wrapping_add(*t as u64 + 1)
        });
        if seed == 0 {
            seed = 0xA0761D6478BD642F;
        }
        for step in 0..horizon {
            seed = seed.rotate_left(9).wrapping_add((step as u64 + 1) * 0x94D049BB133111EB);
            let expert = (seed % 16) as usize;
            out.push(format!("expert_{expert:02}"));
        }
        out
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

#[cfg(test)]
mod tests {
    use super::WorldModel;

    #[test]
    fn delete_actions_have_higher_risk() {
        let model = WorldModel::new();
        let risky = model.simulate("drop table users");
        let safe = model.simulate("validate input and hash password");
        assert!(risky.risk_score > safe.risk_score);
    }

    #[test]
    fn graph_contains_expected_nodes() {
        let model = WorldModel::new();
        assert!(model.graph_node_count() >= 6);
        assert!(model.has_anchor_nodes());
    }

    #[test]
    fn prefetch_prediction_returns_lookahead() {
        let model = WorldModel::new();
        let experts = model.predict_expert_prefetch(&[1, 2, 3], 3);
        assert_eq!(experts.len(), 3);
    }
}
