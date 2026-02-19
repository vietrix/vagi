//! Data distillation pipeline for high-logic corpora.

use regex::Regex;

#[derive(Debug, Clone)]
pub struct DistilledSample {
    pub text: String,
    pub logic_score: f32,
    pub source_tag: String,
}

pub struct DataDistiller {
    min_logic_score: f32,
    code_hint: Regex,
    math_hint: Regex,
}

impl DataDistiller {
    pub fn new(min_logic_score: f32) -> Self {
        Self {
            min_logic_score,
            code_hint: Regex::new(r"(?m)(fn\s+\w+|class\s+\w+|def\s+\w+|=>|\{|\}|;)")
                .expect("valid code regex"),
            math_hint: Regex::new(r"(\d+\s*[+\-*/=]\s*\d+|theorem|proof|lemma|integral)")
                .expect("valid math regex"),
        }
    }

    pub fn distill(&self, samples: &[String]) -> Vec<DistilledSample> {
        let mut out = Vec::new();
        for sample in samples {
            let score = self.logic_score(sample);
            if score >= self.min_logic_score {
                out.push(DistilledSample {
                    text: sample.clone(),
                    logic_score: score,
                    source_tag: self.tag(sample),
                });
            }
        }
        out
    }

    pub fn logic_score(&self, text: &str) -> f32 {
        let lower = text.to_lowercase();
        let mut score: f32 = 0.0;
        if self.code_hint.is_match(text) {
            score += 0.45;
        }
        if self.math_hint.is_match(&lower) {
            score += 0.35;
        }
        let structure_markers = ["because", "therefore", "if", "then", "assert", "verify"];
        for marker in structure_markers {
            if lower.contains(marker) {
                score += 0.06;
            }
        }
        score.min(1.0)
    }

    fn tag(&self, text: &str) -> String {
        if self.code_hint.is_match(text) {
            return "code".to_string();
        }
        if self.math_hint.is_match(&text.to_lowercase()) {
            return "math".to_string();
        }
        "structured_text".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_high_logic_samples() {
        let distiller = DataDistiller::new(0.5);
        let samples = vec![
            "fn solve(x: i32) -> i32 { x + 1 }".to_string(),
            "hello world".to_string(),
        ];
        let out = distiller.distill(&samples);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].source_tag, "code");
    }
}
