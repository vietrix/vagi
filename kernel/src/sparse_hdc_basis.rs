//! Sparse HDC basis projection with XOR + popcount.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const HDC_WORDS: usize = 64; // 4096 bits

#[derive(Debug, Clone)]
pub struct SparseHdcBasis {
    lut: HashMap<String, [u64; HDC_WORDS]>,
}

impl SparseHdcBasis {
    pub fn new(tokens: &[String], seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut lut = HashMap::with_capacity(tokens.len());
        for token in tokens {
            let mut hv = [0_u64; HDC_WORDS];
            for word in &mut hv {
                *word = rng.random();
            }
            lut.insert(token.clone(), hv);
        }
        Self { lut }
    }

    pub fn project_to_hdc(&self, tokens: &[String]) -> [u64; HDC_WORDS] {
        let mut out = [0_u64; HDC_WORDS];
        if tokens.is_empty() {
            return out;
        }

        for (idx, token) in tokens.iter().enumerate() {
            if let Some(hv) = self.lut.get(token) {
                for word_idx in 0..HDC_WORDS {
                    // bind token with lightweight positional permutation.
                    let rotated = hv[word_idx].rotate_left((idx % 63) as u32);
                    out[word_idx] ^= rotated;
                }
            }
        }
        out
    }

    pub fn cosine_like_similarity(a: &[u64; HDC_WORDS], b: &[u64; HDC_WORDS]) -> f32 {
        let mut hamming = 0_u32;
        for idx in 0..HDC_WORDS {
            hamming += (a[idx] ^ b[idx]).count_ones();
        }
        let dim = (HDC_WORDS * 64) as f32;
        1.0 - 2.0 * (hamming as f32 / dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_and_compare() {
        let tokens = vec!["secure".to_string(), "login".to_string(), "nonce".to_string()];
        let basis = SparseHdcBasis::new(&tokens, 7);
        let a = basis.project_to_hdc(&["secure".to_string(), "login".to_string()]);
        let b = basis.project_to_hdc(&["secure".to_string(), "login".to_string()]);
        let c = basis.project_to_hdc(&["nonce".to_string()]);
        assert!(SparseHdcBasis::cosine_like_similarity(&a, &b) > 0.95);
        assert!(SparseHdcBasis::cosine_like_similarity(&a, &c) < 0.5);
    }
}
