//! Hyperdimensional Computing (HDC) module for parallel logic verification.
//!
//! Uses high-dimensional binary vectors (hypervectors) to encode and classify
//! code patterns. Runs alongside the existing static checks in the Verifier as
//! a complementary, constant-time pattern matching engine.
//!
//! Core operations:
//! - **Bind** (XOR): combines two concepts into a new hypervector
//! - **Bundle** (majority vote): creates a set-like union of hypervectors
//! - **Permute** (cyclic shift): encodes position/sequence information
//! - **Similarity** (hamming → cosine): measures semantic closeness

use std::collections::HashMap;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Dimension of hypervectors in bits.
pub const HDC_DIM: usize = 8192;

/// Number of u64 words needed for HDC_DIM bits.
const WORDS: usize = HDC_DIM / 64;

// ─── HyperVector ─────────────────────────────────────────────────────────────

/// Binary hypervector stored as a fixed-length bit array.
///
/// All operations are constant-time O(HDC_DIM/64) — trivially parallelizable
/// and cache-friendly due to the compact bit representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HyperVector {
    bits: [u64; WORDS],
}

impl HyperVector {
    /// Create an all-zeros hypervector.
    pub fn zeros() -> Self {
        Self { bits: [0u64; WORDS] }
    }

    /// Create a random binary hypervector from a deterministic seed.
    pub fn random(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut bits = [0u64; WORDS];
        for word in bits.iter_mut() {
            *word = rng.random();
        }
        Self { bits }
    }

    /// **Bind** operation: XOR. Combines two concepts into a dissimilar vector.
    /// Properties: self-inverse, distributes over bundle.
    pub fn bind(&self, other: &Self) -> Self {
        let mut bits = [0u64; WORDS];
        for i in 0..WORDS {
            bits[i] = self.bits[i] ^ other.bits[i];
        }
        Self { bits }
    }

    /// **Bundle** (majority vote): creates a set-like union.
    /// For each bit position, output 1 if majority of inputs have 1.
    pub fn bundle(vecs: &[Self]) -> Self {
        if vecs.is_empty() {
            return Self::zeros();
        }
        if vecs.len() == 1 {
            return vecs[0].clone();
        }

        let threshold = vecs.len() / 2;
        let mut bits = [0u64; WORDS];

        for word_idx in 0..WORDS {
            let mut result_word = 0u64;
            for bit in 0..64 {
                let mask = 1u64 << bit;
                let count = vecs.iter().filter(|v| v.bits[word_idx] & mask != 0).count();
                if count > threshold {
                    result_word |= mask;
                }
            }
            bits[word_idx] = result_word;
        }

        Self { bits }
    }

    /// **Permute**: cyclic left shift by `shifts` positions.
    /// Encodes positional/sequence information.
    pub fn permute(&self, shifts: usize) -> Self {
        let shifts = shifts % HDC_DIM;
        if shifts == 0 {
            return self.clone();
        }

        let word_shift = shifts / 64;
        let bit_shift = shifts % 64;

        let mut bits = [0u64; WORDS];
        for i in 0..WORDS {
            let src = (i + WORDS - word_shift) % WORDS;
            if bit_shift == 0 {
                bits[i] = self.bits[src];
            } else {
                let prev = (src + WORDS - 1) % WORDS;
                bits[i] = (self.bits[src] << bit_shift) | (self.bits[prev] >> (64 - bit_shift));
            }
        }

        Self { bits }
    }

    /// Hamming distance between two hypervectors.
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        let mut dist = 0u32;
        for i in 0..WORDS {
            dist += (self.bits[i] ^ other.bits[i]).count_ones();
        }
        dist
    }

    /// Cosine-like similarity derived from Hamming distance.
    /// Range: [-1.0, 1.0]. 1.0 = identical, 0.0 = random, -1.0 = complementary.
    pub fn similarity(&self, other: &Self) -> f32 {
        let hamming = self.hamming_distance(other) as f32;
        1.0 - 2.0 * hamming / HDC_DIM as f32
    }

    /// Count of set bits (population count).
    pub fn popcount(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }
}

// ─── HDC Verifier ────────────────────────────────────────────────────────────

/// HDC-based pattern matcher for code verification.
///
/// Encodes code IR into a hypervector and compares it against prototype
/// violation patterns. High similarity → violation detected.
pub struct HdcVerifier {
    /// Codebook: maps token strings to their random hypervectors.
    codebook: HashMap<String, HyperVector>,
    /// Prototype violation patterns.
    prototypes: Vec<(String, HyperVector)>,
}

/// Recursive HDC compressor that maintains a master hypervector over episodes.
pub struct RecursiveHdcMemory {
    codebook: HashMap<String, HyperVector>,
    master: HyperVector,
    recency: f32,
}

impl RecursiveHdcMemory {
    pub fn new(recency: f32) -> Self {
        Self {
            codebook: HashMap::new(),
            master: HyperVector::zeros(),
            recency: recency.clamp(0.0, 1.0),
        }
    }

    pub fn master(&self) -> &HyperVector {
        &self.master
    }

    pub fn master_popcount(&self) -> u32 {
        self.master.popcount()
    }

    fn get_or_create(&mut self, token: &str) -> HyperVector {
        if let Some(hv) = self.codebook.get(token) {
            return hv.clone();
        }
        let mut seed = 0xcbf29ce484222325u64;
        for b in token.bytes() {
            seed ^= b as u64;
            seed = seed.wrapping_mul(0x100000001b3);
        }
        let hv = HyperVector::random(seed);
        self.codebook.insert(token.to_string(), hv.clone());
        hv
    }

    fn encode_text(&mut self, text: &str) -> HyperVector {
        let tokens: Vec<String> = text
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|t| !t.is_empty())
            .map(|s| s.to_lowercase())
            .collect();
        if tokens.is_empty() {
            return HyperVector::zeros();
        }
        let mut vecs = Vec::with_capacity(tokens.len());
        for (idx, tok) in tokens.iter().enumerate() {
            vecs.push(self.get_or_create(tok).permute(idx % 64));
        }
        HyperVector::bundle(&vecs)
    }

    /// Recursively bundle new episode into master vector.
    pub fn ingest_episode(&mut self, episode: &str) {
        let encoded = self.encode_text(episode);
        let mixed = HyperVector::bundle(&[self.master.clone(), encoded]);
        if self.recency >= 0.999 {
            self.master = mixed;
            return;
        }
        // Cheap recency control by blending via repeated bundling of old state.
        let repeat_old = (self.recency * 4.0).round() as usize;
        let mut pool = vec![mixed];
        for _ in 0..repeat_old {
            pool.push(self.master.clone());
        }
        self.master = HyperVector::bundle(&pool);
    }

    /// Bind cue with master and report relevance score.
    pub fn bind_query_relevance(&mut self, cue: &str) -> f32 {
        let cue_vec = self.encode_text(cue);
        let unlocked = self.master.bind(&cue_vec);
        unlocked.similarity(&cue_vec)
    }
}

impl HdcVerifier {
    /// Create the HDC verifier with built-in violation prototypes.
    pub fn new() -> Self {
        let mut verifier = Self {
            codebook: HashMap::new(),
            prototypes: Vec::new(),
        };
        verifier.build_prototypes();
        verifier
    }

    /// Get or create a codebook entry for a token.
    fn get_or_create(&mut self, token: &str) -> HyperVector {
        if let Some(hv) = self.codebook.get(token) {
            return hv.clone();
        }
        // Deterministic seed from token content.
        let seed = {
            let mut hash = 0xcbf29ce484222325u64;
            for byte in token.bytes() {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            hash
        };
        let hv = HyperVector::random(seed);
        self.codebook.insert(token.to_string(), hv.clone());
        hv
    }

    /// Build prototype hypervectors for known violation patterns.
    fn build_prototypes(&mut self) {
        let patterns = [
            (
                "infinite_loop",
                &["while", "true", "loop", "break", "never", "infinite"][..],
            ),
            (
                "unsafe_access",
                &["unsafe", "raw", "pointer", "transmute", "unchecked"],
            ),
            (
                "side_effect_overflow",
                &["write_file", "network_call", "spawn_process", "exec", "eval"],
            ),
            (
                "injection_risk",
                &["eval", "exec", "shell", "sql", "inject", "format"],
            ),
            (
                "resource_exhaustion",
                &["allocate", "unbounded", "oom", "memory", "stack", "overflow"],
            ),
        ];

        for (name, tokens) in patterns {
            let mut components = Vec::new();
            for (pos, token) in tokens.iter().enumerate() {
                let token_hv = self.get_or_create(token);
                let positioned = token_hv.permute(pos);
                components.push(positioned);
            }
            let prototype = HyperVector::bundle(&components);
            self.prototypes.push((name.to_string(), prototype));
        }
    }

    /// Encode a patch IR string into a hypervector.
    pub fn encode_patch(&mut self, patch_ir: &str) -> HyperVector {
        let tokens: Vec<&str> = patch_ir
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|t| !t.is_empty())
            .collect();

        if tokens.is_empty() {
            return HyperVector::zeros();
        }

        let mut components = Vec::with_capacity(tokens.len());
        for (pos, token) in tokens.iter().enumerate() {
            let lower = token.to_lowercase();
            let token_hv = self.get_or_create(&lower);
            let positioned = token_hv.permute(pos % 64); // Wrap position to prevent drift.
            components.push(positioned);
        }

        HyperVector::bundle(&components)
    }

    /// Classify an encoded vector against all prototypes.
    /// Returns (pattern_name, similarity) pairs, sorted by similarity descending.
    pub fn classify(&self, encoded: &HyperVector) -> Vec<(String, f32)> {
        let mut scores: Vec<(String, f32)> = self
            .prototypes
            .iter()
            .map(|(name, proto)| (name.clone(), encoded.similarity(proto)))
            .collect();
        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Check a patch IR for violations. Returns violation names that exceed
    /// the similarity threshold.
    pub fn check(&mut self, patch_ir: &str, threshold: f32) -> Vec<String> {
        let encoded = self.encode_patch(patch_ir);
        self.classify(&encoded)
            .into_iter()
            .filter(|(_, score)| *score > threshold)
            .map(|(name, _)| format!("hdc:{name}"))
            .collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypervector_self_similarity_is_one() {
        let hv = HyperVector::random(42);
        assert!((hv.similarity(&hv) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hypervector_random_vectors_near_zero_similarity() {
        let a = HyperVector::random(1);
        let b = HyperVector::random(2);
        let sim = a.similarity(&b);
        // Random binary vectors should have ~0 cosine similarity.
        assert!(sim.abs() < 0.1, "similarity was {sim}, expected near 0");
    }

    #[test]
    fn bind_is_self_inverse() {
        let a = HyperVector::random(10);
        let b = HyperVector::random(20);
        let bound = a.bind(&b);
        let recovered = bound.bind(&b);
        assert_eq!(a, recovered);
    }

    #[test]
    fn bundle_preserves_similarity() {
        let a = HyperVector::random(100);
        let b = HyperVector::random(200);
        let c = HyperVector::random(300);
        let bundled = HyperVector::bundle(&[a.clone(), b.clone(), c.clone()]);

        // Bundled vector should be more similar to its components than to random.
        let sim_a = bundled.similarity(&a);
        let sim_random = bundled.similarity(&HyperVector::random(999));
        assert!(sim_a > sim_random, "bundle should be similar to component a");
    }

    #[test]
    fn permute_changes_vector() {
        let hv = HyperVector::random(50);
        let permuted = hv.permute(1);
        assert_ne!(hv, permuted);
    }

    #[test]
    fn permute_zero_is_identity() {
        let hv = HyperVector::random(60);
        let permuted = hv.permute(0);
        assert_eq!(hv, permuted);
    }

    #[test]
    fn hdc_verifier_detects_unsafe() {
        let mut verifier = HdcVerifier::new();
        let violations = verifier.check("unsafe { raw_pointer.transmute() }", 0.05);
        // Should detect unsafe_access pattern.
        let has_unsafe = violations.iter().any(|v| v.contains("unsafe"));
        assert!(has_unsafe, "should detect unsafe: violations = {violations:?}");
    }

    #[test]
    fn hdc_verifier_safe_code_has_fewer_violations() {
        let mut verifier = HdcVerifier::new();
        let safe_violations = verifier.check("validate input and hash password", 0.15);
        let unsafe_violations = verifier.check("unsafe transmute raw pointer unchecked exec eval", 0.15);
        assert!(
            safe_violations.len() <= unsafe_violations.len(),
            "safe code should have fewer violations: safe={safe_violations:?} unsafe={unsafe_violations:?}"
        );
    }

    #[test]
    fn recursive_memory_ingests_and_queries() {
        let mut mem = RecursiveHdcMemory::new(0.85);
        mem.ingest_episode("compile auth validator token issue");
        mem.ingest_episode("run logic verifier and causal check");
        let rel = mem.bind_query_relevance("verifier logic");
        assert!(rel.is_finite());
        assert!(mem.master_popcount() > 0);
    }
}
