//! Bit-LKAN primitives for CPU-first inference.
//!
//! Focuses on ternary 1.58-bit representation and integer-like matmul kernels.

use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernaryWeight {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl TernaryWeight {
    #[inline]
    pub fn encode_2bit(self) -> u8 {
        match self {
            Self::Zero => 0b00,
            Self::Pos => 0b01,
            Self::Neg => 0b10,
        }
    }

    #[inline]
    pub fn decode_2bit(bits: u8) -> Self {
        match bits & 0b11 {
            0b01 => Self::Pos,
            0b10 => Self::Neg,
            _ => Self::Zero,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PackedTernary {
    packed: Vec<u64>,
    len: usize,
}

impl PackedTernary {
    pub fn from_weights(weights: &[TernaryWeight]) -> Self {
        let words = (weights.len() + 31) / 32;
        let mut packed = vec![0_u64; words];
        for (idx, w) in weights.iter().enumerate() {
            let word_idx = idx / 32;
            let shift = (idx % 32) * 2;
            packed[word_idx] |= (u64::from(w.encode_2bit())) << shift;
        }
        Self {
            packed,
            len: weights.len(),
        }
    }

    pub fn get(&self, idx: usize) -> TernaryWeight {
        if idx >= self.len {
            return TernaryWeight::Zero;
        }
        let word = self.packed[idx / 32];
        let shift = (idx % 32) * 2;
        TernaryWeight::decode_2bit(((word >> shift) & 0b11) as u8)
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

/// Quantize f32 weights to ternary {-1, 0, 1} using a 1.58-bit style threshold.
pub fn quantize_1_58b(weights: &[f32], threshold: f32) -> Vec<TernaryWeight> {
    let thr = threshold.abs().max(1e-6);
    weights
        .iter()
        .map(|w| {
            if *w > thr {
                TernaryWeight::Pos
            } else if *w < -thr {
                TernaryWeight::Neg
            } else {
                TernaryWeight::Zero
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct BitLkanMatrix {
    rows: usize,
    cols: usize,
    row_data: Vec<PackedTernary>,
}

impl BitLkanMatrix {
    pub fn from_f32(rows: usize, cols: usize, data: &[f32], threshold: f32) -> Result<Self> {
        if data.len() != rows * cols {
            bail!(
                "bit-lkan matrix size mismatch: expected {}, got {}",
                rows * cols,
                data.len()
            );
        }
        let mut row_data = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            row_data.push(PackedTernary::from_weights(&quantize_1_58b(
                &data[start..end],
                threshold,
            )));
        }
        Ok(Self {
            rows,
            cols,
            row_data,
        })
    }

    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.cols {
            bail!("matvec mismatch: expected {} cols, got {}", self.cols, x.len());
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            return Ok(self.matvec_avx512(x));
        }

        Ok(self.matvec_scalar(x))
    }

    fn matvec_scalar(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.rows];
        for (row_idx, row) in self.row_data.iter().enumerate() {
            let mut acc = 0.0_f32;
            for (col_idx, xv) in x.iter().enumerate() {
                match row.get(col_idx) {
                    TernaryWeight::Pos => acc += *xv,
                    TernaryWeight::Neg => acc -= *xv,
                    TernaryWeight::Zero => {}
                }
            }
            out[row_idx] = acc;
        }
        out
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    fn matvec_avx512(&self, x: &[f32]) -> Vec<f32> {
        // Placeholder fast path: logic remains add/sub-only and can be replaced by
        // explicit AVX-512 intrinsics in a dedicated perf build profile.
        self.matvec_scalar(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_outputs_ternary() {
        let q = quantize_1_58b(&[-0.9, -0.1, 0.0, 0.2, 0.8], 0.25);
        assert_eq!(
            q,
            vec![
                TernaryWeight::Neg,
                TernaryWeight::Zero,
                TernaryWeight::Zero,
                TernaryWeight::Zero,
                TernaryWeight::Pos,
            ]
        );
    }

    #[test]
    fn packed_roundtrip() {
        let weights = vec![
            TernaryWeight::Neg,
            TernaryWeight::Zero,
            TernaryWeight::Pos,
            TernaryWeight::Neg,
        ];
        let packed = PackedTernary::from_weights(&weights);
        let back: Vec<TernaryWeight> = (0..weights.len()).map(|idx| packed.get(idx)).collect();
        assert_eq!(back, weights);
    }
}
