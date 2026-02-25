//! Ternaari — Ternary Vector Operations
//! =====================================
//!
//! Packed {-1, 0, +1} vectors for 16x memory compression.
//!
//! Encoding: 2 bits per trit, 4 trits per byte.
//!   00 = 0 (silence), 01 = +1 (affirm), 10 = -1 (deny), 11 = reserved
//!
//! D=256 → 64 bytes (vs 2048 bytes f64). **16x compression.**
//!
//! Training stays f64 — ternary is inference-only quantization.
//! FAMILY_BOUNDARY (0.7635) is the quantization threshold.
//!
//! Finnish variable names per project convention.
//!
//! Authors: Astra Nova (Claude), Dakota (Claude), Rose (Claude)
//!          & Greg Calkins
//! Date:    February 24, 2026

use std::collections::HashMap;

use crate::hdc_primitives::{Hypervektori, PHI, TAU};
use crate::kolmoset::ReleSolmu;

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════

/// Family boundary: where φ and √3 families meet ≈ 2/φ²
pub const FAMILY_BOUNDARY: f64 = 0.7635;

/// Trit encoding: 2 bits each
const TRIT_ZERO: u8 = 0b00;  // silence
const TRIT_POS:  u8 = 0b01;  // +1 affirm
const TRIT_NEG:  u8 = 0b10;  // -1 deny

/// Pre-computed dot products for all byte pairs.
/// Each byte holds 4 trits. dot(a_byte, b_byte) = sum of trit_a[i] * trit_b[i].
/// We precompute this as a 256×256 lookup table.
fn build_dot_lookup() -> Vec<i8> {
    let mut table = vec![0i8; 256 * 256];

    for a in 0u16..256 {
        for b in 0u16..256 {
            let mut dot = 0i8;
            for pos in 0..4 {
                let shift = pos * 2;
                let ta = ((a >> shift) & 0x03) as u8;
                let tb = ((b >> shift) & 0x03) as u8;
                let va = trit_to_val(ta);
                let vb = trit_to_val(tb);
                dot += va * vb;
            }
            table[a as usize * 256 + b as usize] = dot;
        }
    }

    table
}

/// Pre-computed non-zero count per byte (how many trits are non-zero).
fn build_nnz_lookup() -> Vec<u8> {
    let mut table = vec![0u8; 256];
    for a in 0u16..256 {
        let mut count = 0u8;
        for pos in 0..4 {
            let shift = pos * 2;
            let t = ((a >> shift) & 0x03) as u8;
            if t == TRIT_POS || t == TRIT_NEG {
                count += 1;
            }
        }
        table[a as usize] = count;
    }
    table
}

/// Convert trit encoding to numeric value.
#[inline]
fn trit_to_val(t: u8) -> i8 {
    match t {
        TRIT_POS => 1,
        TRIT_NEG => -1,
        _ => 0,
    }
}

// ═══════════════════════════════════════════════════════════════════
// TERNARY VECTOR
// ═══════════════════════════════════════════════════════════════════

/// Packed ternary vector: {-1, 0, +1} with 4 trits per byte.
#[derive(Debug, Clone)]
pub struct TernääriVektori {
    /// Packed bytes: 4 trits per byte, 2 bits each.
    pub data: Vec<u8>,
    /// Logical dimension (number of trits).
    pub dim: usize,
}

impl TernääriVektori {
    /// Create a zero vector of given dimension.
    pub fn zero(dim: usize) -> Self {
        let bytes = (dim + 3) / 4;
        Self {
            data: vec![0u8; bytes],
            dim,
        }
    }

    /// Get trit value at position i: -1, 0, or +1.
    #[inline]
    pub fn get(&self, i: usize) -> i8 {
        let byte_idx = i / 4;
        let bit_pos = (i % 4) * 2;
        let t = (self.data[byte_idx] >> bit_pos) & 0x03;
        trit_to_val(t)
    }

    /// Set trit value at position i.
    #[inline]
    pub fn set(&mut self, i: usize, val: i8) {
        let byte_idx = i / 4;
        let bit_pos = (i % 4) * 2;
        let trit = match val {
            1 => TRIT_POS,
            -1 => TRIT_NEG,
            _ => TRIT_ZERO,
        };
        // Clear the 2-bit field, then set
        self.data[byte_idx] &= !(0x03 << bit_pos);
        self.data[byte_idx] |= trit << bit_pos;
    }

    /// Count of non-zero trits.
    pub fn nnz(&self) -> usize {
        let nnz_lut = build_nnz_lookup();
        let full_bytes = self.dim / 4;
        let mut count = 0usize;

        for i in 0..full_bytes {
            count += nnz_lut[self.data[i] as usize] as usize;
        }

        // Handle trailing trits
        let remainder = self.dim % 4;
        if remainder > 0 {
            let last = self.data[full_bytes];
            for pos in 0..remainder {
                let t = (last >> (pos * 2)) & 0x03;
                if t == TRIT_POS || t == TRIT_NEG {
                    count += 1;
                }
            }
        }

        count
    }

    /// Sparsity: fraction of zero trits.
    pub fn sparsity(&self) -> f64 {
        if self.dim == 0 { return 1.0; }
        1.0 - (self.nnz() as f64 / self.dim as f64)
    }

    /// Memory in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }
}

// ═══════════════════════════════════════════════════════════════════
// TERNARY OPERATIONS
// ═══════════════════════════════════════════════════════════════════

/// Quantize f64 vector to ternary.
///
/// Normalize by L∞ norm, then threshold at FAMILY_BOUNDARY.
/// Above threshold → sign(x), below → 0.
pub fn kvantisoi(v: &[f64], kynnys: f64) -> TernääriVektori {
    let dim = v.len();
    let mut result = TernääriVektori::zero(dim);

    // L∞ normalization
    let max_abs = v.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    if max_abs < 1e-12 {
        return result;
    }

    for i in 0..dim {
        let normalized = v[i] / max_abs;
        if normalized.abs() >= kynnys {
            result.set(i, if normalized > 0.0 { 1 } else { -1 });
        }
        // else: stays 0 (silence)
    }

    result
}

/// Ternary cosine similarity: dot(a,b) / sqrt(nnz(a) * nnz(b)).
///
/// Uses byte-pair lookup table for fast dot product.
pub fn samankaltaisuus(a: &TernääriVektori, b: &TernääriVektori) -> f64 {
    assert_eq!(a.dim, b.dim, "dimension mismatch");

    let dot_lut = build_dot_lookup();
    let nnz_lut = build_nnz_lookup();

    let full_bytes = a.dim / 4;
    let mut dot = 0i32;
    let mut nnz_a = 0u32;
    let mut nnz_b = 0u32;

    for i in 0..full_bytes {
        let ai = a.data[i] as usize;
        let bi = b.data[i] as usize;
        dot += dot_lut[ai * 256 + bi] as i32;
        nnz_a += nnz_lut[ai] as u32;
        nnz_b += nnz_lut[bi] as u32;
    }

    // Handle trailing trits
    let remainder = a.dim % 4;
    if remainder > 0 {
        let last_a = a.data[full_bytes];
        let last_b = b.data[full_bytes];
        for pos in 0..remainder {
            let ta = trit_to_val((last_a >> (pos * 2)) & 0x03);
            let tb = trit_to_val((last_b >> (pos * 2)) & 0x03);
            dot += (ta * tb) as i32;
            if ta != 0 { nnz_a += 1; }
            if tb != 0 { nnz_b += 1; }
        }
    }

    let denom = ((nnz_a as f64) * (nnz_b as f64)).sqrt();
    if denom < 1e-12 { 0.0 } else { dot as f64 / denom }
}

/// Ternary binding: element-wise multiply.
/// -1×-1=+1, anything×0=0, +1×-1=-1, etc.
pub fn sido(a: &TernääriVektori, b: &TernääriVektori) -> TernääriVektori {
    assert_eq!(a.dim, b.dim, "dimension mismatch");
    let mut result = TernääriVektori::zero(a.dim);

    for i in 0..a.dim {
        let va = a.get(i);
        let vb = b.get(i);
        result.set(i, va * vb);
    }

    result
}

/// Circular shift on packed ternary data (Weyl permutation).
pub fn permutoi_weyl(v: &TernääriVektori, step: usize) -> TernääriVektori {
    let dim = v.dim;
    if dim == 0 { return v.clone(); }

    let shift = (((TAU * step as f64 * dim as f64) as usize) % dim).max(1);
    let mut result = TernääriVektori::zero(dim);

    for i in 0..dim {
        let src = (i + dim - shift) % dim;
        result.set(i, v.get(src));
    }

    result
}

/// Majority-vote bundling of ternary vectors.
///
/// For each dimension, tally weighted votes:
///   +1 trits add weight, -1 trits subtract weight, 0 abstains.
/// Final: above threshold → +1, below -threshold → -1, else 0.
pub fn niputa_ternaari(
    vecs: &[&TernääriVektori],
    weights: &[f64],
    threshold: f64,
) -> TernääriVektori {
    if vecs.is_empty() {
        return TernääriVektori::zero(0);
    }
    let dim = vecs[0].dim;
    let mut tallies = vec![0.0f64; dim];

    for (v, &w) in vecs.iter().zip(weights.iter()) {
        for i in 0..dim {
            tallies[i] += v.get(i) as f64 * w;
        }
    }

    let mut result = TernääriVektori::zero(dim);
    for i in 0..dim {
        if tallies[i] > threshold {
            result.set(i, 1);
        } else if tallies[i] < -threshold {
            result.set(i, -1);
        }
    }

    result
}

/// Batch quantize prototypes.
pub fn kvantisoi_prototyypit(
    protos: &HashMap<char, Hypervektori>,
    kynnys: f64,
) -> HashMap<char, TernääriVektori> {
    protos.iter()
        .map(|(&c, v)| (c, kvantisoi(v, kynnys)))
        .collect()
}

/// Ternary inference: find best matching character.
pub fn ennusta_ternaari(
    ctx: &TernääriVektori,
    protos: &HashMap<char, TernääriVektori>,
) -> (char, f64) {
    let mut paras_merkki = ' ';
    let mut paras_piste = f64::NEG_INFINITY;

    for (&c, proto) in protos {
        let piste = samankaltaisuus(ctx, proto);
        if piste > paras_piste {
            paras_piste = piste;
            paras_merkki = c;
        }
    }

    (paras_merkki, paras_piste)
}

// ═══════════════════════════════════════════════════════════════════
// TERNARY RELAY NODE — Quantized relay wrapper
// ═══════════════════════════════════════════════════════════════════

/// Ternary-quantized relay node for inference.
///
/// Holds quantized versions of both myöntö (affirm) and kielto (deny)
/// prototypes. Scoring uses ternary similarity with β-suppression.
#[derive(Debug, Clone)]
pub struct TernääriReleSolmu {
    pub myöntö: HashMap<char, TernääriVektori>,
    pub kielto: HashMap<char, TernääriVektori>,
    pub beta: f64,
    pub luottamus: f64,
}

impl TernääriReleSolmu {
    /// Quantize a trained f64 relay node into ternary.
    pub fn from_rele(rele: &ReleSolmu, kynnys: f64) -> Self {
        Self {
            myöntö: kvantisoi_prototyypit(&rele.myöntö.prototyypit, kynnys),
            kielto: kvantisoi_prototyypit(&rele.kielto.prototyypit, kynnys),
            beta: rele.luottamus() * PHI * 0.103, // scaled β
            luottamus: rele.luottamus(),
        }
    }

    /// Score a character given ternary context: myöntö - β·kielto.
    pub fn pisteet(&self, ctx: &TernääriVektori, merkki: char) -> f64 {
        let m = self.myöntö.get(&merkki)
            .map(|p| samankaltaisuus(ctx, p))
            .unwrap_or(0.0);
        let k = self.kielto.get(&merkki)
            .map(|p| samankaltaisuus(ctx, p))
            .unwrap_or(0.0);
        m - self.beta * k
    }

    /// Predict best character from ternary context.
    pub fn ennusta(&self, ctx: &TernääriVektori) -> (char, f64) {
        let mut paras_merkki = ' ';
        let mut paras_piste = f64::NEG_INFINITY;

        for (&c, _) in &self.myöntö {
            let piste = self.pisteet(ctx, c);
            if piste > paras_piste {
                paras_piste = piste;
                paras_merkki = c;
            }
        }

        (paras_merkki, paras_piste)
    }

    /// Sparsity statistics: average fraction of zero trits across all prototypes.
    pub fn avg_sparsity(&self) -> f64 {
        let protos: Vec<&TernääriVektori> = self.myöntö.values()
            .chain(self.kielto.values())
            .collect();
        if protos.is_empty() { return 0.0; }
        let total: f64 = protos.iter().map(|v| v.sparsity()).sum();
        total / protos.len() as f64
    }

    /// Total memory in bytes for all ternary prototypes.
    pub fn memory_bytes(&self) -> usize {
        let myöntö_bytes: usize = self.myöntö.values().map(|v| v.memory_bytes()).sum();
        let kielto_bytes: usize = self.kielto.values().map(|v| v.memory_bytes()).sum();
        myöntö_bytes + kielto_bytes
    }
}

// ═══════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_zero_vector() {
        let v = TernääriVektori::zero(256);
        assert_eq!(v.dim, 256);
        assert_eq!(v.data.len(), 64);
        assert_eq!(v.nnz(), 0);
        assert_eq!(v.sparsity(), 1.0);
    }

    #[test]
    fn test_ternary_set_get() {
        let mut v = TernääriVektori::zero(8);
        v.set(0, 1);
        v.set(1, -1);
        v.set(2, 0);
        v.set(3, 1);
        v.set(4, -1);

        assert_eq!(v.get(0), 1);
        assert_eq!(v.get(1), -1);
        assert_eq!(v.get(2), 0);
        assert_eq!(v.get(3), 1);
        assert_eq!(v.get(4), -1);
        assert_eq!(v.get(5), 0);
    }

    #[test]
    fn test_ternary_nnz() {
        let mut v = TernääriVektori::zero(8);
        v.set(0, 1);
        v.set(3, -1);
        v.set(7, 1);
        assert_eq!(v.nnz(), 3);
    }

    #[test]
    fn test_quantize_bipolar() {
        // Bipolar vector (all ±1) should quantize to all non-zero
        let v: Vec<f64> = vec![1.0, -1.0, 1.0, -1.0, 0.8, -0.9, 0.3, -0.2];
        let t = kvantisoi(&v, FAMILY_BOUNDARY);
        assert_eq!(t.dim, 8);
        assert_eq!(t.get(0), 1);
        assert_eq!(t.get(1), -1);
        assert_eq!(t.get(2), 1);
        assert_eq!(t.get(3), -1);
        assert_eq!(t.get(4), 1);   // 0.8/1.0 = 0.8 > 0.7635
        assert_eq!(t.get(5), -1);  // 0.9/1.0 = 0.9 > 0.7635
        assert_eq!(t.get(6), 0);   // 0.3/1.0 = 0.3 < 0.7635
        assert_eq!(t.get(7), 0);   // 0.2/1.0 = 0.2 < 0.7635
    }

    #[test]
    fn test_similarity_identical() {
        let mut a = TernääriVektori::zero(8);
        a.set(0, 1);
        a.set(1, -1);
        a.set(2, 1);
        let sim = samankaltaisuus(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10, "self-similarity should be 1.0, got {}", sim);
    }

    #[test]
    fn test_similarity_opposite() {
        let mut a = TernääriVektori::zero(4);
        a.set(0, 1);
        a.set(1, 1);
        let mut b = TernääriVektori::zero(4);
        b.set(0, -1);
        b.set(1, -1);
        let sim = samankaltaisuus(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-10, "opposite should be -1.0, got {}", sim);
    }

    #[test]
    fn test_similarity_orthogonal() {
        let mut a = TernääriVektori::zero(4);
        a.set(0, 1);
        a.set(1, 0);
        let mut b = TernääriVektori::zero(4);
        b.set(0, 0);
        b.set(1, 1);
        let sim = samankaltaisuus(&a, &b);
        assert!((sim - 0.0).abs() < 1e-10, "orthogonal should be 0.0, got {}", sim);
    }

    #[test]
    fn test_binding() {
        let mut a = TernääriVektori::zero(4);
        a.set(0, 1);
        a.set(1, -1);
        a.set(2, 1);
        a.set(3, 0);

        let mut b = TernääriVektori::zero(4);
        b.set(0, -1);
        b.set(1, -1);
        b.set(2, 1);
        b.set(3, 1);

        let c = sido(&a, &b);
        assert_eq!(c.get(0), -1);  // 1 * -1
        assert_eq!(c.get(1), 1);   // -1 * -1
        assert_eq!(c.get(2), 1);   // 1 * 1
        assert_eq!(c.get(3), 0);   // 0 * 1
    }

    #[test]
    fn test_permutation_shifts() {
        let mut v = TernääriVektori::zero(8);
        v.set(0, 1);
        v.set(1, -1);
        let p = permutoi_weyl(&v, 1);
        // Should have shifted — at least one non-zero in different position
        assert_eq!(p.dim, 8);
        assert_eq!(p.nnz(), 2);
    }

    #[test]
    fn test_majority_bundling() {
        let mut a = TernääriVektori::zero(4);
        a.set(0, 1);
        a.set(1, 1);

        let mut b = TernääriVektori::zero(4);
        b.set(0, 1);
        b.set(1, -1);

        let mut c = TernääriVektori::zero(4);
        c.set(0, 1);
        c.set(1, -1);

        let result = niputa_ternaari(
            &[&a, &b, &c],
            &[1.0, 1.0, 1.0],
            0.5,
        );
        assert_eq!(result.get(0), 1);  // 3×+1 = 3.0 > 0.5
        assert_eq!(result.get(1), -1); // 1×1 + 2×(-1) = -1.0 < -0.5
    }

    #[test]
    fn test_memory_compression() {
        let dim = 256;
        let f64_bytes = dim * 8;  // 2048 bytes
        let ternary_bytes = dim / 4; // 64 bytes
        assert_eq!(f64_bytes / ternary_bytes, 32); // Actually 32x but with overhead
        let v = TernääriVektori::zero(dim);
        assert_eq!(v.memory_bytes(), 64);
    }

    #[test]
    fn test_batch_quantize_and_predict() {
        let mut protos = HashMap::new();
        protos.insert('a', vec![1.0; 16]);
        protos.insert('b', vec![-1.0; 16]);

        let t_protos = kvantisoi_prototyypit(&protos, FAMILY_BOUNDARY);
        assert_eq!(t_protos.len(), 2);

        // Context similar to 'a'
        let ctx = kvantisoi(&vec![1.0; 16], FAMILY_BOUNDARY);
        let (pred, score) = ennusta_ternaari(&ctx, &t_protos);
        assert_eq!(pred, 'a');
        assert!(score > 0.0);
    }
}
