//! KontekstiSitoja — Context Binder for N-gram Encoding
//! =====================================================
//! Permute-and-bind context encoding: transforms a window of
//! character hypervectors into a single context-aware query vector.
//!
//! The key insight: cyclic Weyl permutation IS discrete attention.
//!   permute(v, k) says "this vector was k positions back."
//!   Binding (element-wise multiply) composes positioned characters
//!   into a single holographic representation.
//!
//! Example (trigram "the"):
//!   context = permute(t, 2) ⊙ permute(h, 1) ⊙ e
//!
//! This encodes not just WHICH characters but WHERE they were.
//! Asking "what follows THIS specific sequence?" instead of
//! "what does this isolated vector look like?"
//!
//! Multi-scale: bundles unigram + bigram + trigram at φ-weighted
//! ratios, giving the model simultaneous access to all time scales.
//!
//! Finnish variable names per project convention.
//!
//! Authors: Dakota (Claude) & Greg Calkins
//! Date:    February 22, 2026

use std::collections::HashMap;

use crate::hdc_primitives::{
    HdcPeruskäsitteet, Hypervektori,
    ULOTTUVUUS, PHI,
};

// ═══════════════════════════════════════════════════════════════════
// CONTEXT BINDER — KontekstiSitoja
// ═══════════════════════════════════════════════════════════════════
//
// Stateless transformation: no learned parameters.
// Learning happens downstream (output prototypes, swarm dynamics).
// This module is pure geometry — position-aware composition.

/// Context binder: permute-and-bind n-gram encoder.
///
/// Slots between QAMS codebook (char → hypervector) and the swarm.
/// Transforms a window of character hypervectors into a single
/// context-aware query vector using Weyl permutation + binding.
pub struct KontekstiSitoja {
    ulottuvuus: usize,
    /// Maximum context window size (e.g., 3 for trigrams)
    pub ikkuna: usize,
}

impl KontekstiSitoja {
    pub fn new(ikkuna: usize) -> Self {
        Self {
            ulottuvuus: ULOTTUVUUS,
            ikkuna,
        }
    }

    pub fn with_dim(ikkuna: usize, ulottuvuus: usize) -> Self {
        Self { ulottuvuus, ikkuna }
    }

    /// Encode a sequence of character hypervectors into a single context vector.
    ///
    /// The most recent character is at the END of the slice.
    /// Each character is Weyl-permuted by its distance from the present,
    /// then all are bound together (element-wise multiply).
    ///
    /// ```text
    /// context = permute(v[0], n-1) ⊙ permute(v[1], n-2) ⊙ ... ⊙ v[n-1]
    /// ```
    ///
    /// Only the last `ikkuna` (window) characters are used.
    /// Returns a bipolar vector that holographically encodes the sequence.
    pub fn sido_konteksti(
        &self,
        vektorit: &[&[f64]],
        hdc: &HdcPeruskäsitteet,
    ) -> Hypervektori {
        // Truncate to window size
        let alku = if vektorit.len() > self.ikkuna {
            vektorit.len() - self.ikkuna
        } else {
            0
        };
        let ikkuna = &vektorit[alku..];
        let n = ikkuna.len();

        if n == 0 {
            return vec![0.0; self.ulottuvuus];
        }
        if n == 1 {
            return ikkuna[0].to_vec();
        }

        // Oldest character: permuted by (n-1) positions
        let mut konteksti = hdc.permutoi_weyl(ikkuna[0], n - 1);

        // Bind each subsequent character with decreasing permutation
        for i in 1..n {
            let askel = n - 1 - i;
            let permutoitu = if askel > 0 {
                hdc.permutoi_weyl(ikkuna[i], askel)
            } else {
                ikkuna[i].to_vec()
            };
            konteksti = hdc.sido(&konteksti, &permutoitu);
        }

        konteksti
    }

    /// Multi-scale context: bundle unigram + bigram + trigram contexts.
    ///
    /// Gives the model simultaneous access to multiple time scales.
    /// Weighted by powers of φ: trigrams weighted highest because
    /// longer contexts carry more predictive information.
    ///
    /// ```text
    /// multi = bundle(
    ///     1.0  × unigram(c[t]),
    ///     φ    × bigram(c[t-1], c[t]),
    ///     φ²   × trigram(c[t-2], c[t-1], c[t])
    /// )
    /// ```
    pub fn moniskaala_konteksti(
        &self,
        vektorit: &[&[f64]],
        hdc: &mut HdcPeruskäsitteet,
    ) -> Hypervektori {
        let n = vektorit.len();
        if n == 0 {
            return vec![0.0; self.ulottuvuus];
        }

        let mut kontekstit: Vec<Hypervektori> = Vec::new();
        let mut painot: Vec<f64> = Vec::new();

        // Unigram: just the most recent character
        kontekstit.push(vektorit[n - 1].to_vec());
        painot.push(1.0);

        // Bigram: last 2 characters, position-encoded
        if n >= 2 {
            let bigrammi = self.sido_konteksti(&vektorit[n - 2..], hdc);
            kontekstit.push(bigrammi);
            painot.push(PHI);
        }

        // Trigram: last 3 characters, position-encoded
        if n >= 3 && self.ikkuna >= 3 {
            let trigrammi = self.sido_konteksti(&vektorit[n - 3..], hdc);
            kontekstit.push(trigrammi);
            painot.push(PHI * PHI);
        }

        // 4-gram: last 4 characters (if window allows)
        if n >= 4 && self.ikkuna >= 4 {
            let nelogrammi = self.sido_konteksti(&vektorit[n - 4..], hdc);
            kontekstit.push(nelogrammi);
            painot.push(PHI * PHI * PHI);
        }

        // Bundle with φ-weights → sign normalize
        let viitteet: Vec<&[f64]> = kontekstit.iter()
            .map(|v| v.as_slice())
            .collect();
        hdc.niputa(&viitteet, Some(&painot))
    }

    /// Predict: given context vector, find the best-matching character.
    ///
    /// Direct nearest-neighbor lookup in the codebook.
    /// For swarm-based prediction, feed the context vector into the swarm
    /// instead and decode the swarm output — this method bypasses the swarm.
    pub fn ennusta_seuraava(
        &self,
        konteksti: &[f64],
        koodikirja: &HashMap<char, Hypervektori>,
        hdc: &HdcPeruskäsitteet,
    ) -> (char, f64) {
        let mut paras_merkki = ' ';
        let mut paras = f64::NEG_INFINITY;

        for (&merkki, vektori) in koodikirja {
            let pistemäärä = hdc.samankaltaisuus(konteksti, vektori);
            if pistemäärä > paras {
                paras = pistemäärä;
                paras_merkki = merkki;
            }
        }

        (paras_merkki, paras)
    }

    /// Predict with transition prior: combines context similarity
    /// with articulatory transition probability from QAMS.
    ///
    /// ```text
    /// score = (1 - α) × similarity(context, candidate)
    ///       +  α      × transition(prev_char, candidate)
    /// ```
    ///
    /// α=0.15 keeps the transition as gentle guidance while
    /// context similarity dominates.
    pub fn ennusta_siirtymällä(
        &self,
        konteksti: &[f64],
        koodikirja: &HashMap<char, Hypervektori>,
        hdc: &HdcPeruskäsitteet,
        edellinen: Option<char>,
        siirtymä_fn: Option<&dyn Fn(char, char) -> f64>,
        alfa: f64,
    ) -> (char, f64) {
        let mut paras_merkki = ' ';
        let mut paras = f64::NEG_INFINITY;

        for (&merkki, vektori) in koodikirja {
            let sim = hdc.samankaltaisuus(konteksti, vektori);
            let pistemäärä = match (edellinen, siirtymä_fn) {
                (Some(ed), Some(f)) => {
                    let siirtymä = f(ed, merkki);
                    sim * (1.0 - alfa) + siirtymä * alfa
                }
                _ => sim,
            };
            if pistemäärä > paras {
                paras = pistemäärä;
                paras_merkki = merkki;
            }
        }

        (paras_merkki, paras)
    }
}

// ═══════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn luo_hdc() -> HdcPeruskäsitteet {
        HdcPeruskäsitteet::new(ULOTTUVUUS, 42)
    }

    #[test]
    fn test_unigram_is_identity() {
        let hdc = luo_hdc();
        let sitoja = KontekstiSitoja::new(3);
        let mut rng = crate::hdc_primitives::Siemen::new(1);
        let v = rng.bipolaarinen_vektori(ULOTTUVUUS);

        let konteksti = sitoja.sido_konteksti(&[&v], &hdc);
        assert_eq!(konteksti, v,
            "unigram context must equal the raw character vector");
    }

    #[test]
    fn test_order_matters() {
        // "ab" ≠ "ba" — permute-and-bind is order-sensitive
        let mut hdc = luo_hdc();
        let a = hdc.satunnainen_bipolaarinen();
        let b = hdc.satunnainen_bipolaarinen();
        let sitoja = KontekstiSitoja::new(3);

        let ab = sitoja.sido_konteksti(&[&a, &b], &hdc);
        let ba = sitoja.sido_konteksti(&[&b, &a], &hdc);

        let sim = hdc.samankaltaisuus(&ab, &ba);
        assert!(sim.abs() < 0.3,
            "ab and ba should be near-orthogonal, got sim={sim}");
    }

    #[test]
    fn test_deterministic() {
        // Same input → same output (no randomness in binding)
        let mut hdc = luo_hdc();
        let a = hdc.satunnainen_bipolaarinen();
        let b = hdc.satunnainen_bipolaarinen();
        let c = hdc.satunnainen_bipolaarinen();
        let sitoja = KontekstiSitoja::new(3);

        let ctx1 = sitoja.sido_konteksti(&[&a, &b, &c], &hdc);
        let ctx2 = sitoja.sido_konteksti(&[&a, &b, &c], &hdc);
        assert_eq!(ctx1, ctx2, "same input must produce same context");
    }

    #[test]
    fn test_trigram_is_bipolar() {
        // Binding bipolar vectors → bipolar result
        let mut hdc = luo_hdc();
        let a = hdc.satunnainen_bipolaarinen();
        let b = hdc.satunnainen_bipolaarinen();
        let c = hdc.satunnainen_bipolaarinen();
        let sitoja = KontekstiSitoja::new(3);

        let ctx = sitoja.sido_konteksti(&[&a, &b, &c], &hdc);
        assert!(ctx.iter().all(|&x| x == 1.0 || x == -1.0),
            "trigram of bipolar vectors must be bipolar");
    }

    #[test]
    fn test_window_truncation() {
        // With ikkuna=2, a 4-element sequence should only use last 2
        let mut hdc = luo_hdc();
        let a = hdc.satunnainen_bipolaarinen();
        let b = hdc.satunnainen_bipolaarinen();
        let c = hdc.satunnainen_bipolaarinen();
        let d = hdc.satunnainen_bipolaarinen();
        let sitoja = KontekstiSitoja::new(2); // bigram window

        let ctx_full = sitoja.sido_konteksti(&[&a, &b, &c, &d], &hdc);
        let ctx_last2 = sitoja.sido_konteksti(&[&c, &d], &hdc);
        assert_eq!(ctx_full, ctx_last2,
            "ikkuna=2 should truncate to last 2 characters");
    }

    #[test]
    fn test_multiscale_bundles_all_scales() {
        let mut hdc = luo_hdc();
        let a = hdc.satunnainen_bipolaarinen();
        let b = hdc.satunnainen_bipolaarinen();
        let c = hdc.satunnainen_bipolaarinen();
        let sitoja = KontekstiSitoja::new(3);

        let multi = sitoja.moniskaala_konteksti(
            &[&a, &b, &c], &mut hdc,
        );

        // Multi-scale should be bipolar (sign-normalized bundle)
        assert!(multi.iter().all(|&x| x == 1.0 || x == -1.0),
            "multi-scale output must be bipolar");

        // Should correlate with the unigram (c) since it's included
        let sim_c = hdc.samankaltaisuus(&multi, &c);
        assert!(sim_c > 0.0,
            "multi-scale should correlate with unigram: {sim_c}");
    }

    #[test]
    fn test_multiscale_trigram_dominates() {
        // With φ²-weighting, trigram should have the most influence
        let mut hdc = luo_hdc();
        let a = hdc.satunnainen_bipolaarinen();
        let b = hdc.satunnainen_bipolaarinen();
        let c = hdc.satunnainen_bipolaarinen();
        let sitoja = KontekstiSitoja::new(3);

        let trigram = sitoja.sido_konteksti(&[&a, &b, &c], &hdc);
        let multi = sitoja.moniskaala_konteksti(
            &[&a, &b, &c], &mut hdc,
        );

        let sim_tri = hdc.samankaltaisuus(&multi, &trigram);
        let sim_uni = hdc.samankaltaisuus(&multi, &c);
        assert!(sim_tri > sim_uni,
            "trigram (φ² weight) should dominate unigram: tri={sim_tri} uni={sim_uni}");
    }

    #[test]
    fn test_prediction_finds_exact_match() {
        let mut hdc = luo_hdc();
        let mut koodikirja = HashMap::new();
        let v_a = hdc.satunnainen_bipolaarinen();
        let v_b = hdc.satunnainen_bipolaarinen();
        let v_c = hdc.satunnainen_bipolaarinen();
        koodikirja.insert('a', v_a.clone());
        koodikirja.insert('b', v_b.clone());
        koodikirja.insert('c', v_c.clone());
        let sitoja = KontekstiSitoja::new(3);

        // Query with exact vector for 'b' → should predict 'b'
        let (merkki, _) = sitoja.ennusta_seuraava(&v_b, &koodikirja, &hdc);
        assert_eq!(merkki, 'b', "exact match query should return 'b'");
    }

    #[test]
    fn test_shared_prefix_different_context() {
        // "ab?" and "cb?" should produce different predictions
        // because the context encodes the full sequence
        let mut hdc = luo_hdc();
        let a = hdc.satunnainen_bipolaarinen();
        let b = hdc.satunnainen_bipolaarinen();
        let c = hdc.satunnainen_bipolaarinen();
        let sitoja = KontekstiSitoja::new(3);

        let ctx_ab = sitoja.sido_konteksti(&[&a, &b], &hdc);
        let ctx_cb = sitoja.sido_konteksti(&[&c, &b], &hdc);

        let sim = hdc.samankaltaisuus(&ctx_ab, &ctx_cb);
        assert!(sim.abs() < 0.3,
            "different prefixes should produce near-orthogonal contexts: {sim}");
    }

    #[test]
    fn test_empty_context() {
        let hdc = luo_hdc();
        let sitoja = KontekstiSitoja::new(3);
        let ctx = sitoja.sido_konteksti(&[], &hdc);
        assert_eq!(ctx.len(), ULOTTUVUUS);
        assert!(ctx.iter().all(|&x| x == 0.0),
            "empty context should be zero vector");
    }

    #[test]
    fn test_context_self_inverse() {
        // Binding is self-inverse: unbinding a character from a bigram
        // should recover (approximately) the other character
        let mut hdc = luo_hdc();
        let a = hdc.satunnainen_bipolaarinen();
        let b = hdc.satunnainen_bipolaarinen();
        let sitoja = KontekstiSitoja::new(2);

        // bigram = permute(a, 1) ⊙ b
        let bigrammi = sitoja.sido_konteksti(&[&a, &b], &hdc);

        // Unbind b to recover permute(a, 1)
        let palautettu_perm = hdc.irrota(&bigrammi, &b);

        // The recovered vector should be permute(a, 1)
        let odotettu = hdc.permutoi_weyl(&a, 1);
        let sim = hdc.samankaltaisuus(&palautettu_perm, &odotettu);
        assert!((sim - 1.0).abs() < 1e-10,
            "unbinding b from bigram should recover permute(a,1): sim={sim}");
    }
}
