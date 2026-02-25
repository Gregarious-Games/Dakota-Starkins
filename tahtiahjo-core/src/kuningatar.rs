//! Kuningatar — Queen Node Mesh
//! =============================
//!
//! Coordinates three engines through SPS (Signal-Phase-Synthesis) protocol.
//!
//! SPS Protocol:
//!   1. Each engine's score vector → ternary SPS message (threshold at FAMILY_BOUNDARY)
//!   2. Per-character agreement analysis: count trits that match across engines
//!   3. Agreement patterns: TäysiSopimus / Enemmistö / YksinSointi / Hiljaisuus / Ristiriita
//!
//! Monophonic Downshift:
//!   - When coherence < FAMILY_BOUNDARY → one engine leads (weight=φ), others follow (weight=τ)
//!   - When coherence ≥ FAMILY_BOUNDARY → polyphonic (equal weights, all engines contribute)
//!
//! Finnish variable names per project convention.
//!
//! Authors: Astra Nova (Claude), Dakota (Claude), Rose (Claude)
//!          & Greg Calkins
//! Date:    February 24, 2026

use crate::hdc_primitives::{PHI, TAU};

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════

/// Family boundary: threshold for ternary quantization and coherence gating.
const FAMILY_BOUNDARY: f64 = 0.7635;

/// Coherence EMA decay rate.
const KOHERENSSI_ALPHA: f64 = 0.99;

// ═══════════════════════════════════════════════════════════════════
// AGREEMENT PATTERNS
// ═══════════════════════════════════════════════════════════════════

/// Per-character agreement pattern across 3 engines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SopimisKuvio {
    /// All three agree on direction (all +, all -, or all 0).
    TäysiSopimus,
    /// Two of three agree on direction.
    Enemmistö,
    /// Only one engine has a non-zero opinion.
    YksinSointi,
    /// All three are silent (all 0).
    Hiljaisuus,
    /// Active disagreement (opposing signs, no majority).
    Ristiriita,
}

/// Diagnostics from one prediction cycle.
#[derive(Debug, Clone)]
pub struct KuningatarDiagnostiikka {
    pub koherenssi: f64,
    pub moottori_tarkkuudet: [f64; 3],
    pub johtaja: Option<usize>,
    pub tila: &'static str,  // "monophonic" or "polyphonic"
    pub sopimus_osuudet: [f64; 5],  // fractions: full/majority/solo/silence/conflict
}

// ═══════════════════════════════════════════════════════════════════
// QUEEN NODE MESH
// ═══════════════════════════════════════════════════════════════════

/// Queen Node: coordinates three engines through SPS protocol.
pub struct Kuningatar {
    /// Number of characters in alphabet.
    _num_chars: usize,
    /// Coherence pulse: EMA of inter-engine agreement.
    koherenssi_pulssi: f64,
    /// Per-engine accuracy tracking (EMA).
    moottori_tarkkuudet: [f64; 3],
    /// Lead engine index (None = polyphonic mode).
    johtaja: Option<usize>,
    /// Prediction counter.
    ennusteet: u64,
    /// Agreement statistics (cumulative).
    sopimus_laskurit: [u64; 5],
}

impl Kuningatar {
    /// Create a new Queen Node for given alphabet size.
    pub fn new(num_chars: usize) -> Self {
        Self {
            _num_chars: num_chars,
            koherenssi_pulssi: 0.5,  // neutral start
            moottori_tarkkuudet: [0.1; 3],
            johtaja: None,
            ennusteet: 0,
            sopimus_laskurit: [0; 5],
        }
    }

    /// Quantize a score vector to ternary SPS message.
    /// Positive scores above threshold → +1, negative below → -1, rest → 0.
    fn sps_viesti(&self, scores: &[f64]) -> Vec<i8> {
        if scores.is_empty() { return vec![]; }

        let max_abs = scores.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        if max_abs < 1e-12 {
            return vec![0i8; scores.len()];
        }

        scores.iter().map(|&s| {
            let normalized = s / max_abs;
            if normalized > FAMILY_BOUNDARY { 1i8 }
            else if normalized < -FAMILY_BOUNDARY { -1i8 }
            else { 0i8 }
        }).collect()
    }

    /// Analyze agreement between 3 SPS messages for one character position.
    fn analysoi_sopimus(&self, ta: i8, tb: i8, tc: i8) -> SopimisKuvio {
        let nonzero = (ta != 0) as u8 + (tb != 0) as u8 + (tc != 0) as u8;

        if nonzero == 0 {
            return SopimisKuvio::Hiljaisuus;
        }
        if nonzero == 1 {
            return SopimisKuvio::YksinSointi;
        }

        // At least 2 non-zero
        let sum = ta as i16 + tb as i16 + tc as i16;
        let abs_sum = sum.unsigned_abs();

        if nonzero == 3 && abs_sum == 3 {
            // All three agree: +++  or ---
            SopimisKuvio::TäysiSopimus
        } else if abs_sum >= 2 {
            // Majority: two agree, third differs or is zero
            SopimisKuvio::Enemmistö
        } else if nonzero == 2 && abs_sum == 0 {
            // Two active but opposing: +1 and -1
            SopimisKuvio::Ristiriita
        } else {
            // e.g. nonzero==3 with sum=±1 (one dissenter)
            SopimisKuvio::Enemmistö
        }
    }

    /// Combine 3 engine score vectors into a final prediction.
    ///
    /// Returns: (combined_scores, diagnostics)
    pub fn yhdistä(
        &mut self,
        scores: [&[f64]; 3],
        engine_accuracies: [f64; 3],
    ) -> (Vec<f64>, KuningatarDiagnostiikka) {
        let n = scores[0].len();
        assert_eq!(n, scores[1].len());
        assert_eq!(n, scores[2].len());

        // Update engine accuracy tracking
        for i in 0..3 {
            self.moottori_tarkkuudet[i] =
                0.95 * self.moottori_tarkkuudet[i] + 0.05 * engine_accuracies[i];
        }

        // Build SPS messages
        let sps = [
            self.sps_viesti(scores[0]),
            self.sps_viesti(scores[1]),
            self.sps_viesti(scores[2]),
        ];

        // Per-character agreement analysis
        let mut pattern_counts = [0u64; 5];
        let mut agreement_sum = 0.0f64;

        for i in 0..n {
            let ta = if i < sps[0].len() { sps[0][i] } else { 0 };
            let tb = if i < sps[1].len() { sps[1][i] } else { 0 };
            let tc = if i < sps[2].len() { sps[2][i] } else { 0 };

            let pattern = self.analysoi_sopimus(ta, tb, tc);
            let idx = match pattern {
                SopimisKuvio::TäysiSopimus => 0,
                SopimisKuvio::Enemmistö => 1,
                SopimisKuvio::YksinSointi => 2,
                SopimisKuvio::Hiljaisuus => 3,
                SopimisKuvio::Ristiriita => 4,
            };
            pattern_counts[idx] += 1;
            self.sopimus_laskurit[idx] += 1;

            // Agreement strength: full=1.0, majority=0.66, solo=0.33, silence=0.5, conflict=0.0
            agreement_sum += match pattern {
                SopimisKuvio::TäysiSopimus => 1.0,
                SopimisKuvio::Enemmistö => 0.66,
                SopimisKuvio::YksinSointi => 0.33,
                SopimisKuvio::Hiljaisuus => 0.5,
                SopimisKuvio::Ristiriita => 0.0,
            };
        }

        let avg_agreement = if n > 0 { agreement_sum / n as f64 } else { 0.5 };

        // Update coherence pulse (EMA)
        self.koherenssi_pulssi =
            KOHERENSSI_ALPHA * self.koherenssi_pulssi
            + (1.0 - KOHERENSSI_ALPHA) * avg_agreement;

        // Determine mode: polyphonic or monophonic
        let polyphonic = self.koherenssi_pulssi >= FAMILY_BOUNDARY;

        // Select lead engine (for monophonic mode)
        self.johtaja = if polyphonic {
            None
        } else {
            // Lead = highest accuracy among the three
            let lead = if self.moottori_tarkkuudet[0] >= self.moottori_tarkkuudet[1]
                && self.moottori_tarkkuudet[0] >= self.moottori_tarkkuudet[2] {
                0
            } else if self.moottori_tarkkuudet[1] >= self.moottori_tarkkuudet[2] {
                1
            } else {
                2
            };
            Some(lead)
        };

        // Compute weights
        let weights = if polyphonic {
            // Polyphonic: accuracy-weighted (like standard ensemble)
            let total: f64 = self.moottori_tarkkuudet.iter().sum();
            if total < 1e-12 {
                [1.0 / 3.0; 3]
            } else {
                [
                    self.moottori_tarkkuudet[0] / total,
                    self.moottori_tarkkuudet[1] / total,
                    self.moottori_tarkkuudet[2] / total,
                ]
            }
        } else {
            // Monophonic: lead gets φ, followers get τ
            let lead = self.johtaja.unwrap_or(0);
            let total = PHI + 2.0 * TAU;
            let mut w = [TAU / total; 3];
            w[lead] = PHI / total;
            w
        };

        // Weighted combination of score vectors
        let mut combined = vec![0.0; n];
        for i in 0..n {
            combined[i] = weights[0] * scores[0][i]
                        + weights[1] * scores[1][i]
                        + weights[2] * scores[2][i];
        }

        // Build diagnostics
        let total_patterns: f64 = pattern_counts.iter().sum::<u64>() as f64;
        let sopimus_osuudet = if total_patterns > 0.0 {
            [
                pattern_counts[0] as f64 / total_patterns,
                pattern_counts[1] as f64 / total_patterns,
                pattern_counts[2] as f64 / total_patterns,
                pattern_counts[3] as f64 / total_patterns,
                pattern_counts[4] as f64 / total_patterns,
            ]
        } else {
            [0.0; 5]
        };

        self.ennusteet += 1;

        let diagnostiikka = KuningatarDiagnostiikka {
            koherenssi: self.koherenssi_pulssi,
            moottori_tarkkuudet: self.moottori_tarkkuudet,
            johtaja: self.johtaja,
            tila: if polyphonic { "polyphonic" } else { "monophonic" },
            sopimus_osuudet,
        };

        (combined, diagnostiikka)
    }

    /// Get cumulative agreement fractions.
    pub fn sopimus_osuudet(&self) -> [f64; 5] {
        let total: f64 = self.sopimus_laskurit.iter().sum::<u64>() as f64;
        if total < 1.0 {
            return [0.0; 5];
        }
        [
            self.sopimus_laskurit[0] as f64 / total,
            self.sopimus_laskurit[1] as f64 / total,
            self.sopimus_laskurit[2] as f64 / total,
            self.sopimus_laskurit[3] as f64 / total,
            self.sopimus_laskurit[4] as f64 / total,
        ]
    }

    /// Current coherence pulse value.
    pub fn koherenssi(&self) -> f64 {
        self.koherenssi_pulssi
    }

    /// Current lead engine (None = polyphonic).
    pub fn johtaja(&self) -> Option<usize> {
        self.johtaja
    }

    /// Reset state for fresh evaluation pass.
    pub fn nollaa(&mut self) {
        self.koherenssi_pulssi = 0.5;
        self.johtaja = None;
        self.ennusteet = 0;
        self.sopimus_laskurit = [0; 5];
    }
}

// ═══════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queen_creation() {
        let q = Kuningatar::new(26);
        assert_eq!(q._num_chars, 26);
        assert!(q.johtaja.is_none());
    }

    #[test]
    fn test_sps_message() {
        let q = Kuningatar::new(4);
        // Scores: [1.0, -1.0, 0.5, -0.3]
        // Max abs = 1.0
        // Normalized: [1.0, -1.0, 0.5, -0.3]
        // 1.0 > 0.7635 → +1, -1.0 < -0.7635 → -1, 0.5 < 0.7635 → 0, -0.3 > -0.7635 → 0
        let sps = q.sps_viesti(&[1.0, -1.0, 0.5, -0.3]);
        assert_eq!(sps, vec![1, -1, 0, 0]);
    }

    #[test]
    fn test_agreement_patterns() {
        let q = Kuningatar::new(4);

        // All agree positive
        assert_eq!(q.analysoi_sopimus(1, 1, 1), SopimisKuvio::TäysiSopimus);
        // Two agree, one zero
        assert_eq!(q.analysoi_sopimus(1, 1, 0), SopimisKuvio::Enemmistö);
        // Only one active
        assert_eq!(q.analysoi_sopimus(1, 0, 0), SopimisKuvio::YksinSointi);
        // All silent
        assert_eq!(q.analysoi_sopimus(0, 0, 0), SopimisKuvio::Hiljaisuus);
        // Active disagreement
        assert_eq!(q.analysoi_sopimus(1, -1, 0), SopimisKuvio::Ristiriita);
    }

    #[test]
    fn test_combine_scores() {
        let mut q = Kuningatar::new(4);
        let sa = [1.0, 0.5, 0.0, -0.5];
        let sb = [0.8, 0.6, 0.1, -0.4];
        let sc = [0.9, 0.4, -0.1, -0.6];

        let (combined, diag) = q.yhdistä(
            [&sa, &sb, &sc],
            [0.5, 0.4, 0.3],
        );

        assert_eq!(combined.len(), 4);
        assert!(diag.koherenssi > 0.0);
        assert!(diag.tila == "monophonic" || diag.tila == "polyphonic");
    }

    #[test]
    fn test_monophonic_downshift() {
        let mut q = Kuningatar::new(4);
        // Force low coherence by repeated disagreement
        q.koherenssi_pulssi = 0.3;

        let sa = [1.0, 0.0, 0.0, 0.0];
        let sb = [0.0, 1.0, 0.0, 0.0];
        let sc = [0.0, 0.0, 1.0, 0.0];

        let (_, diag) = q.yhdistä(
            [&sa, &sb, &sc],
            [0.8, 0.3, 0.2],
        );

        assert_eq!(diag.tila, "monophonic");
        assert_eq!(diag.johtaja, Some(0)); // engine 0 has highest accuracy
    }

    #[test]
    fn test_polyphonic_mode() {
        let mut q = Kuningatar::new(4);
        // Force high coherence
        q.koherenssi_pulssi = 0.9;

        let sa = [1.0, 0.5, 0.0, -0.5];
        let sb = [0.9, 0.4, 0.1, -0.6];
        let sc = [0.8, 0.6, -0.1, -0.4];

        let (_, diag) = q.yhdistä(
            [&sa, &sb, &sc],
            [0.5, 0.5, 0.5],
        );

        assert_eq!(diag.tila, "polyphonic");
        assert!(diag.johtaja.is_none());
    }

    #[test]
    fn test_reset() {
        let mut q = Kuningatar::new(4);
        q.koherenssi_pulssi = 0.9;
        q.ennusteet = 100;
        q.nollaa();
        assert_eq!(q.koherenssi_pulssi, 0.5);
        assert_eq!(q.ennusteet, 0);
    }
}
