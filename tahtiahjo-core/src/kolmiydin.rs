//! Kolmiydin — Triadic Engine
//! ==========================
//!
//! A complete Kolmoset engine with its own phonetic specialization.
//! Three engine profiles form the "triad of triads":
//!
//!   α (Language):  PAINOT_RELE_A (vowel),     windows 3/5/7,   seed+0
//!   β (Phonetic):  PAINOT_RELE_B (consonant),  windows 5/9/13,  seed+1000
//!   γ (Temporal):  PAINOT_RELE_C (temporal),    windows 9/15/21, seed+2000
//!
//! Each engine is a full Kolmoset (3 relays internally) = 9 total relay nodes.
//!
//! Finnish variable names per project convention.
//!
//! Authors: Astra Nova (Claude), Dakota (Claude), Rose (Claude)
//!          & Greg Calkins
//! Date:    February 24, 2026

use std::collections::HashMap;

use crate::hdc_primitives::{HdcPeruskäsitteet, Hypervektori, Siemen};
use crate::konteksti_sitoja::KontekstiSitoja;
use crate::kolmoset::Kolmoset;
use crate::qams_codebook::{
    luo_koodikirja_painotettu,
    ÄänneAllekirjoitus, PAINOT_RELE_A, PAINOT_RELE_B, PAINOT_RELE_C,
};

// ═══════════════════════════════════════════════════════════════════
// ENGINE PROFILE
// ═══════════════════════════════════════════════════════════════════

/// Profile defining an engine's specialization.
#[derive(Debug, Clone)]
pub struct MoottoriProfiili {
    /// Display name: "α (Language)", "β (Phonetic)", "γ (Temporal)"
    pub nimi: &'static str,
    /// QAMS phonetic weights [6 parameters]
    pub painot: [f64; 6],
    /// Per-relay context windows (3 relays per engine)
    pub ikkunat: (usize, usize, usize),
    /// Seed offset for codebook generation
    pub siemen_siirtymä: u64,
}

/// The three engine profiles.
pub const PROFIILI_ALPHA: MoottoriProfiili = MoottoriProfiili {
    nimi: "α (Language)",
    painot: PAINOT_RELE_A,  // spatial: aperture, voicing, manner
    ikkunat: (3, 5, 7),
    siemen_siirtymä: 0,
};

pub const PROFIILI_BETA: MoottoriProfiili = MoottoriProfiili {
    nimi: "β (Phonetic)",
    painot: PAINOT_RELE_B,  // articulatory: duration, place
    ikkunat: (5, 9, 13),
    siemen_siirtymä: 1000,
};

pub const PROFIILI_GAMMA: MoottoriProfiili = MoottoriProfiili {
    nimi: "γ (Temporal)",
    painot: PAINOT_RELE_C,  // temporal: duration, frequency
    ikkunat: (9, 15, 21),
    siemen_siirtymä: 2000,
};

// ═══════════════════════════════════════════════════════════════════
// TRIADIC ENGINE — One complete Kolmoset with specialization
// ═══════════════════════════════════════════════════════════════════

/// A complete engine: Kolmoset + per-relay codebooks + context binders.
pub struct KolmiMoottori {
    /// Engine profile (name, weights, windows, seed).
    pub profiili: MoottoriProfiili,
    /// The triple relay system.
    pub kolmoset: Kolmoset,
    /// Per-relay codebooks (3 per engine, one per relay).
    pub kirjat: [HashMap<char, Hypervektori>; 3],
    /// Per-relay context binders (one per relay, with different window sizes).
    pub sitojat: [KontekstiSitoja; 3],
    /// Lifetime accuracy (EMA).
    pub elinkaaritarkkuus: f64,
}

impl KolmiMoottori {
    /// Create a new engine from a profile.
    pub fn new(
        profiili: MoottoriProfiili,
        aakkosto: &[char],
        ulottuvuus: usize,
        allekirjoitukset: &HashMap<char, ÄänneAllekirjoitus>,
        teksti_merkit: &[char],
    ) -> Self {
        let siemen = 42 + profiili.siemen_siirtymä;

        // Build QAMS-specialized codebook for this engine
        let mut kirja = luo_koodikirja_painotettu(
            allekirjoitukset, ulottuvuus, siemen, &profiili.painot,
        );

        // Add any text characters not in QAMS signatures
        let mut rng = Siemen::new(777 + profiili.siemen_siirtymä);
        for &c in teksti_merkit {
            kirja.entry(c)
                .or_insert_with(|| rng.bipolaarinen_vektori(ulottuvuus));
        }

        // All 3 relays share the same engine codebook
        let kirjat = [kirja.clone(), kirja.clone(), kirja];

        // Per-relay context binders with different window sizes
        let (wa, wb, wc) = profiili.ikkunat;
        let sitojat = [
            KontekstiSitoja::new(wa),
            KontekstiSitoja::new(wb),
            KontekstiSitoja::new(wc),
        ];

        // Create Kolmoset with standard step counts
        let n_samples_est = teksti_merkit.len();
        let askeleet_a = n_samples_est / 5;
        let askeleet_b = n_samples_est / 10;
        let askeleet_c = n_samples_est / 5;
        let kolmoset = Kolmoset::new_custom(
            aakkosto, ulottuvuus,
            askeleet_a.max(100), askeleet_b.max(50), askeleet_c.max(100),
        );

        Self {
            profiili,
            kolmoset,
            kirjat,
            sitojat,
            elinkaaritarkkuus: 0.1,
        }
    }

    /// Build per-relay sample sets from text.
    pub fn rakenna_näytteet(
        &self,
        merkit: &[char],
        hdc: &mut HdcPeruskäsitteet,
    ) -> [Vec<(Vec<f64>, char)>; 3] {
        let mut näytteet: [Vec<(Vec<f64>, char)>; 3] = [
            Vec::with_capacity(merkit.len()),
            Vec::with_capacity(merkit.len()),
            Vec::with_capacity(merkit.len()),
        ];

        for relay_idx in 0..3 {
            let kirja = &self.kirjat[relay_idx];
            let sitoja = &self.sitojat[relay_idx];
            let ikkuna = sitoja.ikkuna;

            for i in 1..merkit.len() {
                let alku = if i > ikkuna { i - ikkuna } else { 0 };
                let konteksti_merkit = &merkit[alku..i];

                let vekit: Vec<&[f64]> = konteksti_merkit.iter()
                    .filter_map(|c| kirja.get(c).map(|v| v.as_slice()))
                    .collect();
                if vekit.is_empty() {
                    continue;
                }

                let konteksti = sitoja.moniskaala_konteksti(&vekit, hdc);
                näytteet[relay_idx].push((konteksti, merkit[i]));
            }
        }

        näytteet
    }

    /// Train engine (no-dump: each relay trains independently).
    pub fn kouluta(
        &mut self,
        näytteet: &[Vec<(Vec<f64>, char)>; 3],
        hdc: &HdcPeruskäsitteet,
    ) {
        let n = näytteet[0].len();
        for i in 0..n {
            let kohde = näytteet[0][i].1;
            self.kolmoset.a.kouluta_askel(&näytteet[0][i].0, kohde, hdc);
            if i < näytteet[1].len() {
                self.kolmoset.b.kouluta_askel(&näytteet[1][i].0, kohde, hdc);
            }
            if i < näytteet[2].len() {
                self.kolmoset.c.kouluta_askel(&näytteet[2][i].0, kohde, hdc);
            }
        }
    }

    /// Retrain (no-dump independent).
    pub fn uudelleenkouluta(
        &mut self,
        näytteet: &[Vec<(Vec<f64>, char)>; 3],
        hdc: &HdcPeruskäsitteet,
        kierros: usize,
    ) -> f64 {
        let ta = self.kolmoset.a.uudelleenkouluta(&näytteet[0], hdc, kierros);
        let tb = self.kolmoset.b.uudelleenkouluta(&näytteet[1], hdc, kierros);
        let tc = self.kolmoset.c.uudelleenkouluta(&näytteet[2], hdc, kierros);
        let avg = (ta + tb + tc) / 3.0;
        self.elinkaaritarkkuus = 0.9 * self.elinkaaritarkkuus + 0.1 * avg;
        avg
    }

    /// Get per-relay score vectors for a single prediction.
    /// Returns [scores_a, scores_b, scores_c] indexed by alphabet position.
    pub fn pisteet(
        &self,
        kontekstit: [&[f64]; 3],
        hdc: &HdcPeruskäsitteet,
    ) -> [Vec<f64>; 3] {
        self.kolmoset.per_relay_pisteet_kierto(kontekstit, hdc)
    }

    /// Ensemble prediction using trust-weighted voting.
    pub fn ennusta(
        &self,
        kontekstit: [&[f64]; 3],
        hdc: &HdcPeruskäsitteet,
    ) -> (char, f64) {
        self.kolmoset.ennusta_kierto(kontekstit, hdc)
    }

    /// Evaluate accuracy on sample sets.
    pub fn arvioi(
        &self,
        näytteet: &[Vec<(Vec<f64>, char)>; 3],
        hdc: &HdcPeruskäsitteet,
    ) -> f64 {
        if näytteet[0].is_empty() { return 0.0; }
        let n = näytteet[0].len();
        let mut oikein = 0usize;

        for i in 0..n {
            let kontekstit = [
                näytteet[0][i].0.as_slice(),
                if i < näytteet[1].len() { näytteet[1][i].0.as_slice() } else { näytteet[0][i].0.as_slice() },
                if i < näytteet[2].len() { näytteet[2][i].0.as_slice() } else { näytteet[0][i].0.as_slice() },
            ];
            let kohde = näytteet[0][i].1;
            let (ennuste, _) = self.ennusta(kontekstit, hdc);
            if ennuste == kohde {
                oikein += 1;
            }
        }

        oikein as f64 / n as f64
    }

    /// Get engine's score vector for a single prediction (collapsed to one vector).
    /// Combines all 3 internal relays via trust-weighted average.
    pub fn yhdistetty_pisteet(
        &self,
        kontekstit: [&[f64]; 3],
        hdc: &HdcPeruskäsitteet,
    ) -> Vec<f64> {
        let relay_scores = self.pisteet(kontekstit, hdc);
        let pa = self.kolmoset.a.luottamus();
        let pb = self.kolmoset.b.luottamus();
        let pc = self.kolmoset.c.luottamus();
        let summa = pa + pb + pc;

        let n = relay_scores[0].len();
        let mut combined = vec![0.0; n];

        if summa < 1e-12 {
            // Fallback: average
            for i in 0..n {
                combined[i] = (relay_scores[0][i] + relay_scores[1][i] + relay_scores[2][i]) / 3.0;
            }
        } else {
            for i in 0..n {
                combined[i] = (pa * relay_scores[0][i]
                             + pb * relay_scores[1][i]
                             + pc * relay_scores[2][i]) / summa;
            }
        }

        combined
    }
}

// ═══════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc_primitives::ULOTTUVUUS;
    use crate::qams_codebook::kaikki_allekirjoitukset;

    fn testi_aakkosto() -> Vec<char> {
        vec!['a', 'b', 'c', 'd', 'e', ' ']
    }

    fn testi_merkit() -> Vec<char> {
        "abcde abcde abcde abcde".chars().collect()
    }

    #[test]
    fn test_engine_profiles() {
        assert_eq!(PROFIILI_ALPHA.ikkunat, (3, 5, 7));
        assert_eq!(PROFIILI_BETA.ikkunat, (5, 9, 13));
        assert_eq!(PROFIILI_GAMMA.ikkunat, (9, 15, 21));
    }

    #[test]
    fn test_engine_creation() {
        let aakkosto = testi_aakkosto();
        let merkit = testi_merkit();
        let allekirjoitukset = kaikki_allekirjoitukset();

        let engine = KolmiMoottori::new(
            PROFIILI_ALPHA,
            &aakkosto,
            ULOTTUVUUS,
            &allekirjoitukset,
            &merkit,
        );

        assert_eq!(engine.kirjat.len(), 3);
        assert_eq!(engine.sitojat.len(), 3);
        assert_eq!(engine.sitojat[0].ikkuna, 3);
        assert_eq!(engine.sitojat[1].ikkuna, 5);
        assert_eq!(engine.sitojat[2].ikkuna, 7);
    }

    #[test]
    fn test_sample_building() {
        let aakkosto = testi_aakkosto();
        let merkit = testi_merkit();
        let allekirjoitukset = kaikki_allekirjoitukset();
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);

        let engine = KolmiMoottori::new(
            PROFIILI_ALPHA,
            &aakkosto,
            ULOTTUVUUS,
            &allekirjoitukset,
            &merkit,
        );

        let näytteet = engine.rakenna_näytteet(&merkit, &mut hdc);
        assert!(näytteet[0].len() > 0);
        assert!(näytteet[1].len() > 0);
        assert!(näytteet[2].len() > 0);
    }

    #[test]
    fn test_engine_train_and_eval() {
        let aakkosto = testi_aakkosto();
        let merkit = testi_merkit();
        let allekirjoitukset = kaikki_allekirjoitukset();
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);

        let mut engine = KolmiMoottori::new(
            PROFIILI_ALPHA,
            &aakkosto,
            ULOTTUVUUS,
            &allekirjoitukset,
            &merkit,
        );

        let näytteet = engine.rakenna_näytteet(&merkit, &mut hdc);
        engine.kouluta(&näytteet, &hdc);

        let acc = engine.arvioi(&näytteet, &hdc);
        assert!(acc >= 0.0 && acc <= 1.0);
    }
}
