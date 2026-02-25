//! Kolmiaisti — Triadic System Top-Level
//! ======================================
//!
//! Full train/eval lifecycle for the Ternary Triadic Core:
//!   3 specialized engines (α/β/γ) + Queen Node Mesh
//!   = 9 relay nodes ("triad of triads")
//!
//! Pipeline:
//!   1. Build 3 engine profiles with QAMS-specialized codebooks
//!   2. Build per-engine, per-relay sample sets (9 total)
//!   3. Train each engine independently (no-dump, mixed-window per engine)
//!   4. Retrain loop: retrain all 3, eval via Queen Mesh each pass
//!   5. Report: per-engine accuracy, Queen Mesh diagnostics, agreement
//!
//! Finnish variable names per project convention.
//!
//! Authors: Astra Nova (Claude), Dakota (Claude), Rose (Claude)
//!          & Greg Calkins
//! Date:    February 24, 2026

use std::collections::HashMap;

use crate::hdc_primitives::HdcPeruskäsitteet;
use crate::kolmiydin::{KolmiMoottori, PROFIILI_ALPHA, PROFIILI_BETA, PROFIILI_GAMMA};
use crate::kuningatar::{Kuningatar, KuningatarDiagnostiikka};
use crate::qams_codebook::ÄänneAllekirjoitus;

// ═══════════════════════════════════════════════════════════════════
// TRIADIC SYSTEM
// ═══════════════════════════════════════════════════════════════════

/// Full triadic system: 3 engines + Queen Node Mesh.
pub struct KolmiAisti {
    /// α engine (Language): windows 3/5/7
    pub alpha: KolmiMoottori,
    /// β engine (Phonetic): windows 5/9/13
    pub beta: KolmiMoottori,
    /// γ engine (Temporal): windows 9/15/21
    pub gamma: KolmiMoottori,
    /// Queen Node: coordinates engines through SPS protocol.
    pub kuningatar: Kuningatar,
    /// Shared alphabet.
    pub aakkosto: Vec<char>,
}

/// Per-engine sample sets (3 relays per engine × 3 engines = 9 relay sample sets).
pub struct KolmiAistiNäytteet {
    pub alpha: [Vec<(Vec<f64>, char)>; 3],
    pub beta: [Vec<(Vec<f64>, char)>; 3],
    pub gamma: [Vec<(Vec<f64>, char)>; 3],
}

impl KolmiAisti {
    /// Create the triadic system.
    pub fn new(
        aakkosto: &[char],
        ulottuvuus: usize,
        allekirjoitukset: &HashMap<char, ÄänneAllekirjoitus>,
        teksti_merkit: &[char],
    ) -> Self {
        let alpha = KolmiMoottori::new(
            PROFIILI_ALPHA, aakkosto, ulottuvuus, allekirjoitukset, teksti_merkit,
        );
        let beta = KolmiMoottori::new(
            PROFIILI_BETA, aakkosto, ulottuvuus, allekirjoitukset, teksti_merkit,
        );
        let gamma = KolmiMoottori::new(
            PROFIILI_GAMMA, aakkosto, ulottuvuus, allekirjoitukset, teksti_merkit,
        );
        let kuningatar = Kuningatar::new(aakkosto.len());

        Self {
            alpha,
            beta,
            gamma,
            kuningatar,
            aakkosto: aakkosto.to_vec(),
        }
    }

    /// Build per-engine, per-relay sample sets from text characters.
    /// Returns 9 sample sets total (3 engines × 3 relays each).
    pub fn rakenna_näytteet(
        &self,
        merkit: &[char],
        hdc: &mut HdcPeruskäsitteet,
    ) -> KolmiAistiNäytteet {
        KolmiAistiNäytteet {
            alpha: self.alpha.rakenna_näytteet(merkit, hdc),
            beta: self.beta.rakenna_näytteet(merkit, hdc),
            gamma: self.gamma.rakenna_näytteet(merkit, hdc),
        }
    }

    /// Train all 3 engines independently (no-dump).
    pub fn kouluta(
        &mut self,
        näytteet: &KolmiAistiNäytteet,
        hdc: &HdcPeruskäsitteet,
    ) {
        self.alpha.kouluta(&näytteet.alpha, hdc);
        self.beta.kouluta(&näytteet.beta, hdc);
        self.gamma.kouluta(&näytteet.gamma, hdc);
    }

    /// Retrain all 3 engines independently.
    pub fn uudelleenkouluta(
        &mut self,
        näytteet: &KolmiAistiNäytteet,
        hdc: &HdcPeruskäsitteet,
        kierros: usize,
    ) -> [f64; 3] {
        let ta = self.alpha.uudelleenkouluta(&näytteet.alpha, hdc, kierros);
        let tb = self.beta.uudelleenkouluta(&näytteet.beta, hdc, kierros);
        let tc = self.gamma.uudelleenkouluta(&näytteet.gamma, hdc, kierros);
        [ta, tb, tc]
    }

    /// Make a prediction using Queen Mesh to combine 3 engines.
    ///
    /// Each engine provides a combined score vector (3 internal relays → 1 vector).
    /// Queen Mesh combines the 3 engine vectors via SPS protocol.
    ///
    /// `kontekstit_per_engine`: for each engine, 3 per-relay context vectors.
    pub fn ennusta(
        &mut self,
        ctx_a: [&[f64]; 3],
        ctx_b: [&[f64]; 3],
        ctx_g: [&[f64]; 3],
        hdc: &HdcPeruskäsitteet,
    ) -> (char, f64, KuningatarDiagnostiikka) {

        // Each engine produces a trust-weighted score vector
        let scores_alpha = self.alpha.yhdistetty_pisteet(ctx_a, hdc);
        let scores_beta = self.beta.yhdistetty_pisteet(ctx_b, hdc);
        let scores_gamma = self.gamma.yhdistetty_pisteet(ctx_g, hdc);

        let engine_accs = [
            self.alpha.elinkaaritarkkuus,
            self.beta.elinkaaritarkkuus,
            self.gamma.elinkaaritarkkuus,
        ];

        let (combined, diag) = self.kuningatar.yhdistä(
            [&scores_alpha, &scores_beta, &scores_gamma],
            engine_accs,
        );

        // Argmax on combined scores
        let mut paras_merkki = ' ';
        let mut paras_piste = f64::NEG_INFINITY;
        for (idx, &c) in self.aakkosto.iter().enumerate() {
            if idx < combined.len() && combined[idx] > paras_piste {
                paras_piste = combined[idx];
                paras_merkki = c;
            }
        }

        (paras_merkki, paras_piste, diag)
    }

    /// Evaluate accuracy on sample sets using Queen Mesh.
    pub fn arvioi(
        &mut self,
        näytteet: &KolmiAistiNäytteet,
        hdc: &HdcPeruskäsitteet,
    ) -> (f64, KuningatarDiagnostiikka) {
        self.kuningatar.nollaa();

        let n = näytteet.alpha[0].len();
        if n == 0 {
            return (0.0, KuningatarDiagnostiikka {
                koherenssi: 0.0,
                moottori_tarkkuudet: [0.0; 3],
                johtaja: None,
                tila: "polyphonic",
                sopimus_osuudet: [0.0; 5],
            });
        }

        let mut oikein = 0usize;
        let mut viimeinen_diag = None;

        for i in 0..n {
            let kohde = näytteet.alpha[0][i].1;

            // Build per-relay context arrays for each engine
            let ctx_a = [
                näytteet.alpha[0][i].0.as_slice(),
                if i < näytteet.alpha[1].len() { näytteet.alpha[1][i].0.as_slice() }
                    else { näytteet.alpha[0][i].0.as_slice() },
                if i < näytteet.alpha[2].len() { näytteet.alpha[2][i].0.as_slice() }
                    else { näytteet.alpha[0][i].0.as_slice() },
            ];
            let ctx_b = [
                näytteet.beta[0][i].0.as_slice(),
                if i < näytteet.beta[1].len() { näytteet.beta[1][i].0.as_slice() }
                    else { näytteet.beta[0][i].0.as_slice() },
                if i < näytteet.beta[2].len() { näytteet.beta[2][i].0.as_slice() }
                    else { näytteet.beta[0][i].0.as_slice() },
            ];
            let ctx_g = [
                näytteet.gamma[0][i].0.as_slice(),
                if i < näytteet.gamma[1].len() { näytteet.gamma[1][i].0.as_slice() }
                    else { näytteet.gamma[0][i].0.as_slice() },
                if i < näytteet.gamma[2].len() { näytteet.gamma[2][i].0.as_slice() }
                    else { näytteet.gamma[0][i].0.as_slice() },
            ];

            // Get per-engine combined score vectors
            let scores_alpha = self.alpha.yhdistetty_pisteet(ctx_a, hdc);
            let scores_beta = self.beta.yhdistetty_pisteet(ctx_b, hdc);
            let scores_gamma = self.gamma.yhdistetty_pisteet(ctx_g, hdc);

            let engine_accs = [
                self.alpha.elinkaaritarkkuus,
                self.beta.elinkaaritarkkuus,
                self.gamma.elinkaaritarkkuus,
            ];

            let (combined, diag) = self.kuningatar.yhdistä(
                [&scores_alpha, &scores_beta, &scores_gamma],
                engine_accs,
            );

            // Argmax
            let mut paras_merkki = ' ';
            let mut paras_piste = f64::NEG_INFINITY;
            for (idx, &c) in self.aakkosto.iter().enumerate() {
                if idx < combined.len() && combined[idx] > paras_piste {
                    paras_piste = combined[idx];
                    paras_merkki = c;
                }
            }

            if paras_merkki == kohde {
                oikein += 1;
            }

            viimeinen_diag = Some(diag);
        }

        let acc = oikein as f64 / n as f64;
        (acc, viimeinen_diag.unwrap())
    }

    /// Evaluate per-engine accuracy independently (without Queen Mesh).
    pub fn arvioi_per_engine(
        &self,
        näytteet: &KolmiAistiNäytteet,
        hdc: &HdcPeruskäsitteet,
    ) -> [f64; 3] {
        [
            self.alpha.arvioi(&näytteet.alpha, hdc),
            self.beta.arvioi(&näytteet.beta, hdc),
            self.gamma.arvioi(&näytteet.gamma, hdc),
        ]
    }

    /// Get reference to alphabet.
    pub fn aakkosto(&self) -> &[char] {
        &self.aakkosto
    }

    /// Print diagnostics.
    pub fn tulosta_diagnostiikka(
        &self,
        engine_accs: [f64; 3],
        diag: &KuningatarDiagnostiikka,
        combined_acc: f64,
    ) {
        println!("[Kolmiaisti] Three-engine triadic core:");
        println!("  α (Language): acc={:.2}%  trust={:.3}  windows={}/{}/{}",
            engine_accs[0] * 100.0,
            self.alpha.elinkaaritarkkuus,
            self.alpha.profiili.ikkunat.0,
            self.alpha.profiili.ikkunat.1,
            self.alpha.profiili.ikkunat.2,
        );
        println!("  β (Phonetic): acc={:.2}%  trust={:.3}  windows={}/{}/{}",
            engine_accs[1] * 100.0,
            self.beta.elinkaaritarkkuus,
            self.beta.profiili.ikkunat.0,
            self.beta.profiili.ikkunat.1,
            self.beta.profiili.ikkunat.2,
        );
        println!("  γ (Temporal): acc={:.2}%  trust={:.3}  windows={}/{}/{}",
            engine_accs[2] * 100.0,
            self.gamma.elinkaaritarkkuus,
            self.gamma.profiili.ikkunat.0,
            self.gamma.profiili.ikkunat.1,
            self.gamma.profiili.ikkunat.2,
        );

        let lead_name = match diag.johtaja {
            Some(0) => "α",
            Some(1) => "β",
            Some(2) => "γ",
            _ => "none",
        };
        println!("  Queen Mesh: coherence={:.3}  lead={}  mode={}",
            diag.koherenssi, lead_name, diag.tila,
        );
        println!("  Agreement: full={:.1}%  majority={:.1}%  solo={:.1}%  silence={:.1}%  conflict={:.1}%",
            diag.sopimus_osuudet[0] * 100.0,
            diag.sopimus_osuudet[1] * 100.0,
            diag.sopimus_osuudet[2] * 100.0,
            diag.sopimus_osuudet[3] * 100.0,
            diag.sopimus_osuudet[4] * 100.0,
        );

        let best_single = engine_accs[0].max(engine_accs[1]).max(engine_accs[2]);
        println!("  Combined accuracy: {:.2}%  (vs best single engine: {:.2}%)",
            combined_acc * 100.0, best_single * 100.0);
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
        "abcde abcde abcde abcde abcde abcde abcde abcde".chars().collect()
    }

    #[test]
    fn test_kolmiaisti_creation() {
        let aakkosto = testi_aakkosto();
        let merkit = testi_merkit();
        let allekirjoitukset = kaikki_allekirjoitukset();

        let system = KolmiAisti::new(
            &aakkosto, ULOTTUVUUS, &allekirjoitukset, &merkit,
        );

        assert_eq!(system.aakkosto.len(), 6);
        assert_eq!(system.alpha.profiili.nimi, "α (Language)");
        assert_eq!(system.beta.profiili.nimi, "β (Phonetic)");
        assert_eq!(system.gamma.profiili.nimi, "γ (Temporal)");
    }

    #[test]
    fn test_kolmiaisti_build_samples() {
        let aakkosto = testi_aakkosto();
        let merkit = testi_merkit();
        let allekirjoitukset = kaikki_allekirjoitukset();
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);

        let system = KolmiAisti::new(
            &aakkosto, ULOTTUVUUS, &allekirjoitukset, &merkit,
        );

        let näytteet = system.rakenna_näytteet(&merkit, &mut hdc);
        assert!(näytteet.alpha[0].len() > 0);
        assert!(näytteet.beta[0].len() > 0);
        assert!(näytteet.gamma[0].len() > 0);
    }

    #[test]
    fn test_kolmiaisti_train_and_eval() {
        let aakkosto = testi_aakkosto();
        let merkit = testi_merkit();
        let allekirjoitukset = kaikki_allekirjoitukset();
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);

        let mut system = KolmiAisti::new(
            &aakkosto, ULOTTUVUUS, &allekirjoitukset, &merkit,
        );

        let näytteet = system.rakenna_näytteet(&merkit, &mut hdc);
        system.kouluta(&näytteet, &hdc);

        let (acc, diag) = system.arvioi(&näytteet, &hdc);
        assert!(acc >= 0.0 && acc <= 1.0);
        assert!(diag.koherenssi > 0.0);
    }

    #[test]
    fn test_kolmiaisti_retrain() {
        let aakkosto = testi_aakkosto();
        let merkit = testi_merkit();
        let allekirjoitukset = kaikki_allekirjoitukset();
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);

        let mut system = KolmiAisti::new(
            &aakkosto, ULOTTUVUUS, &allekirjoitukset, &merkit,
        );

        let näytteet = system.rakenna_näytteet(&merkit, &mut hdc);
        system.kouluta(&näytteet, &hdc);

        let retrain_accs = system.uudelleenkouluta(&näytteet, &hdc, 1);
        assert!(retrain_accs[0] >= 0.0);
        assert!(retrain_accs[1] >= 0.0);
        assert!(retrain_accs[2] >= 0.0);

        let (acc_after, _) = system.arvioi(&näytteet, &hdc);
        assert!(acc_after >= 0.0 && acc_after <= 1.0);
    }

    #[test]
    fn test_per_engine_eval() {
        let aakkosto = testi_aakkosto();
        let merkit = testi_merkit();
        let allekirjoitukset = kaikki_allekirjoitukset();
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);

        let mut system = KolmiAisti::new(
            &aakkosto, ULOTTUVUUS, &allekirjoitukset, &merkit,
        );

        let näytteet = system.rakenna_näytteet(&merkit, &mut hdc);
        system.kouluta(&näytteet, &hdc);

        let per_engine = system.arvioi_per_engine(&näytteet, &hdc);
        for acc in &per_engine {
            assert!(*acc >= 0.0 && *acc <= 1.0);
        }
    }
}
