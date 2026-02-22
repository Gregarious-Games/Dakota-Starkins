//! LuokkaKertymä — Prototype Accumulator for HDC Language Model
//! =============================================================
//! Standard HDC classification pipeline for next-character prediction.
//!
//! Training:    slide window → encode context → accumulate into prototypes
//! Inference:   encode context → cosine similarity against prototypes → argmax
//! Retraining:  misclassified → subtract from wrong, add to correct
//!
//! After one pass: ~65 prototype vectors, each the superposition of
//! every context that preceded that character in the training text.
//! Two or three retraining passes typically jump accuracy 5-15%.
//!
//! Finnish variable names per project convention.
//!
//! Authors: Dakota (Claude) & Greg Calkins
//! Date:    February 22, 2026

use std::collections::HashMap;

use crate::hdc_primitives::{
    HdcPeruskäsitteet, Hypervektori, Siemen,
    ULOTTUVUUS,
};
use crate::konteksti_sitoja::KontekstiSitoja;

// ═══════════════════════════════════════════════════════════════════
// TRAINING REPORT
// ═══════════════════════════════════════════════════════════════════

/// Results from a training + evaluation run.
#[derive(Debug, Clone)]
pub struct KoulutusRaportti {
    /// Accuracy after single-pass training (before retraining)
    pub tarkkuus_alku: f64,
    /// Accuracy after each retraining pass
    pub tarkkuus_kierrokset: Vec<f64>,
    /// Number of character classes (prototypes)
    pub luokka_lkm: usize,
    /// Total training positions
    pub näyte_lkm: usize,
    /// Top-5 most frequent predicted characters
    pub yleisimmät: Vec<(char, usize)>,
}

// ═══════════════════════════════════════════════════════════════════
// PROTOTYPE ACCUMULATOR — LuokkaKertymä
// ═══════════════════════════════════════════════════════════════════

/// HDC prototype accumulator for next-character prediction.
///
/// Each character class has a prototype vector = superposition of
/// all context vectors that preceded that character in training text.
/// Inference: encode context → cosine similarity → argmax.
pub struct LuokkaKertymä {
    /// Character prototypes: accumulated context vectors
    prototyypit: HashMap<char, Hypervektori>,
    /// Dimension
    ulottuvuus: usize,
}

impl LuokkaKertymä {
    pub fn new(ulottuvuus: usize) -> Self {
        Self {
            prototyypit: HashMap::new(),
            ulottuvuus,
        }
    }

    /// Number of character classes (prototypes).
    pub fn luokka_lkm(&self) -> usize {
        self.prototyypit.len()
    }

    /// Get a prototype vector (for inspection/debugging).
    pub fn prototyyppi(&self, merkki: char) -> Option<&Hypervektori> {
        self.prototyypit.get(&merkki)
    }

    // ════════════════════════════════════════════════════════════════
    // SINGLE-PASS TRAINING
    // ════════════════════════════════════════════════════════════════

    /// Train: slide window across text, accumulate context vectors
    /// into character class prototypes.
    ///
    /// For each position i, the context is the preceding characters
    /// (up to window size). The label is text[i] (the character to predict).
    /// The context vector is added to that character's prototype.
    pub fn kouluta(
        &mut self,
        teksti: &str,
        koodikirja: &HashMap<char, Hypervektori>,
        sitoja: &KontekstiSitoja,
        hdc: &mut HdcPeruskäsitteet,
    ) -> usize {
        let merkit: Vec<char> = teksti.chars()
            .map(|c| c.to_lowercase().next().unwrap_or(c))
            .collect();
        let mut laskuri = 0usize;

        for i in 1..merkit.len() {
            // Build context from preceding characters
            let alku = if i > sitoja.ikkuna { i - sitoja.ikkuna } else { 0 };
            let konteksti_merkit = &merkit[alku..i];

            let vekit: Vec<&[f64]> = konteksti_merkit.iter()
                .filter_map(|c| koodikirja.get(c).map(|v| v.as_slice()))
                .collect();
            if vekit.is_empty() {
                continue;
            }

            let konteksti = sitoja.moniskaala_konteksti(&vekit, hdc);
            let merkki = merkit[i];

            // Accumulate into prototype
            let proto = self.prototyypit
                .entry(merkki)
                .or_insert_with(|| vec![0.0; self.ulottuvuus]);
            for (p, &k) in proto.iter_mut().zip(konteksti.iter()) {
                *p += k;
            }
            laskuri += 1;
        }

        laskuri
    }

    // ════════════════════════════════════════════════════════════════
    // INFERENCE
    // ════════════════════════════════════════════════════════════════

    /// Predict next character given context vectors.
    ///
    /// Encodes context via multi-scale bundling, then finds the
    /// prototype with highest cosine similarity.
    pub fn ennusta(
        &self,
        konteksti_vekit: &[&[f64]],
        sitoja: &KontekstiSitoja,
        hdc: &mut HdcPeruskäsitteet,
    ) -> (char, f64) {
        if konteksti_vekit.is_empty() || self.prototyypit.is_empty() {
            return (' ', 0.0);
        }

        let konteksti = sitoja.moniskaala_konteksti(konteksti_vekit, hdc);
        self.ennusta_vektorista(&konteksti, hdc)
    }

    /// Predict from a pre-encoded context vector.
    fn ennusta_vektorista(
        &self,
        konteksti: &[f64],
        hdc: &HdcPeruskäsitteet,
    ) -> (char, f64) {
        let mut paras_merkki = ' ';
        let mut paras = f64::NEG_INFINITY;

        for (&merkki, proto) in &self.prototyypit {
            let sim = hdc.samankaltaisuus(konteksti, proto);
            if sim > paras {
                paras = sim;
                paras_merkki = merkki;
            }
        }

        (paras_merkki, paras)
    }

    // ════════════════════════════════════════════════════════════════
    // EVALUATION
    // ════════════════════════════════════════════════════════════════

    /// Compute accuracy on text.
    ///
    /// For each position, encode context and predict. Count correct.
    pub fn tarkkuus(
        &self,
        teksti: &str,
        koodikirja: &HashMap<char, Hypervektori>,
        sitoja: &KontekstiSitoja,
        hdc: &mut HdcPeruskäsitteet,
    ) -> f64 {
        let merkit: Vec<char> = teksti.chars()
            .map(|c| c.to_lowercase().next().unwrap_or(c))
            .collect();
        let mut oikein = 0usize;
        let mut yhteensä = 0usize;

        for i in 1..merkit.len() {
            let alku = if i > sitoja.ikkuna { i - sitoja.ikkuna } else { 0 };
            let konteksti_merkit = &merkit[alku..i];

            let vekit: Vec<&[f64]> = konteksti_merkit.iter()
                .filter_map(|c| koodikirja.get(c).map(|v| v.as_slice()))
                .collect();
            if vekit.is_empty() {
                continue;
            }

            let konteksti = sitoja.moniskaala_konteksti(&vekit, hdc);
            let (ennuste, _) = self.ennusta_vektorista(&konteksti, hdc);
            let todellinen = merkit[i];

            if ennuste == todellinen {
                oikein += 1;
            }
            yhteensä += 1;
        }

        if yhteensä == 0 { 0.0 } else { oikein as f64 / yhteensä as f64 }
    }

    // ════════════════════════════════════════════════════════════════
    // RETRAINING
    // ════════════════════════════════════════════════════════════════

    /// Retraining pass: fix misclassifications.
    ///
    /// For every wrong prediction:
    ///   - Subtract context from the wrong prototype (push away)
    ///   - Add context to the correct prototype (pull toward)
    ///
    /// Returns the number of corrections made.
    pub fn uudelleenkouluta(
        &mut self,
        teksti: &str,
        koodikirja: &HashMap<char, Hypervektori>,
        sitoja: &KontekstiSitoja,
        hdc: &mut HdcPeruskäsitteet,
    ) -> usize {
        let merkit: Vec<char> = teksti.chars()
            .map(|c| c.to_lowercase().next().unwrap_or(c))
            .collect();
        let mut korjaukset = 0usize;

        for i in 1..merkit.len() {
            let alku = if i > sitoja.ikkuna { i - sitoja.ikkuna } else { 0 };
            let konteksti_merkit = &merkit[alku..i];

            let vekit: Vec<&[f64]> = konteksti_merkit.iter()
                .filter_map(|c| koodikirja.get(c).map(|v| v.as_slice()))
                .collect();
            if vekit.is_empty() {
                continue;
            }

            let konteksti = sitoja.moniskaala_konteksti(&vekit, hdc);
            let (ennuste, _) = self.ennusta_vektorista(&konteksti, hdc);
            let todellinen = merkit[i];

            if ennuste != todellinen {
                // Subtract from wrong prototype
                if let Some(väärä) = self.prototyypit.get_mut(&ennuste) {
                    for (p, &k) in väärä.iter_mut().zip(konteksti.iter()) {
                        *p -= k;
                    }
                }
                // Add to correct prototype
                let oikea = self.prototyypit
                    .entry(todellinen)
                    .or_insert_with(|| vec![0.0; self.ulottuvuus]);
                for (p, &k) in oikea.iter_mut().zip(konteksti.iter()) {
                    *p += k;
                }
                korjaukset += 1;
            }
        }

        korjaukset
    }

    // ════════════════════════════════════════════════════════════════
    // FULL PIPELINE
    // ════════════════════════════════════════════════════════════════

    /// Full pipeline: single-pass training + N retraining passes.
    ///
    /// Returns a report with accuracy after each phase.
    pub fn kouluta_kokonaan(
        &mut self,
        teksti: &str,
        koodikirja: &HashMap<char, Hypervektori>,
        sitoja: &KontekstiSitoja,
        hdc: &mut HdcPeruskäsitteet,
        uudelleen_kierrokset: usize,
    ) -> KoulutusRaportti {
        // Single-pass training
        let näyte_lkm = self.kouluta(teksti, koodikirja, sitoja, hdc);
        let tarkkuus_alku = self.tarkkuus(teksti, koodikirja, sitoja, hdc);

        // Retraining passes
        let mut tarkkuus_kierrokset = Vec::new();
        for _ in 0..uudelleen_kierrokset {
            self.uudelleenkouluta(teksti, koodikirja, sitoja, hdc);
            let acc = self.tarkkuus(teksti, koodikirja, sitoja, hdc);
            tarkkuus_kierrokset.push(acc);
        }

        // Top-5 most common prototypes by norm
        let mut proto_norms: Vec<(char, f64)> = self.prototyypit.iter()
            .map(|(&c, v)| (c, v.iter().map(|x| x * x).sum::<f64>().sqrt()))
            .collect();
        proto_norms.sort_by(|a, b| b.1.total_cmp(&a.1));
        let yleisimmät: Vec<(char, usize)> = proto_norms.iter()
            .take(5)
            .map(|&(c, n)| (c, n as usize))
            .collect();

        KoulutusRaportti {
            tarkkuus_alku,
            tarkkuus_kierrokset,
            luokka_lkm: self.prototyypit.len(),
            näyte_lkm,
            yleisimmät,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// UTILITY: random codebook from text
// ═══════════════════════════════════════════════════════════════════

/// Generate a random bipolar codebook for all unique characters in text.
///
/// Baseline comparison — no phonetic structure, just random vectors.
pub fn luo_satunnainen_koodikirja(
    teksti: &str,
    ulottuvuus: usize,
    siemen_arvo: u64,
) -> HashMap<char, Hypervektori> {
    let mut rng = Siemen::new(siemen_arvo);
    let mut kirja = HashMap::new();
    for c in teksti.chars() {
        let c_lower = c.to_lowercase().next().unwrap_or(c);
        kirja.entry(c_lower)
            .or_insert_with(|| rng.bipolaarinen_vektori(ulottuvuus));
    }
    kirja
}

// ═══════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // Kalevala opening — enough for a smoke test
    const KALEVALA: &str = "\
Mieleni minun tekevi,
aivoni ajattelevi
lähteäni laulamahan,
saa'ani sanelemahan,
sukuvirttä suoltamahan,
lajivirttä laulamahan.
Sanat suussani sulavat,
puhe'et putoelevat,
kielelleni kerkiävät,
hampahilleni hajoovat.";

    fn luo_hdc() -> HdcPeruskäsitteet {
        HdcPeruskäsitteet::new(ULOTTUVUUS, 42)
    }

    #[test]
    fn test_single_pass_learns() {
        let mut hdc = luo_hdc();
        let sitoja = KontekstiSitoja::new(3);
        let koodikirja = luo_satunnainen_koodikirja(KALEVALA, ULOTTUVUUS, 99);
        let mut kertymä = LuokkaKertymä::new(ULOTTUVUUS);

        let näytteet = kertymä.kouluta(KALEVALA, &koodikirja, &sitoja, &mut hdc);
        assert!(näytteet > 0, "should have training samples");
        assert!(kertymä.luokka_lkm() > 5,
            "should learn multiple character classes, got {}", kertymä.luokka_lkm());

        let tarkkuus = kertymä.tarkkuus(KALEVALA, &koodikirja, &sitoja, &mut hdc);
        assert!(tarkkuus > 0.0,
            "accuracy after training should be > 0%, got {:.1}%", tarkkuus * 100.0);
    }

    #[test]
    fn test_retraining_improves() {
        let mut hdc = luo_hdc();
        let sitoja = KontekstiSitoja::new(3);
        let koodikirja = luo_satunnainen_koodikirja(KALEVALA, ULOTTUVUUS, 99);
        let mut kertymä = LuokkaKertymä::new(ULOTTUVUUS);

        kertymä.kouluta(KALEVALA, &koodikirja, &sitoja, &mut hdc);
        let ennen = kertymä.tarkkuus(KALEVALA, &koodikirja, &sitoja, &mut hdc);

        // Two retraining passes
        kertymä.uudelleenkouluta(KALEVALA, &koodikirja, &sitoja, &mut hdc);
        kertymä.uudelleenkouluta(KALEVALA, &koodikirja, &sitoja, &mut hdc);
        let jälkeen = kertymä.tarkkuus(KALEVALA, &koodikirja, &sitoja, &mut hdc);

        assert!(jälkeen >= ennen,
            "retraining should not decrease accuracy: before={:.1}% after={:.1}%",
            ennen * 100.0, jälkeen * 100.0);
    }

    #[test]
    fn test_full_pipeline_reports() {
        let mut hdc = luo_hdc();
        let sitoja = KontekstiSitoja::new(3);
        let koodikirja = luo_satunnainen_koodikirja(KALEVALA, ULOTTUVUUS, 99);
        let mut kertymä = LuokkaKertymä::new(ULOTTUVUUS);

        let raportti = kertymä.kouluta_kokonaan(
            KALEVALA, &koodikirja, &sitoja, &mut hdc, 3,
        );

        assert!(raportti.näyte_lkm > 100, "should have >100 samples");
        assert!(raportti.luokka_lkm > 10, "should have >10 classes");
        assert_eq!(raportti.tarkkuus_kierrokset.len(), 3, "3 retraining passes");

        // Print results for visibility
        eprintln!("\n  LuokkaKertymä Pipeline (random codebook, Kalevala excerpt):");
        eprintln!("    Classes: {}, Samples: {}", raportti.luokka_lkm, raportti.näyte_lkm);
        eprintln!("    Single-pass accuracy: {:.1}%", raportti.tarkkuus_alku * 100.0);
        for (i, &acc) in raportti.tarkkuus_kierrokset.iter().enumerate() {
            eprintln!("    Retrain pass {}: {:.1}%", i + 1, acc * 100.0);
        }
    }

    #[test]
    fn test_qams_vs_random() {
        let mut hdc = luo_hdc();
        let sitoja = KontekstiSitoja::new(3);

        // Random codebook
        let satunnainen = luo_satunnainen_koodikirja(KALEVALA, ULOTTUVUUS, 99);
        let mut kertymä_r = LuokkaKertymä::new(ULOTTUVUUS);
        kertymä_r.kouluta(KALEVALA, &satunnainen, &sitoja, &mut hdc);
        kertymä_r.uudelleenkouluta(KALEVALA, &satunnainen, &sitoja, &mut hdc);
        let tarkkuus_r = kertymä_r.tarkkuus(KALEVALA, &satunnainen, &sitoja, &mut hdc);

        // QAMS codebook
        let allekirjoitukset = crate::qams_codebook::kaikki_allekirjoitukset();
        let qams = crate::qams_codebook::luo_koodikirja(&allekirjoitukset, ULOTTUVUUS, 42);
        let mut kertymä_q = LuokkaKertymä::new(ULOTTUVUUS);
        kertymä_q.kouluta(KALEVALA, &qams, &sitoja, &mut hdc);
        kertymä_q.uudelleenkouluta(KALEVALA, &qams, &sitoja, &mut hdc);
        let tarkkuus_q = kertymä_q.tarkkuus(KALEVALA, &qams, &sitoja, &mut hdc);

        eprintln!("\n  QAMS vs Random (Kalevala excerpt, 1 retrain):");
        eprintln!("    Random: {:.1}%", tarkkuus_r * 100.0);
        eprintln!("    QAMS:   {:.1}%", tarkkuus_q * 100.0);

        // Both should learn something
        assert!(tarkkuus_r > 0.0, "random should learn: {:.1}%", tarkkuus_r * 100.0);
        assert!(tarkkuus_q > 0.0, "QAMS should learn: {:.1}%", tarkkuus_q * 100.0);
    }

    #[test]
    fn test_prediction_returns_valid_char() {
        let mut hdc = luo_hdc();
        let sitoja = KontekstiSitoja::new(3);
        let koodikirja = luo_satunnainen_koodikirja(KALEVALA, ULOTTUVUUS, 99);
        let mut kertymä = LuokkaKertymä::new(ULOTTUVUUS);
        kertymä.kouluta(KALEVALA, &koodikirja, &sitoja, &mut hdc);

        // Predict from context "mie"
        let vekit: Vec<&[f64]> = ['m', 'i', 'e'].iter()
            .filter_map(|c| koodikirja.get(c).map(|v| v.as_slice()))
            .collect();
        let (merkki, pistemäärä) = kertymä.ennusta(&vekit, &sitoja, &mut hdc);

        assert!(koodikirja.contains_key(&merkki) || kertymä.prototyypit.contains_key(&merkki),
            "predicted char '{}' should exist in training data", merkki);
        assert!(pistemäärä > -1.0, "score should be reasonable: {pistemäärä}");
    }

    #[test]
    fn test_empty_text() {
        let mut hdc = luo_hdc();
        let sitoja = KontekstiSitoja::new(3);
        let koodikirja = luo_satunnainen_koodikirja("abc", ULOTTUVUUS, 99);
        let mut kertymä = LuokkaKertymä::new(ULOTTUVUUS);

        let näytteet = kertymä.kouluta("", &koodikirja, &sitoja, &mut hdc);
        assert_eq!(näytteet, 0);
        assert_eq!(kertymä.luokka_lkm(), 0);
    }
}
