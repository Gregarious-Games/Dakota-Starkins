//! Kaksoisnapainen — Bipyramid Hierarchical Prediction
//! ====================================================
//!
//! Rose's geometric insight: 3 trirectangular bipyramids with 3-fold
//! rotational symmetry. Each bipyramid has 6 triangular faces. Three
//! compound to 18 faces. Two poles along the symmetry axis.
//!
//!   2 poles × 18 faces = 36 + axis = 37
//!
//! This is the exact factorization of our 37-character alphabet.
//!
//! Instead of forcing D=256 to separate 37 characters simultaneously,
//! decompose into two stages:
//!
//!   Stage 1 — Pole Selection (binary: sonorant vs obstruent)
//!   Stage 2 — Face Selection (argmax within winning pole, ~18 chars)
//!
//! D=256 separates 18 classes cleanly (needs ~15). The binary pole
//! selection is nearly free — sonorants and obstruents occupy
//! different regions of phonetic space.
//!
//! Classification rule (from QAMS signatures):
//!   - Akseli (axis): space, newline — the silence channel
//!   - Sonorantti (Pole A): voiced + non-stop (ääntö > 0 AND tapa > -0.5)
//!     → vowels, nasals, approximants, voiced fricatives
//!   - Obstruentti (Pole B): everything else
//!     → stops, unvoiced fricatives, affricates, punctuation, digits
//!
//! Authors: Dakota (Claude), Rose (Claude) & Greg Calkins
//! Date:    February 23, 2026

use std::collections::HashMap;
use crate::qams_codebook::{kaikki_allekirjoitukset, ÄänneAllekirjoitus};

// ═══════════════════════════════════════════════════════════════════
// POLE TYPES
// ═══════════════════════════════════════════════════════════════════

/// The three regions of the bipyramid geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Napa {
    /// Axis — the silence channel (space, newline)
    Akseli,
    /// Pole A — sonorants: vowels, nasals, approximants, voiced fricatives
    Sonorantti,
    /// Pole B — obstruents: stops, unvoiced fricatives, affricates, punctuation
    Obstruentti,
}

// ═══════════════════════════════════════════════════════════════════
// BIPYRAMID MAP
// ═══════════════════════════════════════════════════════════════════

/// Maps each character in the alphabet to its bipyramid pole.
pub struct KaksoisnapainenKartta {
    /// char → pole assignment
    napa_kartta: HashMap<char, Napa>,
}

impl KaksoisnapainenKartta {
    /// Build pole assignments from the QAMS phonetic signatures.
    ///
    /// Rule:
    ///   - space/newline → Akseli
    ///   - ääntö > 0 AND tapa > -0.5 → Sonorantti
    ///   - everything else → Obstruentti
    pub fn new(aakkosto: &[char]) -> Self {
        let allekirjoitukset = kaikki_allekirjoitukset();
        let mut napa_kartta = HashMap::new();

        for &merkki in aakkosto {
            let napa = luokittele_merkki(merkki, allekirjoitukset.get(&merkki));
            napa_kartta.insert(merkki, napa);
        }

        Self { napa_kartta }
    }

    /// Get the pole for a character. Returns Obstruentti for unknown chars.
    pub fn napa(&self, merkki: char) -> Napa {
        self.napa_kartta.get(&merkki).copied().unwrap_or(Napa::Obstruentti)
    }

    /// Get all characters assigned to a given pole.
    pub fn navan_merkit(&self, napa: Napa) -> Vec<char> {
        let mut merkit: Vec<char> = self.napa_kartta.iter()
            .filter(|(_, &n)| n == napa)
            .map(|(&c, _)| c)
            .collect();
        merkit.sort();
        merkit
    }

    /// Print pole assignment summary.
    pub fn tulosta_yhteenveto(&self) {
        let akseli = self.navan_merkit(Napa::Akseli);
        let sonorantti = self.navan_merkit(Napa::Sonorantti);
        let obstruentti = self.navan_merkit(Napa::Obstruentti);

        println!("  [Bipyramid] Pole assignments:");
        println!("    Akseli ({} chars):      {:?}",
            akseli.len(), näytä_merkit(&akseli));
        println!("    Sonorantti ({} chars):  {:?}",
            sonorantti.len(), näytä_merkit(&sonorantti));
        println!("    Obstruentti ({} chars): {:?}",
            obstruentti.len(), näytä_merkit(&obstruentti));
    }

    /// Transition-based pole selection: sum P(c|prev) by pole.
    ///
    /// Uses the bigram transition probabilities from Keskus to determine
    /// which pole the next character is most likely in.
    ///
    /// P(Sonorantti | prev) = Σ P(c | prev) for c ∈ Sonorantti
    ///
    /// Returns (winning_pole, pole_probs, margin).
    /// Margin = difference between top-2 pole probabilities.
    pub fn valitse_napa_siirtymä<F>(
        &self,
        siirtymä_p: F,
    ) -> (Napa, HashMap<Napa, f64>, f64)
    where
        F: Fn(char) -> f64,
    {
        let mut napa_summa: HashMap<Napa, f64> = HashMap::new();

        for (&merkki, &napa) in &self.napa_kartta {
            let p = siirtymä_p(merkki);
            *napa_summa.entry(napa).or_insert(0.0) += p;
        }

        // Sort poles by total probability (descending)
        let mut järjestetty: Vec<(Napa, f64)> = napa_summa.iter()
            .map(|(&n, &s)| (n, s))
            .collect();
        järjestetty.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let voittaja = järjestetty.first().map(|&(n, _)| n).unwrap_or(Napa::Obstruentti);
        let marginaali = if järjestetty.len() >= 2 {
            järjestetty[0].1 - järjestetty[1].1
        } else {
            f64::MAX
        };

        (voittaja, napa_summa, marginaali)
    }
}

// ═══════════════════════════════════════════════════════════════════
// CLASSIFICATION LOGIC
// ═══════════════════════════════════════════════════════════════════

/// Classify a single character into its bipyramid pole.
fn luokittele_merkki(merkki: char, allekirjoitus: Option<&ÄänneAllekirjoitus>) -> Napa {
    // Axis: whitespace characters
    if merkki == ' ' || merkki == '\n' || merkki == '\t' || merkki == '\r' {
        return Napa::Akseli;
    }

    // If we have a QAMS signature, use phonetic classification
    if let Some(alle) = allekirjoitus {
        // Sonorantti: voiced (ääntö > 0) AND not a stop (tapa > -0.5)
        // This captures: vowels, nasals, approximants, voiced fricatives
        // But NOT voiced stops (b,d,g) which have tapa = -1.0
        if alle.ääntö > 0.0 && alle.tapa > -0.5 {
            return Napa::Sonorantti;
        }
        return Napa::Obstruentti;
    }

    // Unknown characters → Obstruentti (conservative default)
    Napa::Obstruentti
}

/// Format characters for display, showing whitespace as names.
fn näytä_merkit(merkit: &[char]) -> String {
    let osat: Vec<String> = merkit.iter().map(|&c| {
        match c {
            ' ' => "SP".to_string(),
            '\n' => "NL".to_string(),
            '\t' => "TAB".to_string(),
            _ => c.to_string(),
        }
    }).collect();
    osat.join(" ")
}

// ═══════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn testi_aakkosto() -> Vec<char> {
        "abcdefghijklmnopqrstuvwxyzäöå .,!?\n".chars().collect()
    }

    #[test]
    fn test_whitespace_is_akseli() {
        let kartta = KaksoisnapainenKartta::new(&testi_aakkosto());
        assert_eq!(kartta.napa(' '), Napa::Akseli);
        assert_eq!(kartta.napa('\n'), Napa::Akseli);
    }

    #[test]
    fn test_vowels_are_sonorantti() {
        let kartta = KaksoisnapainenKartta::new(&testi_aakkosto());
        for vokaali in &['a', 'e', 'i', 'o', 'u', 'ä', 'ö'] {
            assert_eq!(kartta.napa(*vokaali), Napa::Sonorantti,
                "vowel '{}' should be Sonorantti", vokaali);
        }
    }

    #[test]
    fn test_nasals_are_sonorantti() {
        let kartta = KaksoisnapainenKartta::new(&testi_aakkosto());
        for nasaali in &['m', 'n'] {
            assert_eq!(kartta.napa(*nasaali), Napa::Sonorantti,
                "nasal '{}' should be Sonorantti", nasaali);
        }
    }

    #[test]
    fn test_approximants_are_sonorantti() {
        let kartta = KaksoisnapainenKartta::new(&testi_aakkosto());
        for approks in &['l', 'r', 'w', 'y'] {
            assert_eq!(kartta.napa(*approks), Napa::Sonorantti,
                "approximant '{}' should be Sonorantti", approks);
        }
    }

    #[test]
    fn test_unvoiced_stops_are_obstruentti() {
        let kartta = KaksoisnapainenKartta::new(&testi_aakkosto());
        for plosive in &['p', 't', 'k'] {
            assert_eq!(kartta.napa(*plosive), Napa::Obstruentti,
                "unvoiced stop '{}' should be Obstruentti", plosive);
        }
    }

    #[test]
    fn test_voiced_stops_are_obstruentti() {
        // b, d, g are voiced (ääntö=1) but stops (tapa=-1), so Obstruentti
        let kartta = KaksoisnapainenKartta::new(&testi_aakkosto());
        for plosive in &['b', 'd', 'g'] {
            assert_eq!(kartta.napa(*plosive), Napa::Obstruentti,
                "voiced stop '{}' should be Obstruentti (tapa=-1)", plosive);
        }
    }

    #[test]
    fn test_punctuation_is_obstruentti() {
        let kartta = KaksoisnapainenKartta::new(&testi_aakkosto());
        for punct in &['.', ',', '!', '?'] {
            assert_eq!(kartta.napa(*punct), Napa::Obstruentti,
                "punctuation '{}' should be Obstruentti", punct);
        }
    }

    #[test]
    fn test_pole_balance() {
        let kartta = KaksoisnapainenKartta::new(&testi_aakkosto());
        let son = kartta.navan_merkit(Napa::Sonorantti);
        let obs = kartta.navan_merkit(Napa::Obstruentti);
        let aks = kartta.navan_merkit(Napa::Akseli);
        // Should be roughly balanced: ~17 sonorant, ~18 obstruent, 2 axis
        assert!(son.len() >= 10, "Sonorantti should have >= 10 chars, got {}", son.len());
        assert!(obs.len() >= 10, "Obstruentti should have >= 10 chars, got {}", obs.len());
        assert!(aks.len() >= 1, "Akseli should have >= 1 char, got {}", aks.len());
        // Total should match alphabet
        assert_eq!(son.len() + obs.len() + aks.len(), testi_aakkosto().len());
    }

    #[test]
    fn test_pole_selection_transition() {
        let kartta = KaksoisnapainenKartta::new(&testi_aakkosto());
        // Simulate transition probs where sonorants dominate
        let (voittaja, _, marginaali) = kartta.valitse_napa_siirtymä(|c| {
            match c {
                'a' | 'e' | 'i' | 'o' | 'u' => 0.1,  // sonorant vowels dominant
                _ => 0.01,                               // everything else low
            }
        });
        assert_eq!(voittaja, Napa::Sonorantti, "Sonorantti should win with high vowel probs");
        assert!(marginaali > 0.0, "Margin should be positive");
    }
}
