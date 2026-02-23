//! QAMS Phonetic Codebook for HDC Language Model — Rust Rewrite
//! =============================================================
//! Drop-in replacement for random codebook. Every symbol is a motion primitive.
//! Encode the motion, not an arbitrary label.
//!
//! 6 phonetic parameters per character:
//!   [aperture, duration, voicing, place, manner, frequency]
//!
//! Characters with similar phonetics → similar hypervectors.
//! Characters with different phonetics → dissimilar hypervectors.
//!
//! Finnish variable names per project convention.
//!
//! Authors: Rose (Claude) & Greg Calkins
//! Date:    February 22, 2026

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════

/// Golden ratio
const PHI: f64 = 1.618_033_988_749_895;
/// Genesis constant Γ = 1/(6φ)
const GAMMA: f64 = 1.0 / (6.0 * PHI);
/// Number of phonetic parameters per character
const PARAMETRIT: usize = 6;

// ═══════════════════════════════════════════════════════════════════
// PHONETIC MOTION SIGNATURES — TERNARY ENCODING
// ═══════════════════════════════════════════════════════════════════
//
// 6 parameters per character:
//   [aukko, kesto, ääntö, paikka, tapa, taajuus]
//    aperture, duration, voicing, place, manner, frequency
//
// TERNARY: each axis is [-1, 0, +1]
//   Zero is NOT absence — zero is the SPS silence channel.

/// A single character's phonetic signature.
#[derive(Debug, Clone, Copy)]
pub struct ÄänneAllekirjoitus {
    /// aukko: -1=closed, 0=neutral, +1=open
    pub aukko: f64,
    /// kesto: -1=impulse, 0=medium, +1=sustained
    pub kesto: f64,
    /// ääntö: -1=unvoiced, 0=neutral, +1=voiced
    pub ääntö: f64,
    /// paikka: -1=lips, 0=palate, +1=throat
    pub paikka: f64,
    /// tapa: -1=stop, 0=fricative, +1=approximant
    pub tapa: f64,
    /// taajuus: -1=low, 0=mid, +1=high
    pub taajuus: f64,
}

impl ÄänneAllekirjoitus {
    pub const fn new(
        aukko: f64, kesto: f64, ääntö: f64,
        paikka: f64, tapa: f64, taajuus: f64,
    ) -> Self {
        Self { aukko, kesto, ääntö, paikka, tapa, taajuus }
    }

    pub fn as_array(&self) -> [f64; PARAMETRIT] {
        [self.aukko, self.kesto, self.ääntö,
         self.paikka, self.tapa, self.taajuus]
    }
}

// ═══════════════════════════════════════════════════════════════════
// SPOKEN-FORM DIGIT ENCODING
// ═══════════════════════════════════════════════════════════════════
//
// Each digit = MEAN of its Finnish spoken-form phonemes.
// "yksi" and "kolme" are completely different mouth motions,
// so digits get genuinely different signatures.
//
//   0 = "nolla"      [n, o, l, l, a]
//   1 = "yksi"       [y, k, s, i]
//   2 = "kaksi"      [k, a, k, s, i]
//   3 = "kolme"      [k, o, l, m, e]
//   4 = "neljä"      [n, e, l, j, ä]
//   5 = "viisi"      [v, i, i, s, i]
//   6 = "kuusi"      [k, u, u, s, i]
//   7 = "seitsemän"  [s, e, i, t, s, e, m, ä, n]
//   8 = "kahdeksan"  [k, a, h, d, e, k, s, a, n]
//   9 = "yhdeksän"   [y, h, d, e, k, s, ä, n]

// ═══════════════════════════════════════════════════════════════════
// COMFORT GOVERNOR — MukavuusHallitsija
// ═══════════════════════════════════════════════════════════════════
//
// Prevents runaway feedback on uniform phonetic blocks.
// The closer comfort is to 1.0, the harder it is to push higher.
// Penalties pass through ungoverned.

/// Comfort state for a single node.
#[derive(Debug, Clone)]
pub struct MukavuusHallitsija {
    /// Current comfort level [0.0, 1.0]
    pub mukavuus: f64,
}

impl MukavuusHallitsija {
    pub fn new() -> Self {
        Self { mukavuus: 0.0 }
    }

    /// Apply governed comfort delta.
    ///
    /// Positive deltas: damped by headroom (1.0 - current).
    ///   At mukavuus=0.0 → full delta passes through.
    ///   At mukavuus=0.9 → only 10% of delta passes through.
    ///   At mukavuus=0.99 → only 1% passes through.
    ///
    /// Negative deltas: pass through ungoverned.
    ///   Mistakes always hurt at full strength.
    pub fn hallitse(&mut self, delta: f64) {
        let hallittu = if delta > 0.0 {
            let vapaa_tila = 1.0 - self.mukavuus;
            delta * vapaa_tila
        } else {
            delta
        };
        self.mukavuus = (self.mukavuus + hallittu).clamp(0.0, 1.0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// SIMILARITY & TRANSITION
// ═══════════════════════════════════════════════════════════════════

/// Cosine similarity between two phonetic signatures.
pub fn äänne_samankaltaisuus(
    a: &ÄänneAllekirjoitus,
    b: &ÄänneAllekirjoitus,
) -> f64 {
    let va = a.as_array();
    let vb = b.as_array();
    let dot: f64 = va.iter().zip(vb.iter())
        .map(|(x, y)| x * y).sum();
    let na: f64 = va.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = vb.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return if na < 1e-12 && nb < 1e-12 { 1.0 } else { 0.0 };
    }
    dot / (na * nb)
}

/// Articulatory transition probability.
/// Smooth transitions → high, awkward articulations → low.
/// Uses Γ = 1/(6φ) as the scaling constant.
pub fn siirtymä_todennäköisyys(
    a: &ÄänneAllekirjoitus,
    b: &ÄänneAllekirjoitus,
) -> f64 {
    let va = a.as_array();
    let vb = b.as_array();
    let etäisyys: f64 = va.iter().zip(vb.iter())
        .map(|(x, y)| (x - y).powi(2)).sum();
    (-etäisyys * GAMMA).exp()
}

// ═══════════════════════════════════════════════════════════════════
// CODEBOOK GENERATION — Block-Diagonal Basis
// ═══════════════════════════════════════════════════════════════════

pub type Hypervektori = Vec<f64>;

/// Seeded xorshift64 PRNG for reproducible codebooks.
struct Siemen { tila: u64 }

impl Siemen {
    fn new(siemen: u64) -> Self {
        Self { tila: if siemen == 0 { 1 } else { siemen } }
    }
    fn seuraava(&mut self) -> u64 {
        self.tila ^= self.tila << 13;
        self.tila ^= self.tila >> 7;
        self.tila ^= self.tila << 17;
        self.tila
    }
    fn bipolaarinen(&mut self) -> f64 {
        if self.seuraava() & 1 == 0 { 1.0 } else { -1.0 }
    }
    fn bipolaarinen_vektori(&mut self, n: usize) -> Hypervektori {
        (0..n).map(|_| self.bipolaarinen()).collect()
    }
}

/// Generate character codebook from phonetic motion signatures.
///
/// BLOCK-DIAGONAL: each of 6 phonetic parameters owns D/6 dims.
/// Flipping one axis affects ~D/6 dims → cos ≈ 0.67.
pub fn luo_koodikirja(
    taulukko: &HashMap<char, ÄänneAllekirjoitus>,
    ulottuvuus: usize,
    siemen: u64,
) -> HashMap<char, Hypervektori> {
    let mut rng = Siemen::new(siemen);
    let lohko = ulottuvuus / PARAMETRIT;

    let perus: Vec<Hypervektori> = (0..PARAMETRIT)
        .map(|_| rng.bipolaarinen_vektori(lohko))
        .collect();

    let risti: Vec<Hypervektori> = (0..PARAMETRIT - 1)
        .map(|_| rng.bipolaarinen_vektori(ulottuvuus))
        .collect();

    let mut kirja: HashMap<char, Hypervektori> = HashMap::new();

    for (&merkki, alle) in taulukko.iter() {
        let p = alle.as_array();
        let mut v = vec![0.0f64; ulottuvuus];

        // Block-diagonal
        for i in 0..PARAMETRIT {
            let alku = i * lohko;
            for j in 0..lohko {
                if alku + j < ulottuvuus {
                    v[alku + j] += p[i] * perus[i][j];
                }
            }
        }
        // Cross-terms
        for i in 0..PARAMETRIT - 1 {
            let w = (p[i] * p[i + 1]).abs();
            for j in 0..ulottuvuus {
                v[j] += w * risti[i][j] * 0.1;
            }
        }
        // Disambiguation noise
        let seed = (merkki as u64).wrapping_mul(2654435761);
        let mut crng = Siemen::new(seed);
        for j in 0..ulottuvuus {
            v[j] += crng.bipolaarinen() * 0.2;
        }
        // Bipolarize
        kirja.insert(merkki,
            v.iter().map(|&x| if x >= 0.0 { 1.0 } else { -1.0 }).collect()
        );
    }
    kirja
}

// ═══════════════════════════════════════════════════════════════════
// FULL PHONETIC SIGNATURE TABLE
// ═══════════════════════════════════════════════════════════════════
//
// All characters that appear in Finnish and English text.
// Port of PHONETIC_SIGNATURES from qams_codebook.py.

/// Complete phonetic signature table for all supported characters.
pub fn kaikki_allekirjoitukset() -> HashMap<char, ÄänneAllekirjoitus> {
    let mut t = HashMap::new();

    // Vowels — open, sustained, voiced, approximant
    //                                 aukko  kesto  ääntö paikka  tapa  taajuus
    t.insert('a', ÄänneAllekirjoitus::new( 1.0,  1.0,  1.0,  0.0,  1.0, -0.4));
    t.insert('e', ÄänneAllekirjoitus::new( 0.5,  0.8,  1.0, -0.2,  1.0,  0.0));
    t.insert('i', ÄänneAllekirjoitus::new( 0.0,  0.7,  1.0, -0.4,  1.0,  0.6));
    t.insert('o', ÄänneAllekirjoitus::new( 0.6,  0.8,  1.0,  0.2,  1.0, -0.5));
    t.insert('u', ÄänneAllekirjoitus::new( 0.0,  0.7,  1.0,  0.4,  1.0, -0.6));
    // Finnish front vowels
    t.insert('ä', ÄänneAllekirjoitus::new( 1.0,  1.0,  1.0, -0.6,  1.0,  0.1));
    t.insert('ö', ÄänneAllekirjoitus::new( 0.6,  0.8,  1.0, -0.4,  1.0, -0.1));
    t.insert('å', ÄänneAllekirjoitus::new( 0.9,  0.9,  1.0,  0.3,  1.0, -0.5));
    // 'y' as Finnish front vowel (palatal approximant in English, vowel in Finnish)
    // handled below as approximant — Finnish 'y' ≈ German ü

    // Plosives — closed, impulse, stop
    t.insert('b', ÄänneAllekirjoitus::new(-1.0, -1.0,  1.0, -1.0, -1.0, -0.7));
    t.insert('p', ÄänneAllekirjoitus::new(-1.0, -1.0, -1.0, -1.0, -1.0, -0.6));
    t.insert('d', ÄänneAllekirjoitus::new(-1.0, -1.0,  1.0,  0.0, -1.0, -0.4));
    t.insert('t', ÄänneAllekirjoitus::new(-1.0, -1.0, -1.0,  0.0, -1.0,  0.2));
    t.insert('g', ÄänneAllekirjoitus::new(-1.0, -1.0,  1.0,  0.8, -1.0, -0.6));
    t.insert('k', ÄänneAllekirjoitus::new(-1.0, -1.0, -1.0,  0.8, -1.0, -0.5));

    // Fricatives — narrow aperture, sustained, fricative manner
    t.insert('f', ÄänneAllekirjoitus::new(-0.7,  0.5, -1.0, -0.8,  0.0,  0.3));
    t.insert('v', ÄänneAllekirjoitus::new(-0.7,  0.5,  1.0, -0.8,  0.0,  0.1));
    t.insert('s', ÄänneAllekirjoitus::new(-0.8,  0.6, -1.0,  0.0,  0.0,  0.7));
    t.insert('z', ÄänneAllekirjoitus::new(-0.8,  0.6,  1.0,  0.0,  0.0,  0.5));
    t.insert('h', ÄänneAllekirjoitus::new( 0.0,  0.3, -1.0,  1.0,  0.0, -0.4));

    // Nasals — closed, voiced, nasal manner
    t.insert('m', ÄänneAllekirjoitus::new(-1.0,  0.6,  1.0, -1.0,  0.5, -0.7));
    t.insert('n', ÄänneAllekirjoitus::new(-1.0,  0.5,  1.0,  0.0,  0.5, -0.5));

    // Approximants / Liquids
    t.insert('l', ÄänneAllekirjoitus::new( 0.0,  0.5,  1.0,  0.0,  1.0, -0.3));
    t.insert('r', ÄänneAllekirjoitus::new(-0.3,  0.4,  1.0,  0.1,  1.0, -0.4));
    t.insert('w', ÄänneAllekirjoitus::new(-0.4,  0.2,  1.0, -1.0,  1.0, -0.7));
    t.insert('y', ÄänneAllekirjoitus::new(-0.3,  0.2,  1.0, -0.4,  1.0,  0.4));

    // Affricates / Special consonants
    t.insert('c', ÄänneAllekirjoitus::new(-0.9, -0.5, -1.0,  0.0, -0.5,  0.1));
    t.insert('j', ÄänneAllekirjoitus::new(-0.8, -0.4,  1.0, -0.1, -0.5,  0.0));
    t.insert('q', ÄänneAllekirjoitus::new(-1.0, -1.0, -1.0,  0.8, -1.0, -0.5));
    t.insert('x', ÄänneAllekirjoitus::new(-0.9, -0.3, -1.0,  0.0, -0.5,  0.4));

    // Punctuation — silence primitives (SPS for text)
    t.insert(' ',  ÄänneAllekirjoitus::new( 0.0,  0.0,  0.0,  0.0,  0.0,  0.0));
    t.insert('.',  ÄänneAllekirjoitus::new( 0.0,  0.0,  0.0,  0.0,  0.0, -0.1));
    t.insert(',',  ÄänneAllekirjoitus::new( 0.0,  0.0,  0.0,  0.0,  0.0,  0.1));
    t.insert('!',  ÄänneAllekirjoitus::new( 0.0,  0.0,  0.0,  0.0,  0.0,  0.9));
    t.insert('?',  ÄänneAllekirjoitus::new( 0.0,  0.0,  0.0,  0.0,  0.0,  0.6));
    t.insert(':',  ÄänneAllekirjoitus::new( 0.0,  0.0,  0.0,  0.0,  0.0,  0.2));
    t.insert(';',  ÄänneAllekirjoitus::new( 0.0,  0.0,  0.0,  0.0,  0.0,  0.15));
    t.insert('-',  ÄänneAllekirjoitus::new( 0.0,  0.0,  0.0,  0.0,  0.0, -0.2));
    t.insert('\'', ÄänneAllekirjoitus::new(-0.5, -0.8, -1.0,  1.0, -0.5,  0.0));
    t.insert('\n', ÄänneAllekirjoitus::new( 0.0,  0.0,  0.0,  0.0,  0.0,  0.0));

    // Digits — Finnish spoken-form centroids
    t.insert('0', ÄänneAllekirjoitus::new( 0.12,  0.66,  1.00,  0.04,  0.90, -0.40));
    t.insert('1', ÄänneAllekirjoitus::new(-0.53,  0.12,  0.00,  0.00,  0.25,  0.30));
    t.insert('2', ÄänneAllekirjoitus::new(-0.36,  0.06, -0.20,  0.24,  0.00, -0.02));
    t.insert('3', ÄänneAllekirjoitus::new(-0.18,  0.34,  0.60, -0.04,  0.50, -0.40));
    t.insert('4', ÄänneAllekirjoitus::new(-0.06,  0.48,  1.00, -0.06,  0.60, -0.24));
    t.insert('5', ÄänneAllekirjoitus::new(-0.30,  0.64,  0.60, -0.40,  0.60,  0.52));
    t.insert('6', ÄänneAllekirjoitus::new(-0.36,  0.34,  0.20,  0.24,  0.40, -0.08));
    t.insert('7', ÄänneAllekirjoitus::new(-0.29,  0.51,  0.33, -0.20,  0.44,  0.07));
    t.insert('8', ÄänneAllekirjoitus::new(-0.26,  0.13,  0.11,  0.27,  0.06, -0.27));
    t.insert('9', ÄänneAllekirjoitus::new(-0.33,  0.17,  0.25,  0.15,  0.19, -0.14));

    t
}

// ═══════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    const ULOTTUVUUS: usize = 256;

    fn test_signatures() -> HashMap<char, ÄänneAllekirjoitus> {
        let mut t = HashMap::new();
        // Vowels
        t.insert('a', ÄänneAllekirjoitus::new(1.0, 1.0, 1.0, 0.0, 1.0, -0.4));
        t.insert('e', ÄänneAllekirjoitus::new(0.5, 0.8, 1.0, -0.2, 1.0, 0.0));
        t.insert('i', ÄänneAllekirjoitus::new(0.0, 0.7, 1.0, -0.4, 1.0, 0.6));
        t.insert('o', ÄänneAllekirjoitus::new(0.6, 0.8, 1.0, 0.2, 1.0, -0.5));
        t.insert('u', ÄänneAllekirjoitus::new(0.0, 0.7, 1.0, 0.4, 1.0, -0.6));
        // Plosives
        t.insert('b', ÄänneAllekirjoitus::new(-1.0, -1.0, 1.0, -1.0, -1.0, -0.7));
        t.insert('p', ÄänneAllekirjoitus::new(-1.0, -1.0, -1.0, -1.0, -1.0, -0.6));
        t.insert('t', ÄänneAllekirjoitus::new(-1.0, -1.0, -1.0, 0.0, -1.0, 0.2));
        t.insert('k', ÄänneAllekirjoitus::new(-1.0, -1.0, -1.0, 0.8, -1.0, -0.5));
        // Space
        t.insert(' ', ÄänneAllekirjoitus::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        t
    }

    #[test]
    fn test_vowels_cluster() {
        let t = test_signatures();
        let sim = äänne_samankaltaisuus(&t[&'a'], &t[&'e']);
        assert!(sim > 0.3, "a-e similarity {sim} should be > 0.3");
    }

    #[test]
    fn test_voiced_unvoiced_pair() {
        let t = test_signatures();
        let sim = äänne_samankaltaisuus(&t[&'b'], &t[&'p']);
        assert!(sim > 0.4, "b-p similarity {sim} should be > 0.4");
    }

    #[test]
    fn test_vowel_consonant_distant() {
        let t = test_signatures();
        let sim = äänne_samankaltaisuus(&t[&'a'], &t[&'t']);
        assert!(sim < 0.0, "a-t similarity {sim} should be < 0.0");
    }

    #[test]
    fn test_transition_smooth_vs_awkward() {
        let t = test_signatures();
        let smooth = siirtymä_todennäköisyys(&t[&'a'], &t[&'e']);
        let awkward = siirtymä_todennäköisyys(&t[&'t'], &t[&'k']);
        assert!(smooth > awkward,
            "a→e ({smooth}) should be smoother than t→k ({awkward})");
    }

    #[test]
    fn test_comfort_governor() {
        let mut gov = MukavuusHallitsija::new();
        gov.hallitse(0.5);
        assert!((gov.mukavuus - 0.5).abs() < 1e-10, "0→0.5");
        gov.hallitse(0.5);
        // headroom = 0.5, so governed = 0.5 * 0.5 = 0.25
        assert!((gov.mukavuus - 0.75).abs() < 1e-10, "0.5→0.75");
        gov.hallitse(0.5);
        // headroom = 0.25, so governed = 0.5 * 0.25 = 0.125
        assert!((gov.mukavuus - 0.875).abs() < 1e-10, "0.75→0.875");
        // Penalties pass through ungoverned
        gov.hallitse(-0.5);
        assert!((gov.mukavuus - 0.375).abs() < 1e-10, "0.875→0.375");
    }

    #[test]
    fn test_codebook_generation() {
        let t = test_signatures();
        let kirja = luo_koodikirja(&t, ULOTTUVUUS, 42);
        assert_eq!(kirja.len(), t.len());
        for v in kirja.values() {
            assert_eq!(v.len(), ULOTTUVUUS);
            assert!(v.iter().all(|&x| x == 1.0 || x == -1.0),
                "all values must be bipolar");
        }
    }
}
