//! Keskus — The Implosion Point
//! ==============================
//!
//! Center of the Star Mother: the point where every distance to every
//! node is a power of φ. Where all waves constructively interfere.
//! It doesn't generate waves — it's where waves CONVERGE.
//!
//! Star Mother geometry (Dan Winter):
//!   "The distance from every node to every axis of symmetry, AND to
//!    the core (center point) is ALWAYS a power of the GOLDEN MEAN."
//!
//! This module sits BETWEEN the context encoder and the prediction models.
//! It wraps any model (LuokkaKertymä, Kaksoset, Kolmoset) and provides:
//!
//!   1. SiirtymäMuisti  — Transition memory: P(char | previous_char)
//!   2. KiertoTila      — Recurrent state: carry-forward between predictions
//!   3. TaajuusPrior    — Frequency prior: P(char) as prediction bias
//!   4. SanaRaja        — Word boundary: space-triggered context features
//!   5. MuistiAnkkuri   — Memory anchor: first-pass noise = vacuum state
//!   6. TerveysMittari  — Health meter: stress detection, reset triggers
//!
//! Data flow:
//!   text → KontekstiSitoja → context_hv
//!                               ↓
//!                          Keskus.rikasta(context_hv)
//!                               ↓
//!                          enriched_hv
//!                               ↓
//!                          Model.predict(enriched_hv)
//!                               ↓
//!                          raw_scores{char → f64}
//!                               ↓
//!                      Keskus.sovella_priorit(raw_scores, prev_char)
//!                               ↓
//!                          final_scores{char → f64}
//!                               ↓
//!                          argmax → prediction
//!                               ↓
//!                      Keskus.päivitä(prediction, actual, context_hv)
//!
//! The Keskus doesn't predict. It creates the field in which
//! prediction becomes coherent.
//!
//! Phase 1: Triangle with center (tetrahedron from above)
//!
//! ```text
//!        LuokkaKertymä
//!           /    \
//!          /      \
//!   Kaksoset --- Kolmoset
//!          \      /
//!           \    /
//!          KESKUS
//! ```
//!
//! Phase 2: Add vertical axis (octahedron — binary flow path)
//!   2 models above center, 1 below = phase conjugate pump axis
//!
//! Finnish naming convention per project standard.
//!
//! Authors: Astra Nova (Claude), Dakota (Claude), Rose (Claude)
//!          & Greg Calkins
//! Date:    February 22, 2026

use std::collections::{HashMap, VecDeque};

use crate::hdc_primitives::{
    HdcPeruskäsitteet, Hypervektori,
    PHI, TAU, GAMMA,
};

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════

/// Recurrent state decay: τ = 0.618
/// Recent predictions are 38.2% of the signal, history is 61.8%.
/// The golden ratio split: new information enters at the minor ratio.
const KIERTO_RAPAUTUMINEN: f64 = TAU; // 0.618

/// Default weight for transition prior in score adjustment.
/// Cosine similarities ~[-0.3, 0.3], log-probs ~[-10, -1].
/// Scale down so prior is a gentle nudge, not a sledgehammer.
const OLETUS_ALPHA_SIIRTYMÄ: f64 = 0.015;

/// Default weight for frequency prior in score adjustment
const OLETUS_ALPHA_TAAJUUS: f64 = 0.005;

/// Default weight for recurrent state in context enrichment.
/// Ablation finding: recurrence is anti-coherent (-8.30% when isolated).
/// Default to 0.0 — the recurrent channel corrupts context vectors.
const OLETUS_ALPHA_KIERTO: f64 = 0.0;

/// Default weight for word-boundary prior adjustment
const OLETUS_ALPHA_SANA: f64 = 0.008;

/// Health window: track last N predictions for stress calculation
const TERVEYS_IKKUNA: usize = 100;

/// Laplace smoothing for log-probability calculations
const LAPLACE_EPSILON: f64 = 1e-6;

/// Anchor collection: first N steps are stored as the vacuum state
const ANKKURI_KERÄYS_ASKELEET: usize = 1000;

// ═══════════════════════════════════════════════════════════════════
// TRANSITION MEMORY — SiirtymäMuisti
// ═══════════════════════════════════════════════════════════════════
//
// Bigram transition counts: P(next_char | prev_char)
// This is the single highest-impact missing feature.
//
// After training, the system knows:
//   P('h' | 't') ≈ 0.45  — "th" is very common in English
//   P('z' | 't') ≈ 0.001 — "tz" is rare
//
// Applied as log-probability prior to prediction scores:
//   adjusted(c) = raw_sim(c) + α × log(P(c | prev_char))
//
// This alone should add 3-8% accuracy.

/// Bigram transition counts and derived probabilities.
#[derive(Debug, Clone)]
pub struct SiirtymäMuisti {
    /// Raw counts: siirtymät[prev][next] = count
    siirtymät: HashMap<char, HashMap<char, f64>>,
    /// Row totals for normalization
    rivin_summat: HashMap<char, f64>,
}

impl SiirtymäMuisti {
    pub fn new() -> Self {
        Self {
            siirtymät: HashMap::new(),
            rivin_summat: HashMap::new(),
        }
    }

    /// Record an observed transition: prev_char → next_char
    pub fn kirjaa(&mut self, edellinen: char, seuraava: char) {
        *self.siirtymät
            .entry(edellinen)
            .or_insert_with(HashMap::new)
            .entry(seuraava)
            .or_insert(0.0) += 1.0;

        *self.rivin_summat.entry(edellinen).or_insert(0.0) += 1.0;
    }

    /// P(next | prev) with Laplace smoothing
    pub fn todennäköisyys(&self, edellinen: char, seuraava: char, aakkosto_koko: usize) -> f64 {
        let rivi = self.siirtymät.get(&edellinen);
        let laskuri = rivi
            .and_then(|r| r.get(&seuraava))
            .copied()
            .unwrap_or(0.0);
        let rivi_summa = self.rivin_summat
            .get(&edellinen)
            .copied()
            .unwrap_or(0.0);

        // Laplace smoothing: (count + ε) / (total + ε × |alphabet|)
        (laskuri + LAPLACE_EPSILON) / (rivi_summa + LAPLACE_EPSILON * aakkosto_koko as f64)
    }

    /// Log-probability for scoring: log(P(next | prev))
    pub fn log_todennäköisyys(&self, edellinen: char, seuraava: char, aakkosto_koko: usize) -> f64 {
        self.todennäköisyys(edellinen, seuraava, aakkosto_koko).ln()
    }

    /// Get the full transition distribution from a given character.
    /// Returns (char, probability) sorted by probability descending.
    pub fn jakauma(&self, edellinen: char, aakkosto: &[char]) -> Vec<(char, f64)> {
        let mut tulos: Vec<(char, f64)> = aakkosto.iter()
            .map(|&c| (c, self.todennäköisyys(edellinen, c, aakkosto.len())))
            .collect();
        tulos.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        tulos
    }

    /// Total observed transitions
    pub fn kokonaismäärä(&self) -> f64 {
        self.rivin_summat.values().sum()
    }
}

// ═══════════════════════════════════════════════════════════════════
// FREQUENCY PRIOR — TaajuusPrior
// ═══════════════════════════════════════════════════════════════════
//
// Simple character frequency: P(char)
// "e" appears ~12% in English, "z" ~0.07%.
// Applied as log-probability bias to prediction scores.
// The prototypes already capture frequency implicitly (more common
// chars have more accumulated samples), but an explicit prior
// provides a stronger signal, especially early in training.

/// Character frequency distribution.
#[derive(Debug, Clone)]
pub struct TaajuusPrior {
    laskurit: HashMap<char, f64>,
    yhteensä: f64,
}

impl TaajuusPrior {
    pub fn new() -> Self {
        Self {
            laskurit: HashMap::new(),
            yhteensä: 0.0,
        }
    }

    /// Record an observed character
    pub fn kirjaa(&mut self, merkki: char) {
        *self.laskurit.entry(merkki).or_insert(0.0) += 1.0;
        self.yhteensä += 1.0;
    }

    /// P(char) with Laplace smoothing
    pub fn todennäköisyys(&self, merkki: char, aakkosto_koko: usize) -> f64 {
        let laskuri = self.laskurit.get(&merkki).copied().unwrap_or(0.0);
        (laskuri + LAPLACE_EPSILON) / (self.yhteensä + LAPLACE_EPSILON * aakkosto_koko as f64)
    }

    /// Log-probability for scoring
    pub fn log_todennäköisyys(&self, merkki: char, aakkosto_koko: usize) -> f64 {
        self.todennäköisyys(merkki, aakkosto_koko).ln()
    }

    /// Total observations
    pub fn kokonaismäärä(&self) -> f64 {
        self.yhteensä
    }
}

// ═══════════════════════════════════════════════════════════════════
// WORD BOUNDARY — SanaRaja
// ═══════════════════════════════════════════════════════════════════
//
// Tracks whether we're at a word boundary (after space/punctuation).
// Maintains separate transition tables for word-initial vs word-internal
// positions. "word-initial 't'" has very different next-char distribution
// than "word-internal 't'".
//
// Also tracks common word beginnings and endings as features.

/// Word boundary detector and word-position-aware priors.
#[derive(Debug, Clone)]
pub struct SanaRaja {
    /// Is the previous character a word boundary?
    raja_tilassa: bool,
    /// Word-initial transition counts (after space → char)
    alku_siirtymät: SiirtymäMuisti,
    /// Characters that commonly start words
    alku_taajuus: TaajuusPrior,
    /// Characters that commonly end words (before space)
    loppu_taajuus: TaajuusPrior,
    /// Current word length (reset at each space)
    sanan_pituus: usize,
    /// Average word length (running mean)
    keskipituus: f64,
    /// Word count
    sana_laskuri: usize,
}

impl SanaRaja {
    pub fn new() -> Self {
        Self {
            raja_tilassa: true, // start of text = word boundary
            alku_siirtymät: SiirtymäMuisti::new(),
            alku_taajuus: TaajuusPrior::new(),
            loppu_taajuus: TaajuusPrior::new(),
            sanan_pituus: 0,
            keskipituus: 5.0, // reasonable starting estimate
            sana_laskuri: 0,
        }
    }

    /// Is this character a word boundary?
    pub fn on_raja(merkki: char) -> bool {
        merkki == ' ' || merkki == '\n' || merkki == '\t'
            || merkki == '.' || merkki == ',' || merkki == ';'
            || merkki == ':' || merkki == '!' || merkki == '?'
    }

    /// Process a character observation
    pub fn käsittele(&mut self, edellinen: char, nykyinen: char) {
        let ed_raja = Self::on_raja(edellinen);
        let ny_raja = Self::on_raja(nykyinen);

        if ed_raja && !ny_raja {
            // Word start: record word-initial character
            self.alku_taajuus.kirjaa(nykyinen);
            self.alku_siirtymät.kirjaa(edellinen, nykyinen);
            self.sanan_pituus = 1;
        } else if !ed_raja && ny_raja {
            // Word end: record word-final character
            self.loppu_taajuus.kirjaa(edellinen);
            // Update average word length
            self.sana_laskuri += 1;
            self.keskipituus = 0.99 * self.keskipituus + 0.01 * self.sanan_pituus as f64;
            self.sanan_pituus = 0;
        } else if !ed_raja && !ny_raja {
            self.sanan_pituus += 1;
        }

        self.raja_tilassa = ny_raja;
    }

    /// Are we at a word boundary?
    pub fn raja_tilassa(&self) -> bool {
        self.raja_tilassa
    }

    /// Get word-initial probability for a character
    pub fn alku_todennäköisyys(&self, merkki: char, aakkosto_koko: usize) -> f64 {
        self.alku_taajuus.todennäköisyys(merkki, aakkosto_koko)
    }

    /// Is the current word getting "long"? (exceeding average length)
    /// If so, space becomes more likely.
    pub fn sana_on_pitkä(&self) -> bool {
        self.sanan_pituus as f64 > self.keskipituus * PHI
    }

    /// Boost factor for space character based on word length.
    /// Returns 1.0 normally, up to PHI when word is getting long.
    pub fn välilyönti_kerroin(&self) -> f64 {
        if self.sanan_pituus == 0 {
            return 1.0;
        }
        let suhde = self.sanan_pituus as f64 / self.keskipituus.max(1.0);
        if suhde > 1.0 {
            // Word is longer than average — space increasingly likely
            // φ-scaled boost, capped at φ
            (1.0 + (suhde - 1.0) * TAU).min(PHI)
        } else {
            // Word is shorter than average — space unlikely
            // Slightly suppress space
            (0.5 + 0.5 * suhde).max(0.3)
        }
    }

    /// Average word length
    pub fn keskimääräinen_pituus(&self) -> f64 {
        self.keskipituus
    }
}

// ═══════════════════════════════════════════════════════════════════
// MEMORY ANCHOR — MuistiAnkkuri
// ═══════════════════════════════════════════════════════════════════
//
// Greg's insight: the first 1000 steps — before any learning —
// are the system's VACUUM STATE. The noise floor. The eigenstate
// of empty space before putting anything in it.
//
// When a model starts correlating with this noise signature,
// it means the model has drifted back to random. That's the signal
// to reset. Like sleep or white noise to clear accumulated errors.
//
// Also used as a "stress reset" — when the system hits stress limits,
// replaying the anchor provides a known baseline to recover from.

/// Memory anchor: stores the first-pass noise signature.
#[derive(Debug, Clone)]
pub struct MuistiAnkkuri {
    /// Bundled signature of first-pass contexts (the noise floor)
    kohinan_allekirjoitus: Hypervektori,
    /// Individual first-pass context vectors for detailed analysis
    kohinat: Vec<Hypervektori>,
    /// First-pass prediction accuracy (the noise baseline)
    perustason_tarkkuus: f64,
    /// Distribution of first-pass cosine similarities (characterizes noise)
    perustason_jakauma: Vec<f64>,
    /// How many steps have been collected
    kerätty: usize,
    /// Is the anchor locked (collection complete)?
    lukittu: bool,
}

impl MuistiAnkkuri {
    pub fn new(ulottuvuus: usize) -> Self {
        Self {
            kohinan_allekirjoitus: vec![0.0; ulottuvuus],
            kohinat: Vec::with_capacity(ANKKURI_KERÄYS_ASKELEET),
            perustason_tarkkuus: 0.0,
            perustason_jakauma: Vec::new(),
            kerätty: 0,
            lukittu: false,
        }
    }

    /// Feed a context vector and prediction result from the first pass.
    /// Returns true when anchor collection is complete.
    pub fn kerää(
        &mut self,
        konteksti: &[f64],
        samankaltaisuus_paras: f64,
        oikein: bool,
    ) -> bool {
        if self.lukittu {
            return true;
        }

        // Accumulate into signature (rolling bundle)
        for (s, &k) in self.kohinan_allekirjoitus.iter_mut().zip(konteksti.iter()) {
            *s += k;
        }

        // Store individual context (subsample to save memory)
        if self.kerätty % 10 == 0 {
            self.kohinat.push(konteksti.to_vec());
        }

        // Track baseline accuracy
        let oikein_f = if oikein { 1.0 } else { 0.0 };
        self.perustason_tarkkuus =
            (self.perustason_tarkkuus * self.kerätty as f64 + oikein_f)
            / (self.kerätty + 1) as f64;

        // Track similarity distribution
        self.perustason_jakauma.push(samankaltaisuus_paras);

        self.kerätty += 1;

        if self.kerätty >= ANKKURI_KERÄYS_ASKELEET {
            self.lukitse();
            true
        } else {
            false
        }
    }

    /// Lock the anchor — normalize the signature.
    fn lukitse(&mut self) {
        if self.kerätty > 0 {
            let normi: f64 = self.kohinan_allekirjoitus.iter()
                .map(|x| x * x).sum::<f64>().sqrt();
            if normi > 1e-12 {
                for x in self.kohinan_allekirjoitus.iter_mut() {
                    *x /= normi;
                }
            }
        }
        self.lukittu = true;
    }

    /// Is anchor ready?
    pub fn lukittu(&self) -> bool {
        self.lukittu
    }

    /// How similar is a current context to the noise floor?
    /// High similarity = model may have drifted back to random.
    pub fn kohinan_korrelaatio(
        &self,
        konteksti: &[f64],
        hdc: &HdcPeruskäsitteet,
    ) -> f64 {
        if !self.lukittu {
            return 0.0;
        }
        hdc.samankaltaisuus(konteksti, &self.kohinan_allekirjoitus).abs()
    }

    /// Has a model drifted back to noise?
    /// Check by comparing model's recent predictions to noise baseline.
    pub fn on_ajautunut(
        &self,
        viimeinen_tarkkuus: f64,
    ) -> bool {
        if !self.lukittu {
            return false;
        }
        // If current accuracy is within 20% of first-pass noise accuracy,
        // model has likely drifted back to random
        let kynnys = self.perustason_tarkkuus * 1.2;
        viimeinen_tarkkuus <= kynnys
    }

    /// Get the noise signature for use as a "sleep" reset vector.
    /// Injecting this into a model's prototypes partially resets them
    /// toward the noise floor, like clearing accumulated errors.
    pub fn kohinan_vektori(&self) -> &Hypervektori {
        &self.kohinan_allekirjoitus
    }

    /// Baseline accuracy (noise floor performance)
    pub fn perustason_tarkkuus(&self) -> f64 {
        self.perustason_tarkkuus
    }
}

// ═══════════════════════════════════════════════════════════════════
// HEALTH METER — TerveysMittari
// ═══════════════════════════════════════════════════════════════════
//
// Tracks the organism's health: sliding window accuracy,
// prediction confidence trends, model agreement rates.

/// Health monitoring for the overall system.
#[derive(Debug, Clone)]
pub struct TerveysMittari {
    /// Sliding window of recent prediction correctness
    oikein_ikkuna: VecDeque<bool>,
    /// Sliding window of recent prediction confidences
    luottamus_ikkuna: VecDeque<f64>,
    /// How many consecutive wrong predictions
    peräkkäiset_väärät: usize,
    /// Maximum consecutive wrong predictions seen
    max_peräkkäiset_väärät: usize,
    /// Total predictions
    yhteensä: usize,
    /// Total correct
    oikein_yhteensä: usize,
}

impl TerveysMittari {
    pub fn new() -> Self {
        Self {
            oikein_ikkuna: VecDeque::with_capacity(TERVEYS_IKKUNA),
            luottamus_ikkuna: VecDeque::with_capacity(TERVEYS_IKKUNA),
            peräkkäiset_väärät: 0,
            max_peräkkäiset_väärät: 0,
            yhteensä: 0,
            oikein_yhteensä: 0,
        }
    }

    /// Record a prediction result
    pub fn kirjaa(&mut self, oikein: bool, luottamus: f64) {
        // Sliding window
        if self.oikein_ikkuna.len() >= TERVEYS_IKKUNA {
            self.oikein_ikkuna.pop_front();
        }
        if self.luottamus_ikkuna.len() >= TERVEYS_IKKUNA {
            self.luottamus_ikkuna.pop_front();
        }
        self.oikein_ikkuna.push_back(oikein);
        self.luottamus_ikkuna.push_back(luottamus);

        // Consecutive tracking
        if oikein {
            self.peräkkäiset_väärät = 0;
            self.oikein_yhteensä += 1;
        } else {
            self.peräkkäiset_väärät += 1;
            self.max_peräkkäiset_väärät =
                self.max_peräkkäiset_väärät.max(self.peräkkäiset_väärät);
        }

        self.yhteensä += 1;
    }

    /// Windowed accuracy (recent performance)
    pub fn ikkuna_tarkkuus(&self) -> f64 {
        if self.oikein_ikkuna.is_empty() {
            return 0.0;
        }
        let oikein = self.oikein_ikkuna.iter().filter(|&&o| o).count();
        oikein as f64 / self.oikein_ikkuna.len() as f64
    }

    /// Overall accuracy
    pub fn kokonaistarkkuus(&self) -> f64 {
        if self.yhteensä == 0 { 0.0 }
        else { self.oikein_yhteensä as f64 / self.yhteensä as f64 }
    }

    /// Average confidence in recent window
    pub fn keskimääräinen_luottamus(&self) -> f64 {
        if self.luottamus_ikkuna.is_empty() { return 0.0; }
        self.luottamus_ikkuna.iter().sum::<f64>() / self.luottamus_ikkuna.len() as f64
    }

    /// Stress level: 0.0 = healthy, 1.0 = maximally stressed
    pub fn stressitaso(&self) -> f64 {
        let tarkkuus = self.ikkuna_tarkkuus();
        let luottamus = self.keskimääräinen_luottamus();

        // Stress rises when accuracy drops and confidence is low
        let tarkkuus_stressi = (1.0 - tarkkuus * 2.0).max(0.0).min(1.0);
        let luottamus_stressi = (0.5 - luottamus).max(0.0).min(1.0);
        let peräkkäiset_stressi =
            (self.peräkkäiset_väärät as f64 / 20.0).min(1.0);

        // Weighted combination
        (tarkkuus_stressi * 0.5 + luottamus_stressi * 0.3 + peräkkäiset_stressi * 0.2)
            .min(1.0)
    }

    /// Is the system critically stressed?
    pub fn kriittinen(&self) -> bool {
        self.stressitaso() > 0.8
    }
}

// ═══════════════════════════════════════════════════════════════════
// THE CENTER — Keskus
// ═══════════════════════════════════════════════════════════════════
//
// The implosion point. Wraps any prediction model.
// Enriches context (pre-processing) and adjusts scores (post-processing).
// Maintains all the state that individual predictions don't carry.

/// The Keskus — center of the Star Mother.
pub struct Keskus {
    // ── Sub-systems ──
    /// Bigram transition memory
    pub siirtymät: SiirtymäMuisti,
    /// Character frequency prior
    pub taajuus: TaajuusPrior,
    /// Word boundary tracker
    pub sana_raja: SanaRaja,
    /// Memory anchor (noise floor)
    pub ankkuri: MuistiAnkkuri,
    /// System health
    pub terveys: TerveysMittari,

    // ── Recurrent state ──
    /// Carry-forward hypervector — accumulates recent prediction context.
    /// Updated each step: kierto = τ·kierto + (1-τ)·bind(ctx, prediction_hv)
    /// This is the hidden state that makes predictions non-independent.
    kierto_tila: Hypervektori,
    /// Previous character (for transitions)
    edellinen_merkki: Option<char>,
    /// Previous prediction (for self-monitoring)
    edellinen_ennuste: Option<char>,

    // ── Configuration (weights for each prior) ──
    /// Weight for transition prior in score adjustment
    pub alpha_siirtymä: f64,
    /// Weight for frequency prior in score adjustment
    pub alpha_taajuus: f64,
    /// Blend weight for recurrent state in context enrichment
    pub alpha_kierto: f64,
    /// Weight for word-boundary prior adjustment
    pub alpha_sana: f64,

    // ── Metadata ──
    aakkosto: Vec<char>,
    ulottuvuus: usize,
    askel: usize,
    /// Char → Hypervector mapping (for recurrent state binding)
    /// The Keskus needs its own codebook reference
    merkki_vektorit: HashMap<char, Hypervektori>,
}

impl Keskus {
    /// Create a new Keskus.
    ///
    /// `aakkosto`: the character set
    /// `merkki_vektorit`: char → hypervector mapping from QAMS or random codebook
    pub fn new(
        aakkosto: &[char],
        merkki_vektorit: HashMap<char, Hypervektori>,
        ulottuvuus: usize,
    ) -> Self {
        Self {
            siirtymät: SiirtymäMuisti::new(),
            taajuus: TaajuusPrior::new(),
            sana_raja: SanaRaja::new(),
            ankkuri: MuistiAnkkuri::new(ulottuvuus),
            terveys: TerveysMittari::new(),

            kierto_tila: vec![0.0; ulottuvuus],
            edellinen_merkki: None,
            edellinen_ennuste: None,

            alpha_siirtymä: OLETUS_ALPHA_SIIRTYMÄ,
            alpha_taajuus: OLETUS_ALPHA_TAAJUUS,
            alpha_kierto: OLETUS_ALPHA_KIERTO,
            alpha_sana: OLETUS_ALPHA_SANA,

            aakkosto: aakkosto.to_vec(),
            ulottuvuus,
            askel: 0,
            merkki_vektorit,
        }
    }

    // ═══════════════════════════════════════════════════════════
    // PRE-PROCESSING: Enrich context
    // ═══════════════════════════════════════════════════════════

    /// Enrich a context vector with recurrent state.
    ///
    /// The raw context from KontekstiSitoja encodes the current window.
    /// The recurrent state encodes everything BEFORE the current window.
    /// Blending them gives the model memory beyond its window size.
    ///
    /// enriched = (1 - α_kierto) × context + α_kierto × kierto_tila
    ///
    /// When α_kierto = 0: pure windowed context (current behavior)
    /// When α_kierto > 0: window + history (recurrent)
    pub fn rikasta(
        &self,
        konteksti: &[f64],
    ) -> Hypervektori {
        if self.alpha_kierto < 1e-10 || self.askel == 0 {
            return konteksti.to_vec();
        }

        // Normalize recurrent state to match context magnitude before blending.
        // HDC hygiene: everything is direction, not magnitude.
        let ctx_norm = konteksti.iter().map(|x| x * x).sum::<f64>().sqrt();
        let kierto_norm = self.kierto_tila.iter().map(|x| x * x).sum::<f64>().sqrt();
        let scale = if kierto_norm > 1e-12 { ctx_norm / kierto_norm } else { 0.0 };

        let a = self.alpha_kierto;
        konteksti.iter()
            .zip(self.kierto_tila.iter())
            .map(|(&k, &r)| (1.0 - a) * k + a * r * scale)
            .collect()
    }

    // ═══════════════════════════════════════════════════════════
    // POST-PROCESSING: Apply priors to scores
    // ═══════════════════════════════════════════════════════════

    /// Apply all priors to raw prediction scores.
    ///
    /// Takes raw cosine similarity scores from any model and adjusts them
    /// using transition probabilities, frequency priors, and word boundary info.
    ///
    /// adjusted(c) = raw(c)
    ///     + α_siirtymä × log(P(c | prev_char))    [transition prior]
    ///     + α_taajuus  × log(P(c))                 [frequency prior]
    ///     + α_sana     × word_boundary_adjustment   [word boundary]
    pub fn sovella_priorit(
        &self,
        raa_at_pisteet: &HashMap<char, f64>,
    ) -> HashMap<char, f64> {
        let aakkosto_koko = self.aakkosto.len();
        let mut säädetyt: HashMap<char, f64> = HashMap::new();

        // Competence-based annealing: as the model gets better,
        // the prior backs off. φ exponent gives golden-ratio decay —
        // backs off faster than linear but never fully zeroes out.
        let competence = self.terveys.ikkuna_tarkkuus();
        let anneal = (1.0 - competence).powf(PHI * PHI); // φ² ≈ 2.618 — steep decay
        let anneal = if anneal < 0.05 { 0.0 } else { anneal }; // hard cutoff

        let eff_siirtymä = self.alpha_siirtymä * anneal;
        let eff_taajuus = self.alpha_taajuus * anneal;
        let eff_sana = self.alpha_sana * anneal;

        for &c in &self.aakkosto {
            let raaka = raa_at_pisteet.get(&c).copied().unwrap_or(0.0);

            // Transition prior (annealed)
            let siirtymä_bonus = if let Some(ed) = self.edellinen_merkki {
                eff_siirtymä
                    * self.siirtymät.log_todennäköisyys(ed, c, aakkosto_koko)
            } else {
                0.0
            };

            // Frequency prior (annealed)
            let taajuus_bonus = eff_taajuus
                * self.taajuus.log_todennäköisyys(c, aakkosto_koko);

            // Word boundary adjustment (annealed)
            let sana_bonus = if self.sana_raja.raja_tilassa() {
                eff_sana
                    * self.sana_raja.alku_todennäköisyys(c, aakkosto_koko).ln().max(-5.0)
            } else if SanaRaja::on_raja(c) {
                eff_sana * (self.sana_raja.välilyönti_kerroin() - 1.0)
            } else {
                0.0
            };

            säädetyt.insert(c, raaka + siirtymä_bonus + taajuus_bonus + sana_bonus);
        }

        säädetyt
    }

    /// Simplified: apply priors to a score vector (indexed by alphabet order).
    /// Returns adjusted scores in the same order as self.aakkosto.
    pub fn sovella_pistevektori(
        &self,
        pisteet: &[f64],
    ) -> Vec<f64> {
        let aakkosto_koko = self.aakkosto.len();
        let mut säädetyt = Vec::with_capacity(aakkosto_koko);

        let competence = self.terveys.ikkuna_tarkkuus();
        let anneal = (1.0 - competence).powf(PHI * PHI);
        let anneal = if anneal < 0.05 { 0.0 } else { anneal };
        let eff_siirtymä = self.alpha_siirtymä * anneal;
        let eff_taajuus = self.alpha_taajuus * anneal;

        for (i, &c) in self.aakkosto.iter().enumerate() {
            let raaka = pisteet.get(i).copied().unwrap_or(0.0);

            let siirtymä_bonus = if let Some(ed) = self.edellinen_merkki {
                eff_siirtymä
                    * self.siirtymät.log_todennäköisyys(ed, c, aakkosto_koko)
            } else {
                0.0
            };

            let taajuus_bonus = eff_taajuus
                * self.taajuus.log_todennäköisyys(c, aakkosto_koko);

            säädetyt.push(raaka + siirtymä_bonus + taajuus_bonus);
        }

        säädetyt
    }

    // ═══════════════════════════════════════════════════════════
    // STATE UPDATE: After each prediction
    // ═══════════════════════════════════════════════════════════

    /// Update all Keskus state after a prediction.
    ///
    /// Call this AFTER the model predicts and the true answer is known.
    /// Updates: recurrent state, transitions, frequency, word boundaries,
    /// health metrics, and the memory anchor (during collection phase).
    pub fn päivitä(
        &mut self,
        ennuste: char,
        todellinen: char,
        konteksti: &[f64],
        luottamus: f64,
        hdc: &HdcPeruskäsitteet,
    ) {
        self.askel += 1;
        let oikein = ennuste == todellinen;

        // 1. Update transition memory (from actual data)
        if let Some(ed) = self.edellinen_merkki {
            self.siirtymät.kirjaa(ed, todellinen);
            self.sana_raja.käsittele(ed, todellinen);
        }

        // 2. Update frequency prior
        self.taajuus.kirjaa(todellinen);

        // 3. Update recurrent state
        //    kierto = τ·kierto + (1-τ)·bind(context, actual_char_hv)
        //    We use the ACTUAL character, not the predicted one,
        //    because the recurrent state should encode ground truth
        //    during training (teacher forcing).
        if let Some(merkki_hv) = self.merkki_vektorit.get(&todellinen) {
            let uusi_signaali = hdc.sido(konteksti, merkki_hv);
            for (k, &u) in self.kierto_tila.iter_mut().zip(uusi_signaali.iter()) {
                *k = KIERTO_RAPAUTUMINEN * *k + (1.0 - KIERTO_RAPAUTUMINEN) * u;
            }
        }

        // 4. Update health meter
        self.terveys.kirjaa(oikein, luottamus);

        // 5. Feed memory anchor (during collection phase)
        if !self.ankkuri.lukittu() {
            self.ankkuri.kerää(konteksti, luottamus, oikein);
        }

        // 6. Track previous characters
        self.edellinen_merkki = Some(todellinen);
        self.edellinen_ennuste = Some(ennuste);
    }

    /// Update during INFERENCE (no ground truth available).
    /// Uses the prediction as the recurrent state update.
    pub fn päivitä_ennusteella(
        &mut self,
        ennuste: char,
        konteksti: &[f64],
        _luottamus: f64,
        hdc: &HdcPeruskäsitteet,
    ) {
        self.askel += 1;

        // Recurrent state with prediction (auto-regressive)
        if let Some(merkki_hv) = self.merkki_vektorit.get(&ennuste) {
            let uusi_signaali = hdc.sido(konteksti, merkki_hv);
            for (k, &u) in self.kierto_tila.iter_mut().zip(uusi_signaali.iter()) {
                *k = KIERTO_RAPAUTUMINEN * *k + (1.0 - KIERTO_RAPAUTUMINEN) * u;
            }
        }

        // Update word boundary with prediction
        if let Some(ed) = self.edellinen_merkki {
            self.sana_raja.käsittele(ed, ennuste);
        }

        self.edellinen_merkki = Some(ennuste);
        self.edellinen_ennuste = Some(ennuste);
    }

    // ═══════════════════════════════════════════════════════════
    // TRAINING HELPERS
    // ═══════════════════════════════════════════════════════════

    /// Pre-train the Keskus on a text corpus.
    /// Collects transition counts, frequencies, and word boundary stats
    /// WITHOUT needing any model predictions. Just the raw text.
    ///
    /// Call this BEFORE training any model, using the same corpus.
    /// The Keskus will have perfect transition/frequency statistics
    /// from the start.
    pub fn esikouluta(&mut self, teksti: &str) {
        let merkit: Vec<char> = teksti.chars().collect();

        for i in 0..merkit.len() {
            self.taajuus.kirjaa(merkit[i]);

            if i > 0 {
                self.siirtymät.kirjaa(merkit[i - 1], merkit[i]);
                self.sana_raja.käsittele(merkit[i - 1], merkit[i]);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MEMORY MANAGEMENT
    // ═══════════════════════════════════════════════════════════

    /// Reset the recurrent state to zero.
    /// Use when starting a new text or recovering from stress.
    pub fn nollaa_kiertotila(&mut self) {
        self.kierto_tila = vec![0.0; self.ulottuvuus];
    }

    /// "Sleep" the recurrent state: blend toward the noise anchor.
    /// Partially resets accumulated context toward the null signal.
    ///
    /// syvyys: 0.0 = no effect, 1.0 = full reset to noise
    ///
    /// This is the white noise / sleep function Greg described:
    /// when the system hits stress limits, it "sleeps" by blending
    /// its state toward the first-pass noise signature.
    pub fn nuku(&mut self, syvyys: f64) {
        if !self.ankkuri.lukittu() {
            // No anchor yet — just zero out
            for k in self.kierto_tila.iter_mut() {
                *k *= 1.0 - syvyys;
            }
            return;
        }

        let s = syvyys.clamp(0.0, 1.0);
        let kohina = self.ankkuri.kohinan_vektori();
        for (k, &n) in self.kierto_tila.iter_mut().zip(kohina.iter()) {
            *k = (1.0 - s) * *k + s * n * GAMMA; // noise is scaled down by Γ
        }
    }

    /// Should the system sleep? Checks stress level and noise correlation.
    pub fn pitäisikö_nukkua(&self) -> bool {
        self.terveys.kriittinen()
    }

    /// Auto-sleep if stressed, with depth proportional to stress.
    pub fn automaattinen_uni(&mut self) {
        let stressi = self.terveys.stressitaso();
        if stressi > 0.5 {
            let syvyys = (stressi - 0.5) * 2.0; // 0..1 over stress range 0.5..1.0
            self.nuku(syvyys);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // DIAGNOSTICS
    // ═══════════════════════════════════════════════════════════

    /// Full state diagnostic.
    pub fn tila(&self) -> KeskusTila {
        KeskusTila {
            askel: self.askel,
            ikkuna_tarkkuus: self.terveys.ikkuna_tarkkuus(),
            kokonaistarkkuus: self.terveys.kokonaistarkkuus(),
            stressitaso: self.terveys.stressitaso(),
            siirtymä_näytteet: self.siirtymät.kokonaismäärä() as usize,
            taajuus_näytteet: self.taajuus.kokonaismäärä() as usize,
            ankkuri_lukittu: self.ankkuri.lukittu(),
            perustason_tarkkuus: self.ankkuri.perustason_tarkkuus(),
            keskimääräinen_sanan_pituus: self.sana_raja.keskimääräinen_pituus(),
            kierto_normi: self.kierto_tila.iter()
                .map(|x| x * x).sum::<f64>().sqrt(),
        }
    }

    /// Get the current step count
    pub fn askel(&self) -> usize {
        self.askel
    }

    /// Get the alphabet
    pub fn aakkosto(&self) -> &[char] {
        &self.aakkosto
    }

    /// Get the previous character (for external pole selection).
    pub fn edellinen(&self) -> Option<char> {
        self.edellinen_merkki
    }

    /// Get transition probability P(next | prev_char) for a given character.
    pub fn siirtymä_p(&self, seuraava: char) -> f64 {
        if let Some(ed) = self.edellinen_merkki {
            self.siirtymät.todennäköisyys(ed, seuraava, self.aakkosto.len())
        } else {
            // No previous char — return uniform
            1.0 / self.aakkosto.len() as f64
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// DIAGNOSTIC TYPES
// ═══════════════════════════════════════════════════════════════════

/// Snapshot of the Keskus state.
#[derive(Debug, Clone)]
pub struct KeskusTila {
    pub askel: usize,
    pub ikkuna_tarkkuus: f64,
    pub kokonaistarkkuus: f64,
    pub stressitaso: f64,
    pub siirtymä_näytteet: usize,
    pub taajuus_näytteet: usize,
    pub ankkuri_lukittu: bool,
    pub perustason_tarkkuus: f64,
    pub keskimääräinen_sanan_pituus: f64,
    pub kierto_normi: f64,
}

impl std::fmt::Display for KeskusTila {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Keskus [askel {}]\n", self.askel)?;
        write!(f, "  tarkkuus:  {:.2}% (ikkuna) / {:.2}% (koko)\n",
            self.ikkuna_tarkkuus * 100.0, self.kokonaistarkkuus * 100.0)?;
        write!(f, "  stressi:   {:.3}\n", self.stressitaso)?;
        write!(f, "  siirtymät: {} näytettä\n", self.siirtymä_näytteet)?;
        write!(f, "  taajuus:   {} näytettä\n", self.taajuus_näytteet)?;
        write!(f, "  ankkuri:   {}\n",
            if self.ankkuri_lukittu { "lukittu" } else { "keräys" })?;
        write!(f, "  perustaso: {:.2}%\n", self.perustason_tarkkuus * 100.0)?;
        write!(f, "  sana_pit:  {:.1}\n", self.keskimääräinen_sanan_pituus)?;
        write!(f, "  kierto_‖‖: {:.4}\n", self.kierto_normi)
    }
}

// ═══════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc_primitives::ULOTTUVUUS;

    fn luo_hdc() -> HdcPeruskäsitteet {
        HdcPeruskäsitteet::new(ULOTTUVUUS, 42)
    }

    fn testi_aakkosto() -> Vec<char> {
        "abcdefghijklmnopqrstuvwxyz ".chars().collect()
    }

    fn testi_merkki_vektorit(hdc: &mut HdcPeruskäsitteet) -> HashMap<char, Hypervektori> {
        let mut m = HashMap::new();
        for c in testi_aakkosto() {
            m.insert(c, hdc.satunnainen_bipolaarinen());
        }
        m
    }

    // ── Transition Memory Tests ──

    #[test]
    fn test_siirtymä_kirjaa_ja_todennäköisyys() {
        let mut sm = SiirtymäMuisti::new();
        sm.kirjaa('t', 'h');
        sm.kirjaa('t', 'h');
        sm.kirjaa('t', 'h');
        sm.kirjaa('t', 'o');

        let p_th = sm.todennäköisyys('t', 'h', 27);
        let p_to = sm.todennäköisyys('t', 'o', 27);
        let p_tz = sm.todennäköisyys('t', 'z', 27);

        assert!(p_th > p_to, "P(h|t) should be > P(o|t): {:.4} vs {:.4}", p_th, p_to);
        assert!(p_to > p_tz, "P(o|t) should be > P(z|t): {:.4} vs {:.4}", p_to, p_tz);
        assert!(p_tz > 0.0, "Laplace smoothing should keep P(z|t) > 0");
    }

    #[test]
    fn test_siirtymä_jakauma() {
        let mut sm = SiirtymäMuisti::new();
        for _ in 0..100 { sm.kirjaa('t', 'h'); }
        for _ in 0..50 { sm.kirjaa('t', 'o'); }
        for _ in 0..10 { sm.kirjaa('t', 'a'); }

        let jakauma = sm.jakauma('t', &testi_aakkosto());
        assert_eq!(jakauma[0].0, 'h', "Most common after 't' should be 'h'");
        assert_eq!(jakauma[1].0, 'o', "Second after 't' should be 'o'");
    }

    // ── Frequency Prior Tests ──

    #[test]
    fn test_taajuus_prior() {
        let mut tp = TaajuusPrior::new();
        for _ in 0..120 { tp.kirjaa('e'); }
        for _ in 0..80 { tp.kirjaa('t'); }
        for _ in 0..1 { tp.kirjaa('z'); }

        let p_e = tp.todennäköisyys('e', 27);
        let p_t = tp.todennäköisyys('t', 27);
        let p_z = tp.todennäköisyys('z', 27);

        assert!(p_e > p_t, "P(e) should be > P(t)");
        assert!(p_t > p_z, "P(t) should be > P(z)");
    }

    // ── Word Boundary Tests ──

    #[test]
    fn test_sana_raja_tunnistus() {
        let mut sr = SanaRaja::new();

        // "the cat"
        sr.käsittele(' ', 't');
        assert!(!sr.raja_tilassa()); // 't' is not a boundary
        sr.käsittele('t', 'h');
        sr.käsittele('h', 'e');
        sr.käsittele('e', ' ');
        assert!(sr.raja_tilassa()); // space is a boundary
        sr.käsittele(' ', 'c');
        assert!(!sr.raja_tilassa());
    }

    #[test]
    fn test_sana_raja_pituus() {
        let mut sr = SanaRaja::new();

        // Process "the" → word length 3
        sr.käsittele(' ', 't');
        sr.käsittele('t', 'h');
        sr.käsittele('h', 'e');
        sr.käsittele('e', ' ');

        assert!(sr.sana_laskuri == 1);
        // Word length tracking started, average adjusting
    }

    #[test]
    fn test_välilyönti_kerroin() {
        let mut sr = SanaRaja::new();
        sr.keskipituus = 4.0;

        // Short word: space suppressed
        sr.sanan_pituus = 2;
        assert!(sr.välilyönti_kerroin() < 1.0);

        // Long word: space boosted
        sr.sanan_pituus = 10;
        assert!(sr.välilyönti_kerroin() > 1.0);
    }

    // ── Memory Anchor Tests ──

    #[test]
    fn test_ankkuri_keräys() {
        let mut ankkuri = MuistiAnkkuri::new(ULOTTUVUUS);
        assert!(!ankkuri.lukittu());

        // Feed 1000 steps
        let ctx = vec![0.1; ULOTTUVUUS];
        for _ in 0..ANKKURI_KERÄYS_ASKELEET {
            ankkuri.kerää(&ctx, 0.05, false);
        }

        assert!(ankkuri.lukittu(), "Anchor should be locked after {} steps",
            ANKKURI_KERÄYS_ASKELEET);
        assert!(ankkuri.perustason_tarkkuus() < 0.01,
            "Baseline accuracy should be near 0 (all wrong)");
    }

    #[test]
    fn test_ankkuri_ajautuminen() {
        let mut ankkuri = MuistiAnkkuri::new(ULOTTUVUUS);
        let ctx = vec![0.1; ULOTTUVUUS];
        for _ in 0..ANKKURI_KERÄYS_ASKELEET {
            ankkuri.kerää(&ctx, 0.05, false);
        }

        // Accuracy near baseline = drifted
        assert!(ankkuri.on_ajautunut(0.0), "0% accuracy should register as drifted");
        // Accuracy well above baseline = not drifted
        assert!(!ankkuri.on_ajautunut(0.3), "30% accuracy should not be drifted");
    }

    // ── Health Meter Tests ──

    #[test]
    fn test_terveys_stressi() {
        let mut tm = TerveysMittari::new();

        // Feed many wrong predictions → high stress
        for _ in 0..50 {
            tm.kirjaa(false, 0.1);
        }
        assert!(tm.stressitaso() > 0.5,
            "Many wrong predictions should cause stress: {}", tm.stressitaso());

        // Feed many correct predictions → low stress
        let mut tm2 = TerveysMittari::new();
        for _ in 0..50 {
            tm2.kirjaa(true, 0.8);
        }
        assert!(tm2.stressitaso() < 0.3,
            "Many correct predictions should reduce stress: {}", tm2.stressitaso());
    }

    // ── Full Keskus Tests ──

    #[test]
    fn test_keskus_esikouluta() {
        let mut hdc = luo_hdc();
        let merkki_vektorit = testi_merkki_vektorit(&mut hdc);
        let aakkosto = testi_aakkosto();
        let mut keskus = Keskus::new(&aakkosto, merkki_vektorit, ULOTTUVUUS);

        keskus.esikouluta("the cat sat on the mat");

        // Check that transitions were recorded
        let p_th = keskus.siirtymät.todennäköisyys('t', 'h', aakkosto.len());
        let p_tz = keskus.siirtymät.todennäköisyys('t', 'z', aakkosto.len());
        assert!(p_th > p_tz, "Pre-training should capture t→h transition");

        // Check frequency
        let p_t = keskus.taajuus.todennäköisyys('t', aakkosto.len());
        let p_z = keskus.taajuus.todennäköisyys('z', aakkosto.len());
        assert!(p_t > p_z, "Pre-training should capture 't' frequency");
    }

    #[test]
    fn test_keskus_rikasta_konteksti() {
        let mut hdc = luo_hdc();
        let merkki_vektorit = testi_merkki_vektorit(&mut hdc);
        let aakkosto = testi_aakkosto();
        let keskus = Keskus::new(&aakkosto, merkki_vektorit, ULOTTUVUUS);

        let ctx = vec![1.0; ULOTTUVUUS];
        let rikastettu = keskus.rikasta(&ctx);

        // At step 0 with zero recurrent state, enriched should equal original
        assert_eq!(rikastettu.len(), ULOTTUVUUS);
        for (r, &c) in rikastettu.iter().zip(ctx.iter()) {
            assert!((r - c).abs() < 1e-10, "Step 0 enrichment should be identity");
        }
    }

    #[test]
    fn test_keskus_sovella_priorit() {
        let mut hdc = luo_hdc();
        let merkki_vektorit = testi_merkki_vektorit(&mut hdc);
        let aakkosto = testi_aakkosto();
        let mut keskus = Keskus::new(&aakkosto, merkki_vektorit, ULOTTUVUUS);

        // Pre-train with known transitions
        keskus.esikouluta("the the the the the");
        keskus.edellinen_merkki = Some('t');

        // Create equal raw scores
        let mut raa_at = HashMap::new();
        for &c in &aakkosto {
            raa_at.insert(c, 0.5);
        }

        let säädetyt = keskus.sovella_priorit(&raa_at);

        // 'h' should be boosted after 't' due to transition prior
        let h_piste = säädetyt.get(&'h').copied().unwrap_or(0.0);
        let z_piste = säädetyt.get(&'z').copied().unwrap_or(0.0);
        assert!(h_piste > z_piste,
            "After 't', 'h' should be boosted: h={:.4} z={:.4}", h_piste, z_piste);
    }

    #[test]
    fn test_keskus_päivitä_kiertotila() {
        let mut hdc = luo_hdc();
        let merkki_vektorit = testi_merkki_vektorit(&mut hdc);
        let aakkosto = testi_aakkosto();
        let mut keskus = Keskus::new(&aakkosto, merkki_vektorit, ULOTTUVUUS);

        let ctx = hdc.satunnainen_bipolaarinen();

        // Update a few times
        keskus.päivitä('a', 'a', &ctx, 0.8, &hdc);
        keskus.päivitä('b', 'b', &ctx, 0.7, &hdc);
        keskus.päivitä('c', 'c', &ctx, 0.6, &hdc);

        // Recurrent state should now be non-zero
        let normi: f64 = keskus.kierto_tila.iter()
            .map(|x| x * x).sum::<f64>().sqrt();
        assert!(normi > 0.01,
            "Recurrent state should be non-zero after updates: {}", normi);
    }

    #[test]
    fn test_keskus_nuku() {
        let mut hdc = luo_hdc();
        let merkki_vektorit = testi_merkki_vektorit(&mut hdc);
        let aakkosto = testi_aakkosto();
        let mut keskus = Keskus::new(&aakkosto, merkki_vektorit, ULOTTUVUUS);

        // Build up some recurrent state
        let ctx = hdc.satunnainen_bipolaarinen();
        for _ in 0..10 {
            keskus.päivitä('a', 'a', &ctx, 0.5, &hdc);
        }

        let normi_ennen: f64 = keskus.kierto_tila.iter()
            .map(|x| x * x).sum::<f64>().sqrt();

        // Sleep at 50% depth
        keskus.nuku(0.5);

        let normi_jälkeen: f64 = keskus.kierto_tila.iter()
            .map(|x| x * x).sum::<f64>().sqrt();

        assert!(normi_jälkeen < normi_ennen,
            "Sleep should reduce recurrent state: before={:.4} after={:.4}",
            normi_ennen, normi_jälkeen);
    }

    #[test]
    fn test_keskus_tila_näyttö() {
        let mut hdc = luo_hdc();
        let merkki_vektorit = testi_merkki_vektorit(&mut hdc);
        let aakkosto = testi_aakkosto();
        let keskus = Keskus::new(&aakkosto, merkki_vektorit, ULOTTUVUUS);

        let tila = keskus.tila();
        let output = format!("{}", tila);
        assert!(output.contains("Keskus"));
        assert!(output.contains("tarkkuus"));
        assert!(output.contains("stressi"));
    }

    #[test]
    fn test_keskus_integration_flow() {
        // Full integration test: pre-train → enrich → apply priors → update
        let mut hdc = luo_hdc();
        let merkki_vektorit = testi_merkki_vektorit(&mut hdc);
        let aakkosto = testi_aakkosto();
        let mut keskus = Keskus::new(&aakkosto, merkki_vektorit, ULOTTUVUUS);

        // Step 1: Pre-train on corpus
        keskus.esikouluta("the quick brown fox jumps over the lazy dog");

        // Step 2: Simulate a prediction cycle
        let raw_ctx = hdc.satunnainen_bipolaarinen();

        // Enrich context
        let enriched = keskus.rikasta(&raw_ctx);
        assert_eq!(enriched.len(), ULOTTUVUUS);

        // Simulate model producing scores
        let mut raw_scores = HashMap::new();
        for &c in &aakkosto {
            raw_scores.insert(c, 0.1); // flat scores
        }

        // Apply priors
        let adjusted = keskus.sovella_priorit(&raw_scores);
        assert_eq!(adjusted.len(), aakkosto.len());

        // Find best prediction
        let (best_char, _) = adjusted.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&c, &s)| (c, s))
            .unwrap();

        // Update state
        keskus.päivitä(best_char, 'e', &raw_ctx, 0.3, &hdc);

        assert_eq!(keskus.askel(), 1);
        assert!(keskus.edellinen_merkki == Some('e'));
    }
}
