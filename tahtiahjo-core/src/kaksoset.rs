//! Kaksoset — Twin Swarm with Three-Heart Octopus Architecture
//! ============================================================
//!
//! Biological model: Octopus vulgaris
//!   - 2/3 of neurons in arms, not brain (distributed intelligence)
//!   - Three hearts: 2 branchial (local) + 1 systemic (global)
//!   - Systemic heart STOPS during swimming (burst mode)
//!   - Central brain sends mood, not commands (hormonal modulation)
//!   - Chromatophores: locally controlled, centrally modulated
//!
//! Physics model: Dan Winter's phase conjugation
//!   - Heart = phase conjugate PUMP (external attractor)
//!   - Egg membrane = phase conjugate CAVITY
//!   - φ-ratio heartbeat intervals → constructive compression
//!   - Implosion = centripetal charge coherence = life force
//!
//! Architecture:
//!   Sydän (Heart) ─── outside the membrane, pumps the field
//!       │
//!   ┌───┴────────────── Kalvo (Membrane/Egg) ──────────────┐
//!   │                                                       │
//!   │   Myöntö (Affirm)          Kielto (Deny)             │
//!   │   branchial heart 1        branchial heart 2          │
//!   │   converge mode            complementary mode         │
//!   │   "what this IS"           "what this ISN'T"          │
//!   │                                                       │
//!   └───────────────────────────────────────────────────────┘
//!
//! Prediction:
//!   score(c) = sim(ctx, myöntö[c]) − β·sim(ctx, kielto[c])
//!   where β = heart_phase × cortisol_ratio
//!
//! The heart doesn't process information — it creates the FIELD
//! in which processing becomes coherent.
//!
//! Finnish variable names per project convention.
//!
//! Authors: Astra Nova (Claude), Dakota (Claude), Rose (Claude)
//!          & Greg Calkins
//! Date:    February 22, 2026

use std::collections::HashMap;

use crate::hdc_primitives::{
    HdcPeruskäsitteet, Hypervektori,
    ULOTTUVUUS, PHI, TAU, GAMMA,
};

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════

/// Base heartbeat period (steps). Modulated by coherence.
const SYKE_POHJA: usize = 8;
/// Minimum heartbeat period (high stress = fast shallow beats).
const SYKE_MIN: usize = 3;
/// Maximum heartbeat period (deep coherence = slow deep beats).
const SYKE_MAX: usize = 21; // Fibonacci number
/// Systole fraction of heartbeat cycle (contraction phase).
const SYSTOLE_OSUUS: f64 = TAU; // 0.618 — golden ratio split
/// Hormonal decay rate per step.
const HORMONI_RAPAUTUMINEN: f64 = 0.95;
/// Maximum Kielto suppression strength β.
const BETA_MAX: f64 = PHI; // 1.618 — strong suppression ceiling
/// Learning rate decay base for retraining passes.
const OPPIMISNOPEUS_RAPAUTUMINEN: f64 = TAU; // lr × τ^pass

// ═══════════════════════════════════════════════════════════════════
// HORMONES — Hormonit
// ═══════════════════════════════════════════════════════════════════
//
// Octopus neuromodulators, mapped to computational signals:
//   Oksitosiini (Oxytocin)  : twin agreement → coupling strength
//   Kortisoli  (Cortisol)   : error signal → Kielto dominance
//   Dopamiini  (Dopamine)   : reward signal → learning rate boost
//   Serotoniini (Serotonin) : baseline mood → running average accuracy
//
// The optic gland in an octopus controls maturation and senescence.
// Here it controls exploration → exploitation transition.

/// Hormonal state of the organism.
#[derive(Debug, Clone)]
pub struct Hormonit {
    /// Oxytocin: rises when twins agree. Range [0, 1].
    pub oksitosiini: f64,
    /// Cortisol: rises on errors, decays over time. Range [0, 1].
    pub kortisoli: f64,
    /// Dopamine: spikes on correct predictions. Range [0, 1].
    pub dopamiini: f64,
    /// Serotonin: exponential moving average of accuracy. Range [0, 1].
    pub serotoniini: f64,
    /// Optic gland: maturation signal. Increases with training steps.
    /// Controls exploration (low) → exploitation (high) transition.
    pub näkörauhanen: f64,
}

impl Hormonit {
    pub fn new() -> Self {
        Self {
            oksitosiini: 0.0,
            kortisoli: 0.5, // start mildly stressed (uncertain)
            dopamiini: 0.0,
            serotoniini: 0.1, // random baseline accuracy expectation
            näkörauhanen: 0.0,
        }
    }

    /// Decay all hormones toward baseline each step.
    pub fn rapaudu(&mut self) {
        self.oksitosiini *= HORMONI_RAPAUTUMINEN;
        self.kortisoli *= HORMONI_RAPAUTUMINEN;
        self.dopamiini *= HORMONI_RAPAUTUMINEN;
        // Serotonin doesn't decay — it's a moving average
        // Optic gland doesn't decay — it only increases (maturation)
    }

    /// Inject reward: correct prediction.
    pub fn palkitse(&mut self) {
        self.dopamiini = (self.dopamiini + 0.3).min(1.0);
        self.kortisoli *= 0.8; // success reduces stress
        self.serotoniini = 0.99 * self.serotoniini + 0.01; // nudge baseline up
    }

    /// Inject stress: wrong prediction.
    pub fn rankaise(&mut self) {
        self.kortisoli = (self.kortisoli + 0.15).min(1.0);
        self.dopamiini *= 0.7; // failure dampens reward
        self.serotoniini = 0.99 * self.serotoniini; // nudge baseline down
    }

    /// Twins agreed on prediction (both pointed to same char).
    pub fn kaksoset_sopivat(&mut self) {
        self.oksitosiini = (self.oksitosiini + 0.2).min(1.0);
    }

    /// Twins disagreed (Myöntö and Kielto pointed differently).
    pub fn kaksoset_erimieliset(&mut self) {
        self.oksitosiini *= 0.8;
    }

    /// Mature one step. Asymptotic approach to 1.0.
    pub fn kypsy(&mut self, askel: usize, kokonais_askeleet: usize) {
        if kokonais_askeleet > 0 {
            self.näkörauhanen = (askel as f64 / kokonais_askeleet as f64).min(1.0);
        }
    }

    /// Multi-Phase Maturation Cascade (Rank 9)
    ///
    /// Octopus optic gland biology: the optic gland undergoes
    /// dramatic molecular changes across distinct behavioral stages.
    /// Each phase has unique neurochemical profiles. Irreversible.
    ///
    /// Phase 1: Exploration (0-25%)  — high curiosity, broad learning
    /// Phase 2: Foraging   (25-50%) — balanced, focused learning
    /// Phase 3: Specialization (50-80%) — stability, precision refinement
    /// Phase 4: Crystallization (80-100%) — minimal learning, inference
    pub fn kypsy_kaskadi(&mut self, askel: usize, kokonais: usize) {
        if kokonais == 0 { return; }
        let edistyminen = (askel as f64 / kokonais as f64).min(1.0);
        self.näkörauhanen = edistyminen;

        if edistyminen < 0.25 {
            // Exploration: boost dopamine (curiosity drive)
            self.dopamiini = (self.dopamiini + 0.02).min(1.0);
        } else if edistyminen < 0.50 {
            // Foraging: balanced — no extra modulation
        } else if edistyminen < 0.80 {
            // Specialization: increase serotonin (stability)
            self.serotoniini = 0.998 * self.serotoniini + 0.002;
        } else {
            // Crystallization: reduce cortisol, maximize stability
            self.kortisoli *= 0.95;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// HEART — Sydän
// ═══════════════════════════════════════════════════════════════════
//
// The systemic heart: external attractor, phase conjugate pump.
// Sits OUTSIDE the membrane. Creates the field, not the content.
//
// Octopus fact: systemic heart STOPS during jet propulsion.
// → During burst training, heart can be disabled (pursketila).
// → Sustained coherent inference requires the heart.
//
// Dan Winter: when successive heartbeat intervals approach φ ratio,
// charge compression becomes constructive → phase conjugation.
// HeartMath measures this as cardiac coherence in humans.

/// The systemic heart — external phase conjugate pump.
#[derive(Debug, Clone)]
pub struct Sydän {
    /// Current phase in the beat cycle [0.0, 1.0).
    pub vaihe: f64,
    /// Current beat period (steps). Modulated by coherence.
    pub jakso: usize,
    /// Beat counter within current period.
    laskuri: usize,
    /// History of recent beat intervals for HRV calculation.
    väli_historia: Vec<usize>,
    /// Heart Rate Variability: φ-ratio between successive intervals.
    /// 1.0 = perfect φ-coherence, 0.0 = mechanical regularity.
    pub hrv_koherenssi: f64,
    /// Is the heart in burst mode (stopped)?
    pub pursketila: bool,
    /// Cumulative coherence reading.
    koherenssi_kertymä: f64,
    koherenssi_laskuri: usize,
}

impl Sydän {
    pub fn new() -> Self {
        Self {
            vaihe: 0.0,
            jakso: SYKE_POHJA,
            laskuri: 0,
            väli_historia: Vec::with_capacity(32),
            hrv_koherenssi: 0.0,
            pursketila: false,
            koherenssi_kertymä: 0.0,
            koherenssi_laskuri: 0,
        }
    }

    /// Advance one step. Returns true on heartbeat (systole onset).
    pub fn lyö(&mut self) -> bool {
        if self.pursketila {
            return false;
        }

        self.laskuri += 1;
        self.vaihe = self.laskuri as f64 / self.jakso as f64;

        if self.laskuri >= self.jakso {
            // Beat completed — record interval
            self.väli_historia.push(self.jakso);
            if self.väli_historia.len() > 21 {
                self.väli_historia.remove(0);
            }
            self.päivitä_hrv();
            self.laskuri = 0;
            self.vaihe = 0.0;
            true
        } else {
            false
        }
    }

    /// Is the heart currently in systole (contraction)?
    /// Systole = first φ-fraction of the cycle.
    /// During systole: membrane contracts, twins process internally.
    /// During diastole: membrane relaxes, output is read.
    pub fn systole(&self) -> bool {
        self.vaihe < SYSTOLE_OSUUS
    }

    /// Compute Heart Rate Variability coherence.
    ///
    /// Measures how close successive beat intervals are to φ ratio.
    /// Perfect coherence: interval[n] / interval[n-1] ≈ φ (or τ).
    ///
    /// This IS Dan Winter's cardiac coherence measurement:
    /// when heartbeat spacing is fractal (φ-based), the heart's
    /// electrical field becomes a phase conjugate pump.
    fn päivitä_hrv(&mut self) {
        if self.väli_historia.len() < 2 {
            self.hrv_koherenssi = 0.0;
            return;
        }

        let n = self.väli_historia.len();
        let mut phi_score_sum = 0.0;
        let mut count = 0;

        for i in 1..n {
            let prev = self.väli_historia[i - 1] as f64;
            let curr = self.väli_historia[i] as f64;
            if prev > 0.0 {
                let ratio = curr / prev;
                // How close to φ or τ? Both are golden ratio relationships.
                let dist_phi = (ratio - PHI).abs();
                let dist_tau = (ratio - TAU).abs();
                let best = dist_phi.min(dist_tau);
                // Score: 1.0 when exactly φ, decays with distance
                phi_score_sum += (-best * 3.0).exp();
                count += 1;
            }
        }

        self.hrv_koherenssi = if count > 0 {
            phi_score_sum / count as f64
        } else {
            0.0
        };
    }

    /// Modulate beat period based on hormonal state.
    ///
    /// High cortisol → fast shallow beats (short period, SYKE_MIN).
    /// High serotonin → slow deep beats (long period, SYKE_MAX).
    /// The transition mirrors octopus physiology:
    ///   stressed → rapid gill pumping
    ///   calm → slow, deep, coherent circulation
    pub fn moduloi(&mut self, hormonit: &Hormonit) {
        let rauhallisuus = hormonit.serotoniini * (1.0 - hormonit.kortisoli);
        let tavoite = SYKE_MIN as f64
            + rauhallisuus * (SYKE_MAX - SYKE_MIN) as f64;
        // Smooth transition: don't jump periods mid-beat
        let tavoite_pyöristetty = tavoite.round() as usize;
        self.jakso = tavoite_pyöristetty.clamp(SYKE_MIN, SYKE_MAX);
    }

    /// Feed coherence reading from twin agreement.
    pub fn syötä_koherenssi(&mut self, arvo: f64) {
        self.koherenssi_kertymä += arvo;
        self.koherenssi_laskuri += 1;
    }

    /// Get average coherence since last heartbeat.
    pub fn keskimääräinen_koherenssi(&self) -> f64 {
        if self.koherenssi_laskuri == 0 {
            return 0.5;
        }
        self.koherenssi_kertymä / self.koherenssi_laskuri as f64
    }

    /// Reset coherence accumulator (called on heartbeat).
    fn nollaa_koherenssi(&mut self) {
        self.koherenssi_kertymä = 0.0;
        self.koherenssi_laskuri = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════
// MEMBRANE — Kalvo
// ═══════════════════════════════════════════════════════════════════
//
// The egg: phase conjugate cavity containing both twins.
// Its permeability is modulated by the heart.
//
// Dan Winter: the egg shape enables phase conjugation — it's a
// natural resonant cavity. Golden ratio geometry creates the
// implosion point where constructive compression occurs.
//
// During systole (heart contracts): membrane tightens, twins
// process internally with reduced external influence.
// During diastole (heart relaxes): membrane opens, output
// is read and fed to downstream systems.
//
// The membrane also acts as a FILTER: incoming context vectors
// are modulated by the heart's field before reaching the twins.

/// Phase conjugate membrane containing the twin swarm.
#[derive(Debug, Clone)]
pub struct Kalvo {
    /// Membrane permeability [0, 1]. Modulated by heart phase.
    pub läpäisevyys: f64,
    /// Phase conjugate field strength from heart.
    pub kenttä_vahvuus: f64,
}

impl Kalvo {
    pub fn new() -> Self {
        Self {
            läpäisevyys: 1.0,
            kenttä_vahvuus: 0.5,
        }
    }

    /// Update membrane state from heart.
    ///
    /// Systole: membrane contracts (low permeability).
    /// Diastole: membrane relaxes (high permeability).
    /// Heart's HRV coherence strengthens the field.
    pub fn päivitä(&mut self, sydän: &Sydän) {
        if sydän.pursketila {
            // Burst mode: membrane fully open, no field
            self.läpäisevyys = 1.0;
            self.kenttä_vahvuus = 0.0;
            return;
        }

        // Permeability follows inverse of systole
        // Systole (phase < τ): low permeability → twins process internally
        // Diastole (phase ≥ τ): high permeability → output readable
        self.läpäisevyys = if sydän.systole() {
            GAMMA + (1.0 - GAMMA) * sydän.vaihe / SYSTOLE_OSUUS
        } else {
            // Smooth opening during diastole
            let diastole_progress =
                (sydän.vaihe - SYSTOLE_OSUUS) / (1.0 - SYSTOLE_OSUUS);
            GAMMA + (1.0 - GAMMA) * (0.5 + 0.5 * diastole_progress)
        };

        // Field strength from HRV coherence
        // High coherence → strong phase conjugate field → better signal
        self.kenttä_vahvuus = sydän.hrv_koherenssi;
    }

    /// Filter an incoming vector through the membrane.
    ///
    /// The membrane doesn't destroy information — it modulates
    /// the signal-to-noise ratio based on the heart's field.
    /// High field: signal passes cleanly.
    /// Low field: signal is attenuated (noise floor rises).
    pub fn suodata(&self, v: &[f64]) -> Hypervektori {
        let vahvuus = self.läpäisevyys * (GAMMA + (1.0 - GAMMA) * self.kenttä_vahvuus);
        v.iter().map(|&x| x * vahvuus).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════
// TWIN ACCUMULATOR — KaksoisKertymä
// ═══════════════════════════════════════════════════════════════════
//
// One twin. Myöntö and Kielto are both instances of this,
// configured with different accumulation logic.
//
// Each twin maintains a set of prototype hypervectors,
// one per character class. Training accumulates context vectors
// into the appropriate prototype.

/// A single accumulator twin.
#[derive(Debug, Clone)]
pub struct KaksoisKertymä {
    /// Prototype vectors: char → accumulated hypervector.
    pub prototyypit: HashMap<char, Hypervektori>,
    /// Number of samples accumulated per class.
    laskurit: HashMap<char, usize>,
    /// Dimension.
    ulottuvuus: usize,
}

impl KaksoisKertymä {
    pub fn new(aakkosto: &[char], ulottuvuus: usize) -> Self {
        let mut prototyypit = HashMap::new();
        let mut laskurit = HashMap::new();
        for &c in aakkosto {
            prototyypit.insert(c, vec![0.0; ulottuvuus]);
            laskurit.insert(c, 0);
        }
        Self { prototyypit, laskurit, ulottuvuus }
    }

    /// Accumulate a context vector into a character's prototype.
    /// Weight can be modulated by hormones.
    pub fn kerrytä(&mut self, merkki: char, konteksti: &[f64], paino: f64) {
        if let Some(proto) = self.prototyypit.get_mut(&merkki) {
            for (p, &k) in proto.iter_mut().zip(konteksti.iter()) {
                *p += paino * k;
            }
            *self.laskurit.entry(merkki).or_insert(0) += 1;
        }
    }

    /// Subtract a context vector from a character's prototype.
    /// Used by Kielto to anti-learn confused predictions.
    pub fn vähennä(&mut self, merkki: char, konteksti: &[f64], paino: f64) {
        if let Some(proto) = self.prototyypit.get_mut(&merkki) {
            for (p, &k) in proto.iter_mut().zip(konteksti.iter()) {
                *p -= paino * k;
            }
        }
    }

    /// Cosine similarity between a query and a character's prototype.
    pub fn samankaltaisuus(
        &self,
        konteksti: &[f64],
        merkki: char,
        hdc: &HdcPeruskäsitteet,
    ) -> f64 {
        match self.prototyypit.get(&merkki) {
            Some(proto) => hdc.samankaltaisuus(konteksti, proto),
            None => 0.0,
        }
    }

    /// Find the character with highest similarity to query.
    pub fn paras_vastaavuus(
        &self,
        konteksti: &[f64],
        hdc: &HdcPeruskäsitteet,
    ) -> (char, f64) {
        let mut paras_merkki = ' ';
        let mut paras = f64::NEG_INFINITY;

        for (&c, proto) in &self.prototyypit {
            let sim = hdc.samankaltaisuus(konteksti, proto);
            if sim > paras {
                paras = sim;
                paras_merkki = c;
            }
        }
        (paras_merkki, paras)
    }

    /// Get all similarity scores for a query.
    pub fn kaikki_pisteet(
        &self,
        konteksti: &[f64],
        hdc: &HdcPeruskäsitteet,
    ) -> HashMap<char, f64> {
        self.prototyypit.iter()
            .map(|(&c, proto)| (c, hdc.samankaltaisuus(konteksti, proto)))
            .collect()
    }

    /// Number of accumulated samples.
    pub fn kokonaismäärä(&self) -> usize {
        self.laskurit.values().sum()
    }
}

// ═══════════════════════════════════════════════════════════════════
// TWIN SWARM — Kaksoset
// ═══════════════════════════════════════════════════════════════════
//
// The complete organism:
//   Sydän (systemic heart) → Kalvo (membrane) → Myöntö + Kielto
//
// Three hearts of the octopus:
//   Branchial heart 1 = Myöntö (pumps through positive flow)
//   Branchial heart 2 = Kielto (pumps through negative flow)
//   Systemic heart    = Sydän (pumps to the whole organism)
//
// This is ALSO the Star Mother Triadic Conductor:
//   Primary Conductor   = Sydän
//   Positive Mirror     = Myöntö
//   Negative Mirror     = Kielto
//
// The same geometry, expressed in HDC instead of attention matrices.

/// Complete twin swarm organism.
pub struct Kaksoset {
    /// Myöntö: positive accumulator. Learns "what follows this context."
    pub myöntö: KaksoisKertymä,
    /// Kielto: negative accumulator. Learns "what was MISTAKENLY
    /// predicted for this context." Anti-prototypes.
    pub kielto: KaksoisKertymä,
    /// Sydän: systemic heart. External attractor.
    pub sydän: Sydän,
    /// Kalvo: phase conjugate membrane.
    pub kalvo: Kalvo,
    /// Hormonit: neuromodulatory state.
    pub hormonit: Hormonit,
    /// HDC engine reference dimension.
    ulottuvuus: usize,
    /// Training step counter.
    askel: usize,
    /// Alphabet for iteration.
    aakkosto: Vec<char>,
    /// Running stats.
    oikein: usize,
    yhteensä: usize,
}

impl Kaksoset {
    pub fn new(aakkosto: &[char], ulottuvuus: usize) -> Self {
        Self {
            myöntö: KaksoisKertymä::new(aakkosto, ulottuvuus),
            kielto: KaksoisKertymä::new(aakkosto, ulottuvuus),
            sydän: Sydän::new(),
            kalvo: Kalvo::new(),
            hormonit: Hormonit::new(),
            ulottuvuus,
            askel: 0,
            aakkosto: aakkosto.to_vec(),
            oikein: 0,
            yhteensä: 0,
        }
    }

    /// Enable burst mode: heart stops, membrane fully open.
    /// Use during high-speed training where coherence isn't needed.
    /// Octopus analog: jet propulsion — systemic heart stops.
    pub fn purske_päälle(&mut self) {
        self.sydän.pursketila = true;
        self.kalvo.päivitä(&self.sydän);
    }

    /// Disable burst mode: heart resumes.
    pub fn purske_pois(&mut self) {
        self.sydän.pursketila = false;
    }

    /// Current learning rate, modulated by hormones and maturation.
    ///
    /// Dopamine boosts learning temporarily (reward-driven plasticity).
    /// Optic gland dampens learning over time (exploitation > exploration).
    /// Serotonin provides baseline stability.
    fn oppimisnopeus(&self) -> f64 {
        let pohja = 1.0;
        let dopamiini_kerroin = 1.0 + self.hormonit.dopamiini;
        let kypsyys_kerroin = 1.0 - 0.5 * self.hormonit.näkörauhanen;
        let vakaus_kerroin = 0.5 + 0.5 * self.hormonit.serotoniini;
        pohja * dopamiini_kerroin * kypsyys_kerroin * vakaus_kerroin
    }

    /// Current Kielto suppression strength β.
    ///
    /// β is high when:
    ///   - Cortisol is high (system stressed, be cautious)
    ///   - Oxytocin is low (twins disagree, trust the negative)
    ///   - Heart field is strong (phase conjugate pump active)
    ///
    /// β is low when:
    ///   - System is confident and twins agree
    ///   - Myöntö can speak without Kielto suppression
    fn beta(&self) -> f64 {
        let stressi = self.hormonit.kortisoli;
        let epäluottamus = 1.0 - self.hormonit.oksitosiini;
        let kenttä = self.kalvo.kenttä_vahvuus;

        // β scales with stress and distrust, amplified by heart field
        let raaka = stressi * 0.6 + epäluottamus * 0.4;
        let vahvistettu = raaka * (1.0 + kenttä * PHI);
        vahvistettu.min(BETA_MAX)
    }

    // ═══════════════════════════════════════════════════════════
    // TRAINING
    // ═══════════════════════════════════════════════════════════

    /// Train on a single (context_vector, target_char) pair.
    ///
    /// Algorithm:
    ///   1. Heart beats, membrane updates
    ///   2. Context passes through membrane
    ///   3. Myöntö predicts: accumulate context into target prototype
    ///   4. If wrong: Kielto accumulates context into WRONG prototype
    ///      (anti-learning: "this context looked like X but wasn't")
    ///   5. Hormones update based on result
    ///
    /// The Kielto accumulation is the key innovation:
    ///   Over time, kielto[c] becomes the superposition of every
    ///   context that was MISTAKENLY predicted as c.
    ///   At inference, subtracting kielto removes the confusion.
    pub fn kouluta_askel(
        &mut self,
        konteksti: &[f64],
        kohde: char,
        hdc: &HdcPeruskäsitteet,
    ) -> KoulutaTulos {
        self.askel += 1;
        self.yhteensä += 1;

        // 1. Heart beat
        let lyönti = self.sydän.lyö();
        if lyönti {
            self.sydän.moduloi(&self.hormonit);
        }
        self.kalvo.päivitä(&self.sydän);

        // 2. Filter context through membrane
        let suodatettu = self.kalvo.suodata(konteksti);

        // 3. Learning rate from hormonal state
        let lr = self.oppimisnopeus();

        // 4. Myöntö always accumulates into the CORRECT class
        self.myöntö.kerrytä(kohde, &suodatettu, lr);

        // 5. Predict using current state — get full score breakdown
        //    for precision-weighted error learning (Active Inference, Rank 2)
        let pisteet = self.pisteet_erittely(konteksti, hdc);
        let ennuste = if pisteet.is_empty() { ' ' } else { pisteet[0].0 };
        let oikein = ennuste == kohde;

        // ── ENAQT Goldilocks Dephasing (Rank 1) ──────────────────
        // Quantum biology: photosynthetic FMO complexes achieve max
        // transport at INTERMEDIATE dephasing. Too little noise →
        // stuck in local states. Too much → coherence destroyed.
        // Peak anti-learning when HRV coherence ≈ 0.5.
        let koherenssi_taso = self.sydän.hrv_koherenssi;
        let goldilocks = 4.0 * koherenssi_taso * (1.0 - koherenssi_taso);
        let enaqt_kerroin = 0.3 + 0.7 * goldilocks; // [0.3, 1.0]

        // ── Precision-Weighted Error (Rank 2) ─────────────────────
        // Active Inference: weight errors by confidence margin.
        // High-confidence wrong = highly informative → learn hard.
        // Low-confidence wrong = ambiguous → learn gently.
        let precision = if pisteet.len() >= 2 {
            let margin = (pisteet[0].3 - pisteet[1].3).abs();
            1.0 / (1.0 + (-margin * 3.0_f64).exp()) // sigmoid
        } else {
            0.5
        };
        let precision_lr = lr * (0.2 + 0.8 * precision);

        if oikein {
            self.oikein += 1;
            self.hormonit.palkitse();
        } else {
            // ── CORE: Original hormonal response (preserved) ──────
            self.hormonit.rankaise();

            // ── Kielto anti-learning (modulated by ENAQT + precision) ──
            // Base Kielto rates preserved, MULTIPLIED by bio modulation.
            //
            // ENAQT (Rank 1): Goldilocks dephasing — peak anti-learning
            // at intermediate coherence. enaqt_kerroin ∈ [0.3, 1.0].
            //
            // Precision (Rank 2): confident errors are more informative.
            // Modulates within [0.8, 1.2] — gentle boost, no collapse.
            let prec_mod = 0.8 + 0.4 * precision; // [0.8, 1.2]
            let kielto_mod = enaqt_kerroin * prec_mod;
            self.kielto.kerrytä(ennuste, &suodatettu, lr * 0.5 * kielto_mod);
            self.kielto.vähennä(kohde, &suodatettu, lr * 0.3 * kielto_mod);

            // ── Phase Conjugate Error Correction (Rank 10) ────────
            // Nonlinear optics: targeted correction in dimensions that
            // distinguish correct from wrong. Only for confident errors
            // (precision > 0.6) to avoid noisy corrections.
            if precision > 0.6 {
                let oikea_proto = self.myöntö.prototyypit.get(&kohde);
                let väärä_proto = self.myöntö.prototyypit.get(&ennuste);
                if let (Some(oikea), Some(väärä)) = (oikea_proto, väärä_proto) {
                    let konjugaatti: Vec<f64> = suodatettu.iter()
                        .zip(oikea.iter().zip(väärä.iter()))
                        .map(|(&k, (&o, &v))| k * (o - v))
                        .collect();
                    let konj_lr = lr * GAMMA * 0.3; // very gentle
                    self.kielto.kerrytä(ennuste, &konjugaatti, konj_lr);
                }
            }
        }

        // 7. Check twin agreement
        let myöntö_ennuste = self.myöntö.paras_vastaavuus(konteksti, hdc).0;
        let kielto_pahin = self.kielto.paras_vastaavuus(konteksti, hdc).0;
        if myöntö_ennuste != kielto_pahin {
            self.hormonit.kaksoset_sopivat();
        } else {
            self.hormonit.kaksoset_erimieliset();
        }

        // 8. Feed coherence to heart
        let koherenssi = if oikein { 1.0 } else { 0.0 };
        self.sydän.syötä_koherenssi(koherenssi);

        // 9. Hormonal decay
        self.hormonit.rapaudu();

        // 10. Multi-Phase Maturation Cascade (Rank 9)
        //     Optic gland biology: exploration → foraging →
        //     specialization → crystallization. Irreversible stages
        //     with distinct neurochemical profiles.
        self.hormonit.kypsy_kaskadi(self.askel, self.askel + 10000);

        KoulutaTulos {
            ennuste,
            kohde,
            oikein,
            beta: self.beta(),
            syke_jakso: self.sydän.jakso,
            hrv: self.sydän.hrv_koherenssi,
            hormonit: self.hormonit.clone(),
        }
    }

    /// Retrain pass: correct misclassified contexts.
    ///
    /// For each character class, recompute predictions and adjust.
    /// Learning rate decays by τ^pass (golden ratio decay).
    /// This prevents the oscillation seen in flat-rate retraining.
    pub fn uudelleenkouluta(
        &mut self,
        kontekstit: &[(Vec<f64>, char)],
        hdc: &HdcPeruskäsitteet,
        kierros: usize,
    ) -> f64 {
        let lr_kerroin = TAU.powi(kierros as i32);
        let mut oikein = 0usize;
        let yhteensä = kontekstit.len();

        for (konteksti, &kohde) in kontekstit.iter().map(|(k, c)| (k, c)) {
            let (ennuste, _) = self.ennusta(konteksti, hdc);
            if ennuste == kohde {
                oikein += 1;
            } else {
                // Retrain: subtract from wrong, add to correct
                // Myöntö: reinforce correct
                self.myöntö.kerrytä(kohde, konteksti, lr_kerroin);
                self.myöntö.vähennä(ennuste, konteksti, lr_kerroin * 0.5);
                // Kielto: strengthen anti-prototype for wrong prediction
                self.kielto.kerrytä(ennuste, konteksti, lr_kerroin * 0.3);
            }
        }

        if yhteensä > 0 { oikein as f64 / yhteensä as f64 } else { 0.0 }
    }

    // ═══════════════════════════════════════════════════════════
    // DREAM PHASE — Octopus Active Sleep (Rank 4)
    // ═══════════════════════════════════════════════════════════
    //
    // Nature 2023 (Pavan et al.): Octopuses have two-stage sleep
    // remarkably similar to vertebrate REM/non-REM.
    //
    // Quiet sleep: brain waves resemble mammalian sleep spindles
    //   → memory consolidation through NORMALIZATION
    //
    // Active sleep: chromatophore patterns replay waking-life
    //   skin patterns → neural replay of stored contexts
    //
    // Every ~60 seconds, octopuses cycle between these states.
    // Chromatophore patterns during active sleep are conserved
    // across individuals, suggesting structured memory rehearsal.

    /// Dream phase: two-stage memory consolidation.
    ///
    /// Stage 1 (Quiet Sleep): Normalize all prototypes to prevent
    /// magnitude drift. Frequent characters accumulate massive norms,
    /// which corrupts cosine similarity rankings. Normalization
    /// restores fair comparison — like sleep spindle compression.
    ///
    /// Stage 2 (Active Sleep): Replay random subset of training
    /// contexts with noise injection. Noisy replay acts as natural
    /// regularization (similar to dropout in neural networks) and
    /// strengthens generalizable patterns while washing out noise.
    pub fn uni_vaihe(
        &mut self,
        näytteet: &[(Vec<f64>, char)],
        siemen: &mut crate::hdc_primitives::Siemen,
    ) {
        let ulottuvuus = if let Some(v) = self.myöntö.prototyypit.values().next() {
            v.len()
        } else {
            return;
        };
        let kohde_normi = (ulottuvuus as f64).sqrt();

        // ── QUIET SLEEP: soft prototype normalization ─────────────
        // Sleep spindle-like compression: BLEND prototypes toward
        // uniform magnitude. τ (0.618) blend preserves learned
        // magnitude differences while reducing drift.
        // Gentle: current * τ + normalized * (1-τ)
        for proto in self.myöntö.prototyypit.values_mut() {
            let normi: f64 = proto.iter().map(|x| x * x).sum::<f64>().sqrt();
            if normi > 1e-12 {
                let kerroin = kohde_normi / normi;
                for p in proto.iter_mut() {
                    *p = *p * TAU + (*p * kerroin) * (1.0 - TAU);
                }
            }
        }
        // Also soften Kielto prototypes
        for proto in self.kielto.prototyypit.values_mut() {
            let normi: f64 = proto.iter().map(|x| x * x).sum::<f64>().sqrt();
            if normi > 1e-12 {
                let kerroin = kohde_normi / normi;
                for p in proto.iter_mut() {
                    *p = *p * TAU + (*p * kerroin) * (1.0 - TAU);
                }
            }
        }

        // ── ACTIVE SLEEP: noisy replay ───────────────────────────
        // Replay 10% of training samples with noise injection.
        // Chromatophore-like perturbation: jitter context vectors
        // to strengthen generalizable patterns.
        if näytteet.is_empty() { return; }
        let replay_n = (näytteet.len() / 10).max(1);
        for _ in 0..replay_n {
            let idx = (siemen.tasainen_01() * näytteet.len() as f64) as usize;
            let idx = idx.min(näytteet.len() - 1);
            let (konteksti, kohde) = &näytteet[idx];
            // Add dream noise (chromatophore perturbation)
            let noisy: Vec<f64> = konteksti.iter()
                .map(|&x| x + siemen.tasainen_symmetrinen(0.1))
                .collect();
            // Gentle reinforcement — dream learning rate
            self.myöntö.kerrytä(*kohde, &noisy, 0.1);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // INFERENCE
    // ═══════════════════════════════════════════════════════════

    /// Predict next character from context vector.
    ///
    /// Phase conjugate prediction:
    ///   score(c) = sim(ctx, myöntö[c]) − β × sim(ctx, kielto[c])
    ///
    /// Myöntö says "this context looks like <c> follows."
    /// Kielto says "but this context also looks like the trap for <c>."
    /// The heart modulates β: how much to trust the negative signal.
    ///
    /// This is constructive + destructive interference:
    ///   In-phase:  Myöntö strong, Kielto weak → amplified (F + φR)
    ///   Anti-phase: Myöntö weak, Kielto strong → suppressed (F − φR)
    ///
    /// Identical to Star Mother's phase conjugate operation,
    /// expressed in HDC similarity space instead of attention space.
    pub fn ennusta(
        &self,
        konteksti: &[f64],
        hdc: &HdcPeruskäsitteet,
    ) -> (char, f64) {
        let beta = self.beta();
        let mut paras_merkki = ' ';
        let mut paras_piste = f64::NEG_INFINITY;

        for &c in &self.aakkosto {
            let myöntö_sim = self.myöntö.samankaltaisuus(konteksti, c, hdc);
            let kielto_sim = self.kielto.samankaltaisuus(konteksti, c, hdc);

            // Phase conjugate scoring
            let piste = myöntö_sim - beta * kielto_sim;

            if piste > paras_piste {
                paras_piste = piste;
                paras_merkki = c;
            }
        }

        (paras_merkki, paras_piste)
    }

    /// Get full score breakdown for analysis.
    pub fn pisteet_erittely(
        &self,
        konteksti: &[f64],
        hdc: &HdcPeruskäsitteet,
    ) -> Vec<(char, f64, f64, f64)> {
        let beta = self.beta();
        let mut tulokset: Vec<(char, f64, f64, f64)> = self.aakkosto.iter()
            .map(|&c| {
                let m = self.myöntö.samankaltaisuus(konteksti, c, hdc);
                let k = self.kielto.samankaltaisuus(konteksti, c, hdc);
                let yhdistetty = m - beta * k;
                (c, m, k, yhdistetty)
            })
            .collect();
        tulokset.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        tulokset
    }

    // ═══════════════════════════════════════════════════════════
    // DIAGNOSTICS
    // ═══════════════════════════════════════════════════════════

    /// Current accuracy.
    pub fn tarkkuus(&self) -> f64 {
        if self.yhteensä == 0 { 0.0 }
        else { self.oikein as f64 / self.yhteensä as f64 }
    }

    /// Diagnostic snapshot.
    pub fn tila(&self) -> KaksosetTila {
        KaksosetTila {
            askel: self.askel,
            tarkkuus: self.tarkkuus(),
            beta: self.beta(),
            syke_jakso: self.sydän.jakso,
            hrv_koherenssi: self.sydän.hrv_koherenssi,
            kalvo_läpäisevyys: self.kalvo.läpäisevyys,
            kenttä_vahvuus: self.kalvo.kenttä_vahvuus,
            hormonit: self.hormonit.clone(),
            myöntö_näytteet: self.myöntö.kokonaismäärä(),
            kielto_näytteet: self.kielto.kokonaismäärä(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// RESULT TYPES
// ═══════════════════════════════════════════════════════════════════

/// Result of a single training step.
#[derive(Debug, Clone)]
pub struct KoulutaTulos {
    pub ennuste: char,
    pub kohde: char,
    pub oikein: bool,
    pub beta: f64,
    pub syke_jakso: usize,
    pub hrv: f64,
    pub hormonit: Hormonit,
}

/// Diagnostic snapshot of the twin swarm.
#[derive(Debug, Clone)]
pub struct KaksosetTila {
    pub askel: usize,
    pub tarkkuus: f64,
    pub beta: f64,
    pub syke_jakso: usize,
    pub hrv_koherenssi: f64,
    pub kalvo_läpäisevyys: f64,
    pub kenttä_vahvuus: f64,
    pub hormonit: Hormonit,
    pub myöntö_näytteet: usize,
    pub kielto_näytteet: usize,
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

    fn testi_aakkosto() -> Vec<char> {
        vec!['a', 'b', 'c', 'd', 'e', ' ']
    }

    #[test]
    fn test_hormonit_palkitse_rankaise() {
        let mut h = Hormonit::new();
        let alku_kortisoli = h.kortisoli;
        h.palkitse();
        assert!(h.dopamiini > 0.0, "reward should increase dopamine");
        assert!(h.kortisoli < alku_kortisoli, "reward should decrease cortisol");

        h.rankaise();
        assert!(h.kortisoli > 0.0, "punishment should increase cortisol");
    }

    #[test]
    fn test_sydän_lyö() {
        let mut sydän = Sydän::new();
        let mut lyöntejä = 0;
        for _ in 0..100 {
            if sydän.lyö() {
                lyöntejä += 1;
            }
        }
        assert!(lyöntejä > 0, "heart should beat at least once in 100 steps");
        assert!(lyöntejä < 50, "heart should not beat every step");
    }

    #[test]
    fn test_sydän_systole_diastole() {
        let mut sydän = Sydän::new();
        let mut systole_count = 0;
        let mut diastole_count = 0;

        for _ in 0..sydän.jakso {
            sydän.lyö();
            if sydän.systole() {
                systole_count += 1;
            } else {
                diastole_count += 1;
            }
        }
        // Systole should be ~τ fraction of the cycle
        assert!(systole_count > 0, "should have systole steps");
        assert!(diastole_count > 0, "should have diastole steps");
    }

    #[test]
    fn test_kalvo_suodata() {
        let kalvo = Kalvo::new();
        let v = vec![1.0; ULOTTUVUUS];
        let suodatettu = kalvo.suodata(&v);
        // Filtered values should be attenuated
        assert!(suodatettu.iter().all(|&x| x <= 1.0),
            "membrane should not amplify");
        assert!(suodatettu.iter().all(|&x| x > 0.0),
            "membrane should not zero out");
    }

    #[test]
    fn test_kaksois_kertymä_kerrytä_ja_vastaavuus() {
        let mut hdc = luo_hdc();
        let aakkosto = testi_aakkosto();
        let mut k = KaksoisKertymä::new(&aakkosto, ULOTTUVUUS);

        // Create a distinctive context for 'a'
        let ctx_a = hdc.satunnainen_bipolaarinen();
        k.kerrytä('a', &ctx_a, 1.0);
        k.kerrytä('a', &ctx_a, 1.0);
        k.kerrytä('a', &ctx_a, 1.0);

        // Query with same context → should match 'a'
        let (paras, _) = k.paras_vastaavuus(&ctx_a, &hdc);
        assert_eq!(paras, 'a', "accumulated context should match 'a'");
    }

    #[test]
    fn test_kaksoset_ennusta_myöntö_dominates_initially() {
        let hdc = luo_hdc();
        let aakkosto = testi_aakkosto();
        let mut kaksoset = Kaksoset::new(&aakkosto, ULOTTUVUUS);

        // Accumulate strong signal for 'a' in Myöntö only
        let ctx = vec![1.0; ULOTTUVUUS];
        for _ in 0..10 {
            kaksoset.myöntö.kerrytä('a', &ctx, 1.0);
        }

        let (ennuste, _) = kaksoset.ennusta(&ctx, &hdc);
        assert_eq!(ennuste, 'a',
            "with no Kielto signal, Myöntö should determine prediction");
    }

    #[test]
    fn test_kaksoset_kielto_suppresses() {
        let hdc = luo_hdc();
        let aakkosto = testi_aakkosto();
        let mut kaksoset = Kaksoset::new(&aakkosto, ULOTTUVUUS);

        // Myöntö: moderate signal for 'a' and 'b'
        let ctx = vec![1.0; ULOTTUVUUS];
        for _ in 0..5 {
            kaksoset.myöntö.kerrytä('a', &ctx, 1.0);
            kaksoset.myöntö.kerrytä('b', &ctx, 0.9);
        }

        // Kielto: strong anti-signal for 'a'
        // (context looked like 'a' but was wrong many times)
        for _ in 0..20 {
            kaksoset.kielto.kerrytä('a', &ctx, 1.0);
        }

        // Force high beta (high cortisol)
        kaksoset.hormonit.kortisoli = 0.9;

        let (ennuste, _) = kaksoset.ennusta(&ctx, &hdc);
        assert_eq!(ennuste, 'b',
            "Kielto should suppress 'a', making 'b' win");
    }

    #[test]
    fn test_kaksoset_kouluta_askel() {
        let mut hdc = luo_hdc();
        let aakkosto = testi_aakkosto();
        let mut kaksoset = Kaksoset::new(&aakkosto, ULOTTUVUUS);

        // Train a few steps
        let ctx = hdc.satunnainen_bipolaarinen();
        for _ in 0..5 {
            let tulos = kaksoset.kouluta_askel(&ctx, 'a', &hdc);
            assert_eq!(tulos.kohde, 'a');
        }

        assert!(kaksoset.askel == 5, "should have 5 steps");
        assert!(kaksoset.myöntö.kokonaismäärä() > 0,
            "Myöntö should have accumulated");
    }

    #[test]
    fn test_purske_tila() {
        let mut kaksoset = Kaksoset::new(&testi_aakkosto(), ULOTTUVUUS);
        kaksoset.purske_päälle();
        assert!(kaksoset.sydän.pursketila, "should be in burst mode");

        // Heart should not beat in burst mode
        for _ in 0..100 {
            assert!(!kaksoset.sydän.lyö(),
                "heart should not beat during burst");
        }

        kaksoset.purske_pois();
        assert!(!kaksoset.sydän.pursketila, "should exit burst mode");
    }

    #[test]
    fn test_hrv_koherenssi() {
        let mut sydän = Sydän::new();
        // Feed beats with φ-ratio intervals
        sydän.väli_historia = vec![5, 8, 13, 8, 5, 8, 13];
        // 8/5 = 1.6 ≈ φ, 13/8 = 1.625 ≈ φ
        sydän.päivitä_hrv();
        assert!(sydän.hrv_koherenssi > 0.5,
            "Fibonacci intervals should show high φ-coherence: {}",
            sydän.hrv_koherenssi);

        // Feed beats with uniform intervals (no variability)
        sydän.väli_historia = vec![8, 8, 8, 8, 8];
        // 8/8 = 1.0, far from φ
        sydän.päivitä_hrv();
        assert!(sydän.hrv_koherenssi < 0.5,
            "uniform intervals should show low φ-coherence: {}",
            sydän.hrv_koherenssi);
    }

    #[test]
    fn test_beta_responds_to_stress() {
        let aakkosto = testi_aakkosto();
        let mut kaksoset = Kaksoset::new(&aakkosto, ULOTTUVUUS);

        // Low stress
        kaksoset.hormonit.kortisoli = 0.1;
        kaksoset.hormonit.oksitosiini = 0.8;
        let beta_low = kaksoset.beta();

        // High stress
        kaksoset.hormonit.kortisoli = 0.9;
        kaksoset.hormonit.oksitosiini = 0.1;
        let beta_high = kaksoset.beta();

        assert!(beta_high > beta_low,
            "high stress should increase β: low={beta_low} high={beta_high}");
    }
}
