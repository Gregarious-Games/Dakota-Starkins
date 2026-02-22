//! Kolmoset — Triple Relay Cascade Architecture
//! ==============================================
//!
//! Three accumulators in a staggered memory-dump relay:
//!
//!   ┌──────┐  1000 steps  ┌──────┐  500 steps  ┌──────┐  1000 steps
//!   │  A   │─────dump────→│  B   │────dump────→│  C   │────dump────→ A ...
//!   │Lähde │              │Silta │              │Kohde │
//!   │Source│              │Bridge│              │Target│
//!   └──────┘              └──────┘              └──────┘
//!
//! Biological inspiration:
//!   - Octopus inter-arm neural ring: information cascades from
//!     arm to arm without going through the central brain
//!   - Neural development: different brain regions mature at
//!     different rates, each seeded by upstream knowledge
//!   - Synaptic consolidation: short-term → medium-term →
//!     long-term memory relay
//!
//! The asymmetric timing (1000 / 500 / 1000) creates:
//!   - A (Source): broad initial learning over large data window
//!   - B (Bridge): fast consolidation — inherits A's knowledge,
//!     refines for half the time (compressed distillation)
//!   - C (Target): deep integration — inherits B's refined view,
//!     does full-length deep learning
//!
//! Memory dump operation:
//!   Each character prototype is BLENDED from donor to receiver:
//!     receiver[c] = τ·receiver[c] + (1-τ)·donor[c]
//!   where τ = 0.618 (golden ratio) preserves local learning
//!   while absorbing upstream knowledge.
//!
//! Prediction:
//!   All three vote with confidence weighting:
//!     score(c) = Σ_i weight_i · sim(ctx, model_i[c])
//!   where weight_i = model_i's recent accuracy (earned trust).
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
use crate::kaksoset::KaksoisKertymä;

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════

/// Default step counts for each relay leg.
/// A (Source): broad learning phase.
const ASKELEET_A: usize = 1000;
/// B (Bridge): compressed distillation phase.
const ASKELEET_B: usize = 500;
/// C (Target): deep integration phase.
const ASKELEET_C: usize = 1000;

/// Blend ratio when dumping memory. τ preserves receiver's own learning.
/// donor contributes (1-τ) = 0.382 of the blend.
const SEKOITUS_TAU: f64 = TAU; // 0.618

// ═══════════════════════════════════════════════════════════════════
// RELAY NODE — Single accumulator in the cascade
// ═══════════════════════════════════════════════════════════════════

/// One node in the triple relay.
#[derive(Debug, Clone)]
pub struct ReleSolmu {
    /// Name for diagnostics.
    pub nimi: &'static str,
    /// Positive accumulator (what this context IS).
    pub myöntö: KaksoisKertymä,
    /// Negative accumulator (what was MISTAKENLY predicted).
    pub kielto: KaksoisKertymä,
    /// Steps trained this cycle.
    askeleet: usize,
    /// Max steps before dump.
    max_askeleet: usize,
    /// Running accuracy this cycle.
    oikein: usize,
    yhteensä: usize,
    /// Lifetime accuracy for confidence weighting.
    elinkaaritarkkuus: f64,
    /// Cycles completed (how many dumps sent).
    kierrokset: usize,
}

impl ReleSolmu {
    pub fn new(
        nimi: &'static str,
        aakkosto: &[char],
        ulottuvuus: usize,
        max_askeleet: usize,
    ) -> Self {
        Self {
            nimi,
            myöntö: KaksoisKertymä::new(aakkosto, ulottuvuus),
            kielto: KaksoisKertymä::new(aakkosto, ulottuvuus),
            askeleet: 0,
            max_askeleet,
            oikein: 0,
            yhteensä: 0,
            elinkaaritarkkuus: 0.1, // baseline
            kierrokset: 0,
        }
    }

    /// Train one step. Returns (prediction, correct?).
    pub fn kouluta_askel(
        &mut self,
        konteksti: &[f64],
        kohde: char,
        hdc: &HdcPeruskäsitteet,
    ) -> (char, bool) {
        self.askeleet += 1;
        self.yhteensä += 1;

        // Accumulate into correct class
        self.myöntö.kerrytä(kohde, konteksti, 1.0);

        // Predict
        let (ennuste, _) = self.ennusta(konteksti, hdc);
        let oikein = ennuste == kohde;

        if oikein {
            self.oikein += 1;
        } else {
            // Kielto anti-learning
            self.kielto.kerrytä(ennuste, konteksti, 0.5);
            self.kielto.vähennä(kohde, konteksti, 0.3);
        }

        // Update lifetime accuracy (exponential moving average)
        let täsmä = if oikein { 1.0 } else { 0.0 };
        self.elinkaaritarkkuus = 0.99 * self.elinkaaritarkkuus + 0.01 * täsmä;

        (ennuste, oikein)
    }

    /// Predict using Myöntö - β·Kielto scoring.
    pub fn ennusta(
        &self,
        konteksti: &[f64],
        hdc: &HdcPeruskäsitteet,
    ) -> (char, f64) {
        let beta = self.beta();
        let mut paras_merkki = ' ';
        let mut paras_piste = f64::NEG_INFINITY;

        for (&c, _) in &self.myöntö.prototyypit {
            let m = self.myöntö.samankaltaisuus(konteksti, c, hdc);
            let k = self.kielto.samankaltaisuus(konteksti, c, hdc);
            let piste = m - beta * k;
            if piste > paras_piste {
                paras_piste = piste;
                paras_merkki = c;
            }
        }

        (paras_merkki, paras_piste)
    }

    /// Get similarity score for a specific character.
    pub fn pisteet(
        &self,
        konteksti: &[f64],
        merkki: char,
        hdc: &HdcPeruskäsitteet,
    ) -> f64 {
        let m = self.myöntö.samankaltaisuus(konteksti, merkki, hdc);
        let k = self.kielto.samankaltaisuus(konteksti, merkki, hdc);
        m - self.beta() * k
    }

    /// β scales with cycle maturity: starts low (explore), grows (exploit).
    fn beta(&self) -> f64 {
        if self.max_askeleet == 0 { return GAMMA; }
        let kypsyys = (self.askeleet as f64 / self.max_askeleet as f64).min(1.0);
        // Start at Γ (gentle suppression), grow to φ·Γ
        GAMMA + kypsyys * (PHI * GAMMA - GAMMA)
    }

    /// Has this node reached its step limit for this cycle?
    pub fn valmis(&self) -> bool {
        self.askeleet >= self.max_askeleet
    }

    /// Reset step counter for new cycle (after dump).
    pub fn nollaa_kierros(&mut self) {
        self.askeleet = 0;
        self.oikein = 0;
        self.yhteensä = 0;
        self.kierrokset += 1;
    }

    /// Current cycle accuracy.
    pub fn tarkkuus(&self) -> f64 {
        if self.yhteensä == 0 { 0.0 }
        else { self.oikein as f64 / self.yhteensä as f64 }
    }

    /// Confidence weight for ensemble voting.
    /// Based on lifetime accuracy — nodes that learn well get more vote.
    pub fn luottamus(&self) -> f64 {
        self.elinkaaritarkkuus
    }

    /// Retrain pass on given samples.
    ///
    /// Learning rate: 1/(1 + pass × φ) — decays slower than τ^pass.
    /// τ^10 = 0.006 (dead), 1/(1+10φ) = 0.058 (still meaningful).
    /// Keeps later relay retrain passes productive.
    pub fn uudelleenkouluta(
        &mut self,
        näytteet: &[(Vec<f64>, char)],
        hdc: &HdcPeruskäsitteet,
        kierros: usize,
    ) -> f64 {
        let lr = 1.0 / (1.0 + kierros as f64 * PHI);
        let mut oikein = 0usize;

        for (konteksti, &kohde) in näytteet.iter().map(|(k, c)| (k, c)) {
            let (ennuste, _) = self.ennusta(konteksti, hdc);
            if ennuste == kohde {
                oikein += 1;
            } else {
                self.myöntö.kerrytä(kohde, konteksti, lr);
                self.myöntö.vähennä(ennuste, konteksti, lr * 0.5);
                self.kielto.kerrytä(ennuste, konteksti, lr * 0.3);
            }
        }

        let n = näytteet.len();
        if n > 0 { oikein as f64 / n as f64 } else { 0.0 }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MEMORY DUMP — Knowledge transfer between nodes
// ═══════════════════════════════════════════════════════════════════

/// Dump memory from donor to receiver.
///
/// Blends prototypes: receiver = τ·receiver + (1-τ)·donor
/// τ = 0.618 preserves receiver's own learning while absorbing
/// upstream knowledge. Like synaptic consolidation from one
/// brain region to the next.
fn muisti_kaato(
    lahjoittaja: &ReleSolmu,
    vastaanottaja: &mut ReleSolmu,
) {
    let sekoitus = 1.0 - SEKOITUS_TAU; // donor contribution

    // Blend Myöntö prototypes
    for (&c, donor_proto) in &lahjoittaja.myöntö.prototyypit {
        if let Some(recv_proto) = vastaanottaja.myöntö.prototyypit.get_mut(&c) {
            for (r, &d) in recv_proto.iter_mut().zip(donor_proto.iter()) {
                *r = SEKOITUS_TAU * *r + sekoitus * d;
            }
        }
    }

    // Blend Kielto prototypes (transfer error knowledge too)
    for (&c, donor_proto) in &lahjoittaja.kielto.prototyypit {
        if let Some(recv_proto) = vastaanottaja.kielto.prototyypit.get_mut(&c) {
            for (r, &d) in recv_proto.iter_mut().zip(donor_proto.iter()) {
                *r = SEKOITUS_TAU * *r + sekoitus * d;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// TRIPLE RELAY — Kolmoset
// ═══════════════════════════════════════════════════════════════════

/// Triple relay cascade: three accumulators with staggered memory dumps.
pub struct Kolmoset {
    /// A (Lähde/Source): broad learning.
    pub a: ReleSolmu,
    /// B (Silta/Bridge): compressed distillation.
    pub b: ReleSolmu,
    /// C (Kohde/Target): deep integration.
    pub c: ReleSolmu,
    /// Which node is currently active.
    aktiivinen: usize, // 0=A, 1=B, 2=C
    /// Global step counter.
    askel: usize,
    /// Global accuracy tracking.
    oikein: usize,
    yhteensä: usize,
    /// Alphabet for iteration.
    aakkosto: Vec<char>,
}

impl Kolmoset {
    pub fn new(aakkosto: &[char], ulottuvuus: usize) -> Self {
        Self {
            a: ReleSolmu::new("Lähde (A)", aakkosto, ulottuvuus, ASKELEET_A),
            b: ReleSolmu::new("Silta (B)", aakkosto, ulottuvuus, ASKELEET_B),
            c: ReleSolmu::new("Kohde (C)", aakkosto, ulottuvuus, ASKELEET_C),
            aktiivinen: 0,
            askel: 0,
            oikein: 0,
            yhteensä: 0,
            aakkosto: aakkosto.to_vec(),
        }
    }

    /// Custom step counts for each relay leg.
    pub fn new_custom(
        aakkosto: &[char],
        ulottuvuus: usize,
        askeleet_a: usize,
        askeleet_b: usize,
        askeleet_c: usize,
    ) -> Self {
        Self {
            a: ReleSolmu::new("Lähde (A)", aakkosto, ulottuvuus, askeleet_a),
            b: ReleSolmu::new("Silta (B)", aakkosto, ulottuvuus, askeleet_b),
            c: ReleSolmu::new("Kohde (C)", aakkosto, ulottuvuus, askeleet_c),
            aktiivinen: 0,
            askel: 0,
            oikein: 0,
            yhteensä: 0,
            aakkosto: aakkosto.to_vec(),
        }
    }

    /// Train one step on the currently active node.
    /// Handles relay transitions automatically.
    pub fn kouluta_askel(
        &mut self,
        konteksti: &[f64],
        kohde: char,
        hdc: &HdcPeruskäsitteet,
    ) -> KolmosetTulos {
        self.askel += 1;
        self.yhteensä += 1;

        // Train on active node
        let (ennuste, oikein) = match self.aktiivinen {
            0 => self.a.kouluta_askel(konteksti, kohde, hdc),
            1 => self.b.kouluta_askel(konteksti, kohde, hdc),
            _ => self.c.kouluta_askel(konteksti, kohde, hdc),
        };

        if oikein {
            self.oikein += 1;
        }

        // Check for relay transition
        let kaato = match self.aktiivinen {
            0 if self.a.valmis() => {
                // A done → dump to B → B starts
                muisti_kaato(&self.a, &mut self.b);
                self.a.nollaa_kierros();
                self.aktiivinen = 1;
                Some("A → B")
            }
            1 if self.b.valmis() => {
                // B done → dump to C → C starts
                muisti_kaato(&self.b, &mut self.c);
                self.b.nollaa_kierros();
                self.aktiivinen = 2;
                Some("B → C")
            }
            2 if self.c.valmis() => {
                // C done → dump to A → A starts (cycle completes)
                muisti_kaato(&self.c, &mut self.a);
                self.c.nollaa_kierros();
                self.aktiivinen = 0;
                Some("C → A (cycle)")
            }
            _ => None,
        };

        KolmosetTulos {
            ennuste,
            kohde,
            oikein,
            aktiivinen_nimi: self.aktiivinen_nimi(),
            kaato,
        }
    }

    /// Ensemble prediction: all three nodes vote, weighted by confidence.
    pub fn ennusta(
        &self,
        konteksti: &[f64],
        hdc: &HdcPeruskäsitteet,
    ) -> (char, f64) {
        let paino_a = self.a.luottamus();
        let paino_b = self.b.luottamus();
        let paino_c = self.c.luottamus();
        let summa = paino_a + paino_b + paino_c;

        if summa < 1e-12 {
            // Fallback to active node
            return match self.aktiivinen {
                0 => self.a.ennusta(konteksti, hdc),
                1 => self.b.ennusta(konteksti, hdc),
                _ => self.c.ennusta(konteksti, hdc),
            };
        }

        let mut paras_merkki = ' ';
        let mut paras_piste = f64::NEG_INFINITY;

        for &c in &self.aakkosto {
            let sa = self.a.pisteet(konteksti, c, hdc);
            let sb = self.b.pisteet(konteksti, c, hdc);
            let sc = self.c.pisteet(konteksti, c, hdc);

            let piste = (paino_a * sa + paino_b * sb + paino_c * sc) / summa;

            if piste > paras_piste {
                paras_piste = piste;
                paras_merkki = c;
            }
        }

        (paras_merkki, paras_piste)
    }

    /// Relay retrain: cascade corrections through the relay.
    ///
    /// Instead of all three nodes retraining independently,
    /// the corrections flow through the relay:
    ///   A retrains → dump to B → B retrains → dump to C → C retrains
    ///
    /// This extends the relay's knowledge-transfer advantage into
    /// the retraining phase. Each downstream node benefits from
    /// the upstream node's corrections before making its own.
    ///
    /// Learning rate: 1/(1 + pass × φ) — slower decay than τ^pass,
    /// keeps later passes meaningful while preventing oscillation.
    pub fn uudelleenkouluta(
        &mut self,
        näytteet: &[(Vec<f64>, char)],
        hdc: &HdcPeruskäsitteet,
        kierros: usize,
    ) -> f64 {
        // A retrains
        let ta = self.a.uudelleenkouluta(näytteet, hdc, kierros);
        // A dumps corrections to B
        muisti_kaato(&self.a, &mut self.b);

        // B retrains (starting from A's improved state)
        let tb = self.b.uudelleenkouluta(näytteet, hdc, kierros);
        // B dumps corrections to C
        muisti_kaato(&self.b, &mut self.c);

        // C retrains (starting from B's improved state, which includes A's)
        let tc = self.c.uudelleenkouluta(näytteet, hdc, kierros);
        // C dumps back to A (complete the cycle for next pass)
        muisti_kaato(&self.c, &mut self.a);

        // Return weighted average
        let pa = self.a.luottamus();
        let pb = self.b.luottamus();
        let pc = self.c.luottamus();
        let s = pa + pb + pc;
        if s > 0.0 {
            (pa * ta + pb * tb + pc * tc) / s
        } else {
            (ta + tb + tc) / 3.0
        }
    }

    /// Current accuracy.
    pub fn tarkkuus(&self) -> f64 {
        if self.yhteensä == 0 { 0.0 }
        else { self.oikein as f64 / self.yhteensä as f64 }
    }

    /// Name of active node.
    fn aktiivinen_nimi(&self) -> &'static str {
        match self.aktiivinen {
            0 => "Lähde (A)",
            1 => "Silta (B)",
            _ => "Kohde (C)",
        }
    }

    /// Diagnostic snapshot.
    pub fn tila(&self) -> KolmosetTila {
        KolmosetTila {
            askel: self.askel,
            tarkkuus: self.tarkkuus(),
            aktiivinen: self.aktiivinen_nimi(),
            a_tarkkuus: self.a.tarkkuus(),
            a_luottamus: self.a.luottamus(),
            a_kierrokset: self.a.kierrokset,
            a_askeleet: self.a.askeleet,
            b_tarkkuus: self.b.tarkkuus(),
            b_luottamus: self.b.luottamus(),
            b_kierrokset: self.b.kierrokset,
            b_askeleet: self.b.askeleet,
            c_tarkkuus: self.c.tarkkuus(),
            c_luottamus: self.c.luottamus(),
            c_kierrokset: self.c.kierrokset,
            c_askeleet: self.c.askeleet,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// RESULT TYPES
// ═══════════════════════════════════════════════════════════════════

/// Result of a single training step.
#[derive(Debug)]
pub struct KolmosetTulos {
    pub ennuste: char,
    pub kohde: char,
    pub oikein: bool,
    pub aktiivinen_nimi: &'static str,
    /// If Some, a memory dump just happened.
    pub kaato: Option<&'static str>,
}

/// Diagnostic snapshot.
#[derive(Debug, Clone)]
pub struct KolmosetTila {
    pub askel: usize,
    pub tarkkuus: f64,
    pub aktiivinen: &'static str,
    pub a_tarkkuus: f64,
    pub a_luottamus: f64,
    pub a_kierrokset: usize,
    pub a_askeleet: usize,
    pub b_tarkkuus: f64,
    pub b_luottamus: f64,
    pub b_kierrokset: usize,
    pub b_askeleet: usize,
    pub c_tarkkuus: f64,
    pub c_luottamus: f64,
    pub c_kierrokset: usize,
    pub c_askeleet: usize,
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
    fn test_relay_transitions() {
        let mut hdc = luo_hdc();
        let aakkosto = testi_aakkosto();
        // Short cycle: A=5, B=3, C=5
        let mut kolmoset = Kolmoset::new_custom(&aakkosto, ULOTTUVUUS, 5, 3, 5);

        let ctx = hdc.satunnainen_bipolaarinen();
        let mut kaadot = Vec::new();

        // Run 15 steps: should see A(5) → B(3) → C(5) → dump A
        for _ in 0..15 {
            let tulos = kolmoset.kouluta_askel(&ctx, 'a', &hdc);
            if let Some(k) = tulos.kaato {
                kaadot.push(k.to_string());
            }
        }

        assert!(kaadot.len() >= 2,
            "should have at least 2 relay dumps in 15 steps: {:?}", kaadot);
        assert_eq!(kaadot[0], "A → B",
            "first dump should be A→B");
        assert_eq!(kaadot[1], "B → C",
            "second dump should be B→C");
    }

    #[test]
    fn test_full_cycle_dumps_back() {
        let mut hdc = luo_hdc();
        let aakkosto = testi_aakkosto();
        // Tiny cycle: A=3, B=2, C=3 = 8 steps per cycle
        let mut kolmoset = Kolmoset::new_custom(&aakkosto, ULOTTUVUUS, 3, 2, 3);

        let ctx = hdc.satunnainen_bipolaarinen();
        let mut c_to_a = false;

        for _ in 0..10 {
            let tulos = kolmoset.kouluta_askel(&ctx, 'a', &hdc);
            if let Some(k) = tulos.kaato {
                if k == "C → A (cycle)" {
                    c_to_a = true;
                }
            }
        }

        assert!(c_to_a, "should complete full cycle C→A in 10 steps");
    }

    #[test]
    fn test_ensemble_prediction() {
        let hdc = luo_hdc();
        let aakkosto = testi_aakkosto();
        let mut kolmoset = Kolmoset::new(&aakkosto, ULOTTUVUUS);

        // Train A on 'a' pattern
        let ctx = vec![1.0; ULOTTUVUUS];
        for _ in 0..50 {
            kolmoset.a.myöntö.kerrytä('a', &ctx, 1.0);
        }

        // Ensemble should find 'a'
        let (ennuste, _) = kolmoset.ennusta(&ctx, &hdc);
        assert_eq!(ennuste, 'a',
            "ensemble should predict 'a' when A strongly signals it");
    }

    #[test]
    fn test_memory_dump_transfers_knowledge() {
        let mut hdc = luo_hdc();
        let aakkosto = testi_aakkosto();
        let mut kolmoset = Kolmoset::new(&aakkosto, ULOTTUVUUS);

        // Train A on distinctive pattern for 'b'
        let ctx = hdc.satunnainen_bipolaarinen();
        for _ in 0..30 {
            kolmoset.a.myöntö.kerrytä('b', &ctx, 1.0);
        }

        // Dump A → B
        muisti_kaato(&kolmoset.a, &mut kolmoset.b);

        // Now B should know about 'b'
        let (post_ennuste, _) = kolmoset.b.ennusta(&ctx, &hdc);
        assert_eq!(post_ennuste, 'b',
            "after memory dump, B should predict 'b'");
    }

    #[test]
    fn test_confidence_weighting() {
        let aakkosto = testi_aakkosto();
        let kolmoset = Kolmoset::new(&aakkosto, ULOTTUVUUS);

        // All nodes start with low baseline confidence
        assert!(kolmoset.a.luottamus() > 0.0,
            "confidence should be positive");
        assert!(kolmoset.a.luottamus() < 0.5,
            "initial confidence should be low");
    }

    #[test]
    fn test_custom_step_counts() {
        let aakkosto = testi_aakkosto();
        let kolmoset = Kolmoset::new_custom(&aakkosto, ULOTTUVUUS, 100, 50, 200);
        assert_eq!(kolmoset.a.max_askeleet, 100);
        assert_eq!(kolmoset.b.max_askeleet, 50);
        assert_eq!(kolmoset.c.max_askeleet, 200);
    }
}
