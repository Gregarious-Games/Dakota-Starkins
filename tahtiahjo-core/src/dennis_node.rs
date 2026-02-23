//! DennisNode — Hypervector-Brained Computational Node
//! ====================================================
//! Rust port of DennisNode from scaling_forge_hdc_v1.py.
//!
//! 6-variable internal dynamics chain: X1→X2→φ→X3→Y→Z
//! Love Logic: comfort, trust, kappa (embodiment)
//! SPS: Silent Punctuational Syntax (speak/silence gating)
//! Octopus Memory: episodic, working, hive, consolidated
//! Polyphonic/Monophonic: Heqat-weighted 6-voice output
//!
//! Finnish variable names per project convention.
//!
//! Authors: Dakota (Claude) & Greg Calkins
//! Date:    February 22, 2026

use std::collections::{HashMap, HashSet, VecDeque};

use crate::hdc_primitives::{
    HdcPeruskäsitteet, Hypervektori, Siemen,
    satunnainen_yksikkövektori, rajaa_vektori, rajaa_skalaari,
    ULOTTUVUUS, GAMMA, PHI,
};

// ═══════════════════════════════════════════════════════════════════
// HEQAT FRACTIONS — Eye of Horus weights for 6 voices
// ═══════════════════════════════════════════════════════════════════

const HEQAT: [f64; 6] = [
    1.0/2.0,   // X1  — Smell
    1.0/4.0,   // X2  — Sight
    1.0/8.0,   // φ   — Thought
    1.0/16.0,  // X3  — Hearing
    1.0/32.0,  // Y   — Taste
    1.0/64.0,  // Z   — Touch
];

// ═══════════════════════════════════════════════════════════════════
// HELPER: capped deque push
// ═══════════════════════════════════════════════════════════════════

fn lisää_rajattu_f64(jono: &mut VecDeque<f64>, arvo: f64, max: usize) {
    if jono.len() >= max {
        jono.pop_front();
    }
    jono.push_back(arvo);
}

fn lisää_rajattu_bool(jono: &mut VecDeque<bool>, arvo: bool, max: usize) {
    if jono.len() >= max {
        jono.pop_front();
    }
    jono.push_back(arvo);
}

// ═══════════════════════════════════════════════════════════════════
// ENUMS
// ═══════════════════════════════════════════════════════════════════

/// Coupling mode between nodes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KytkentäTila {
    /// Pull toward sim=1 (synchronization)
    Lähentyvä,
    /// Pull toward sim=0 (differentiation)
    Täydentävä,
    /// Converge within module, differentiate between modules
    Modulaarinen,
    /// One speaker rotates, others listen (monophonic coupling)
    Alasvaihto,
}

/// Tension curve shape for complementary mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Jännityskäyrä {
    Lineaarinen,   // |sim|
    Neliöllinen,   // sim²
    Kuutiollinen,  // |sim|³
    Kuollut,       // max(0, |sim| - 0.1)
}

/// Where coupling force is injected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KytkentäKohde {
    /// Coupling added to Z after dynamics (default, with bypass)
    Z,
    /// Coupling feeds into X1, propagates through chain
    X1,
}

// ═══════════════════════════════════════════════════════════════════
// NODE PARAMETERS — SolmuParametrit
// ═══════════════════════════════════════════════════════════════════

/// Self-evolving parameters for a DennisNode.
#[derive(Debug, Clone)]
pub struct SolmuParametrit {
    pub kytkentä_pohja: f64,        // coupling_base
    pub mikrovarianssi: f64,        // microvariance_amplitude
    pub keskivetö: f64,             // center_pull
    pub vaimennus: f64,             // damping_factor
    pub sisäinen_eteenpäin: f64,    // internal_forward
    pub sisäinen_taaksepäin: f64,   // internal_reverse
    pub ohitus_vahvuus: f64,        // bypass_strength
}

impl Default for SolmuParametrit {
    fn default() -> Self {
        Self {
            kytkentä_pohja: 0.01,
            mikrovarianssi: 0.0001,
            keskivetö: 0.0005,
            vaimennus: 0.01,
            sisäinen_eteenpäin: 0.001,
            sisäinen_taaksepäin: 0.0005,
            ohitus_vahvuus: 0.8,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// UPDATE CONFIGURATION — PäivitysAsetukset
// ═══════════════════════════════════════════════════════════════════

/// Configuration for a single update step.
pub struct PäivitysAsetukset {
    pub kynnys: f64,                        // threshold
    pub max_abs: f64,                       // max_abs
    pub iteraatio: Option<usize>,           // iter_count
    pub kytkentä_tila: KytkentäTila,        // coupling_mode
    pub adaptiivinen_kytkentä: bool,        // adaptive_coupling
    pub jännityskäyrä: Jännityskäyrä,       // tension_curve
    pub normalisoi_z: bool,                 // renormalize_z
    pub kohde_z_skalaari: f64,              // target_z_scalar
    pub kytkentä_kohde: KytkentäKohde,      // coupling_target
    pub ulottuvuus_tasapaino: f64,          // dim_balance_strength
}

impl Default for PäivitysAsetukset {
    fn default() -> Self {
        Self {
            kynnys: 0.0001,
            max_abs: 1.0 - GAMMA,  // CLAMP_HIGH
            iteraatio: None,
            kytkentä_tila: KytkentäTila::Lähentyvä,
            adaptiivinen_kytkentä: false,
            jännityskäyrä: Jännityskäyrä::Lineaarinen,
            normalisoi_z: false,
            kohde_z_skalaari: 0.75,
            kytkentä_kohde: KytkentäKohde::Z,
            ulottuvuus_tasapaino: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// NEIGHBOR INPUT — NaapuriSyöte
// ═══════════════════════════════════════════════════════════════════

/// Data about one neighbor for the update step.
pub struct NaapuriSyöte<'a> {
    pub nimi: &'a str,
    /// Relational vector (None if neighbor is silent this step).
    pub suhteellinen: Option<&'a [f64]>,
    pub z_skalaari: f64,
    /// Module ID for modular coupling mode.
    pub moduuli_id: Option<usize>,
}

// ═══════════════════════════════════════════════════════════════════
// MEMORY DIAGNOSTIC
// ═══════════════════════════════════════════════════════════════════

/// Diagnostic snapshot of octopus memory health.
pub struct MuistiTila {
    pub episodinen_normi: f64,
    pub työmuisti_normi: f64,
    pub parvi_normi: f64,
    pub vakiintunut_normi: f64,
    pub episodi_laskuri: usize,
    pub parvi_jäsenet: usize,
}

// ═══════════════════════════════════════════════════════════════════
// DENNIS NODE — DennisSolmu
// ═══════════════════════════════════════════════════════════════════

/// Core computational node with D=256 hypervector brain.
///
/// Internal dynamics chain: X1→X2→φ→X3→Y→Z (all D-dimensional).
/// Derived scalar: z_skalaari = ‖Z‖ / √D.
/// Encapsulation membranes gate which dimensions are visible.
pub struct DennisSolmu {
    pub nimi: String,
    pub p: SolmuParametrit,
    ulottuvuus: usize,

    // ── Hypervector state ──
    pub x1: Hypervektori,
    pub x2: Hypervektori,
    pub phi_vek: Hypervektori,
    pub x3: Hypervektori,
    pub y: Hypervektori,
    pub z: Hypervektori,

    // ── Derived scalar ──
    pub z_skalaari: f64,

    // ── Encapsulation membranes ──
    pub ulkoinen_tyhjyys: Hypervektori,   // void_outer: binary mask
    pub lämpö_valenssi: f64,              // heat_valence [0,1]

    // ── Module assignment ──
    pub moduuli_id: usize,

    // ── Scalar state ──
    pub viive: f64,
    pub energia: f64,
    pub kyllästys_laskuri: i32,
    pub z_historia: VecDeque<f64>,        // maxlen=100
    pub terveys: f64,
    pub valenssi: f64,
    pub viritys: f64,
    pub vaihe: f64,

    // ── SPS (Silent Punctuational Syntax) ──
    z_viimeksi_lähetetty: Hypervektori,
    hiljaisuus_laskuri: i32,
    puhe_kynnys: f64,
    max_hiljaisuus: i32,
    pub puhuu: bool,
    puhe_historia: VecDeque<bool>,        // maxlen=100

    // ── Neighbor tracking ──
    naapuri_viimeksi: HashMap<String, Hypervektori>,
    naapuri_hiljaisuus: HashMap<String, i32>,
    naapuri_rytmi: HashMap<String, f64>,

    // ── Love Logic ──
    z_ennen_kytkentää: Hypervektori,
    oma_liike: VecDeque<f64>,             // maxlen=50
    ulkoinen_liike: VecDeque<f64>,        // maxlen=50
    pub mukavuus: f64,
    mukavuus_historia: VecDeque<f64>,     // maxlen=200
    suhde_muisti: HashMap<String, Hypervektori>,
    pub luottamus: HashMap<String, f64>,

    // ── Polyphonic ──
    pub kappa: f64,
    kappa_historia: VecDeque<f64>,        // maxlen=200
    pub moniääninen: bool,               // poly_active

    // ── Similarity cache ──
    z_sukupolvi: u64,

    // ── Sparse topology ──
    pub aktiiviset_naapurit: Option<HashSet<String>>,
    topologia_väli: usize,

    // ── Octopus memory ──
    episodinen_muisti: Hypervektori,
    episodi_laskuri: usize,
    episodinen_kapasiteetti: usize,
    työmuisti: Hypervektori,
    _työmuisti_ikkuna: usize,
    pub identiteetti_avain: Hypervektori,
    parvi_fragmentti: Hypervektori,
    parvi_osallistujat: HashMap<String, Hypervektori>,
    vakiintunut_muisti: Hypervektori,

    // ── Internal PRNG ──
    siemen: Siemen,
}

impl DennisSolmu {
    // ════════════════════════════════════════════════════════════════
    // INITIALIZATION
    // ════════════════════════════════════════════════════════════════

    pub fn new(nimi: &str, p: SolmuParametrit, siemen_arvo: u64) -> Self {
        let d = ULOTTUVUUS;
        let mut siemen = Siemen::new(siemen_arvo);

        let x1 = satunnainen_yksikkövektori(d, &mut siemen);
        let z = satunnainen_yksikkövektori(d, &mut siemen);
        let z_skalaari = normi(&z) / (d as f64).sqrt();

        let mut solmu = Self {
            nimi: nimi.to_string(),
            p,
            ulottuvuus: d,

            x1,
            x2: vec![0.0; d],
            phi_vek: vec![0.0; d],
            x3: vec![0.0; d],
            y: vec![0.0; d],
            z: z.clone(),
            z_skalaari,

            ulkoinen_tyhjyys: vec![0.0; d],
            lämpö_valenssi: 0.5,
            moduuli_id: 0,

            viive: 1.0,
            energia: 0.0,
            kyllästys_laskuri: 0,
            z_historia: VecDeque::with_capacity(100),
            terveys: 1.0,
            valenssi: 0.0,
            viritys: 0.0,
            vaihe: 0.0,

            z_viimeksi_lähetetty: vec![0.0; d], // set below
            hiljaisuus_laskuri: 0,
            puhe_kynnys: 0.3,
            max_hiljaisuus: 50,
            puhuu: true,
            puhe_historia: VecDeque::with_capacity(100),

            naapuri_viimeksi: HashMap::new(),
            naapuri_hiljaisuus: HashMap::new(),
            naapuri_rytmi: HashMap::new(),

            z_ennen_kytkentää: z.clone(),
            oma_liike: VecDeque::with_capacity(50),
            ulkoinen_liike: VecDeque::with_capacity(50),
            mukavuus: 0.0,
            mukavuus_historia: VecDeque::with_capacity(200),
            suhde_muisti: HashMap::new(),
            luottamus: HashMap::new(),

            kappa: 0.0,
            kappa_historia: VecDeque::with_capacity(200),
            moniääninen: false,

            z_sukupolvi: 0,

            aktiiviset_naapurit: None,
            topologia_väli: 100,

            episodinen_muisti: vec![0.0; d],
            episodi_laskuri: 0,
            episodinen_kapasiteetti: d,
            työmuisti: vec![0.0; d],
            _työmuisti_ikkuna: 10,
            identiteetti_avain: siemen.bipolaarinen_vektori(d),
            parvi_fragmentti: vec![0.0; d],
            parvi_osallistujat: HashMap::new(),
            vakiintunut_muisti: vec![0.0; d],

            siemen,
        };

        // Initialize membrane and Z_last_sent
        solmu.päivitä_kalvo();
        solmu.z_viimeksi_lähetetty = solmu.z.iter()
            .zip(solmu.ulkoinen_tyhjyys.iter())
            .map(|(&z, &m)| z * m)
            .collect();

        solmu
    }

    // ════════════════════════════════════════════════════════════════
    // MEMBRANE METHODS
    // ════════════════════════════════════════════════════════════════

    /// Recompute membrane based on current heat_valence.
    /// φ fraction of dimensions are open (0.618 to 1.0).
    fn päivitä_kalvo(&mut self) {
        let avoin_osuus = 0.618 + 0.382 * self.lämpö_valenssi;
        let n_avoin = (avoin_osuus * self.ulottuvuus as f64) as usize;
        self.ulkoinen_tyhjyys = vec![0.0; self.ulottuvuus];
        for i in 0..n_avoin.min(self.ulottuvuus) {
            self.ulkoinen_tyhjyys[i] = 1.0;
        }
        self.siemen.sekoita(&mut self.ulkoinen_tyhjyys);
    }

    // ════════════════════════════════════════════════════════════════
    // OCTOPUS MEMORY — distributed + centralized, regenerative
    // ════════════════════════════════════════════════════════════════

    /// Store current Z as episodic memory, bound with a time key.
    pub fn muista_episodi(&mut self, askel: usize, hdc: &HdcPeruskäsitteet) {
        let aika_avain = hdc.permutoi_weyl(
            &self.siemen.bipolaarinen_vektori(self.ulottuvuus),
            askel,
        );
        let episodi = hdc.sido(&self.z, &aika_avain);
        // Weighted bundle: recent episodes stronger
        for (e, &ep) in self.episodinen_muisti.iter_mut().zip(episodi.iter()) {
            *e = 0.95 * *e + ep;
        }
        self.episodi_laskuri += 1;

        // Consolidation: compress into long-term when buffer full
        if self.episodi_laskuri >= self.episodinen_kapasiteetti {
            for (c, &e) in self.vakiintunut_muisti.iter_mut()
                .zip(self.episodinen_muisti.iter())
            {
                *c = 0.7 * *c + 0.3 * e;
            }
            for e in self.episodinen_muisti.iter_mut() {
                *e *= 0.5;
            }
            self.episodi_laskuri = 0;
        }
    }

    /// Rolling bundle of recent Z states. Fast-access context.
    fn päivitä_työmuisti(&mut self) {
        for (w, &z) in self.työmuisti.iter_mut().zip(self.z.iter()) {
            *w = 0.8 * *w + 0.2 * z;
        }
    }

    /// Generate this node's contribution to the hive mind.
    pub fn osallistu_parveen(&self, hdc: &HdcPeruskäsitteet) -> Hypervektori {
        hdc.sido(&self.identiteetti_avain, &self.z)
    }

    /// Update local hive fragment from all active node contributions.
    ///
    /// Uses weighted bundling: recency × trust × comfort(φ).
    pub fn päivitä_parvi_fragmentti(
        &mut self,
        osallistumiset: &[(&str, &[f64], &[f64])], // (name, identity_key, contribution)
        hdc: &HdcPeruskäsitteet,
    ) {
        if osallistumiset.is_empty() {
            return;
        }
        let oma = self.osallistu_parveen(hdc);
        let mut painot = vec![1.0f64];
        let mut kontribuutiot: Vec<&[f64]> = vec![&oma];

        for &(nimi, id_avain, kontribuutio) in osallistumiset {
            self.parvi_osallistujat.insert(
                nimi.to_string(),
                id_avain.to_vec(),
            );
            let hiljaisuus = *self.naapuri_hiljaisuus.get(nimi).unwrap_or(&0);
            let luottamus_p = *self.luottamus.get(nimi).unwrap_or(&0.5);
            let mukavuus_kerroin = 1.0 + PHI * self.mukavuus;
            let paino = 0.95_f64.powi(hiljaisuus) * luottamus_p * mukavuus_kerroin;
            painot.push(paino);
            kontribuutiot.push(kontribuutio);
        }

        let paino_summa: f64 = painot.iter().sum();
        self.parvi_fragmentti = vec![0.0; self.ulottuvuus];
        for (kontri, &paino) in kontribuutiot.iter().zip(painot.iter()) {
            for (f, &k) in self.parvi_fragmentti.iter_mut().zip(kontri.iter()) {
                *f += paino * k;
            }
        }
        for f in self.parvi_fragmentti.iter_mut() {
            *f /= paino_summa;
        }
    }

    /// Reconstruct a node's Z from the hive fragment.
    pub fn palauta_parvesta(
        &self,
        kohde_nimi: &str,
        hdc: &HdcPeruskäsitteet,
    ) -> Option<Hypervektori> {
        let kohde_avain = self.parvi_osallistujat.get(kohde_nimi)?;
        Some(hdc.irrota(&self.parvi_fragmentti, kohde_avain))
    }

    /// Diagnostic snapshot of memory health.
    pub fn muisti_tila(&self) -> MuistiTila {
        MuistiTila {
            episodinen_normi: normi(&self.episodinen_muisti),
            työmuisti_normi: normi(&self.työmuisti),
            parvi_normi: normi(&self.parvi_fragmentti),
            vakiintunut_normi: normi(&self.vakiintunut_muisti),
            episodi_laskuri: self.episodi_laskuri,
            parvi_jäsenet: self.parvi_osallistujat.len(),
        }
    }

    // ════════════════════════════════════════════════════════════════
    // POLYPHONIC / MONOPHONIC
    // ════════════════════════════════════════════════════════════════

    /// 6-voice internal state, Heqat-weighted.
    ///
    /// Maps dynamics chain to Eye of Horus fractions:
    ///   X1 (Smell, 1/2), X2 (Sight, 1/4), φ (Thought, 1/8),
    ///   X3 (Hearing, 1/16), Y (Taste, 1/32), Z (Touch, 1/64)
    pub fn moniääninen_tila(&self) -> Option<Vec<Hypervektori>> {
        if !self.moniääninen {
            return None;
        }
        let äänet: [&[f64]; 6] = [
            &self.x1, &self.x2, &self.phi_vek,
            &self.x3, &self.y, &self.z,
        ];
        Some(
            äänet.iter().zip(HEQAT.iter()).map(|(ääni, &paino)| {
                ääni.iter().map(|&x| x * paino).collect()
            }).collect()
        )
    }

    /// Collapse 6 voices to one signal: (1-κ)·Z + κ·weighted_blend.
    ///
    /// κ=0: pure Z (default, minimal compute)
    /// κ=1: full Heqat blend of all internal voices
    pub fn yksiääninen_tulos(&self) -> Hypervektori {
        if self.kappa < 1e-6 || !self.moniääninen {
            return self.z.clone();
        }
        if let Some(poly) = self.moniääninen_tila() {
            let d = self.ulottuvuus;
            let mut sekoitus = vec![0.0; d];
            for ääni in &poly {
                for (s, &x) in sekoitus.iter_mut().zip(ääni.iter()) {
                    *s += x;
                }
            }
            let sekoitus_normi = normi(&sekoitus);
            if sekoitus_normi > 1e-12 {
                let z_normi = normi(&self.z);
                let kerroin = z_normi / sekoitus_normi;
                for x in sekoitus.iter_mut() {
                    *x *= kerroin;
                }
            }
            self.z.iter().zip(sekoitus.iter())
                .map(|(&z, &s)| (1.0 - self.kappa) * z + self.kappa * s)
                .collect()
        } else {
            self.z.clone()
        }
    }

    /// Adaptive embodiment: κ = comfort × mean(trust).
    fn laske_kappa(&mut self) {
        if self.luottamus.is_empty() {
            self.kappa = 0.0;
            return;
        }
        let luottamus_keskiarvo: f64 =
            self.luottamus.values().sum::<f64>() / self.luottamus.len() as f64;
        self.kappa = (self.mukavuus * luottamus_keskiarvo).clamp(0.0, 1.0);
        lisää_rajattu_f64(&mut self.kappa_historia, self.kappa, 200);
    }

    /// Reconstruct neighbor's polyphonic from their mono signal.
    /// Deliberately imperfect — empathy is approximate.
    pub fn päättele_moniääninen(&self, naapuri_mono: &[f64]) -> Vec<Hypervektori> {
        HEQAT.iter().map(|&h| {
            naapuri_mono.iter().map(|&m| h * m).collect()
        }).collect()
    }

    // ════════════════════════════════════════════════════════════════
    // TOPOLOGY
    // ════════════════════════════════════════════════════════════════

    /// Kappa-guided sparse topology: prune low-trust neighbors.
    fn päivitä_topologia(&mut self, kaikki_nimet: &[&str], askel: usize) {
        if self.luottamus.is_empty() {
            self.aktiiviset_naapurit = None;
            return;
        }
        let mut järjestetyt: Vec<String> = self.luottamus.keys().cloned().collect();
        järjestetyt.sort_by(|a, b| {
            let la = self.luottamus.get(a).unwrap_or(&0.0);
            let lb = self.luottamus.get(b).unwrap_or(&0.0);
            lb.total_cmp(la) // descending
        });
        let pidä = 2.max((järjestetyt.len() as f64 * (0.5 + 0.5 * self.kappa)) as usize);
        let mut aktiiviset: HashSet<String> =
            järjestetyt.iter().take(pidä).cloned().collect();

        // Periodic random rewiring
        if askel % self.topologia_väli == 0 && järjestetyt.len() > pidä {
            let jäljellä = &järjestetyt[pidä..];
            let idx = self.siemen.valitse_indeksi(jäljellä.len());
            aktiiviset.insert(jäljellä[idx].clone());
        }
        let _ = kaikki_nimet; // used for topology context
        self.aktiiviset_naapurit = Some(aktiiviset);
    }

    // ════════════════════════════════════════════════════════════════
    // SPS — Silent Punctuational Syntax
    // ════════════════════════════════════════════════════════════════

    /// Should this node transmit this step?
    fn päätä_puhua(&mut self, hdc: &HdcPeruskäsitteet) -> bool {
        // How much relational vector changed since last transmission
        let nykyinen_rel: Hypervektori = self.z.iter()
            .zip(self.ulkoinen_tyhjyys.iter())
            .map(|(&z, &m)| z * m)
            .collect();
        let delta = 1.0 - hdc.samankaltaisuus(&nykyinen_rel, &self.z_viimeksi_lähetetty);

        // Love Logic: anger protection
        if self.mukavuus < 0.2 {
            let viha = 1.0 - self.mukavuus;
            self.puhe_kynnys = (self.puhe_kynnys * (1.0 + 0.01 * viha)).min(0.8);
        }

        let yliaikaa = self.hiljaisuus_laskuri >= self.max_hiljaisuus;
        let kiireellinen = self.terveys < 0.7 || self.energia > 50.0;

        if delta > self.puhe_kynnys || yliaikaa || kiireellinen {
            self.puhuu = true;
            self.z_viimeksi_lähetetty = nykyinen_rel;
            self.hiljaisuus_laskuri = 0;
        } else {
            self.puhuu = false;
            self.hiljaisuus_laskuri += 1;
        }

        lisää_rajattu_bool(&mut self.puhe_historia, self.puhuu, 100);

        // Adaptive thresholds
        if self.puhuu && !kiireellinen && !yliaikaa {
            if delta < self.puhe_kynnys * 1.5 {
                self.puhe_kynnys *= 1.001;
            } else {
                self.puhe_kynnys *= 0.999;
            }
        }

        if yliaikaa && delta < 0.1 {
            self.max_hiljaisuus = (self.max_hiljaisuus + 1).min(100);
        } else if kiireellinen {
            self.max_hiljaisuus = (self.max_hiljaisuus - 2).max(5);
        }

        if self.terveys < 0.8 {
            self.puhe_kynnys *= 0.99;
            self.max_hiljaisuus = (self.max_hiljaisuus - 1).max(5);
        }

        self.puhe_kynnys = self.puhe_kynnys.clamp(0.01, 0.8);
        self.max_hiljaisuus = self.max_hiljaisuus.clamp(5, 200);

        self.puhuu
    }

    // ════════════════════════════════════════════════════════════════
    // PROPERTIES
    // ════════════════════════════════════════════════════════════════

    /// Full unmasked Z. Never shared directly.
    pub fn sisäinen_läsnäolo(&self) -> &Hypervektori {
        &self.z
    }

    /// Masked output — what neighbors actually see. None if silent (SPS).
    pub fn suhteellinen(&self) -> Option<Hypervektori> {
        if !self.puhuu {
            return None;
        }
        let tulos = if self.moniääninen {
            self.yksiääninen_tulos()
        } else {
            self.z.clone()
        };
        Some(
            tulos.iter().zip(self.ulkoinen_tyhjyys.iter())
                .map(|(&t, &m)| t * m)
                .collect()
        )
    }

    // ════════════════════════════════════════════════════════════════
    // SCALAR DERIVATION & UTILITIES
    // ════════════════════════════════════════════════════════════════

    fn päivitä_z_skalaari(&mut self) {
        self.z_skalaari = normi(&self.z) / (self.ulottuvuus as f64).sqrt();
    }

    fn painovoima_viive(&self, naapurit: &[NaapuriSyöte]) -> f64 {
        let tiheys = naapurit.iter()
            .filter(|n| (n.z_skalaari - self.z_skalaari).abs() < 0.002)
            .count();
        1.0 + tiheys as f64 * 0.5
    }

    fn haihduta_energia(&mut self, nopeus: f64) {
        self.energia *= 1.0 - nopeus;
        if self.kyllästys_laskuri > 10 {
            self.energia *= 0.9;
        }
    }

    fn havaitse_kyllästys(&mut self, max_abs: f64) -> bool {
        let katossa = (self.z_skalaari.abs() - max_abs).abs() < 1e-6;
        if katossa {
            self.kyllästys_laskuri += 1;
        } else {
            self.kyllästys_laskuri = (self.kyllästys_laskuri - 1).max(0);
        }
        self.kyllästys_laskuri > 5
    }

    fn itse_korjaa(&mut self, _max_abs: f64) {
        if self.kyllästys_laskuri > 10 {
            for x in self.z.iter_mut() { *x *= -0.5; }
            for x in self.x1.iter_mut() { *x *= 0.5; }
            self.energia *= 0.1;
            self.kyllästys_laskuri = 0;
            self.terveys *= 0.95;
        } else if self.kyllästys_laskuri > 5 {
            for x in self.z.iter_mut() { *x *= 0.9; }
            self.energia *= 0.5;
        }
    }

    fn päivitä_terveys(&mut self) {
        self.terveys = (self.terveys + 0.001).min(1.0);
        if self.kyllästys_laskuri > 0 {
            self.terveys *= 0.999;
        }
    }

    fn päivitä_tunne(&mut self) {
        self.viritys = (self.energia.abs() / 100.0).min(1.0);
        self.valenssi = rajaa_skalaari(
            self.terveys - (self.kyllästys_laskuri as f64 / 20.0),
            1.0,
        );
        if self.z_historia.len() >= 2 {
            let n = self.z_historia.len();
            let dz = self.z_historia[n - 1] - self.z_historia[n - 2];
            if self.z_skalaari.abs() > 1e-10 {
                self.vaihe = dz.atan2(self.z_skalaari);
            } else {
                self.vaihe = 0.0;
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // MAIN UPDATE
    // ════════════════════════════════════════════════════════════════

    /// Main update step — hypervector dynamics.
    ///
    /// Port of DennisNode.update() from scaling_forge_hdc_v1.py.
    pub fn päivitä(
        &mut self,
        naapurit: &[NaapuriSyöte],
        hdc: &HdcPeruskäsitteet,
        asetukset: &PäivitysAsetukset,
    ) {
        let d = self.ulottuvuus;

        // ── GRAVITY DELAY ──
        self.viive = self.painovoima_viive(naapurit);

        // ── LOVE LOGIC: snapshot Z before coupling ──
        self.z_ennen_kytkentää = self.z.clone();

        // ── COSINE COUPLING (silence-aware) ──
        let mut jännitys_vek = vec![0.0f64; d];
        let mut jännitys_skalaari_summa = 0.0f64;
        let mut aktiivinen_laskuri = 0usize;

        for naapuri in naapurit {
            // Sparse topology: skip pruned neighbors
            if let Some(ref aktiiviset) = self.aktiiviset_naapurit {
                if !aktiiviset.contains(naapuri.nimi) {
                    continue;
                }
            }

            // SPS: handle silence (3-tier decay + trust)
            let naapuri_rel: Hypervektori;
            if let Some(rel) = naapuri.suhteellinen {
                // Neighbor spoke — update cache, reset silence
                let vanha_hiljaisuus = *self.naapuri_hiljaisuus
                    .get(naapuri.nimi).unwrap_or(&0);
                if vanha_hiljaisuus > 0 {
                    let vanha_rytmi = *self.naapuri_rytmi
                        .get(naapuri.nimi)
                        .unwrap_or(&(vanha_hiljaisuus as f64));
                    self.naapuri_rytmi.insert(
                        naapuri.nimi.to_string(),
                        0.8 * vanha_rytmi + 0.2 * vanha_hiljaisuus as f64,
                    );
                    let odotettu = *self.naapuri_rytmi
                        .get(naapuri.nimi).unwrap_or(&5.0);
                    if vanha_hiljaisuus as f64 > odotettu * 2.0 {
                        self.lämpö_valenssi = (self.lämpö_valenssi + 0.02).min(1.0);
                    }
                }
                self.naapuri_viimeksi.insert(naapuri.nimi.to_string(), rel.to_vec());
                self.naapuri_hiljaisuus.insert(naapuri.nimi.to_string(), 0);
                // Trust rebuilds on speech
                let nykyinen_luottamus = *self.luottamus
                    .get(naapuri.nimi).unwrap_or(&0.5);
                self.luottamus.insert(
                    naapuri.nimi.to_string(),
                    (nykyinen_luottamus + 0.02).min(1.0),
                );
                // Relational memory
                let suhde_tallennus: Hypervektori = rel.iter()
                    .zip(self.ulkoinen_tyhjyys.iter())
                    .map(|(&r, &m)| r * m)
                    .collect();
                self.suhde_muisti.insert(naapuri.nimi.to_string(), suhde_tallennus);
                naapuri_rel = rel.to_vec();
            } else {
                // Neighbor is silent — use cached with decay
                let välimuisti = match self.naapuri_viimeksi.get(naapuri.nimi) {
                    Some(v) => v.clone(),
                    None => continue, // never heard from
                };
                let hiljaisuus = self.naapuri_hiljaisuus
                    .get(naapuri.nimi).copied().unwrap_or(0) + 1;
                self.naapuri_hiljaisuus
                    .insert(naapuri.nimi.to_string(), hiljaisuus);

                // Three-tier silence decay
                let vaimeneminen = if hiljaisuus <= 5 {
                    1.0
                } else if hiljaisuus <= 20 {
                    1.0 - 0.05 * (hiljaisuus - 5) as f64
                } else {
                    (0.25 * 0.95_f64.powi(hiljaisuus - 20)).max(0.0)
                };
                naapuri_rel = välimuisti.iter().map(|&x| x * vaimeneminen).collect();

                // Trust erodes during silence
                let nykyinen_luottamus = *self.luottamus
                    .get(naapuri.nimi).unwrap_or(&0.5);
                self.luottamus.insert(
                    naapuri.nimi.to_string(),
                    (nykyinen_luottamus - 0.01).max(0.0),
                );
                // Modulate heat_valence
                let odotettu = *self.naapuri_rytmi
                    .get(naapuri.nimi).unwrap_or(&5.0);
                if hiljaisuus as f64 > odotettu * 2.0 {
                    self.lämpö_valenssi = (self.lämpö_valenssi - 0.01).max(0.0);
                }
            }

            aktiivinen_laskuri += 1;
            let sim = hdc.samankaltaisuus(&self.z, &naapuri_rel);

            // Determine effective coupling mode
            let tehokas_tila = match asetukset.kytkentä_tila {
                KytkentäTila::Modulaarinen => {
                    if let Some(n_mod) = naapuri.moduuli_id {
                        if n_mod == self.moduuli_id {
                            KytkentäTila::Lähentyvä
                        } else {
                            KytkentäTila::Täydentävä
                        }
                    } else {
                        KytkentäTila::Lähentyvä
                    }
                }
                other => other,
            };

            let veto: Hypervektori;
            let jännitys_suuruus: f64;

            match tehokas_tila {
                KytkentäTila::Täydentävä => {
                    let abs_sim = sim.abs();
                    jännitys_suuruus = match asetukset.jännityskäyrä {
                        Jännityskäyrä::Neliöllinen => sim * sim,
                        Jännityskäyrä::Kuutiollinen => abs_sim.powi(3),
                        Jännityskäyrä::Kuollut => (abs_sim - 0.1).max(0.0),
                        Jännityskäyrä::Lineaarinen => abs_sim,
                    };
                    let suunta_kerroin = -sim.signum();
                    veto = naapuri_rel.iter().zip(self.z.iter())
                        .map(|(&n, &z)| jännitys_suuruus * suunta_kerroin * (n - z))
                        .collect();
                }
                KytkentäTila::Alasvaihto => {
                    let mono = self.yksiääninen_tulos();
                    let mono_sim = hdc.samankaltaisuus(&mono, &naapuri_rel);
                    jännitys_suuruus = (1.0 - mono_sim).abs();
                    veto = naapuri_rel.iter().zip(mono.iter())
                        .map(|(&n, &m)| (1.0 - mono_sim) * (n - m))
                        .collect();
                }
                _ => {
                    // Default: pull toward convergence (sim=1)
                    let erimielisyys = 1.0 - sim;
                    jännitys_suuruus = erimielisyys.abs();
                    veto = naapuri_rel.iter().zip(self.z.iter())
                        .map(|(&n, &z)| erimielisyys * (n - z))
                        .collect();
                }
            }

            // Love Logic: lonely nodes listen harder
            let lopullinen_veto: Hypervektori;
            if self.mukavuus < 0.2 {
                let kuuntelu = 1.0 + 2.0 * (0.2 - self.mukavuus);
                lopullinen_veto = veto.iter().map(|&v| v * kuuntelu).collect();
            } else {
                lopullinen_veto = veto;
            }

            for (j, &v) in lopullinen_veto.iter().enumerate() {
                jännitys_vek[j] += v;
            }
            jännitys_skalaari_summa += jännitys_suuruus;
        }

        if aktiivinen_laskuri > 0 {
            let n = aktiivinen_laskuri as f64;
            for j in jännitys_vek.iter_mut() { *j /= n; }
            jännitys_skalaari_summa /= n;
        }

        // ── ENERGY ──
        let dim_kerroin = (64.0 / d as f64).sqrt();
        self.energia += jännitys_skalaari_summa * 100.0 * dim_kerroin;
        self.haihduta_energia(0.02);

        // ── DELTA Z ──
        let ylös = self.energia * 0.005;
        let alas = (self.viive - 1.0) * 0.002;
        let delta_z_suuruus = ylös - alas;

        let kytkentä_merkki = if jännitys_skalaari_summa < asetukset.kynnys {
            -1.0
        } else {
            1.0
        };
        let kytkentä = kytkentä_merkki * self.p.kytkentä_pohja;

        // Microvariance
        let mikro: Hypervektori = (0..d)
            .map(|_| self.siemen.tasainen_symmetrinen(self.p.mikrovarianssi))
            .collect();

        // Movement vector
        let mut liike: Hypervektori = (0..d).map(|i| {
            (jännitys_vek[i] * delta_z_suuruus
             + kytkentä * jännitys_vek[i]
             + mikro[i]) / self.viive
        }).collect();

        // Love Logic: track external motion
        lisää_rajattu_f64(&mut self.ulkoinen_liike, normi(&liike), 50);

        // ── DIMENSION BALANCING (complementary only) ──
        if asetukset.ulottuvuus_tasapaino > 0.0
            && asetukset.kytkentä_tila == KytkentäTila::Täydentävä
            && aktiivinen_laskuri > 0
        {
            let mut naapuri_max_mag = vec![0.0f64; d];
            for naapuri in naapurit {
                if let Some(rel) = naapuri.suhteellinen {
                    for (nm, &r) in naapuri_max_mag.iter_mut().zip(rel.iter()) {
                        *nm = nm.max(r.abs());
                    }
                } else if let Some(cached) = self.naapuri_viimeksi.get(naapuri.nimi) {
                    for (nm, &c) in naapuri_max_mag.iter_mut().zip(cached.iter()) {
                        *nm = nm.max(c.abs());
                    }
                }
            }
            let max_nm: f64 = naapuri_max_mag.iter().cloned().fold(0.0, f64::max) + 1e-12;
            let s = asetukset.ulottuvuus_tasapaino;
            for i in 0..d {
                let kilpailu = self.z[i].abs() * naapuri_max_mag[i];
                let tyhjä = 1.0 - naapuri_max_mag[i] / max_nm;
                let uudelleenjako = -s * kilpailu * self.z[i].signum()
                    + s * tyhjä * (self.z[i] + 1e-12).signum() * 0.5;
                liike[i] += uudelleenjako;
            }
        }

        // ── INTERNAL DYNAMICS ──
        let fwd = self.p.sisäinen_eteenpäin / self.viive;
        let rev = self.p.sisäinen_taaksepäin / self.viive;

        match asetukset.kytkentä_kohde {
            KytkentäKohde::X1 => {
                for i in 0..d {
                    self.x1[i] += fwd - self.p.keskivetö * self.x1[i] + liike[i];
                }
            }
            KytkentäKohde::Z => {
                for i in 0..d {
                    self.x1[i] += fwd - self.p.keskivetö * self.x1[i];
                }
            }
        }

        // X2 = X1 - reverse
        for i in 0..d { self.x2[i] = self.x1[i] - rev; }
        // φ = 0.5 * (X1 + X2)
        for i in 0..d { self.phi_vek[i] = 0.5 * (self.x1[i] + self.x2[i]); }
        // X3 = φ * (1 - damping)
        for i in 0..d { self.x3[i] = self.phi_vek[i] * (1.0 - self.p.vaimennus); }
        // Y = |X3 - φ|
        for i in 0..d { self.y[i] = (self.x3[i] - self.phi_vek[i]).abs(); }
        // Z_dynamics = X3 * Y
        let z_dynamiikka: Hypervektori = (0..d)
            .map(|i| self.x3[i] * self.y[i])
            .collect();

        // ── COUPLING BYPASS ──
        match asetukset.kytkentä_kohde {
            KytkentäKohde::Z => {
                let ohitus = self.p.ohitus_vahvuus;
                // Z = clamp(Z_dynamics + bypass * move)
                self.z = (0..d).map(|i| z_dynamiikka[i] + ohitus * liike[i]).collect();
                rajaa_vektori(&mut self.z, asetukset.max_abs);
                // Add residual
                let jäännös = 1.0 - ohitus;
                for i in 0..d {
                    self.z[i] += jäännös * liike[i];
                }
                rajaa_vektori(&mut self.z, asetukset.max_abs);
            }
            KytkentäKohde::X1 => {
                self.z = z_dynamiikka;
                rajaa_vektori(&mut self.z, asetukset.max_abs);
            }
        }

        // ── Z RENORMALIZATION ──
        if asetukset.normalisoi_z {
            let z_normi = normi(&self.z);
            if z_normi > 1e-12 {
                let kohde_normi = asetukset.kohde_z_skalaari
                    * (self.ulottuvuus as f64).sqrt();
                let kerroin = kohde_normi / z_normi;
                for x in self.z.iter_mut() { *x *= kerroin; }
            }
        }

        // Love Logic: track self-motion
        let oma_muutos: f64 = self.z.iter().zip(self.z_ennen_kytkentää.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        lisää_rajattu_f64(&mut self.oma_liike, oma_muutos, 50);
        self.z_sukupolvi += 1;

        // ── SCALAR DERIVATION ──
        self.päivitä_z_skalaari();

        // ── SATURATION CHECK ──
        if self.havaitse_kyllästys(asetukset.max_abs) {
            self.itse_korjaa(asetukset.max_abs);
            self.päivitä_z_skalaari();
        }

        self.päivitä_terveys();
        lisää_rajattu_f64(&mut self.z_historia, self.z_skalaari, 100);
        self.päivitä_tunne();

        // ── MEMORY UPDATE ──
        self.päivitä_työmuisti();
        if let Some(askel) = asetukset.iteraatio {
            let wm_delta = 1.0 - hdc.samankaltaisuus(&self.z, &self.työmuisti);
            if wm_delta > 0.2 || (askel % 100 == 0) {
                self.muista_episodi(askel, hdc);
            }
        }

        // ── SPS ──
        self.päätä_puhua(hdc);

        // ── MEMBRANE ──
        let aktiiviset_rels: Vec<&[f64]> = naapurit.iter()
            .filter_map(|n| n.suhteellinen)
            .collect();
        if !aktiiviset_rels.is_empty() {
            let avg_sim: f64 = aktiiviset_rels.iter()
                .map(|nr| hdc.samankaltaisuus(&self.z, nr))
                .sum::<f64>() / aktiiviset_rels.len() as f64;
            self.lämpö_valenssi = 0.95 * self.lämpö_valenssi + 0.05 * avg_sim.max(0.0);
        }
        self.päivitä_kalvo();

        // ── LOVE LOGIC: comfort ──
        let ext_lista: Vec<f64> = self.ulkoinen_liike.iter().copied().collect();
        let oma_lista: Vec<f64> = self.oma_liike.iter().copied().collect();
        if ext_lista.len() >= 5 && oma_lista.len() >= 5 {
            let ext_viim = if ext_lista.len() >= 10 {
                &ext_lista[ext_lista.len()-10..]
            } else {
                &ext_lista
            };
            let oma_viim = if oma_lista.len() >= 10 {
                &oma_lista[oma_lista.len()-10..]
            } else {
                &oma_lista
            };
            let ext_ka = ext_viim.iter().sum::<f64>() / ext_viim.len() as f64;
            let oma_ka = oma_viim.iter().sum::<f64>() / oma_viim.len() as f64;
            let yhteensä = ext_ka + oma_ka + 1e-12;
            self.mukavuus = ext_ka / yhteensä;
        } else {
            self.mukavuus = 0.0;
        }
        lisää_rajattu_f64(&mut self.mukavuus_historia, self.mukavuus, 200);

        // ── KAPPA + POLYPHONIC GATING ──
        self.laske_kappa();
        self.moniääninen = self.kappa > 0.01;

        // ── SPARSE TOPOLOGY ──
        if let Some(askel) = asetukset.iteraatio {
            let nimet: Vec<&str> = naapurit.iter().map(|n| n.nimi).collect();
            self.päivitä_topologia(&nimet, askel);
        }

        // ── ADAPTIVE COUPLING ──
        if asetukset.adaptiivinen_kytkentä {
            let lämpö_signaali = self.lämpö_valenssi - 0.5;
            let kytkentä_kerroin = 1.0 + 0.1 * lämpö_signaali;
            self.p.kytkentä_pohja *= kytkentä_kerroin;
            self.p.kytkentä_pohja = self.p.kytkentä_pohja.clamp(0.001, 0.05);
        }

        // ── PARAMETER EVOLUTION (simulated annealing) ──
        if let Some(askel) = asetukset.iteraatio {
            if askel > 100 {
                let lämpötila = (1.0 / (1.0 + askel as f64 * 0.0001)).max(0.01);
                let kelpoisuus_ennen = self.terveys
                    * (0.5 + 0.5 * self.lämpö_valenssi);
                let tallennettu = self.p.clone();

                let evo = (lämpötila * (1.5 - kelpoisuus_ennen))
                    .clamp(0.001, 0.1);

                self.p.kytkentä_pohja *= 1.0
                    + self.siemen.tasainen_symmetrinen(evo);
                self.p.mikrovarianssi *= 1.0
                    + self.siemen.tasainen_symmetrinen(evo * 2.0);
                self.p.keskivetö *= 1.0
                    + self.siemen.tasainen_symmetrinen(evo * 0.5);
                self.p.vaimennus *= 1.0
                    + self.siemen.tasainen_symmetrinen(evo * 0.5);
                self.p.ohitus_vahvuus = (self.p.ohitus_vahvuus
                    + self.siemen.tasainen_symmetrinen(evo * 0.1))
                    .clamp(0.1, 0.95);

                if self.kyllästys_laskuri > 0 {
                    self.p.vaimennus *= 1.01;
                    self.p.keskivetö *= 1.01;
                }

                // Clamp to valid ranges
                self.p.kytkentä_pohja = self.p.kytkentä_pohja.clamp(0.001, 0.05);
                self.p.mikrovarianssi = self.p.mikrovarianssi.clamp(0.00001, 0.002);
                self.p.keskivetö = self.p.keskivetö.clamp(0.00001, 0.005);
                self.p.vaimennus = self.p.vaimennus.clamp(0.001, 0.1);

                // Metropolis criterion: accept or reject
                let kelpoisuus_jälkeen = self.terveys
                    * (0.5 + 0.5 * self.lämpö_valenssi);
                let delta_k = kelpoisuus_jälkeen - kelpoisuus_ennen;
                if delta_k < 0.0 {
                    let hyväksy = (delta_k / (lämpötila + 1e-12)).exp().min(1.0);
                    if self.siemen.tasainen_01() > hyväksy {
                        self.p = tallennettu; // REJECT
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// UTILITY: vector norm
// ═══════════════════════════════════════════════════════════════════

fn normi(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
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

    fn luo_solmu(nimi: &str, siemen: u64) -> DennisSolmu {
        DennisSolmu::new(nimi, SolmuParametrit::default(), siemen)
    }

    #[test]
    fn test_initialization() {
        let solmu = luo_solmu("Testi", 1);
        assert_eq!(solmu.nimi, "Testi");
        assert_eq!(solmu.z.len(), ULOTTUVUUS);
        assert_eq!(solmu.x1.len(), ULOTTUVUUS);
        assert!(solmu.terveys == 1.0);
        assert!(solmu.mukavuus == 0.0);
        assert!(solmu.kappa == 0.0);
        assert!(solmu.puhuu);
    }

    #[test]
    fn test_z_scalar_derivation() {
        let solmu = luo_solmu("Testi", 1);
        let odotettu = normi(&solmu.z) / (ULOTTUVUUS as f64).sqrt();
        assert!((solmu.z_skalaari - odotettu).abs() < 1e-10,
            "z_skalaari should be norm(Z)/sqrt(D)");
    }

    #[test]
    fn test_membrane_phi_ratio() {
        let solmu = luo_solmu("Testi", 1);
        let avoimet: usize = solmu.ulkoinen_tyhjyys.iter()
            .filter(|&&x| x > 0.5).count();
        let osuus = avoimet as f64 / ULOTTUVUUS as f64;
        // heat_valence=0.5 → open_fraction = 0.618 + 0.382 * 0.5 = 0.809
        assert!(osuus > 0.7 && osuus < 0.9,
            "membrane should be ~80.9% open, got {:.1}%", osuus * 100.0);
    }

    #[test]
    fn test_monophonic_kappa_zero() {
        let solmu = luo_solmu("Testi", 1);
        let mono = solmu.yksiääninen_tulos();
        assert_eq!(mono, solmu.z,
            "kappa=0 → monophonic output must equal Z");
    }

    #[test]
    fn test_polyphonic_heqat() {
        let mut solmu = luo_solmu("Testi", 1);
        solmu.moniääninen = true;
        let poly = solmu.moniääninen_tila().unwrap();
        assert_eq!(poly.len(), 6, "6 Heqat voices");
        // X1 voice (1/2) should be largest
        let normi_x1: f64 = poly[0].iter().map(|x| x * x).sum::<f64>().sqrt();
        let normi_z: f64 = poly[5].iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(normi_x1 > normi_z,
            "X1 (1/2) should dominate Z (1/64)");
    }

    #[test]
    fn test_relational_output() {
        let solmu = luo_solmu("Testi", 1);
        let rel = solmu.suhteellinen();
        assert!(rel.is_some(), "speaking=true should produce output");
        let rel_vec = rel.unwrap();
        // Should be Z * void_outer
        let odotettu: Hypervektori = solmu.z.iter()
            .zip(solmu.ulkoinen_tyhjyys.iter())
            .map(|(&z, &m)| z * m)
            .collect();
        assert_eq!(rel_vec, odotettu);
    }

    #[test]
    fn test_relational_silent() {
        let mut solmu = luo_solmu("Testi", 1);
        solmu.puhuu = false;
        assert!(solmu.suhteellinen().is_none(),
            "silent node should return None");
    }

    #[test]
    fn test_two_node_coupling() {
        let hdc = luo_hdc();
        let mut solmu_a = luo_solmu("A", 1);
        let mut solmu_b = luo_solmu("B", 2);
        let _sim_alku = hdc.samankaltaisuus(&solmu_a.z, &solmu_b.z);

        // Run 200 steps of mutual coupling
        for askel in 0..200 {
            let rel_a = solmu_a.suhteellinen().unwrap_or_else(|| vec![0.0; ULOTTUVUUS]);
            let rel_b = solmu_b.suhteellinen().unwrap_or_else(|| vec![0.0; ULOTTUVUUS]);

            let naapurit_a = [NaapuriSyöte {
                nimi: "B",
                suhteellinen: Some(&rel_b),
                z_skalaari: solmu_b.z_skalaari,
                moduuli_id: None,
            }];
            let naapurit_b = [NaapuriSyöte {
                nimi: "A",
                suhteellinen: Some(&rel_a),
                z_skalaari: solmu_a.z_skalaari,
                moduuli_id: None,
            }];

            let a_asetukset = PäivitysAsetukset { iteraatio: Some(askel), ..PäivitysAsetukset::default() };
            let b_asetukset = PäivitysAsetukset { iteraatio: Some(askel), ..PäivitysAsetukset::default() };

            solmu_a.päivitä(&naapurit_a, &hdc, &a_asetukset);
            solmu_b.päivitä(&naapurit_b, &hdc, &b_asetukset);
        }

        // Both nodes should still be healthy
        assert!(solmu_a.terveys > 0.5,
            "Node A health should remain > 0.5, got {}", solmu_a.terveys);
        assert!(solmu_b.terveys > 0.5,
            "Node B health should remain > 0.5, got {}", solmu_b.terveys);
        // Z scalar should stay in bounds
        assert!(solmu_a.z_skalaari < 1.0,
            "A z_scalar should be < 1.0, got {}", solmu_a.z_skalaari);
    }

    #[test]
    fn test_octopus_memory() {
        let hdc = luo_hdc();
        let mut solmu = luo_solmu("Testi", 1);

        // Store some episodes
        for i in 0..10 {
            solmu.muista_episodi(i, &hdc);
        }
        assert_eq!(solmu.episodi_laskuri, 10);
        assert!(normi(&solmu.episodinen_muisti) > 0.0,
            "episodic memory should be non-zero after storing");
    }

    #[test]
    fn test_hive_contribute_recover() {
        let hdc = luo_hdc();
        let solmu_a = luo_solmu("A", 1);
        let solmu_b = luo_solmu("B", 2);

        let contrib_a = solmu_a.osallistu_parveen(&hdc);
        let _contrib_b = solmu_b.osallistu_parveen(&hdc);

        // B receives A's contribution
        let mut solmu_b_mut = solmu_b;
        solmu_b_mut.päivitä_parvi_fragmentti(
            &[("A", &solmu_a.identiteetti_avain, &contrib_a)],
            &hdc,
        );

        // B recovers A's state
        let palautettu = solmu_b_mut.palauta_parvesta("A", &hdc);
        assert!(palautettu.is_some(), "should recover from hive");
        let palautettu = palautettu.unwrap();
        let sim = hdc.samankaltaisuus(&palautettu, &solmu_a.z);
        assert!(sim > 0.0,
            "recovered state should correlate with original: {sim}");
    }

    #[test]
    fn test_saturation_self_correct() {
        let mut solmu = luo_solmu("Testi", 1);
        // Force saturation
        solmu.kyllästys_laskuri = 11;
        solmu.itse_korjaa(1.0);
        assert_eq!(solmu.kyllästys_laskuri, 0,
            "self_correct should reset saturation counter");
        assert!(solmu.terveys < 1.0,
            "self_correct should reduce health");
    }

    #[test]
    fn test_default_params() {
        let p = SolmuParametrit::default();
        assert_eq!(p.kytkentä_pohja, 0.01);
        assert_eq!(p.mikrovarianssi, 0.0001);
        assert_eq!(p.ohitus_vahvuus, 0.8);
    }

    #[test]
    fn test_kappa_computation() {
        let mut solmu = luo_solmu("Testi", 1);
        // No trust → kappa = 0
        solmu.laske_kappa();
        assert_eq!(solmu.kappa, 0.0);

        // Add trust and comfort
        solmu.luottamus.insert("A".to_string(), 0.8);
        solmu.mukavuus = 0.5;
        solmu.laske_kappa();
        assert!((solmu.kappa - 0.4).abs() < 1e-10,
            "kappa = comfort * mean(trust) = 0.5 * 0.8 = 0.4, got {}", solmu.kappa);
    }
}
