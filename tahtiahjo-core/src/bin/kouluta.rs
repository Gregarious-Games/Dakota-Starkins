//! Kouluta — Train and evaluate HDC language model on text files.
//!
//! Usage:
//!   cargo run --bin kouluta -- <text_file> [options]
//!
//! Options:
//!   --chars=N         Max characters to train on (default: 10000)
//!   --retrain=N       Number of retraining passes (default: 3)
//!   --window=N        Context window size (default: 3)
//!   --codebook=TYPE   random | qams (default: qams)
//!   --kaksoset        Use twin accumulator with heart/membrane physics
//!   --kolmoset        Use triple relay cascade (A→B→C→A)
//!   --keskus          Wrap Kolmoset with Keskus (transition priors + recurrence)
//!   --alpha-siirtyma=F  Override transition prior weight (default: 0.015)
//!   --alpha-taajuus=F   Override frequency prior weight (default: 0.005)
//!   --alpha-kierto=F    Override recurrent blend weight (default: 0.05)
//!   --alpha-sana=F      Override word-boundary weight (default: 0.008)
//!   --compare         Run all architectures side-by-side
//!
//! Examples:
//!   cargo run --release --bin kouluta -- ../kalevala.txt --chars=10000 --codebook=qams
//!   cargo run --release --bin kouluta -- ../kalevala.txt --kaksoset
//!   cargo run --release --bin kouluta -- ../kalevala.txt --kolmoset
//!   cargo run --release --bin kouluta -- ../kalevala.txt --kolmoset --keskus
//!   cargo run --release --bin kouluta -- ../kalevala.txt --compare

use std::collections::HashMap;
use std::env;
use std::fs;

use tahtiahjo_core::hdc_primitives::{HdcPeruskäsitteet, Hypervektori, ULOTTUVUUS};
use tahtiahjo_core::konteksti_sitoja::KontekstiSitoja;
use tahtiahjo_core::luokka_kertyma::{LuokkaKertymä, luo_satunnainen_koodikirja};
use tahtiahjo_core::qams_codebook::{kaikki_allekirjoitukset, luo_koodikirja};
use tahtiahjo_core::kaksoset::Kaksoset;
use tahtiahjo_core::kolmoset::Kolmoset;
use tahtiahjo_core::keskus::Keskus;
use tahtiahjo_core::kaksoisnapainen::KaksoisnapainenKartta;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: kouluta <text_file> [--chars=N] [--retrain=N] [--window=N] [--codebook=random|qams] [--kaksoset] [--compare]");
        std::process::exit(1);
    }

    let tiedosto = &args[1];
    let mut max_merkit: usize = 10_000;
    let mut uudelleen: usize = 3;
    let mut ikkuna: usize = 3;
    let mut koodikirja_tyyppi = "qams".to_string();
    let mut käytä_kaksoset = false;
    let mut käytä_kolmoset = false;
    let mut käytä_keskus = false;
    let mut vertaa = false;

    // Alpha overrides for Keskus (None = use defaults)
    let mut alpha_siirtyma: Option<f64> = None;
    let mut alpha_taajuus: Option<f64> = None;
    let mut alpha_kierto: Option<f64> = None;
    let mut alpha_sana: Option<f64> = None;

    // Bipyramid options
    let mut käytä_bipyramid = false;
    let mut bipyramid_threshold: f64 = 0.01;

    // Dimension override (default uses ULOTTUVUUS=256)
    let mut ulottuvuus: usize = ULOTTUVUUS;

    for arg in &args[2..] {
        if let Some(val) = arg.strip_prefix("--chars=") {
            max_merkit = val.parse().unwrap_or(10_000);
        } else if let Some(val) = arg.strip_prefix("--retrain=") {
            uudelleen = val.parse().unwrap_or(3);
        } else if let Some(val) = arg.strip_prefix("--window=") {
            ikkuna = val.parse().unwrap_or(3);
        } else if let Some(val) = arg.strip_prefix("--codebook=") {
            koodikirja_tyyppi = val.to_string();
        } else if arg == "--kaksoset" {
            käytä_kaksoset = true;
        } else if arg == "--kolmoset" {
            käytä_kolmoset = true;
        } else if arg == "--keskus" {
            käytä_keskus = true;
        } else if arg == "--compare" {
            vertaa = true;
        } else if let Some(val) = arg.strip_prefix("--alpha-siirtyma=") {
            alpha_siirtyma = val.parse().ok();
        } else if let Some(val) = arg.strip_prefix("--alpha-taajuus=") {
            alpha_taajuus = val.parse().ok();
        } else if let Some(val) = arg.strip_prefix("--alpha-kierto=") {
            alpha_kierto = val.parse().ok();
        } else if let Some(val) = arg.strip_prefix("--alpha-sana=") {
            alpha_sana = val.parse().ok();
        } else if arg == "--bipyramid" {
            käytä_bipyramid = true;
        } else if let Some(val) = arg.strip_prefix("--bipyramid-threshold=") {
            bipyramid_threshold = val.parse().unwrap_or(0.01);
            käytä_bipyramid = true;
        } else if let Some(val) = arg.strip_prefix("--dim=") {
            ulottuvuus = val.parse().unwrap_or(ULOTTUVUUS);
        }
    }

    // Read text
    let raaka = fs::read_to_string(tiedosto)
        .unwrap_or_else(|e| {
            eprintln!("Error reading {}: {}", tiedosto, e);
            std::process::exit(1);
        });
    let teksti: String = raaka.chars().take(max_merkit).collect();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Tähtiahjo — HDC Language Model Training");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  File:       {}", tiedosto);
    println!("  Characters: {} (of {} available)", teksti.len(), raaka.len());
    println!("  Window:     {}", ikkuna);
    println!("  Codebook:   {}", koodikirja_tyyppi);
    println!("  Retrain:    {} passes", uudelleen);
    println!("  Dimension:  D={}", ulottuvuus);
    if vertaa {
        println!("  Mode:       COMPARE (all architectures)");
    } else if käytä_kolmoset && käytä_keskus && käytä_bipyramid {
        println!("  Mode:       KOLMOSET + KESKUS + BIPYRAMID (2-stage pole prediction)");
    } else if käytä_kolmoset && käytä_keskus {
        println!("  Mode:       KOLMOSET + KESKUS (relay + priors + recurrence)");
    } else if käytä_kolmoset {
        println!("  Mode:       KOLMOSET (triple relay cascade)");
    } else if käytä_kaksoset {
        println!("  Mode:       KAKSOSET (twin accumulator)");
    } else {
        println!("  Mode:       STANDARD (LuokkaKertymä)");
    }
    println!("───────────────────────────────────────────────────────────────");

    // Build codebook
    let koodikirja: HashMap<char, Hypervektori> = if koodikirja_tyyppi == "qams" {
        println!("  Building QAMS phonetic codebook...");
        let allekirjoitukset = kaikki_allekirjoitukset();
        let mut kirja = luo_koodikirja(&allekirjoitukset, ulottuvuus, 42);
        // Add any characters from text that aren't in QAMS signatures
        let mut rng = tahtiahjo_core::hdc_primitives::Siemen::new(777);
        for c in teksti.chars() {
            let c_lower = c.to_lowercase().next().unwrap_or(c);
            kirja.entry(c_lower)
                .or_insert_with(|| rng.bipolaarinen_vektori(ulottuvuus));
        }
        kirja
    } else {
        println!("  Building random codebook...");
        luo_satunnainen_koodikirja(&teksti, ulottuvuus, 99)
    };
    println!("  Codebook size: {} characters", koodikirja.len());

    let mut hdc = HdcPeruskäsitteet::new(ulottuvuus, 42);
    let sitoja = KontekstiSitoja::new(ikkuna);

    if vertaa {
        // ═══════════════════════════════════════════════════════════════
        // COMPARE MODE: run all architectures side-by-side
        // ═══════════════════════════════════════════════════════════════
        let tarkkuus_lk = kouluta_luokkakertymä(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen, ulottuvuus,
        );
        let tarkkuus_kk = kouluta_kaksoset(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen, ulottuvuus,
        );
        let tarkkuus_km = kouluta_kolmoset(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen, ulottuvuus,
        );

        println!("\n═══════════════════════════════════════════════════════════════");
        println!("  COMPARISON RESULTS");
        println!("───────────────────────────────────────────────────────────────");
        println!("  LuokkaKertymä:  {:.2}%", tarkkuus_lk * 100.0);
        println!("  Kaksoset:       {:.2}%", tarkkuus_kk * 100.0);
        println!("  Kolmoset:       {:.2}%", tarkkuus_km * 100.0);
        let paras = tarkkuus_lk.max(tarkkuus_kk).max(tarkkuus_km);
        if paras == tarkkuus_km {
            println!("  >>> Kolmoset wins: +{:.2}% over LuokkaKertymä",
                (tarkkuus_km - tarkkuus_lk) * 100.0);
        } else if paras == tarkkuus_kk {
            println!("  >>> Kaksoset wins: +{:.2}% over LuokkaKertymä",
                (tarkkuus_kk - tarkkuus_lk) * 100.0);
        } else {
            println!("  >>> LuokkaKertymä wins");
        }
        println!("═══════════════════════════════════════════════════════════════");
    } else if käytä_kolmoset && käytä_keskus && käytä_bipyramid {
        // ═══════════════════════════════════════════════════════════════
        // KOLMOSET + KESKUS + BIPYRAMID MODE
        // ═══════════════════════════════════════════════════════════════
        let alpha_overrides = (alpha_siirtyma, alpha_taajuus, alpha_kierto, alpha_sana);
        let tarkkuus = kouluta_kolmoset_keskus_kaksoisnapainen(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen,
            alpha_overrides, bipyramid_threshold, ulottuvuus,
        );
        tulosta_lopputulos(tarkkuus);
    } else if käytä_kolmoset && käytä_keskus {
        // ═══════════════════════════════════════════════════════════════
        // KOLMOSET + KESKUS MODE
        // ═══════════════════════════════════════════════════════════════
        let alpha_overrides = (alpha_siirtyma, alpha_taajuus, alpha_kierto, alpha_sana);
        let tarkkuus = kouluta_kolmoset_keskus(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen, alpha_overrides, ulottuvuus,
        );
        tulosta_lopputulos(tarkkuus);
    } else if käytä_kolmoset {
        // ═══════════════════════════════════════════════════════════════
        // KOLMOSET MODE
        // ═══════════════════════════════════════════════════════════════
        let tarkkuus = kouluta_kolmoset(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen, ulottuvuus,
        );
        tulosta_lopputulos(tarkkuus);
    } else if käytä_kaksoset {
        // ═══════════════════════════════════════════════════════════════
        // KAKSOSET MODE
        // ═══════════════════════════════════════════════════════════════
        let tarkkuus = kouluta_kaksoset(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen, ulottuvuus,
        );
        tulosta_lopputulos(tarkkuus);
    } else {
        // ═══════════════════════════════════════════════════════════════
        // STANDARD MODE (LuokkaKertymä)
        // ═══════════════════════════════════════════════════════════════
        let tarkkuus = kouluta_luokkakertymä(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen, ulottuvuus,
        );
        tulosta_lopputulos(tarkkuus);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// STANDARD PIPELINE — LuokkaKertymä
// ═══════════════════════════════════════════════════════════════════════

fn kouluta_luokkakertymä(
    teksti: &str,
    koodikirja: &HashMap<char, Hypervektori>,
    sitoja: &KontekstiSitoja,
    hdc: &mut HdcPeruskäsitteet,
    uudelleen: usize,
    ulottuvuus: usize,
) -> f64 {
    let mut kertymä = LuokkaKertymä::new(ulottuvuus);

    println!("\n  [LuokkaKertymä] Training (single pass)...");
    let näytteet = kertymä.kouluta(teksti, koodikirja, sitoja, hdc);
    let tarkkuus_alku = kertymä.tarkkuus(teksti, koodikirja, sitoja, hdc);
    println!("    Samples:  {}", näytteet);
    println!("    Classes:  {}", kertymä.luokka_lkm());
    println!("    Accuracy: {:.2}%", tarkkuus_alku * 100.0);

    for kierros in 1..=uudelleen {
        println!("\n  [LuokkaKertymä] Retraining pass {}...", kierros);
        let korjaukset = kertymä.uudelleenkouluta(teksti, koodikirja, sitoja, hdc);
        let tarkkuus = kertymä.tarkkuus(teksti, koodikirja, sitoja, hdc);
        println!("    Corrections: {}", korjaukset);
        println!("    Accuracy:    {:.2}%", tarkkuus * 100.0);
    }

    kertymä.tarkkuus(teksti, koodikirja, sitoja, hdc)
}

// ═══════════════════════════════════════════════════════════════════════
// TWIN PIPELINE — Kaksoset
// ═══════════════════════════════════════════════════════════════════════

/// Build (context_vector, target_char) pairs from text.
fn rakenna_näytteet(
    teksti: &str,
    koodikirja: &HashMap<char, Hypervektori>,
    sitoja: &KontekstiSitoja,
    hdc: &mut HdcPeruskäsitteet,
) -> Vec<(Vec<f64>, char)> {
    let merkit: Vec<char> = teksti.chars()
        .map(|c| c.to_lowercase().next().unwrap_or(c))
        .collect();
    let mut näytteet = Vec::with_capacity(merkit.len());

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
        näytteet.push((konteksti, merkit[i]));
    }

    näytteet
}

/// Evaluate Kaksoset accuracy on pre-built samples.
fn arvioi_kaksoset(
    kaksoset: &Kaksoset,
    näytteet: &[(Vec<f64>, char)],
    hdc: &HdcPeruskäsitteet,
) -> f64 {
    if näytteet.is_empty() {
        return 0.0;
    }
    let oikein = näytteet.iter()
        .filter(|(konteksti, kohde)| {
            let (ennuste, _) = kaksoset.ennusta(konteksti, hdc);
            ennuste == *kohde
        })
        .count();
    oikein as f64 / näytteet.len() as f64
}

fn kouluta_kaksoset(
    teksti: &str,
    koodikirja: &HashMap<char, Hypervektori>,
    sitoja: &KontekstiSitoja,
    hdc: &mut HdcPeruskäsitteet,
    uudelleen: usize,
    ulottuvuus: usize,
) -> f64 {
    // Build all (context, target) samples
    println!("\n  [Kaksoset] Building context vectors...");
    let näytteet = rakenna_näytteet(teksti, koodikirja, sitoja, hdc);
    println!("    Samples: {}", näytteet.len());

    // Extract alphabet from samples
    let mut aakkosto: Vec<char> = näytteet.iter().map(|(_, c)| *c).collect();
    aakkosto.sort();
    aakkosto.dedup();
    println!("    Alphabet: {} characters", aakkosto.len());

    // Create twin swarm
    let mut kaksoset = Kaksoset::new(&aakkosto, ulottuvuus);

    // ── Single-pass training ──────────────────────────────────────
    println!("\n  [Kaksoset] Training (single pass, heart active)...");
    for (konteksti, kohde) in &näytteet {
        kaksoset.kouluta_askel(konteksti, *kohde, hdc);
    }

    let tila = kaksoset.tila();
    let tarkkuus_alku = arvioi_kaksoset(&kaksoset, &näytteet, hdc);
    println!("    Online accuracy:    {:.2}% (during training)", tila.tarkkuus * 100.0);
    println!("    Evaluated accuracy: {:.2}% (post-training)", tarkkuus_alku * 100.0);
    println!("    Myöntö samples:     {}", tila.myöntö_näytteet);
    println!("    Kielto samples:     {}", tila.kielto_näytteet);
    println!("    Heart period:       {} steps", tila.syke_jakso);
    println!("    HRV φ-coherence:    {:.4}", tila.hrv_koherenssi);
    println!("    β (suppression):    {:.4}", tila.beta);
    println!("    Membrane perm:      {:.4}", tila.kalvo_läpäisevyys);
    println!("    ─── Hormones ───");
    println!("    Oxytocin:           {:.4}", tila.hormonit.oksitosiini);
    println!("    Cortisol:           {:.4}", tila.hormonit.kortisoli);
    println!("    Dopamine:           {:.4}", tila.hormonit.dopamiini);
    println!("    Serotonin:          {:.4}", tila.hormonit.serotoniini);
    println!("    Optic gland:        {:.4}", tila.hormonit.näkörauhanen);

    // ── Dream phase: octopus active sleep ──────────────────────────
    println!("\n  [Kaksoset] Dream phase (quiet sleep + active replay)...");
    let mut uni_siemen = tahtiahjo_core::hdc_primitives::Siemen::new(314);
    kaksoset.uni_vaihe(&näytteet, &mut uni_siemen);
    let tarkkuus_uni = arvioi_kaksoset(&kaksoset, &näytteet, hdc);
    println!("    Post-dream accuracy: {:.2}%", tarkkuus_uni * 100.0);

    // ── Retraining passes (no dream between — preserves corrections) ─
    for kierros in 1..=uudelleen {
        println!("\n  [Kaksoset] Retraining pass {}...", kierros);
        let tarkkuus = kaksoset.uudelleenkouluta(&näytteet, hdc, kierros);
        let tarkkuus_eval = arvioi_kaksoset(&kaksoset, &näytteet, hdc);
        println!("    Retrain accuracy:   {:.2}%", tarkkuus * 100.0);
        println!("    Evaluated accuracy: {:.2}%", tarkkuus_eval * 100.0);
    }

    // ── Final dream: consolidate after all retraining ─────────────
    println!("\n  [Kaksoset] Final dream phase (consolidation)...");
    kaksoset.uni_vaihe(&näytteet, &mut uni_siemen);
    let tarkkuus_final_dream = arvioi_kaksoset(&kaksoset, &näytteet, hdc);
    println!("    Post-dream accuracy: {:.2}%", tarkkuus_final_dream * 100.0);

    // ── Final diagnostics ─────────────────────────────────────────
    let loppu_tila = kaksoset.tila();
    let loppu_tarkkuus = arvioi_kaksoset(&kaksoset, &näytteet, hdc);
    println!("\n  [Kaksoset] Final state:");
    println!("    β: {:.4}  HRV: {:.4}  Heart: {} steps",
        loppu_tila.beta, loppu_tila.hrv_koherenssi, loppu_tila.syke_jakso);

    loppu_tarkkuus
}

// ═══════════════════════════════════════════════════════════════════════
// TRIPLE RELAY PIPELINE — Kolmoset
// ═══════════════════════════════════════════════════════════════════════

/// Evaluate Kolmoset ensemble accuracy on pre-built samples.
fn arvioi_kolmoset(
    kolmoset: &Kolmoset,
    näytteet: &[(Vec<f64>, char)],
    hdc: &HdcPeruskäsitteet,
) -> f64 {
    if näytteet.is_empty() {
        return 0.0;
    }
    let oikein = näytteet.iter()
        .filter(|(konteksti, kohde)| {
            let (ennuste, _) = kolmoset.ennusta(konteksti, hdc);
            ennuste == *kohde
        })
        .count();
    oikein as f64 / näytteet.len() as f64
}

fn kouluta_kolmoset(
    teksti: &str,
    koodikirja: &HashMap<char, Hypervektori>,
    sitoja: &KontekstiSitoja,
    hdc: &mut HdcPeruskäsitteet,
    uudelleen: usize,
    ulottuvuus: usize,
) -> f64 {
    // Build all (context, target) samples
    println!("\n  [Kolmoset] Building context vectors...");
    let näytteet = rakenna_näytteet(teksti, koodikirja, sitoja, hdc);
    println!("    Samples: {}", näytteet.len());

    // Extract alphabet
    let mut aakkosto: Vec<char> = näytteet.iter().map(|(_, c)| *c).collect();
    aakkosto.sort();
    aakkosto.dedup();
    println!("    Alphabet: {} characters", aakkosto.len());

    // Create triple relay
    // Scale relay legs proportionally to data size
    let n = näytteet.len();
    let askeleet_a = n / 5;         // ~20% broad learning
    let askeleet_b = n / 10;        // ~10% compressed distillation
    let askeleet_c = n / 5;         // ~20% deep integration
    println!("    Relay legs: A={}, B={}, C={}", askeleet_a, askeleet_b, askeleet_c);
    let mut kolmoset = Kolmoset::new_custom(
        &aakkosto, ulottuvuus, askeleet_a, askeleet_b, askeleet_c,
    );

    // ── Single-pass training with relay ──────────────────────────
    println!("\n  [Kolmoset] Training (relay cascade)...");
    let mut kaadot = 0usize;
    for (konteksti, kohde) in &näytteet {
        let tulos = kolmoset.kouluta_askel(konteksti, *kohde, hdc);
        if tulos.kaato.is_some() {
            kaadot += 1;
        }
    }

    let tila = kolmoset.tila();
    let tarkkuus_alku = arvioi_kolmoset(&kolmoset, &näytteet, hdc);
    println!("    Online accuracy:     {:.2}% (during training)", tila.tarkkuus * 100.0);
    println!("    Ensemble accuracy:   {:.2}% (post-training)", tarkkuus_alku * 100.0);
    println!("    Memory dumps:        {} relay transitions", kaadot);
    println!("    Active node:         {}", tila.aktiivinen);
    println!("    ─── Node Details ───");
    println!("    A (Source):  acc={:.1}%  trust={:.3}  cycles={}  steps={}",
        tila.a_tarkkuus * 100.0, tila.a_luottamus, tila.a_kierrokset, tila.a_askeleet);
    println!("    B (Bridge):  acc={:.1}%  trust={:.3}  cycles={}  steps={}",
        tila.b_tarkkuus * 100.0, tila.b_luottamus, tila.b_kierrokset, tila.b_askeleet);
    println!("    C (Target):  acc={:.1}%  trust={:.3}  cycles={}  steps={}",
        tila.c_tarkkuus * 100.0, tila.c_luottamus, tila.c_kierrokset, tila.c_askeleet);

    // ── Retraining passes ─────────────────────────────────────────
    for kierros in 1..=uudelleen {
        println!("\n  [Kolmoset] Retraining pass {}...", kierros);
        let tarkkuus = kolmoset.uudelleenkouluta(&näytteet, hdc, kierros);
        let tarkkuus_eval = arvioi_kolmoset(&kolmoset, &näytteet, hdc);
        println!("    Retrain accuracy:   {:.2}%", tarkkuus * 100.0);
        println!("    Ensemble accuracy:  {:.2}%", tarkkuus_eval * 100.0);
    }

    // ── Final ─────────────────────────────────────────────────────
    let loppu = arvioi_kolmoset(&kolmoset, &näytteet, hdc);
    let loppu_tila = kolmoset.tila();
    println!("\n  [Kolmoset] Final state:");
    println!("    A trust={:.3}  B trust={:.3}  C trust={:.3}",
        loppu_tila.a_luottamus, loppu_tila.b_luottamus, loppu_tila.c_luottamus);

    loppu
}

// ═══════════════════════════════════════════════════════════════════════
// KOLMOSET + KESKUS PIPELINE
// ═══════════════════════════════════════════════════════════════════════

/// Evaluate Kolmoset+Keskus: enrich context, get raw scores, apply priors.
/// Sequential because Keskus has state (transitions, recurrent carry-forward).
fn arvioi_kolmoset_keskus(
    kolmoset: &Kolmoset,
    keskus: &mut Keskus,
    näytteet: &[(Vec<f64>, char)],
    hdc: &HdcPeruskäsitteet,
) -> f64 {
    if näytteet.is_empty() {
        return 0.0;
    }
    let mut oikein = 0usize;

    for (konteksti, kohde) in näytteet {
        // 1. Enrich context with recurrent state
        let rikastettu = keskus.rikasta(konteksti);

        // 2. Get raw scores from Kolmoset on enriched context
        let raa_at = kolmoset.kaikki_pisteet(&rikastettu, hdc);

        // 3. Apply transition + frequency + word boundary priors
        let säädetyt = keskus.sovella_priorit(&raa_at);

        // 4. Argmax on adjusted scores
        let (ennuste, paras_piste) = säädetyt.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&c, &s)| (c, s))
            .unwrap_or((' ', 0.0));

        if ennuste == *kohde {
            oikein += 1;
        }

        // 5. Update Keskus state (recurrent, transitions, health)
        let luottamus = paras_piste.abs().min(1.0);
        keskus.päivitä(ennuste, *kohde, konteksti, luottamus, hdc);
    }

    oikein as f64 / näytteet.len() as f64
}

/// Apply alpha overrides to a Keskus instance.
fn sovella_alpha_yliajot(
    keskus: &mut Keskus,
    overrides: (Option<f64>, Option<f64>, Option<f64>, Option<f64>),
) {
    if let Some(v) = overrides.0 { keskus.alpha_siirtymä = v; }
    if let Some(v) = overrides.1 { keskus.alpha_taajuus = v; }
    if let Some(v) = overrides.2 { keskus.alpha_kierto = v; }
    if let Some(v) = overrides.3 { keskus.alpha_sana = v; }
}

fn kouluta_kolmoset_keskus(
    teksti: &str,
    koodikirja: &HashMap<char, Hypervektori>,
    sitoja: &KontekstiSitoja,
    hdc: &mut HdcPeruskäsitteet,
    uudelleen: usize,
    alpha_overrides: (Option<f64>, Option<f64>, Option<f64>, Option<f64>),
    ulottuvuus: usize,
) -> f64 {
    // Build all (context, target) samples
    println!("\n  [Kolmoset+Keskus] Building context vectors...");
    let näytteet = rakenna_näytteet(teksti, koodikirja, sitoja, hdc);
    println!("    Samples: {}", näytteet.len());

    // Extract alphabet
    let mut aakkosto: Vec<char> = näytteet.iter().map(|(_, c)| *c).collect();
    aakkosto.sort();
    aakkosto.dedup();
    println!("    Alphabet: {} characters", aakkosto.len());

    // Create triple relay (same as vanilla Kolmoset)
    let n = näytteet.len();
    let askeleet_a = n / 5;
    let askeleet_b = n / 10;
    let askeleet_c = n / 5;
    println!("    Relay legs: A={}, B={}, C={}", askeleet_a, askeleet_b, askeleet_c);
    let mut kolmoset = Kolmoset::new_custom(
        &aakkosto, ulottuvuus, askeleet_a, askeleet_b, askeleet_c,
    );

    // ── Create Keskus ───────────────────────────────────────────────
    let mut keskus = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
    sovella_alpha_yliajot(&mut keskus, alpha_overrides);

    // Display active alphas
    println!("\n  [Keskus] Alphas: siirtymä={:.4} taajuus={:.4} kierto={:.4} sana={:.4}",
        keskus.alpha_siirtymä, keskus.alpha_taajuus, keskus.alpha_kierto, keskus.alpha_sana);

    // ── CRITICAL: Pre-train Keskus on raw corpus ────────────────────
    // Perfect bigram statistics from step 1.
    println!("  [Keskus] Pre-training on corpus (transitions + frequencies)...");
    let teksti_lower: String = teksti.chars()
        .map(|c| c.to_lowercase().next().unwrap_or(c))
        .collect();
    keskus.esikouluta(&teksti_lower);
    let keskus_tila = keskus.tila();
    println!("    Transitions: {} observed", keskus_tila.siirtymä_näytteet);
    println!("    Frequencies: {} observed", keskus_tila.taajuus_näytteet);
    println!("    Avg word length: {:.1}", keskus_tila.keskimääräinen_sanan_pituus);

    // ── Single-pass Kolmoset training (no Keskus here — raw training) ─
    println!("\n  [Kolmoset] Training (relay cascade)...");
    let mut kaadot = 0usize;
    for (konteksti, kohde) in &näytteet {
        let tulos = kolmoset.kouluta_askel(konteksti, *kohde, hdc);
        if tulos.kaato.is_some() {
            kaadot += 1;
        }
    }

    let tila = kolmoset.tila();
    println!("    Online accuracy:     {:.2}% (during training)", tila.tarkkuus * 100.0);
    println!("    Memory dumps:        {} relay transitions", kaadot);
    println!("    ─── Node Details ───");
    println!("    A (Source):  acc={:.1}%  trust={:.3}  cycles={}  steps={}",
        tila.a_tarkkuus * 100.0, tila.a_luottamus, tila.a_kierrokset, tila.a_askeleet);
    println!("    B (Bridge):  acc={:.1}%  trust={:.3}  cycles={}  steps={}",
        tila.b_tarkkuus * 100.0, tila.b_luottamus, tila.b_kierrokset, tila.b_askeleet);
    println!("    C (Target):  acc={:.1}%  trust={:.3}  cycles={}  steps={}",
        tila.c_tarkkuus * 100.0, tila.c_luottamus, tila.c_kierrokset, tila.c_askeleet);

    // ── Evaluate: Kolmoset raw vs Kolmoset+Keskus ───────────────────
    let tarkkuus_raaka = arvioi_kolmoset(&kolmoset, &näytteet, hdc);
    println!("\n  [Kolmoset] Raw ensemble accuracy:   {:.2}%", tarkkuus_raaka * 100.0);

    // Fresh Keskus for evaluation (reset recurrent state)
    let mut keskus_eval = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
    sovella_alpha_yliajot(&mut keskus_eval, alpha_overrides);
    keskus_eval.esikouluta(&teksti_lower);
    let tarkkuus_keskus = arvioi_kolmoset_keskus(
        &kolmoset, &mut keskus_eval, &näytteet, hdc,
    );
    println!("  [Kolmoset+Keskus] Enriched accuracy: {:.2}%  (Δ = {:+.2}%)",
        tarkkuus_keskus * 100.0, (tarkkuus_keskus - tarkkuus_raaka) * 100.0);

    // ── Retraining passes (Kolmoset retrains, Keskus re-evaluates) ──
    let mut paras_keskus = tarkkuus_keskus;
    for kierros in 1..=uudelleen {
        println!("\n  [Kolmoset] Retraining pass {}...", kierros);
        let tarkkuus_retrain = kolmoset.uudelleenkouluta(&näytteet, hdc, kierros);
        let tarkkuus_raaka = arvioi_kolmoset(&kolmoset, &näytteet, hdc);
        println!("    Retrain accuracy:   {:.2}%", tarkkuus_retrain * 100.0);
        println!("    Raw ensemble:       {:.2}%", tarkkuus_raaka * 100.0);

        // Re-evaluate with fresh Keskus (recurrent state from scratch each pass)
        let mut keskus_pass = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
        sovella_alpha_yliajot(&mut keskus_pass, alpha_overrides);
        keskus_pass.esikouluta(&teksti_lower);
        let tarkkuus_k = arvioi_kolmoset_keskus(
            &kolmoset, &mut keskus_pass, &näytteet, hdc,
        );
        println!("    Keskus-enriched:    {:.2}%  (Δ = {:+.2}%)",
            tarkkuus_k * 100.0, (tarkkuus_k - tarkkuus_raaka) * 100.0);

        if tarkkuus_k > paras_keskus {
            paras_keskus = tarkkuus_k;
        }
    }

    // ── Final ─────────────────────────────────────────────────────
    let loppu_raaka = arvioi_kolmoset(&kolmoset, &näytteet, hdc);
    let mut keskus_final = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
    sovella_alpha_yliajot(&mut keskus_final, alpha_overrides);
    keskus_final.esikouluta(&teksti_lower);
    let loppu_keskus = arvioi_kolmoset_keskus(
        &kolmoset, &mut keskus_final, &näytteet, hdc,
    );

    println!("\n  [Final] Kolmoset raw:     {:.2}%", loppu_raaka * 100.0);
    println!("  [Final] Kolmoset+Keskus:  {:.2}%  (Δ = {:+.2}%)",
        loppu_keskus * 100.0, (loppu_keskus - loppu_raaka) * 100.0);
    println!("  [Final] Peak Keskus:      {:.2}%", paras_keskus * 100.0);

    let keskus_tila = keskus_final.tila();
    println!("  [Keskus] Stress: {:.3}  Window acc: {:.2}%  Recurrent ‖‖: {:.4}",
        keskus_tila.stressitaso, keskus_tila.ikkuna_tarkkuus * 100.0, keskus_tila.kierto_normi);

    loppu_keskus
}

// ═══════════════════════════════════════════════════════════════════════
// KOLMOSET + KESKUS + BIPYRAMID PIPELINE
// ═══════════════════════════════════════════════════════════════════════

/// Evaluate with bipyramid 2-stage prediction:
///   1. Get all scores (same as flat: rikasta → kaikki_pisteet → sovella_priorit)
///   2. Average scores by pole → pick winning pole
///   3. If margin < threshold → fallback to flat argmax
///   4. Otherwise: argmax within winning pole only
///
/// Returns (accuracy, pole_correct_count, fallback_count).
fn arvioi_kolmoset_keskus_kaksoisnapainen(
    kolmoset: &Kolmoset,
    keskus: &mut Keskus,
    näytteet: &[(Vec<f64>, char)],
    hdc: &HdcPeruskäsitteet,
    kartta: &KaksoisnapainenKartta,
    kynnys: f64,
) -> (f64, usize, usize) {
    if näytteet.is_empty() {
        return (0.0, 0, 0);
    }
    let mut oikein = 0usize;
    let mut napa_oikein = 0usize;
    let mut varatiet = 0usize;

    for (konteksti, kohde) in näytteet {
        // 1. Enrich context with recurrent state
        let rikastettu = keskus.rikasta(konteksti);

        // 2. Get raw scores from Kolmoset on enriched context
        let raa_at = kolmoset.kaikki_pisteet(&rikastettu, hdc);

        // 3. Apply transition + frequency + word boundary priors
        let säädetyt = keskus.sovella_priorit(&raa_at);

        // 4. Bipyramid pole selection — use transition probabilities
        let kohde_napa = kartta.napa(*kohde);
        let (voittaja_napa, _napa_pisteet, marginaali) = kartta.valitse_napa_siirtymä(
            |c| keskus.siirtymä_p(c),
        );

        if voittaja_napa == kohde_napa {
            napa_oikein += 1;
        }

        // 5. Decide: bipyramid or flat fallback
        let ennuste = if marginaali < kynnys {
            // Low confidence → flat argmax (fallback)
            varatiet += 1;
            säädetyt.iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(&c, _)| c)
                .unwrap_or(' ')
        } else {
            // High confidence → argmax within winning pole only
            säädetyt.iter()
                .filter(|(&c, _)| kartta.napa(c) == voittaja_napa)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(&c, _)| c)
                .unwrap_or_else(|| {
                    // Pole empty? Fall back to flat
                    säädetyt.iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(&c, _)| c)
                        .unwrap_or(' ')
                })
        };

        if ennuste == *kohde {
            oikein += 1;
        }

        // 6. Update Keskus state
        let luottamus = säädetyt.get(&ennuste).copied().unwrap_or(0.0).abs().min(1.0);
        keskus.päivitä(ennuste, *kohde, konteksti, luottamus, hdc);
    }

    let tarkkuus = oikein as f64 / näytteet.len() as f64;
    (tarkkuus, napa_oikein, varatiet)
}

fn kouluta_kolmoset_keskus_kaksoisnapainen(
    teksti: &str,
    koodikirja: &HashMap<char, Hypervektori>,
    sitoja: &KontekstiSitoja,
    hdc: &mut HdcPeruskäsitteet,
    uudelleen: usize,
    alpha_overrides: (Option<f64>, Option<f64>, Option<f64>, Option<f64>),
    bipyramid_kynnys: f64,
    ulottuvuus: usize,
) -> f64 {
    // Build all (context, target) samples
    println!("\n  [Kolmoset+Keskus+Bipyramid] Building context vectors...");
    let näytteet = rakenna_näytteet(teksti, koodikirja, sitoja, hdc);
    println!("    Samples: {}", näytteet.len());

    // Extract alphabet
    let mut aakkosto: Vec<char> = näytteet.iter().map(|(_, c)| *c).collect();
    aakkosto.sort();
    aakkosto.dedup();
    println!("    Alphabet: {} characters", aakkosto.len());

    // Build bipyramid map
    let kartta = KaksoisnapainenKartta::new(&aakkosto);
    kartta.tulosta_yhteenveto();
    println!("    Bipyramid threshold: {:.4}", bipyramid_kynnys);

    // Create triple relay (same as vanilla Kolmoset)
    let n = näytteet.len();
    let askeleet_a = n / 5;
    let askeleet_b = n / 10;
    let askeleet_c = n / 5;
    println!("    Relay legs: A={}, B={}, C={}", askeleet_a, askeleet_b, askeleet_c);
    let mut kolmoset = Kolmoset::new_custom(
        &aakkosto, ulottuvuus, askeleet_a, askeleet_b, askeleet_c,
    );

    // ── Create Keskus ───────────────────────────────────────────────
    let mut keskus = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
    sovella_alpha_yliajot(&mut keskus, alpha_overrides);

    println!("\n  [Keskus] Alphas: siirtymä={:.4} taajuus={:.4} kierto={:.4} sana={:.4}",
        keskus.alpha_siirtymä, keskus.alpha_taajuus, keskus.alpha_kierto, keskus.alpha_sana);

    // ── Pre-train Keskus on raw corpus ────────────────────────────
    println!("  [Keskus] Pre-training on corpus (transitions + frequencies)...");
    let teksti_lower: String = teksti.chars()
        .map(|c| c.to_lowercase().next().unwrap_or(c))
        .collect();
    keskus.esikouluta(&teksti_lower);
    let keskus_tila = keskus.tila();
    println!("    Transitions: {} observed", keskus_tila.siirtymä_näytteet);
    println!("    Frequencies: {} observed", keskus_tila.taajuus_näytteet);
    println!("    Avg word length: {:.1}", keskus_tila.keskimääräinen_sanan_pituus);

    // ── Single-pass Kolmoset training (no Keskus — raw training) ─
    println!("\n  [Kolmoset] Training (relay cascade)...");
    let mut kaadot = 0usize;
    for (konteksti, kohde) in &näytteet {
        let tulos = kolmoset.kouluta_askel(konteksti, *kohde, hdc);
        if tulos.kaato.is_some() {
            kaadot += 1;
        }
    }

    let tila = kolmoset.tila();
    println!("    Online accuracy:     {:.2}% (during training)", tila.tarkkuus * 100.0);
    println!("    Memory dumps:        {} relay transitions", kaadot);
    println!("    ─── Node Details ───");
    println!("    A (Source):  acc={:.1}%  trust={:.3}  cycles={}  steps={}",
        tila.a_tarkkuus * 100.0, tila.a_luottamus, tila.a_kierrokset, tila.a_askeleet);
    println!("    B (Bridge):  acc={:.1}%  trust={:.3}  cycles={}  steps={}",
        tila.b_tarkkuus * 100.0, tila.b_luottamus, tila.b_kierrokset, tila.b_askeleet);
    println!("    C (Target):  acc={:.1}%  trust={:.3}  cycles={}  steps={}",
        tila.c_tarkkuus * 100.0, tila.c_luottamus, tila.c_kierrokset, tila.c_askeleet);

    // ── Evaluate: flat vs bipyramid ─────────────────────────────────
    let tarkkuus_raaka = arvioi_kolmoset(&kolmoset, &näytteet, hdc);
    println!("\n  [Kolmoset] Raw ensemble accuracy:   {:.2}%", tarkkuus_raaka * 100.0);

    // Flat Keskus evaluation (for comparison)
    let mut keskus_flat = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
    sovella_alpha_yliajot(&mut keskus_flat, alpha_overrides);
    keskus_flat.esikouluta(&teksti_lower);
    let tarkkuus_flat = arvioi_kolmoset_keskus(
        &kolmoset, &mut keskus_flat, &näytteet, hdc,
    );
    println!("  [Flat Keskus] Enriched accuracy:     {:.2}%  (Δ = {:+.2}%)",
        tarkkuus_flat * 100.0, (tarkkuus_flat - tarkkuus_raaka) * 100.0);

    // Bipyramid evaluation
    let mut keskus_bipy = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
    sovella_alpha_yliajot(&mut keskus_bipy, alpha_overrides);
    keskus_bipy.esikouluta(&teksti_lower);
    let (tarkkuus_bp, napa_oikein, varatiet) = arvioi_kolmoset_keskus_kaksoisnapainen(
        &kolmoset, &mut keskus_bipy, &näytteet, hdc, &kartta, bipyramid_kynnys,
    );
    let napa_tarkkuus = napa_oikein as f64 / näytteet.len() as f64;
    let varatie_osuus = varatiet as f64 / näytteet.len() as f64;
    println!("  [Bipyramid] Enriched accuracy:       {:.2}%  (Δ vs flat = {:+.2}%)",
        tarkkuus_bp * 100.0, (tarkkuus_bp - tarkkuus_flat) * 100.0);
    println!("  [Bipyramid] Pole accuracy:           {:.2}%", napa_tarkkuus * 100.0);
    println!("  [Bipyramid] Fallback rate:           {:.2}%", varatie_osuus * 100.0);

    // ── Retraining passes ────────────────────────────────────────────
    let mut paras_bp = tarkkuus_bp;
    let mut paras_flat = tarkkuus_flat;
    for kierros in 1..=uudelleen {
        println!("\n  [Kolmoset] Retraining pass {}...", kierros);
        let tarkkuus_retrain = kolmoset.uudelleenkouluta(&näytteet, hdc, kierros);
        let tarkkuus_raaka = arvioi_kolmoset(&kolmoset, &näytteet, hdc);
        println!("    Retrain accuracy:   {:.2}%", tarkkuus_retrain * 100.0);
        println!("    Raw ensemble:       {:.2}%", tarkkuus_raaka * 100.0);

        // Flat re-evaluation
        let mut keskus_pass_flat = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
        sovella_alpha_yliajot(&mut keskus_pass_flat, alpha_overrides);
        keskus_pass_flat.esikouluta(&teksti_lower);
        let tarkkuus_f = arvioi_kolmoset_keskus(
            &kolmoset, &mut keskus_pass_flat, &näytteet, hdc,
        );

        // Bipyramid re-evaluation
        let mut keskus_pass_bp = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
        sovella_alpha_yliajot(&mut keskus_pass_bp, alpha_overrides);
        keskus_pass_bp.esikouluta(&teksti_lower);
        let (tarkkuus_b, napa_ok, varatiet_p) = arvioi_kolmoset_keskus_kaksoisnapainen(
            &kolmoset, &mut keskus_pass_bp, &näytteet, hdc, &kartta, bipyramid_kynnys,
        );

        let napa_t = napa_ok as f64 / näytteet.len() as f64;
        let var_t = varatiet_p as f64 / näytteet.len() as f64;
        println!("    Flat Keskus:        {:.2}%  (Δ = {:+.2}%)",
            tarkkuus_f * 100.0, (tarkkuus_f - tarkkuus_raaka) * 100.0);
        println!("    Bipyramid:          {:.2}%  (Δ vs flat = {:+.2}%)  pole={:.1}%  fallback={:.1}%",
            tarkkuus_b * 100.0, (tarkkuus_b - tarkkuus_f) * 100.0,
            napa_t * 100.0, var_t * 100.0);

        if tarkkuus_b > paras_bp { paras_bp = tarkkuus_b; }
        if tarkkuus_f > paras_flat { paras_flat = tarkkuus_f; }
    }

    // ── Final ─────────────────────────────────────────────────────
    let loppu_raaka = arvioi_kolmoset(&kolmoset, &näytteet, hdc);

    let mut keskus_final_flat = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
    sovella_alpha_yliajot(&mut keskus_final_flat, alpha_overrides);
    keskus_final_flat.esikouluta(&teksti_lower);
    let loppu_flat = arvioi_kolmoset_keskus(
        &kolmoset, &mut keskus_final_flat, &näytteet, hdc,
    );

    let mut keskus_final_bp = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
    sovella_alpha_yliajot(&mut keskus_final_bp, alpha_overrides);
    keskus_final_bp.esikouluta(&teksti_lower);
    let (loppu_bp, loppu_napa_ok, loppu_varatiet) = arvioi_kolmoset_keskus_kaksoisnapainen(
        &kolmoset, &mut keskus_final_bp, &näytteet, hdc, &kartta, bipyramid_kynnys,
    );

    let loppu_napa_t = loppu_napa_ok as f64 / näytteet.len() as f64;
    let loppu_var_t = loppu_varatiet as f64 / näytteet.len() as f64;

    println!("\n  [Final] Kolmoset raw:     {:.2}%", loppu_raaka * 100.0);
    println!("  [Final] Flat Keskus:      {:.2}%  (Δ = {:+.2}%)",
        loppu_flat * 100.0, (loppu_flat - loppu_raaka) * 100.0);
    println!("  [Final] Bipyramid:        {:.2}%  (Δ vs flat = {:+.2}%)",
        loppu_bp * 100.0, (loppu_bp - loppu_flat) * 100.0);
    println!("  [Final] Peak flat:        {:.2}%", paras_flat * 100.0);
    println!("  [Final] Peak bipyramid:   {:.2}%", paras_bp * 100.0);
    println!("  [Final] Pole accuracy:    {:.2}%", loppu_napa_t * 100.0);
    println!("  [Final] Fallback rate:    {:.2}%", loppu_var_t * 100.0);

    let keskus_tila = keskus_final_bp.tila();
    println!("  [Keskus] Stress: {:.3}  Window acc: {:.2}%  Recurrent ‖‖: {:.4}",
        keskus_tila.stressitaso, keskus_tila.ikkuna_tarkkuus * 100.0, keskus_tila.kierto_normi);

    loppu_bp
}

// ═══════════════════════════════════════════════════════════════════════
// OUTPUT
// ═══════════════════════════════════════════════════════════════════════

fn tulosta_lopputulos(tarkkuus: f64) {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  FINAL ACCURACY: {:.2}%", tarkkuus * 100.0);
    if tarkkuus >= 0.25 {
        println!("  CLEARED bigram threshold (25%)");
    } else {
        println!("  Below bigram threshold (25%) — {:.1}% to go",
            (0.25 - tarkkuus) * 100.0);
    }
    println!("═══════════════════════════════════════════════════════════════");
}
