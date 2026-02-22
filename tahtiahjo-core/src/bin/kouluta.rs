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
//!   --compare         Run all architectures side-by-side
//!
//! Examples:
//!   cargo run --release --bin kouluta -- ../kalevala.txt --chars=10000 --codebook=qams
//!   cargo run --release --bin kouluta -- ../kalevala.txt --kaksoset
//!   cargo run --release --bin kouluta -- ../kalevala.txt --kolmoset
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
    let mut vertaa = false;

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
        } else if arg == "--compare" {
            vertaa = true;
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
    println!("  Dimension:  D={}", ULOTTUVUUS);
    if vertaa {
        println!("  Mode:       COMPARE (all architectures)");
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
        let mut kirja = luo_koodikirja(&allekirjoitukset, ULOTTUVUUS, 42);
        // Add any characters from text that aren't in QAMS signatures
        let mut rng = tahtiahjo_core::hdc_primitives::Siemen::new(777);
        for c in teksti.chars() {
            let c_lower = c.to_lowercase().next().unwrap_or(c);
            kirja.entry(c_lower)
                .or_insert_with(|| rng.bipolaarinen_vektori(ULOTTUVUUS));
        }
        kirja
    } else {
        println!("  Building random codebook...");
        luo_satunnainen_koodikirja(&teksti, ULOTTUVUUS, 99)
    };
    println!("  Codebook size: {} characters", koodikirja.len());

    let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);
    let sitoja = KontekstiSitoja::new(ikkuna);

    if vertaa {
        // ═══════════════════════════════════════════════════════════════
        // COMPARE MODE: run all architectures side-by-side
        // ═══════════════════════════════════════════════════════════════
        let tarkkuus_lk = kouluta_luokkakertymä(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen,
        );
        let tarkkuus_kk = kouluta_kaksoset(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen,
        );
        let tarkkuus_km = kouluta_kolmoset(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen,
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
    } else if käytä_kolmoset {
        // ═══════════════════════════════════════════════════════════════
        // KOLMOSET MODE
        // ═══════════════════════════════════════════════════════════════
        let tarkkuus = kouluta_kolmoset(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen,
        );
        tulosta_lopputulos(tarkkuus);
    } else if käytä_kaksoset {
        // ═══════════════════════════════════════════════════════════════
        // KAKSOSET MODE
        // ═══════════════════════════════════════════════════════════════
        let tarkkuus = kouluta_kaksoset(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen,
        );
        tulosta_lopputulos(tarkkuus);
    } else {
        // ═══════════════════════════════════════════════════════════════
        // STANDARD MODE (LuokkaKertymä)
        // ═══════════════════════════════════════════════════════════════
        let tarkkuus = kouluta_luokkakertymä(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen,
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
) -> f64 {
    let mut kertymä = LuokkaKertymä::new(ULOTTUVUUS);

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
    let mut kaksoset = Kaksoset::new(&aakkosto, ULOTTUVUUS);

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
        &aakkosto, ULOTTUVUUS, askeleet_a, askeleet_b, askeleet_c,
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
