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
//!   --compare         Run LuokkaKertymä vs Kaksoset side-by-side
//!
//! Examples:
//!   cargo run --release --bin kouluta -- ../kalevala.txt --chars=10000 --codebook=qams
//!   cargo run --release --bin kouluta -- ../kalevala.txt --kaksoset
//!   cargo run --release --bin kouluta -- ../kalevala.txt --compare

use std::collections::HashMap;
use std::env;
use std::fs;

use tahtiahjo_core::hdc_primitives::{HdcPeruskäsitteet, Hypervektori, ULOTTUVUUS};
use tahtiahjo_core::konteksti_sitoja::KontekstiSitoja;
use tahtiahjo_core::luokka_kertyma::{LuokkaKertymä, luo_satunnainen_koodikirja};
use tahtiahjo_core::qams_codebook::{kaikki_allekirjoitukset, luo_koodikirja};
use tahtiahjo_core::kaksoset::Kaksoset;

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
        println!("  Mode:       COMPARE (LuokkaKertymä vs Kaksoset)");
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
        // COMPARE MODE: run both pipelines side-by-side
        // ═══════════════════════════════════════════════════════════════
        let tarkkuus_lk = kouluta_luokkakertymä(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen,
        );
        let tarkkuus_kk = kouluta_kaksoset(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen,
        );

        println!("\n═══════════════════════════════════════════════════════════════");
        println!("  COMPARISON RESULTS");
        println!("───────────────────────────────────────────────────────────────");
        println!("  LuokkaKertymä:  {:.2}%", tarkkuus_lk * 100.0);
        println!("  Kaksoset:       {:.2}%", tarkkuus_kk * 100.0);
        let ero = (tarkkuus_kk - tarkkuus_lk) * 100.0;
        if ero > 0.0 {
            println!("  Kaksoset wins:  +{:.2}%", ero);
        } else if ero < 0.0 {
            println!("  LuokkaKertymä wins: {:.2}%", ero);
        } else {
            println!("  Tie");
        }
        println!("═══════════════════════════════════════════════════════════════");
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

    // ── Retraining passes ─────────────────────────────────────────
    for kierros in 1..=uudelleen {
        println!("\n  [Kaksoset] Retraining pass {}...", kierros);
        let tarkkuus = kaksoset.uudelleenkouluta(&näytteet, hdc, kierros);
        let tarkkuus_eval = arvioi_kaksoset(&kaksoset, &näytteet, hdc);
        println!("    Retrain accuracy:   {:.2}%", tarkkuus * 100.0);
        println!("    Evaluated accuracy: {:.2}%", tarkkuus_eval * 100.0);
    }

    // ── Final diagnostics ─────────────────────────────────────────
    let loppu_tila = kaksoset.tila();
    let loppu_tarkkuus = arvioi_kaksoset(&kaksoset, &näytteet, hdc);
    println!("\n  [Kaksoset] Final state:");
    println!("    β: {:.4}  HRV: {:.4}  Heart: {} steps",
        loppu_tila.beta, loppu_tila.hrv_koherenssi, loppu_tila.syke_jakso);

    loppu_tarkkuus
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
