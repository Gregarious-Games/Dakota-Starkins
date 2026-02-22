//! Kouluta — Train and evaluate HDC language model on text files.
//!
//! Usage:
//!   cargo run --bin kouluta -- <text_file> [--chars=N] [--retrain=N] [--window=N] [--codebook=random|qams]
//!
//! Example:
//!   cargo run --release --bin kouluta -- ../kalevala.txt --chars=10000 --retrain=3 --codebook=qams

use std::collections::HashMap;
use std::env;
use std::fs;

use tahtiahjo_core::hdc_primitives::{HdcPeruskäsitteet, Hypervektori, ULOTTUVUUS};
use tahtiahjo_core::konteksti_sitoja::KontekstiSitoja;
use tahtiahjo_core::luokka_kertyma::{LuokkaKertymä, luo_satunnainen_koodikirja};
use tahtiahjo_core::qams_codebook::{kaikki_allekirjoitukset, luo_koodikirja};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: kouluta <text_file> [--chars=N] [--retrain=N] [--window=N] [--codebook=random|qams]");
        std::process::exit(1);
    }

    let tiedosto = &args[1];
    let mut max_merkit: usize = 10_000;
    let mut uudelleen: usize = 3;
    let mut ikkuna: usize = 3;
    let mut koodikirja_tyyppi = "qams".to_string();

    for arg in &args[2..] {
        if let Some(val) = arg.strip_prefix("--chars=") {
            max_merkit = val.parse().unwrap_or(10_000);
        } else if let Some(val) = arg.strip_prefix("--retrain=") {
            uudelleen = val.parse().unwrap_or(3);
        } else if let Some(val) = arg.strip_prefix("--window=") {
            ikkuna = val.parse().unwrap_or(3);
        } else if let Some(val) = arg.strip_prefix("--codebook=") {
            koodikirja_tyyppi = val.to_string();
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
    let mut kertymä = LuokkaKertymä::new(ULOTTUVUUS);

    // Train
    println!("\n  Training (single pass)...");
    let näytteet = kertymä.kouluta(&teksti, &koodikirja, &sitoja, &mut hdc);
    let tarkkuus_alku = kertymä.tarkkuus(&teksti, &koodikirja, &sitoja, &mut hdc);
    println!("    Samples:  {}", näytteet);
    println!("    Classes:  {}", kertymä.luokka_lkm());
    println!("    Accuracy: {:.2}%", tarkkuus_alku * 100.0);

    // Retrain
    for kierros in 1..=uudelleen {
        println!("\n  Retraining pass {}...", kierros);
        let korjaukset = kertymä.uudelleenkouluta(&teksti, &koodikirja, &sitoja, &mut hdc);
        let tarkkuus = kertymä.tarkkuus(&teksti, &koodikirja, &sitoja, &mut hdc);
        println!("    Corrections: {}", korjaukset);
        println!("    Accuracy:    {:.2}%", tarkkuus * 100.0);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    let loppu = kertymä.tarkkuus(&teksti, &koodikirja, &sitoja, &mut hdc);
    println!("  FINAL ACCURACY: {:.2}%", loppu * 100.0);
    if loppu >= 0.25 {
        println!("  ✓ CLEARED bigram threshold (25%)");
    } else {
        println!("  · Below bigram threshold (25%) — {:.1}% to go",
            (0.25 - loppu) * 100.0);
    }
    println!("═══════════════════════════════════════════════════════════════");
}
