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

use tahtiahjo_core::hdc_primitives::{HdcPeruskäsitteet, Hypervektori, ULOTTUVUUS, rotate_codebook};
use tahtiahjo_core::konteksti_sitoja::KontekstiSitoja;
use tahtiahjo_core::luokka_kertyma::{LuokkaKertymä, luo_satunnainen_koodikirja};
use tahtiahjo_core::qams_codebook::{
    kaikki_allekirjoitukset, luo_koodikirja, luo_koodikirja_painotettu,
    ÄänneAllekirjoitus, PAINOT_RELE_A, PAINOT_RELE_B, PAINOT_RELE_C,
};
use tahtiahjo_core::kaksoset::Kaksoset;
use tahtiahjo_core::kolmoset::Kolmoset;
use tahtiahjo_core::keskus::Keskus;
use tahtiahjo_core::kaksoisnapainen::KaksoisnapainenKartta;
use tahtiahjo_core::phase_harmonic::PhaseHarmonizer;

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

    // Golden-mean damping coefficient (0.0 = off)
    let mut vaimennus: f64 = 0.0;

    // Phase conjugate relay rotation
    let mut käytä_rotation = false;
    let mut rotation_plane: (usize, usize) = (0, 1);

    // QAMS phonetic specialization
    let mut käytä_qams_specialize = false;

    // Phase Conjugate Voting
    let mut käytä_pcv = false;

    // No relay dump (parallel independent training)
    let mut no_dump = false;

    // Phase Harmonic Engine
    let mut käytä_phase_harmonic = false;
    let mut käytä_harmonic_hybrid = false;
    let mut käytä_harmonic_compare = false;
    let mut käytä_d18 = false;
    let mut käytä_blend_fallback = false;
    let mut käytä_align_temp = false;
    let mut käytä_align_keskus = false;
    let mut käytä_ece_weights = false;
    let mut käytä_entropy_gate = false;

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
        } else if let Some(val) = arg.strip_prefix("--damping=") {
            vaimennus = val.parse().unwrap_or(0.0);
        } else if arg == "--rotation" {
            käytä_rotation = true;
        } else if let Some(val) = arg.strip_prefix("--rotation-plane=") {
            let parts: Vec<&str> = val.split(',').collect();
            if parts.len() == 2 {
                let i = parts[0].parse().unwrap_or(0);
                let j = parts[1].parse().unwrap_or(1);
                rotation_plane = (i, j);
            }
            käytä_rotation = true;
        } else if arg == "--qams-specialize" || arg == "--qams-spec" {
            käytä_qams_specialize = true;
        } else if arg == "--pcv" {
            käytä_pcv = true;
        } else if arg == "--no-dump" {
            no_dump = true;
        } else if arg == "--phase-harmonic" {
            käytä_phase_harmonic = true;
        } else if arg == "--harmonic-hybrid" {
            käytä_harmonic_hybrid = true;
        } else if arg == "--harmonic-compare" {
            käytä_harmonic_compare = true;
        } else if arg == "--d18" {
            käytä_d18 = true;
        } else if arg == "--blend-fallback" {
            käytä_blend_fallback = true;
        } else if arg == "--align-temp" {
            käytä_align_temp = true;
        } else if arg == "--align-keskus" {
            käytä_align_keskus = true;
            käytä_align_temp = true; // align-keskus implies align-temp
        } else if arg == "--ece-weights" {
            käytä_ece_weights = true;
        } else if arg == "--entropy-gate" {
            käytä_entropy_gate = true;
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
    if vaimennus > 0.0 {
        println!("  Damping:    ψ={:.6} (golden-mean drift brake)", vaimennus);
    }
    if vertaa {
        println!("  Mode:       COMPARE (all architectures)");
    } else if käytä_kolmoset && käytä_harmonic_compare {
        println!("  Mode:       KOLMOSET + HARMONIC COMPARE (4-way: raw/keskus/harmonic/hybrid)");
        if käytä_d18 {
            println!("  Harmonic:   D-18 mode (9 conjugate pairs)");
        }
    } else if käytä_kolmoset && käytä_harmonic_hybrid {
        println!("  Mode:       KOLMOSET + PHASE HARMONIC + KESKUS (hybrid)");
        if käytä_d18 {
            println!("  Harmonic:   D-18 mode (9 conjugate pairs)");
        }
    } else if käytä_kolmoset && käytä_phase_harmonic {
        println!("  Mode:       KOLMOSET + PHASE HARMONIC (pure phase-lock)");
        if käytä_d18 {
            println!("  Harmonic:   D-18 mode (9 conjugate pairs)");
        }
    } else if käytä_kolmoset && (käytä_rotation || käytä_qams_specialize || käytä_pcv || no_dump) {
        let mut mode_parts = vec!["KOLMOSET"];
        if käytä_rotation { mode_parts.push("ROTATION"); }
        if käytä_qams_specialize { mode_parts.push("QAMS-SPEC"); }
        if käytä_pcv { mode_parts.push("PCV"); }
        if no_dump { mode_parts.push("NO-DUMP"); }
        println!("  Mode:       {} (per-relay diversity)", mode_parts.join(" + "));
        if käytä_rotation {
            println!("  Rotation:   plane=({},{}) at 0°/120°/240°", rotation_plane.0, rotation_plane.1);
        }
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
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen, ulottuvuus, vaimennus,
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
    } else if käytä_kolmoset && käytä_harmonic_compare {
        // ═══════════════════════════════════════════════════════════════
        // KOLMOSET + HARMONIC COMPARE (4-way: raw/keskus/harmonic/hybrid)
        // ═══════════════════════════════════════════════════════════════
        let alpha_overrides = (alpha_siirtyma, alpha_taajuus, alpha_kierto, alpha_sana);
        let tarkkuus = kouluta_kolmoset_harmonic_compare(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen,
            alpha_overrides, ulottuvuus, vaimennus, käytä_d18,
        );
        tulosta_lopputulos(tarkkuus);
    } else if käytä_kolmoset && käytä_harmonic_hybrid {
        // ═══════════════════════════════════════════════════════════════
        // KOLMOSET + PHASE HARMONIC + KESKUS HYBRID
        // ═══════════════════════════════════════════════════════════════
        let alpha_overrides = (alpha_siirtyma, alpha_taajuus, alpha_kierto, alpha_sana);
        let tarkkuus = kouluta_kolmoset_harmonic(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen,
            alpha_overrides, ulottuvuus, vaimennus, käytä_d18, true,
        );
        tulosta_lopputulos(tarkkuus);
    } else if käytä_kolmoset && käytä_phase_harmonic {
        // ═══════════════════════════════════════════════════════════════
        // KOLMOSET + PURE PHASE HARMONIC
        // ═══════════════════════════════════════════════════════════════
        let tarkkuus = kouluta_kolmoset_harmonic(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen,
            (None, None, None, None), ulottuvuus, vaimennus, käytä_d18, false,
        );
        tulosta_lopputulos(tarkkuus);
    } else if käytä_kolmoset && (käytä_rotation || käytä_qams_specialize || käytä_pcv || no_dump) {
        // ═══════════════════════════════════════════════════════════════
        // KOLMOSET + ROTATION / QAMS SPECIALIZATION / PCV MODE
        // ═══════════════════════════════════════════════════════════════
        let allekirjoitukset = kaikki_allekirjoitukset();
        let tarkkuus = kouluta_kolmoset_kierto(
            &teksti, &allekirjoitukset, &koodikirja, &sitoja, &mut hdc,
            uudelleen, ulottuvuus, vaimennus,
            if käytä_rotation { Some(rotation_plane) } else { None },
            käytä_qams_specialize,
            käytä_pcv,
            no_dump,
            käytä_blend_fallback,
            käytä_align_temp,
            käytä_align_keskus,
            käytä_ece_weights,
            käytä_entropy_gate,
        );
        tulosta_lopputulos(tarkkuus);
    } else if käytä_kolmoset && käytä_keskus && käytä_bipyramid {
        // ═══════════════════════════════════════════════════════════════
        // KOLMOSET + KESKUS + BIPYRAMID MODE
        // ═══════════════════════════════════════════════════════════════
        let alpha_overrides = (alpha_siirtyma, alpha_taajuus, alpha_kierto, alpha_sana);
        let tarkkuus = kouluta_kolmoset_keskus_kaksoisnapainen(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen,
            alpha_overrides, bipyramid_threshold, ulottuvuus, vaimennus,
        );
        tulosta_lopputulos(tarkkuus);
    } else if käytä_kolmoset && käytä_keskus {
        // ═══════════════════════════════════════════════════════════════
        // KOLMOSET + KESKUS MODE
        // ═══════════════════════════════════════════════════════════════
        let alpha_overrides = (alpha_siirtyma, alpha_taajuus, alpha_kierto, alpha_sana);
        let tarkkuus = kouluta_kolmoset_keskus(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen, alpha_overrides, ulottuvuus, vaimennus,
        );
        tulosta_lopputulos(tarkkuus);
    } else if käytä_kolmoset {
        // ═══════════════════════════════════════════════════════════════
        // KOLMOSET MODE
        // ═══════════════════════════════════════════════════════════════
        let tarkkuus = kouluta_kolmoset(
            &teksti, &koodikirja, &sitoja, &mut hdc, uudelleen, ulottuvuus, vaimennus,
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
    vaimennus: f64,
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

    // Apply golden-mean damping if requested
    if vaimennus > 0.0 {
        kolmoset.aseta_vaimennus(vaimennus);
        println!("    Damping: ψ={:.6}", vaimennus);
    }

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
// KOLMOSET + ROTATION / QAMS SPECIALIZATION PIPELINE
// ═══════════════════════════════════════════════════════════════════════

/// Evaluate Kolmoset with per-relay contexts on pre-built sample sets.
fn arvioi_kolmoset_kierto(
    kolmoset: &Kolmoset,
    näytteet: [&[(Vec<f64>, char)]; 3],
    hdc: &HdcPeruskäsitteet,
    pcv: bool,
) -> f64 {
    if näytteet[0].is_empty() {
        return 0.0;
    }
    let oikein = (0..näytteet[0].len())
        .filter(|&i| {
            let kontekstit = [
                näytteet[0][i].0.as_slice(),
                näytteet[1][i].0.as_slice(),
                näytteet[2][i].0.as_slice(),
            ];
            let kohde = näytteet[0][i].1;
            let (ennuste, _) = if pcv {
                kolmoset.ennusta_kierto_pcv(kontekstit, hdc)
            } else {
                kolmoset.ennusta_kierto(kontekstit, hdc)
            };
            ennuste == kohde
        })
        .count();
    oikein as f64 / näytteet[0].len() as f64
}

fn kouluta_kolmoset_kierto(
    teksti: &str,
    allekirjoitukset: &HashMap<char, ÄänneAllekirjoitus>,
    koodikirja_oletus: &HashMap<char, Hypervektori>,
    sitoja: &KontekstiSitoja,
    hdc: &mut HdcPeruskäsitteet,
    uudelleen: usize,
    ulottuvuus: usize,
    vaimennus: f64,
    rotation_plane: Option<(usize, usize)>,
    qams_specialize: bool,
    pcv: bool,
    no_dump: bool,
    blend_fallback: bool,
    align_temp: bool,
    align_keskus: bool,
    ece_weights: bool,
    entropy_gate: bool,
) -> f64 {
    // ── Build per-relay codebooks ─────────────────────────────────
    println!("\n  [Kierto] Building per-relay codebooks...");

    let mut kirjat: [HashMap<char, Hypervektori>; 3] = [
        koodikirja_oletus.clone(),
        koodikirja_oletus.clone(),
        koodikirja_oletus.clone(),
    ];

    if qams_specialize {
        // QAMS specialization: each relay emphasizes different phonetic dimensions
        println!("    QAMS specialization: A=vowel, B=consonant, C=temporal");
        kirjat[0] = luo_koodikirja_painotettu(allekirjoitukset, ulottuvuus, 42, &PAINOT_RELE_A);
        kirjat[1] = luo_koodikirja_painotettu(allekirjoitukset, ulottuvuus, 42, &PAINOT_RELE_B);
        kirjat[2] = luo_koodikirja_painotettu(allekirjoitukset, ulottuvuus, 42, &PAINOT_RELE_C);

        // Add any characters from text that aren't in QAMS signatures
        let mut rng = tahtiahjo_core::hdc_primitives::Siemen::new(777);
        for c in teksti.chars() {
            let c_lower = c.to_lowercase().next().unwrap_or(c);
            for kirja in kirjat.iter_mut() {
                kirja.entry(c_lower)
                    .or_insert_with(|| rng.bipolaarinen_vektori(ulottuvuus));
            }
        }
    }

    if let Some((i, j)) = rotation_plane {
        // Givens rotation: 120° spacing per relay (0°, 120°, 240°)
        let theta_120 = 2.0 * std::f64::consts::PI / 3.0;
        println!("    Givens rotation: plane=({},{}) at 0°/120°/240°", i, j);
        // Relay A: 0° (identity — no rotation needed)
        // Relay B: 120°
        kirjat[1] = rotate_codebook(&kirjat[1], i, j, theta_120);
        // Relay C: 240°
        kirjat[2] = rotate_codebook(&kirjat[2], i, j, 2.0 * theta_120);
    }

    println!("    Codebook sizes: A={}, B={}, C={}",
        kirjat[0].len(), kirjat[1].len(), kirjat[2].len());

    // ── Build per-relay sample sets ──────────────────────────────
    println!("  [Kierto] Building per-relay context vectors...");
    let näytteet_a = rakenna_näytteet(teksti, &kirjat[0], sitoja, hdc);
    let näytteet_b = rakenna_näytteet(teksti, &kirjat[1], sitoja, hdc);
    let näytteet_c = rakenna_näytteet(teksti, &kirjat[2], sitoja, hdc);
    println!("    Samples per relay: A={}, B={}, C={}",
        näytteet_a.len(), näytteet_b.len(), näytteet_c.len());

    // Extract alphabet (same across all relays — same text)
    let mut aakkosto: Vec<char> = näytteet_a.iter().map(|(_, c)| *c).collect();
    aakkosto.sort();
    aakkosto.dedup();
    println!("    Alphabet: {} characters", aakkosto.len());

    // ── Create Kolmoset ─────────────────────────────────────────
    let n = näytteet_a.len();
    let askeleet_a = n / 5;
    let askeleet_b = n / 10;
    let askeleet_c = n / 5;
    println!("    Relay legs: A={}, B={}, C={}", askeleet_a, askeleet_b, askeleet_c);
    let mut kolmoset = Kolmoset::new_custom(
        &aakkosto, ulottuvuus, askeleet_a, askeleet_b, askeleet_c,
    );

    if vaimennus > 0.0 {
        kolmoset.aseta_vaimennus(vaimennus);
        println!("    Damping: ψ={:.6}", vaimennus);
    }

    // ── Single-pass training with per-relay contexts ─────────────
    let mut kaadot = 0usize;
    if no_dump {
        // Parallel independent training: all 3 relays see every sample
        println!("\n  [Kierto] Training (parallel independent, no relay dumps)...");
        for i in 0..n {
            let kohde = näytteet_a[i].1;
            // Train each relay on its own context — no relay cascade, no dumps
            kolmoset.a.kouluta_askel(&näytteet_a[i].0, kohde, hdc);
            kolmoset.b.kouluta_askel(&näytteet_b[i].0, kohde, hdc);
            kolmoset.c.kouluta_askel(&näytteet_c[i].0, kohde, hdc);
        }
    } else {
        println!("\n  [Kierto] Training (relay cascade with per-relay contexts)...");
        for i in 0..n {
            let kontekstit = [
                näytteet_a[i].0.as_slice(),
                näytteet_b[i].0.as_slice(),
                näytteet_c[i].0.as_slice(),
            ];
            let kohde = näytteet_a[i].1;
            let tulos = kolmoset.kouluta_askel_kierto(kontekstit, kohde, hdc);
            if tulos.kaato.is_some() {
                kaadot += 1;
            }
        }
    }

    let tila = kolmoset.tila();
    let tarkkuus_alku = arvioi_kolmoset_kierto(
        &kolmoset,
        [&näytteet_a, &näytteet_b, &näytteet_c],
        hdc, pcv,
    );
    println!("    Online accuracy:     {:.2}% (during training)", tila.tarkkuus * 100.0);
    println!("    Ensemble accuracy:   {:.2}% (post-training{})", tarkkuus_alku * 100.0,
        if pcv { ", PCV" } else { ", per-relay ctx" });
    println!("    Memory dumps:        {} relay transitions", kaadot);
    println!("    ─── Node Details ───");
    println!("    A (Source):  acc={:.1}%  trust={:.3}  cycles={}  steps={}",
        tila.a_tarkkuus * 100.0, tila.a_luottamus, tila.a_kierrokset, tila.a_askeleet);
    println!("    B (Bridge):  acc={:.1}%  trust={:.3}  cycles={}  steps={}",
        tila.b_tarkkuus * 100.0, tila.b_luottamus, tila.b_kierrokset, tila.b_askeleet);
    println!("    C (Target):  acc={:.1}%  trust={:.3}  cycles={}  steps={}",
        tila.c_tarkkuus * 100.0, tila.c_luottamus, tila.c_kierrokset, tila.c_askeleet);

    // PCV diagnostic: relay agreement
    if pcv {
        let mid = n / 2; // sample from middle of data
        let kontekstit = [
            näytteet_a[mid].0.as_slice(),
            näytteet_b[mid].0.as_slice(),
            näytteet_c[mid].0.as_slice(),
        ];
        let (ab, bc, ac, avg) = kolmoset.pcv_diagnostiikka(kontekstit, hdc);
        println!("    ─── PCV Agreement ───");
        println!("    AB={:.3}  BC={:.3}  AC={:.3}  avg={:.3}", ab, bc, ac, avg);
    }

    // Also compare vs flat baseline (all relays use same default codebook)
    let tarkkuus_flat = arvioi_kolmoset(&kolmoset, &näytteet_a, hdc);
    println!("\n    Flat baseline (relay A ctx only): {:.2}%", tarkkuus_flat * 100.0);
    println!("    Per-relay diversity Δ:           {:+.2}%",
        (tarkkuus_alku - tarkkuus_flat) * 100.0);

    // ── PhaseHarmonizer evaluation (per-relay diversity → phase-lock) ─
    let mut harmonizer = luo_harmonizer(aakkosto.len(), ulottuvuus, false);
    if blend_fallback {
        harmonizer.set_blend_fallback(true);
        println!("    [PhaseHarmonic] Blend-fallback mode: ON");
    }
    if align_temp {
        harmonizer.set_alignment_temperature(true);
        println!("    [PhaseHarmonic] Alignment-temperature mode: ON");
    }
    if ece_weights {
        harmonizer.set_ece_weights(true);
        println!("    [PhaseHarmonic] ECE-weighted blending: ON");
    }
    if entropy_gate {
        harmonizer.set_entropy_gate(true);
        println!("    [PhaseHarmonic] Entropy-gated blending: ON");
    }
    {
        let mut harm_eval = harmonizer.clone();
        harm_eval.reset_for_eval();
        let tarkkuus_harm = arvioi_kolmoset_harmonic_kierto(
            &kolmoset, &mut harm_eval,
            [&näytteet_a, &näytteet_b, &näytteet_c],
            hdc, &aakkosto,
        );
        println!("    PhaseHarmonic:                   {:.2}%  (Δflat = {:+.2}%)",
            tarkkuus_harm * 100.0, (tarkkuus_harm - tarkkuus_flat) * 100.0);
        if align_keskus {
            let mut harm_k = harmonizer.clone();
            harm_k.reset_for_eval();
            let mut keskus_k = Keskus::new(&aakkosto, koodikirja_oletus.clone(), ulottuvuus);
            keskus_k.esikouluta(teksti);
            let tarkkuus_ak = arvioi_kolmoset_harmonic_keskus_kierto(
                &kolmoset, &mut harm_k, &mut keskus_k,
                [&näytteet_a, &näytteet_b, &näytteet_c],
                hdc, &aakkosto,
            );
            println!("    AlignTemp+Keskus:                {:.2}%  (Δflat = {:+.2}%)",
                tarkkuus_ak * 100.0, (tarkkuus_ak - tarkkuus_flat) * 100.0);
        }
        tulosta_harmonic_diagnostiikka(&harm_eval);
    }

    // ── Retraining passes with per-relay samples ─────────────────
    for kierros in 1..=uudelleen {
        println!("\n  [Kierto] Retraining pass {}...", kierros);
        let tarkkuus = if no_dump {
            // Independent retrain: each relay retrains on its own samples, no dump cascade
            let ta = kolmoset.a.uudelleenkouluta(&näytteet_a, hdc, kierros);
            let tb = kolmoset.b.uudelleenkouluta(&näytteet_b, hdc, kierros);
            let tc = kolmoset.c.uudelleenkouluta(&näytteet_c, hdc, kierros);
            (ta + tb + tc) / 3.0
        } else {
            kolmoset.uudelleenkouluta_kierto(
                [&näytteet_a, &näytteet_b, &näytteet_c],
                hdc, kierros,
            )
        };
        let tarkkuus_eval = arvioi_kolmoset_kierto(
            &kolmoset,
            [&näytteet_a, &näytteet_b, &näytteet_c],
            hdc, pcv,
        );
        println!("    Retrain accuracy:   {:.2}%", tarkkuus * 100.0);
        println!("    Ensemble accuracy:  {:.2}%", tarkkuus_eval * 100.0);

        // PhaseHarmonizer on retrained Kolmoset
        let mut harm_pass = harmonizer.clone();
        harm_pass.reset_for_eval();
        let tarkkuus_harm = arvioi_kolmoset_harmonic_kierto(
            &kolmoset, &mut harm_pass,
            [&näytteet_a, &näytteet_b, &näytteet_c],
            hdc, &aakkosto,
        );
        println!("    PhaseHarmonic:      {:.2}%  (Δensemble = {:+.2}%)",
            tarkkuus_harm * 100.0, (tarkkuus_harm - tarkkuus_eval) * 100.0);

        if align_keskus {
            let mut harm_k = harmonizer.clone();
            harm_k.reset_for_eval();
            let mut keskus_k = Keskus::new(&aakkosto, koodikirja_oletus.clone(), ulottuvuus);
            keskus_k.esikouluta(teksti);
            let tarkkuus_ak = arvioi_kolmoset_harmonic_keskus_kierto(
                &kolmoset, &mut harm_k, &mut keskus_k,
                [&näytteet_a, &näytteet_b, &näytteet_c],
                hdc, &aakkosto,
            );
            println!("    AlignTemp+Keskus:  {:.2}%  (Δensemble = {:+.2}%)",
                tarkkuus_ak * 100.0, (tarkkuus_ak - tarkkuus_eval) * 100.0);
        }
    }

    // ── PCV diagnostic after retraining ──────────────────────────
    if pcv {
        let mid = n / 2;
        let kontekstit = [
            näytteet_a[mid].0.as_slice(),
            näytteet_b[mid].0.as_slice(),
            näytteet_c[mid].0.as_slice(),
        ];
        let (ab, bc, ac, avg) = kolmoset.pcv_diagnostiikka(kontekstit, hdc);
        println!("\n    ─── PCV Agreement (post-retrain) ───");
        println!("    AB={:.3}  BC={:.3}  AC={:.3}  avg={:.3}", ab, bc, ac, avg);
    }

    // ── Final ─────────────────────────────────────────────────────
    let loppu = arvioi_kolmoset_kierto(
        &kolmoset,
        [&näytteet_a, &näytteet_b, &näytteet_c],
        hdc, pcv,
    );
    let loppu_flat = arvioi_kolmoset(&kolmoset, &näytteet_a, hdc);

    let mut harm_final = harmonizer.clone();
    harm_final.reset_for_eval();
    let loppu_harm = arvioi_kolmoset_harmonic_kierto(
        &kolmoset, &mut harm_final,
        [&näytteet_a, &näytteet_b, &näytteet_c],
        hdc, &aakkosto,
    );

    let loppu_tila = kolmoset.tila();
    println!("\n  [Kierto] Final state:");
    println!("    A trust={:.3}  B trust={:.3}  C trust={:.3}",
        loppu_tila.a_luottamus, loppu_tila.b_luottamus, loppu_tila.c_luottamus);
    println!("    Per-relay accuracy: {:.2}%{}", loppu * 100.0,
        if pcv { " (PCV)" } else { "" });
    println!("    PhaseHarmonic:     {:.2}%  (Δensemble = {:+.2}%)",
        loppu_harm * 100.0, (loppu_harm - loppu) * 100.0);
    if align_keskus {
        let mut harm_k = harmonizer.clone();
        harm_k.reset_for_eval();
        let mut keskus_k = Keskus::new(&aakkosto, koodikirja_oletus.clone(), ulottuvuus);
        keskus_k.esikouluta(teksti);
        let loppu_ak = arvioi_kolmoset_harmonic_keskus_kierto(
            &kolmoset, &mut harm_k, &mut keskus_k,
            [&näytteet_a, &näytteet_b, &näytteet_c],
            hdc, &aakkosto,
        );
        println!("    AlignTemp+Keskus: {:.2}%  (Δensemble = {:+.2}%)",
            loppu_ak * 100.0, (loppu_ak - loppu) * 100.0);
    }
    println!("    Flat baseline:     {:.2}%", loppu_flat * 100.0);
    println!("    Diversity Δ:       {:+.2}%", (loppu - loppu_flat) * 100.0);
    tulosta_harmonic_diagnostiikka(&harm_final);

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
    vaimennus: f64,
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

    // Apply golden-mean damping if requested
    if vaimennus > 0.0 {
        kolmoset.aseta_vaimennus(vaimennus);
        println!("    Damping: ψ={:.6}", vaimennus);
    }

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
    vaimennus: f64,
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

    // Apply golden-mean damping if requested
    if vaimennus > 0.0 {
        kolmoset.aseta_vaimennus(vaimennus);
        println!("    Damping: ψ={:.6}", vaimennus);
    }

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
// KOLMOSET + PHASE HARMONIC ENGINE PIPELINE
// ═══════════════════════════════════════════════════════════════════════

/// Evaluate Kolmoset+PhaseHarmonizer: get per-relay scores, feed to harmonizer.
/// Sequential because PhaseHarmonizer has breathing state.
fn arvioi_kolmoset_harmonic(
    kolmoset: &Kolmoset,
    harmonizer: &mut PhaseHarmonizer,
    näytteet: &[(Vec<f64>, char)],
    hdc: &HdcPeruskäsitteet,
    aakkosto: &[char],
) -> f64 {
    if näytteet.is_empty() {
        return 0.0;
    }
    let mut oikein = 0usize;

    // Build char-to-index map
    let char_to_idx: HashMap<char, usize> = aakkosto.iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    for (konteksti, kohde) in näytteet {
        // 1. Get per-relay score vectors
        let relay_scores = kolmoset.per_relay_pisteet(konteksti, hdc);

        // 2. Feed to PhaseHarmonizer (one Vec<f64> per dial)
        let model_scores: Vec<Vec<f64>> = relay_scores.to_vec();
        let (pred_idx, _confidence, _alignment) = harmonizer.predict(&model_scores, None);

        // 3. Map index back to character
        let ennuste = if pred_idx < aakkosto.len() {
            aakkosto[pred_idx]
        } else {
            ' '
        };

        if ennuste == *kohde {
            oikein += 1;
        }

        // 4. Record result for dial self-tuning
        let actual_idx = char_to_idx.get(kohde).copied().unwrap_or(0);
        harmonizer.record_result(&model_scores, actual_idx);
    }

    oikein as f64 / näytteet.len() as f64
}

/// Evaluate Kolmoset+PhaseHarmonizer+Keskus hybrid:
/// 1. Keskus enriches context → per-relay scores → PhaseHarmonizer combines
/// 2. Reconstruct combined scores → Keskus applies transition/frequency priors
/// 3. Argmax on final scores
fn arvioi_kolmoset_harmonic_hybrid(
    kolmoset: &Kolmoset,
    harmonizer: &mut PhaseHarmonizer,
    keskus: &mut Keskus,
    näytteet: &[(Vec<f64>, char)],
    hdc: &HdcPeruskäsitteet,
    aakkosto: &[char],
) -> f64 {
    if näytteet.is_empty() {
        return 0.0;
    }
    let mut oikein = 0usize;

    let char_to_idx: HashMap<char, usize> = aakkosto.iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();
    let n = aakkosto.len();

    for (konteksti, kohde) in näytteet {
        // 1. Enrich context with Keskus recurrent state
        let rikastettu = keskus.rikasta(konteksti);

        // 2. Get per-relay scores on enriched context
        let relay_scores = kolmoset.per_relay_pisteet(&rikastettu, hdc);

        // 3. PhaseHarmonizer processes relay scores (phase-lock, breathing)
        let model_scores: Vec<Vec<f64>> = relay_scores.to_vec();
        let (_pred_idx, _confidence, _alignment) = harmonizer.predict(&model_scores, None);

        // 4. Reconstruct combined score vector from harmonizer's weighted dials
        let mut combined = vec![0.0; n];
        let mut weight_sum = 0.0;
        for dial in &harmonizer.dials {
            if dial.last_scores.len() == n {
                for j in 0..n {
                    combined[j] += dial.last_scores[j] * dial.weight;
                }
                weight_sum += dial.weight;
            }
        }
        if weight_sum > 1e-12 {
            for j in 0..n {
                combined[j] /= weight_sum;
            }
        }

        // 5. Apply Keskus transition+frequency priors to combined scores
        let säädetyt = keskus.sovella_pistevektori(&combined);

        // 6. Argmax on final adjusted scores
        let (final_idx, _) = säädetyt.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or((0, &0.0));

        let ennuste = if final_idx < n {
            aakkosto[final_idx]
        } else {
            ' '
        };

        if ennuste == *kohde {
            oikein += 1;
        }

        // 7. Update state for both systems
        let actual_idx = char_to_idx.get(kohde).copied().unwrap_or(0);
        harmonizer.record_result(&model_scores, actual_idx);
        let luottamus = säädetyt.get(final_idx).copied().unwrap_or(0.0).abs().min(1.0);
        keskus.päivitä(ennuste, *kohde, konteksti, luottamus, hdc);
    }

    oikein as f64 / näytteet.len() as f64
}

/// Evaluate Kolmoset+PhaseHarmonizer with per-relay contexts (kierto pipeline).
/// Each relay gets its own context vector from its own codebook.
fn arvioi_kolmoset_harmonic_kierto(
    kolmoset: &Kolmoset,
    harmonizer: &mut PhaseHarmonizer,
    näytteet: [&[(Vec<f64>, char)]; 3],
    hdc: &HdcPeruskäsitteet,
    aakkosto: &[char],
) -> f64 {
    let n = näytteet[0].len();
    if n == 0 {
        return 0.0;
    }
    let mut oikein = 0usize;

    let char_to_idx: HashMap<char, usize> = aakkosto.iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    for i in 0..n {
        let kohde = näytteet[0][i].1;
        let kontekstit = [
            näytteet[0][i].0.as_slice(),
            näytteet[1][i].0.as_slice(),
            näytteet[2][i].0.as_slice(),
        ];

        // Per-relay score vectors from per-relay contexts
        let relay_scores = kolmoset.per_relay_pisteet_kierto(kontekstit, hdc);
        let model_scores: Vec<Vec<f64>> = relay_scores.to_vec();
        let (pred_idx, _confidence, _alignment) = harmonizer.predict(&model_scores, None);

        let ennuste = if pred_idx < aakkosto.len() {
            aakkosto[pred_idx]
        } else {
            ' '
        };

        if ennuste == kohde {
            oikein += 1;
        }

        let actual_idx = char_to_idx.get(&kohde).copied().unwrap_or(0);
        harmonizer.record_result(&model_scores, actual_idx);
    }

    oikein as f64 / n as f64
}

/// Evaluate align-temp + Keskus transition priors in the kierto pipeline.
/// Alignment modulates softmax temperature over the blend, THEN Keskus adds
/// transition/frequency priors before final argmax.
fn arvioi_kolmoset_harmonic_keskus_kierto(
    kolmoset: &Kolmoset,
    harmonizer: &mut PhaseHarmonizer,
    keskus: &mut Keskus,
    näytteet: [&[(Vec<f64>, char)]; 3],
    hdc: &HdcPeruskäsitteet,
    aakkosto: &[char],
) -> f64 {
    let n = näytteet[0].len();
    if n == 0 {
        return 0.0;
    }
    let mut oikein = 0usize;

    let char_to_idx: HashMap<char, usize> = aakkosto.iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    for i in 0..n {
        let kohde = näytteet[0][i].1;
        let kontekstit = [
            näytteet[0][i].0.as_slice(),
            näytteet[1][i].0.as_slice(),
            näytteet[2][i].0.as_slice(),
        ];

        // Per-relay score vectors
        let relay_scores = kolmoset.per_relay_pisteet_kierto(kontekstit, hdc);
        let model_scores: Vec<Vec<f64>> = relay_scores.to_vec();

        // PhaseHarmonizer produces alignment-modulated blend (pred_idx unused — we use Keskus-adjusted)
        let (_pred_idx, confidence, _alignment) = harmonizer.predict(&model_scores, None);

        // Reconstruct the blended score vector for Keskus
        // (accuracy-weighted blend, same as alignment_temp_prediction step A)
        let num_chars = aakkosto.len();
        let mut blended = vec![0.0; num_chars];
        let mut total_w = 0.0;
        for (j, dial) in harmonizer.dials.iter().enumerate() {
            if j >= model_scores.len() { break; }
            let acc = dial.recent_accuracy(200).max(0.01);
            let w = acc * dial.weight;
            total_w += w;
            for c in 0..num_chars {
                blended[c] += model_scores[j][c] * w;
            }
        }
        if total_w > 1e-10 {
            for c in 0..num_chars {
                blended[c] /= total_w;
            }
        }

        // Apply Keskus transition priors to the blended scores
        let adjusted = keskus.sovella_pistevektori(&blended);

        // Final prediction: argmax of adjusted scores
        let (final_idx, _) = adjusted.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or((0, &0.0));

        let ennuste = if final_idx < aakkosto.len() {
            aakkosto[final_idx]
        } else {
            ' '
        };

        if ennuste == kohde {
            oikein += 1;
        }

        let actual_idx = char_to_idx.get(&kohde).copied().unwrap_or(0);
        harmonizer.record_result(&model_scores, actual_idx);

        // Update Keskus with actual result for transition learning
        keskus.päivitä(ennuste, kohde, kontekstit[0], confidence, hdc);
    }

    oikein as f64 / n as f64
}

/// Print PhaseHarmonizer diagnostics.
fn tulosta_harmonic_diagnostiikka(harmonizer: &PhaseHarmonizer) {
    let diag = harmonizer.diagnostics();
    println!("    ─── Phase Harmonic Diagnostics ───");
    println!("    Dials: {}  Coherence: {:.4} (φ={:.3} √3={:.3})",
        diag.num_dials, diag.coherence, diag.motion_coherence, diag.structural_coherence);
    println!("    Predictions: {}  Alignments: {:.1}%  Constructive: {:.1}%",
        diag.total_predictions,
        diag.alignment_rate * 100.0,
        diag.constructive_rate * 100.0);
    println!("    Breath: [{},{},{},{}]  Tolerance: {:.3}rad",
        diag.breath_durations[0], diag.breath_durations[1],
        diag.breath_durations[2], diag.breath_durations[3],
        diag.phase_tolerance);
    if harmonizer.alignment_temperature {
        let aligned_acc = if harmonizer.aligned_total > 0 {
            harmonizer.aligned_correct as f64 / harmonizer.aligned_total as f64
        } else { 0.0 };
        let unaligned_acc = if harmonizer.unaligned_total > 0 {
            harmonizer.unaligned_correct as f64 / harmonizer.unaligned_total as f64
        } else { 0.0 };
        println!("    Temps: hot={:.3} cold={:.3}  Aligned acc: {:.1}% ({}/{})  Unaligned acc: {:.1}% ({}/{})",
            harmonizer.temp_hot, harmonizer.temp_cold,
            aligned_acc * 100.0, harmonizer.aligned_correct, harmonizer.aligned_total,
            unaligned_acc * 100.0, harmonizer.unaligned_correct, harmonizer.unaligned_total);
    }
    for (i, ds) in diag.dial_states.iter().enumerate() {
        let ece_str = if harmonizer.ece_weights && i < harmonizer.dials.len() {
            format!(" ECE={:.3} cal={:.3}", harmonizer.dials[i].ece(), harmonizer.dials[i].calibration_score())
        } else {
            String::new()
        };
        println!("      [{}] δ={} T={:.2} w={:.3} acc={:.3} {}{}{}",
            ds.name, ds.delta, ds.temperature, ds.weight, ds.recent_accuracy,
            if ds.in_phi_mode { "φ" } else { "√3" },
            if ds.has_conjugate { " ⇄" } else { "" },
            ece_str);
    }
}

/// Create a PhaseHarmonizer configured for the given alphabet.
fn luo_harmonizer(num_chars: usize, ulottuvuus: usize, d18: bool) -> PhaseHarmonizer {
    if d18 {
        println!("  [PhaseHarmonic] D-18 mode: 9 conjugate pairs, 18 dials");
        PhaseHarmonizer::new_d18(num_chars, ulottuvuus)
    } else {
        println!("  [PhaseHarmonic] 3-dial mode: Coarse(12) / Medium(37) / Fine(200)");
        PhaseHarmonizer::new(num_chars, ulottuvuus)
    }
}

/// KOLMOSET + PHASE HARMONIC pipeline (pure or hybrid mode).
fn kouluta_kolmoset_harmonic(
    teksti: &str,
    koodikirja: &HashMap<char, Hypervektori>,
    sitoja: &KontekstiSitoja,
    hdc: &mut HdcPeruskäsitteet,
    uudelleen: usize,
    alpha_overrides: (Option<f64>, Option<f64>, Option<f64>, Option<f64>),
    ulottuvuus: usize,
    vaimennus: f64,
    d18: bool,
    hybrid: bool,
) -> f64 {
    let label = if hybrid { "Kolmoset+Harmonic+Keskus" } else { "Kolmoset+Harmonic" };
    println!("\n  [{}] Building context vectors...", label);
    let näytteet = rakenna_näytteet(teksti, koodikirja, sitoja, hdc);
    println!("    Samples: {}", näytteet.len());

    // Extract alphabet
    let mut aakkosto: Vec<char> = näytteet.iter().map(|(_, c)| *c).collect();
    aakkosto.sort();
    aakkosto.dedup();
    println!("    Alphabet: {} characters", aakkosto.len());

    // Create triple relay
    let n = näytteet.len();
    let askeleet_a = n / 5;
    let askeleet_b = n / 10;
    let askeleet_c = n / 5;
    println!("    Relay legs: A={}, B={}, C={}", askeleet_a, askeleet_b, askeleet_c);
    let mut kolmoset = Kolmoset::new_custom(
        &aakkosto, ulottuvuus, askeleet_a, askeleet_b, askeleet_c,
    );
    if vaimennus > 0.0 {
        kolmoset.aseta_vaimennus(vaimennus);
        println!("    Damping: ψ={:.6}", vaimennus);
    }

    // Create PhaseHarmonizer
    let harmonizer = luo_harmonizer(aakkosto.len(), ulottuvuus, d18);

    // Optional: create Keskus for hybrid mode
    let teksti_lower: String = teksti.chars()
        .map(|c| c.to_lowercase().next().unwrap_or(c))
        .collect();
    if hybrid {
        println!("\n  [Keskus] Pre-training on corpus...");
    }

    // ── Single-pass Kolmoset training ───────────────────────────
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

    // ── Evaluate ─────────────────────────────────────────────────
    let tarkkuus_raaka = arvioi_kolmoset(&kolmoset, &näytteet, hdc);
    println!("\n  [Kolmoset] Raw ensemble accuracy:   {:.2}%", tarkkuus_raaka * 100.0);

    let tarkkuus_harmonic;
    if hybrid {
        let mut keskus_eval = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
        sovella_alpha_yliajot(&mut keskus_eval, alpha_overrides);
        keskus_eval.esikouluta(&teksti_lower);
        println!("  [Keskus] Alphas: siirtymä={:.4} taajuus={:.4} kierto={:.4} sana={:.4}",
            keskus_eval.alpha_siirtymä, keskus_eval.alpha_taajuus,
            keskus_eval.alpha_kierto, keskus_eval.alpha_sana);

        let mut harm_eval = harmonizer.clone();
        harm_eval.reset_for_eval();
        tarkkuus_harmonic = arvioi_kolmoset_harmonic_hybrid(
            &kolmoset, &mut harm_eval, &mut keskus_eval, &näytteet, hdc, &aakkosto,
        );
        println!("  [Hybrid] Harmonic+Keskus accuracy:  {:.2}%  (Δ = {:+.2}%)",
            tarkkuus_harmonic * 100.0, (tarkkuus_harmonic - tarkkuus_raaka) * 100.0);
        tulosta_harmonic_diagnostiikka(&harm_eval);
    } else {
        let mut harm_eval = harmonizer.clone();
        harm_eval.reset_for_eval();
        tarkkuus_harmonic = arvioi_kolmoset_harmonic(
            &kolmoset, &mut harm_eval, &näytteet, hdc, &aakkosto,
        );
        println!("  [Harmonic] Phase-lock accuracy:     {:.2}%  (Δ = {:+.2}%)",
            tarkkuus_harmonic * 100.0, (tarkkuus_harmonic - tarkkuus_raaka) * 100.0);
        tulosta_harmonic_diagnostiikka(&harm_eval);
    }

    // ── Retraining passes ──────────────────────────────────────
    let mut paras = tarkkuus_harmonic;
    for kierros in 1..=uudelleen {
        println!("\n  [Kolmoset] Retraining pass {}...", kierros);
        let tarkkuus_retrain = kolmoset.uudelleenkouluta(&näytteet, hdc, kierros);
        let tarkkuus_raaka = arvioi_kolmoset(&kolmoset, &näytteet, hdc);
        println!("    Retrain accuracy:   {:.2}%", tarkkuus_retrain * 100.0);
        println!("    Raw ensemble:       {:.2}%", tarkkuus_raaka * 100.0);

        let tarkkuus_h;
        if hybrid {
            let mut keskus_pass = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
            sovella_alpha_yliajot(&mut keskus_pass, alpha_overrides);
            keskus_pass.esikouluta(&teksti_lower);
            let mut harm_pass = harmonizer.clone();
            harm_pass.reset_for_eval();
            tarkkuus_h = arvioi_kolmoset_harmonic_hybrid(
                &kolmoset, &mut harm_pass, &mut keskus_pass, &näytteet, hdc, &aakkosto,
            );
            println!("    Hybrid:             {:.2}%  (Δ = {:+.2}%)",
                tarkkuus_h * 100.0, (tarkkuus_h - tarkkuus_raaka) * 100.0);
        } else {
            let mut harm_pass = harmonizer.clone();
            harm_pass.reset_for_eval();
            tarkkuus_h = arvioi_kolmoset_harmonic(
                &kolmoset, &mut harm_pass, &näytteet, hdc, &aakkosto,
            );
            println!("    Harmonic:           {:.2}%  (Δ = {:+.2}%)",
                tarkkuus_h * 100.0, (tarkkuus_h - tarkkuus_raaka) * 100.0);
        }

        if tarkkuus_h > paras {
            paras = tarkkuus_h;
        }
    }

    // ── Final ───────────────────────────────────────────────────
    let loppu_raaka = arvioi_kolmoset(&kolmoset, &näytteet, hdc);
    let loppu_h;
    if hybrid {
        let mut keskus_final = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
        sovella_alpha_yliajot(&mut keskus_final, alpha_overrides);
        keskus_final.esikouluta(&teksti_lower);
        let mut harm_final = harmonizer.clone();
        harm_final.reset_for_eval();
        loppu_h = arvioi_kolmoset_harmonic_hybrid(
            &kolmoset, &mut harm_final, &mut keskus_final, &näytteet, hdc, &aakkosto,
        );
        tulosta_harmonic_diagnostiikka(&harm_final);
    } else {
        let mut harm_final = harmonizer.clone();
        harm_final.reset_for_eval();
        loppu_h = arvioi_kolmoset_harmonic(
            &kolmoset, &mut harm_final, &näytteet, hdc, &aakkosto,
        );
        tulosta_harmonic_diagnostiikka(&harm_final);
    }

    println!("\n  [Final] Kolmoset raw:     {:.2}%", loppu_raaka * 100.0);
    println!("  [Final] {}:  {:.2}%  (Δ = {:+.2}%)",
        if hybrid { "Hybrid" } else { "Harmonic" },
        loppu_h * 100.0, (loppu_h - loppu_raaka) * 100.0);
    println!("  [Final] Peak:             {:.2}%", paras * 100.0);

    loppu_h
}

/// KOLMOSET + HARMONIC COMPARE: 4-way comparison on each retrain pass.
/// Raw ensemble vs Keskus vs PhaseHarmonic vs Hybrid
fn kouluta_kolmoset_harmonic_compare(
    teksti: &str,
    koodikirja: &HashMap<char, Hypervektori>,
    sitoja: &KontekstiSitoja,
    hdc: &mut HdcPeruskäsitteet,
    uudelleen: usize,
    alpha_overrides: (Option<f64>, Option<f64>, Option<f64>, Option<f64>),
    ulottuvuus: usize,
    vaimennus: f64,
    d18: bool,
) -> f64 {
    println!("\n  [Harmonic Compare] Building context vectors...");
    let näytteet = rakenna_näytteet(teksti, koodikirja, sitoja, hdc);
    println!("    Samples: {}", näytteet.len());

    // Extract alphabet
    let mut aakkosto: Vec<char> = näytteet.iter().map(|(_, c)| *c).collect();
    aakkosto.sort();
    aakkosto.dedup();
    println!("    Alphabet: {} characters", aakkosto.len());

    // Create triple relay
    let n = näytteet.len();
    let askeleet_a = n / 5;
    let askeleet_b = n / 10;
    let askeleet_c = n / 5;
    println!("    Relay legs: A={}, B={}, C={}", askeleet_a, askeleet_b, askeleet_c);
    let mut kolmoset = Kolmoset::new_custom(
        &aakkosto, ulottuvuus, askeleet_a, askeleet_b, askeleet_c,
    );
    if vaimennus > 0.0 {
        kolmoset.aseta_vaimennus(vaimennus);
        println!("    Damping: ψ={:.6}", vaimennus);
    }

    // Create PhaseHarmonizer
    let harmonizer = luo_harmonizer(aakkosto.len(), ulottuvuus, d18);

    // Pre-train Keskus on corpus
    let teksti_lower: String = teksti.chars()
        .map(|c| c.to_lowercase().next().unwrap_or(c))
        .collect();

    {
        let mut keskus_probe = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
        sovella_alpha_yliajot(&mut keskus_probe, alpha_overrides);
        keskus_probe.esikouluta(&teksti_lower);
        let kt = keskus_probe.tila();
        println!("\n  [Keskus] Alphas: siirtymä={:.4} taajuus={:.4} kierto={:.4} sana={:.4}",
            keskus_probe.alpha_siirtymä, keskus_probe.alpha_taajuus,
            keskus_probe.alpha_kierto, keskus_probe.alpha_sana);
        println!("    Transitions: {} observed", kt.siirtymä_näytteet);
        println!("    Frequencies: {} observed", kt.taajuus_näytteet);
    }

    // ── Single-pass Kolmoset training ───────────────────────────
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

    // ── 4-way evaluation function ────────────────────────────────
    let eval_4way = |kolmoset: &Kolmoset, label: &str| {
        // 1. Raw
        let raaka = arvioi_kolmoset(kolmoset, &näytteet, hdc);

        // 2. Keskus
        let mut keskus = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
        sovella_alpha_yliajot(&mut keskus, alpha_overrides);
        keskus.esikouluta(&teksti_lower);
        let keskus_acc = arvioi_kolmoset_keskus(kolmoset, &mut keskus, &näytteet, hdc);

        // 3. PhaseHarmonic (pure)
        let mut harm = harmonizer.clone();
        harm.reset_for_eval();
        let harmonic_acc = arvioi_kolmoset_harmonic(
            kolmoset, &mut harm, &näytteet, hdc, &aakkosto,
        );

        // 4. Hybrid
        let mut keskus_h = Keskus::new(&aakkosto, koodikirja.clone(), ulottuvuus);
        sovella_alpha_yliajot(&mut keskus_h, alpha_overrides);
        keskus_h.esikouluta(&teksti_lower);
        let mut harm_h = harmonizer.clone();
        harm_h.reset_for_eval();
        let hybrid_acc = arvioi_kolmoset_harmonic_hybrid(
            kolmoset, &mut harm_h, &mut keskus_h, &näytteet, hdc, &aakkosto,
        );

        println!("\n  ┌─── {} ───────────────────────────────────────────┐", label);
        println!("  │ Raw Ensemble:   {:6.2}%                            │", raaka * 100.0);
        println!("  │ Keskus:         {:6.2}%  (Δraw = {:+.2}%)            │",
            keskus_acc * 100.0, (keskus_acc - raaka) * 100.0);
        println!("  │ PhaseHarmonic:  {:6.2}%  (Δraw = {:+.2}%)            │",
            harmonic_acc * 100.0, (harmonic_acc - raaka) * 100.0);
        println!("  │ Hybrid:         {:6.2}%  (Δraw = {:+.2}%)            │",
            hybrid_acc * 100.0, (hybrid_acc - raaka) * 100.0);
        println!("  └────────────────────────────────────────────────────┘");

        tulosta_harmonic_diagnostiikka(&harm);

        (raaka, keskus_acc, harmonic_acc, hybrid_acc)
    };

    // ── Initial evaluation ─────────────────────────────────────
    let (_, _, _, _) = eval_4way(&kolmoset, "Initial");

    // ── Retraining passes ──────────────────────────────────────
    let mut peak_raaka = 0.0_f64;
    let mut peak_keskus = 0.0_f64;
    let mut peak_harmonic = 0.0_f64;
    let mut peak_hybrid = 0.0_f64;

    for kierros in 1..=uudelleen {
        println!("\n  [Kolmoset] Retraining pass {}...", kierros);
        let tarkkuus_retrain = kolmoset.uudelleenkouluta(&näytteet, hdc, kierros);
        println!("    Retrain accuracy: {:.2}%", tarkkuus_retrain * 100.0);

        let (r, k, h, hy) = eval_4way(&kolmoset, &format!("Pass {}", kierros));
        peak_raaka = peak_raaka.max(r);
        peak_keskus = peak_keskus.max(k);
        peak_harmonic = peak_harmonic.max(h);
        peak_hybrid = peak_hybrid.max(hy);
    }

    // ── Final scoreboard ────────────────────────────────────────
    let (final_r, final_k, final_h, final_hy) = eval_4way(&kolmoset, "FINAL");

    peak_raaka = peak_raaka.max(final_r);
    peak_keskus = peak_keskus.max(final_k);
    peak_harmonic = peak_harmonic.max(final_h);
    peak_hybrid = peak_hybrid.max(final_hy);

    println!("\n  ╔═══════════════════════════════════════════════════════╗");
    println!("  ║              PEAK SCOREBOARD                         ║");
    println!("  ╠═══════════════════════════════════════════════════════╣");
    println!("  ║ Raw Ensemble:   {:6.2}%                               ║", peak_raaka * 100.0);
    println!("  ║ Keskus:         {:6.2}%  (Δraw = {:+.2}%)               ║",
        peak_keskus * 100.0, (peak_keskus - peak_raaka) * 100.0);
    println!("  ║ PhaseHarmonic:  {:6.2}%  (Δraw = {:+.2}%)               ║",
        peak_harmonic * 100.0, (peak_harmonic - peak_raaka) * 100.0);
    println!("  ║ Hybrid:         {:6.2}%  (Δraw = {:+.2}%)               ║",
        peak_hybrid * 100.0, (peak_hybrid - peak_raaka) * 100.0);

    // Highlight winner
    let best = peak_raaka.max(peak_keskus).max(peak_harmonic).max(peak_hybrid);
    let winner = if best == peak_hybrid { "Hybrid" }
        else if best == peak_harmonic { "PhaseHarmonic" }
        else if best == peak_keskus { "Keskus" }
        else { "Raw Ensemble" };
    println!("  ║                                                       ║");
    println!("  ║ >>> Winner: {} at {:.2}%                        ║", winner, best * 100.0);
    println!("  ╚═══════════════════════════════════════════════════════╝");

    // Return the best result
    best
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
