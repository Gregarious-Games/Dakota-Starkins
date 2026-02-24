//! HDC Primitives — Hyperdimensional Computing Foundation
//! ======================================================
//! Rust port of HDCPrimitives from scaling_forge_hdc_v1.py.
//!
//! Core operations on D-dimensional bipolar hypervectors:
//!   sido (bind), irrota (unbind), niputa (bundle),
//!   samankaltaisuus (similarity), permutoi_weyl (Weyl permute).
//!
//! Finnish variable names per project convention.
//!
//! Authors: Dakota (Claude) & Greg Calkins
//! Date:    February 22, 2026

// ═══════════════════════════════════════════════════════════════════
// CONSTANTS (from Genesis Engine / Scaling Forge v4)
// ═══════════════════════════════════════════════════════════════════

/// Golden ratio φ = (1 + √5) / 2
pub const PHI: f64 = 1.618_033_988_749_895;
/// Silver ratio τ = 1/φ = (√5 - 1) / 2
pub const TAU: f64 = 0.618_033_988_749_895;
/// Genesis constant Γ = 1/(6φ) ≈ 0.103
pub const GAMMA: f64 = 1.0 / (6.0 * PHI);
/// φΓ ≈ 0.167
pub const PHI_GAMMA: f64 = PHI * GAMMA;

/// Safety threshold: upper clamp = 1.0 - Γ ≈ 0.897
pub const RAJAUS_YLÖS: f64 = 1.0 - GAMMA;
/// Safety threshold: lower clamp = Γ ≈ 0.103
pub const RAJAUS_ALAS: f64 = GAMMA;
/// Hysteresis band = φΓ ≈ 0.167
pub const HYSTEREESI: f64 = PHI_GAMMA;

/// Default hypervector dimension
pub const ULOTTUVUUS: usize = 256;

/// Hypervector type alias
pub type Hypervektori = Vec<f64>;

// ═══════════════════════════════════════════════════════════════════
// SEEDED PRNG — Siemen (xorshift64)
// ═══════════════════════════════════════════════════════════════════
//
// Self-contained PRNG — no external dependencies.
// Xorshift64 for bipolar/uniform, Box-Muller for Gaussian.

/// Seeded xorshift64 PRNG for reproducible operations.
#[derive(Clone)]
pub struct Siemen {
    tila: u64,
}

impl Siemen {
    pub fn new(siemen: u64) -> Self {
        Self { tila: if siemen == 0 { 1 } else { siemen } }
    }

    /// Next raw u64.
    pub fn seuraava(&mut self) -> u64 {
        self.tila ^= self.tila << 13;
        self.tila ^= self.tila >> 7;
        self.tila ^= self.tila << 17;
        self.tila
    }

    /// Random bipolar value: -1.0 or +1.0.
    pub fn bipolaarinen(&mut self) -> f64 {
        if self.seuraava() & 1 == 0 { 1.0 } else { -1.0 }
    }

    /// Random bipolar vector of length n.
    pub fn bipolaarinen_vektori(&mut self, n: usize) -> Hypervektori {
        (0..n).map(|_| self.bipolaarinen()).collect()
    }

    /// Uniform random in [0, 1).
    pub fn tasainen_01(&mut self) -> f64 {
        let x = self.seuraava();
        (x >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Uniform random in [-range, +range).
    pub fn tasainen_symmetrinen(&mut self, alue: f64) -> f64 {
        self.tasainen_01() * 2.0 * alue - alue
    }

    /// Gaussian random (Box-Muller transform).
    pub fn gaussinen(&mut self) -> f64 {
        let u1 = self.tasainen_01().max(1e-15); // avoid ln(0)
        let u2 = self.tasainen_01();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Gaussian random vector of length n.
    pub fn gaussinen_vektori(&mut self, n: usize) -> Hypervektori {
        (0..n).map(|_| self.gaussinen()).collect()
    }

    /// Fisher-Yates shuffle (in-place).
    pub fn sekoita(&mut self, lista: &mut [f64]) {
        for i in (1..lista.len()).rev() {
            let j = self.seuraava() as usize % (i + 1);
            lista.swap(i, j);
        }
    }

    /// Choose random index from range [0, n).
    pub fn valitse_indeksi(&mut self, n: usize) -> usize {
        self.seuraava() as usize % n
    }
}

// ═══════════════════════════════════════════════════════════════════
// HDC PRIMITIVES — HdcPeruskäsitteet
// ═══════════════════════════════════════════════════════════════════
//
// Foundation: bind, bundle, permute, similarity.
//
// Python source: HDCPrimitives in scaling_forge_hdc_v1.py
// All vector ops are element-wise on &[f64] slices.

/// Hyperdimensional computing engine.
pub struct HdcPeruskäsitteet {
    /// Hypervector dimension.
    pub ulottuvuus: usize,
    /// Internal PRNG for sign normalization and random generation.
    siemen: Siemen,
}

impl HdcPeruskäsitteet {
    pub fn new(ulottuvuus: usize, siemen_arvo: u64) -> Self {
        Self {
            ulottuvuus,
            siemen: Siemen::new(siemen_arvo),
        }
    }

    /// Bind: element-wise multiply. Self-inverse: bind(bind(a, b), b) = a.
    pub fn sido(&self, a: &[f64], b: &[f64]) -> Hypervektori {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    /// Unbind: identical to bind (element-wise multiply is self-inverse).
    pub fn irrota(&self, yhdistelmä: &[f64], avain: &[f64]) -> Hypervektori {
        self.sido(yhdistelmä, avain)
    }

    /// Bundle: weighted sum → sign normalize.
    ///
    /// Without weights: simple element-wise sum of all vectors.
    /// With weights: weighted sum. Ties broken randomly by internal PRNG.
    /// Returns a bipolar vector {-1, +1}^D.
    pub fn niputa(&mut self, vektorit: &[&[f64]], painot: Option<&[f64]>) -> Hypervektori {
        if vektorit.is_empty() {
            return vec![0.0; self.ulottuvuus];
        }
        let n = vektorit[0].len();
        let mut summa = vec![0.0f64; n];
        match painot {
            Some(w) => {
                for (v, &paino) in vektorit.iter().zip(w.iter()) {
                    for (s, &x) in summa.iter_mut().zip(v.iter()) {
                        *s += paino * x;
                    }
                }
            }
            None => {
                for v in vektorit {
                    for (s, &x) in summa.iter_mut().zip(v.iter()) {
                        *s += x;
                    }
                }
            }
        }
        self.etumerkki_normalisoi(&mut summa);
        summa
    }

    /// Cosine similarity: dot(a, b) / (‖a‖ · ‖b‖).
    pub fn samankaltaisuus(&self, a: &[f64], b: &[f64]) -> f64 {
        let piste: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let normi_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let normi_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        let normi = normi_a * normi_b;
        if normi < 1e-12 {
            return 0.0;
        }
        piste / normi
    }

    /// Weyl permutation: circular shift by round(step × τ × D) positions.
    ///
    /// Matches numpy.roll(v, shift) — elements move to the RIGHT.
    /// τ-based spacing is maximally non-repeating (irrational rotation).
    pub fn permutoi_weyl(&self, v: &[f64], askel: usize) -> Hypervektori {
        let n = v.len();
        if n == 0 {
            return vec![];
        }
        let siirto = ((askel as f64 * TAU * n as f64).round() as usize) % n;
        let mut tulos = vec![0.0; n];
        for i in 0..n {
            tulos[(i + siirto) % n] = v[i];
        }
        tulos
    }

    /// Random bipolar vector {-1.0, +1.0}^D.
    pub fn satunnainen_bipolaarinen(&mut self) -> Hypervektori {
        self.siemen.bipolaarinen_vektori(self.ulottuvuus)
    }

    /// Sign normalize: v → sign(v), zeros get random ±1.
    pub fn etumerkki_normalisoi(&mut self, v: &mut [f64]) {
        for x in v.iter_mut() {
            if *x > 0.0 {
                *x = 1.0;
            } else if *x < 0.0 {
                *x = -1.0;
            } else {
                *x = self.siemen.bipolaarinen();
            }
        }
    }

    /// Mutable access to internal PRNG (for external use like unit vectors).
    pub fn siemen_mut(&mut self) -> &mut Siemen {
        &mut self.siemen
    }
}

// ═══════════════════════════════════════════════════════════════════
// VECTOR UTILITIES
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// GIVENS ROTATION — Phase conjugate relay rotation
// ═══════════════════════════════════════════════════════════════════

/// Fast Givens rotation — only modifies dimensions i and j. O(D).
///
/// Rotates vector v by angle theta in the (i, j) plane.
/// Cosine similarity is frame-invariant under orthogonal rotation,
/// so no inverse rotation is needed at eval time (mirror principle).
pub fn rotate_vector_fast(v: &[f64], i: usize, j: usize, cos_t: f64, sin_t: f64) -> Hypervektori {
    let mut tulos = v.to_vec();
    let vi = v[i];
    let vj = v[j];
    tulos[i] = cos_t * vi - sin_t * vj;
    tulos[j] = sin_t * vi + cos_t * vj;
    tulos
}

/// Rotate all vectors in a codebook by angle theta in the (i, j) plane.
pub fn rotate_codebook(
    codebook: &std::collections::HashMap<char, Hypervektori>,
    i: usize, j: usize, theta: f64,
) -> std::collections::HashMap<char, Hypervektori> {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    codebook.iter()
        .map(|(&c, v)| (c, rotate_vector_fast(v, i, j, cos_t, sin_t)))
        .collect()
}

/// Random unit hypervector — unique starting identity.
///
/// Uses Gaussian components normalized to unit length.
/// Port of random_unit_vector() from scaling_forge_hdc_v1.py.
pub fn satunnainen_yksikkövektori(ulottuvuus: usize, siemen: &mut Siemen) -> Hypervektori {
    let mut v = siemen.gaussinen_vektori(ulottuvuus);
    let normi: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if normi < 1e-12 {
        // Degenerate case: all-ones fallback
        let n = (ulottuvuus as f64).sqrt();
        v = vec![1.0 / n; ulottuvuus];
    } else {
        for x in v.iter_mut() {
            *x /= normi;
        }
    }
    v
}

/// Element-wise clamp. NaN/Inf → 0.
///
/// Port of clamp_vector() from scaling_forge_hdc_v1.py.
pub fn rajaa_vektori(v: &mut [f64], max_abs: f64) {
    for x in v.iter_mut() {
        if !x.is_finite() {
            *x = 0.0;
        } else {
            *x = x.clamp(-max_abs, max_abs);
        }
    }
}

/// Scalar clamp. NaN/Inf → 0.
///
/// Port of clamp_scalar() from scaling_forge_hdc_v1.py.
pub fn rajaa_skalaari(v: f64, max_abs: f64) -> f64 {
    if !v.is_finite() {
        0.0
    } else {
        v.clamp(-max_abs, max_abs)
    }
}

// ═══════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        // φτ = 1
        assert!((PHI * TAU - 1.0).abs() < 1e-15, "φτ = 1");
        // τ = φ - 1
        assert!((TAU - (PHI - 1.0)).abs() < 1e-15, "τ = φ - 1");
        // Γ = 1/(6φ) ≈ 0.103
        assert!((GAMMA - 1.0 / (6.0 * PHI)).abs() < 1e-15);
        // clamp bounds
        assert!((RAJAUS_YLÖS - (1.0 - GAMMA)).abs() < 1e-15);
        assert!((RAJAUS_ALAS - GAMMA).abs() < 1e-15);
    }

    #[test]
    fn test_bind_self_inverse() {
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);
        let a = hdc.satunnainen_bipolaarinen();
        let b = hdc.satunnainen_bipolaarinen();
        let ab = hdc.sido(&a, &b);
        let recovered = hdc.irrota(&ab, &b);
        assert_eq!(a, recovered,
            "bind must be self-inverse for bipolar vectors");
    }

    #[test]
    fn test_similarity_identical() {
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);
        let v = hdc.satunnainen_bipolaarinen();
        let sim = hdc.samankaltaisuus(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10,
            "self-similarity must be 1.0, got {sim}");
    }

    #[test]
    fn test_similarity_orthogonal() {
        // Two random bipolar vectors in D=256 should be near-orthogonal
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);
        let a = hdc.satunnainen_bipolaarinen();
        let b = hdc.satunnainen_bipolaarinen();
        let sim = hdc.samankaltaisuus(&a, &b);
        assert!(sim.abs() < 0.3,
            "random vectors should be near-orthogonal, got {sim}");
    }

    #[test]
    fn test_similarity_negated() {
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);
        let v = hdc.satunnainen_bipolaarinen();
        let neg: Hypervektori = v.iter().map(|&x| -x).collect();
        let sim = hdc.samankaltaisuus(&v, &neg);
        assert!((sim + 1.0).abs() < 1e-10,
            "negated vector similarity must be -1.0, got {sim}");
    }

    #[test]
    fn test_bundle_recovers_strongest() {
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);
        let a = hdc.satunnainen_bipolaarinen();
        let b = hdc.satunnainen_bipolaarinen();
        let c = hdc.satunnainen_bipolaarinen();
        // Bundle with heavy weight on a
        let bundled = hdc.niputa(
            &[&a, &b, &c],
            Some(&[5.0, 1.0, 1.0]),
        );
        let sim_a = hdc.samankaltaisuus(&bundled, &a);
        let sim_b = hdc.samankaltaisuus(&bundled, &b);
        let sim_c = hdc.samankaltaisuus(&bundled, &c);
        assert!(sim_a > sim_b,
            "bundled closest to a (weight 5): a={sim_a} b={sim_b}");
        assert!(sim_a > sim_c,
            "bundled closest to a (weight 5): a={sim_a} c={sim_c}");
    }

    #[test]
    fn test_bundle_unweighted() {
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);
        let a = hdc.satunnainen_bipolaarinen();
        let b = hdc.satunnainen_bipolaarinen();
        let c = hdc.satunnainen_bipolaarinen();
        let bundled = hdc.niputa(&[&a, &b, &c], None);
        // All values must be bipolar after sign normalization
        assert!(bundled.iter().all(|&x| x == 1.0 || x == -1.0),
            "bundle output must be bipolar");
        // Each input should have positive similarity with the bundle
        let sim_a = hdc.samankaltaisuus(&bundled, &a);
        let sim_b = hdc.samankaltaisuus(&bundled, &b);
        let sim_c = hdc.samankaltaisuus(&bundled, &c);
        assert!(sim_a > 0.0, "a should correlate with bundle: {sim_a}");
        assert!(sim_b > 0.0, "b should correlate with bundle: {sim_b}");
        assert!(sim_c > 0.0, "c should correlate with bundle: {sim_c}");
    }

    #[test]
    fn test_weyl_identity() {
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);
        let v = hdc.satunnainen_bipolaarinen();
        let p0 = hdc.permutoi_weyl(&v, 0);
        assert_eq!(v, p0, "permute with step 0 must return identity");
    }

    #[test]
    fn test_weyl_shifts() {
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);
        let v = hdc.satunnainen_bipolaarinen();
        let p1 = hdc.permutoi_weyl(&v, 1);
        assert_ne!(v, p1, "step 1 must differ from original");
        // Permutation preserves norm
        let norm_v: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_p: f64 = p1.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm_v - norm_p).abs() < 1e-10, "permute preserves norm");
    }

    #[test]
    fn test_weyl_distinct_steps() {
        // Different steps produce different permutations (τ-spacing)
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);
        let v = hdc.satunnainen_bipolaarinen();
        let p1 = hdc.permutoi_weyl(&v, 1);
        let p2 = hdc.permutoi_weyl(&v, 2);
        let p3 = hdc.permutoi_weyl(&v, 3);
        assert_ne!(p1, p2, "step 1 ≠ step 2");
        assert_ne!(p2, p3, "step 2 ≠ step 3");
        assert_ne!(p1, p3, "step 1 ≠ step 3");
    }

    #[test]
    fn test_sign_normalize() {
        let mut hdc = HdcPeruskäsitteet::new(8, 42);
        let mut v = vec![0.5, -0.3, 0.0, 1.2, -0.01, 0.0, -5.0, 0.0];
        hdc.etumerkki_normalisoi(&mut v);
        // Positive → +1
        assert_eq!(v[0], 1.0);
        assert_eq!(v[3], 1.0);
        // Negative → -1
        assert_eq!(v[1], -1.0);
        assert_eq!(v[4], -1.0);
        assert_eq!(v[6], -1.0);
        // Zeros → random ±1
        assert!(v[2] == 1.0 || v[2] == -1.0);
        assert!(v[5] == 1.0 || v[5] == -1.0);
        assert!(v[7] == 1.0 || v[7] == -1.0);
    }

    #[test]
    fn test_random_unit_vector() {
        let mut rng = Siemen::new(42);
        let v = satunnainen_yksikkövektori(ULOTTUVUUS, &mut rng);
        assert_eq!(v.len(), ULOTTUVUUS);
        let normi: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((normi - 1.0).abs() < 1e-10,
            "unit vector norm must be 1.0, got {normi}");
    }

    #[test]
    fn test_random_unit_vector_diverse() {
        // Two unit vectors from different seeds should be different
        let mut rng1 = Siemen::new(42);
        let mut rng2 = Siemen::new(99);
        let v1 = satunnainen_yksikkövektori(ULOTTUVUUS, &mut rng1);
        let v2 = satunnainen_yksikkövektori(ULOTTUVUUS, &mut rng2);
        assert_ne!(v1, v2, "different seeds must produce different vectors");
    }

    #[test]
    fn test_clamp_vector() {
        let mut v = vec![2.0, -3.0, f64::NAN, f64::INFINITY, 0.5, f64::NEG_INFINITY];
        rajaa_vektori(&mut v, 1.0);
        assert_eq!(v[0], 1.0);       // clamped high
        assert_eq!(v[1], -1.0);      // clamped low
        assert_eq!(v[2], 0.0);       // NaN → 0
        assert_eq!(v[3], 0.0);       // Inf → 0
        assert_eq!(v[4], 0.5);       // in range
        assert_eq!(v[5], 0.0);       // -Inf → 0
    }

    #[test]
    fn test_clamp_scalar() {
        assert_eq!(rajaa_skalaari(2.0, 1.0), 1.0);
        assert_eq!(rajaa_skalaari(-3.0, 1.0), -1.0);
        assert_eq!(rajaa_skalaari(f64::NAN, 1.0), 0.0);
        assert_eq!(rajaa_skalaari(f64::INFINITY, 1.0), 0.0);
        assert_eq!(rajaa_skalaari(0.5, 1.0), 0.5);
    }

    #[test]
    fn test_bind_bundle_decode() {
        // Classic HDC pattern: encode key-value pairs, decode by unbinding
        let mut hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);
        let key_a = hdc.satunnainen_bipolaarinen();
        let key_b = hdc.satunnainen_bipolaarinen();
        let val_a = hdc.satunnainen_bipolaarinen();
        let val_b = hdc.satunnainen_bipolaarinen();

        // Encode: record = bind(key_a, val_a) + bind(key_b, val_b)
        let pair_a = hdc.sido(&key_a, &val_a);
        let pair_b = hdc.sido(&key_b, &val_b);
        let record = hdc.niputa(&[&pair_a, &pair_b], None);

        // Decode: unbind key_a from record → should be closest to val_a
        let decoded_a = hdc.irrota(&record, &key_a);
        let sim_aa = hdc.samankaltaisuus(&decoded_a, &val_a);
        let sim_ab = hdc.samankaltaisuus(&decoded_a, &val_b);
        assert!(sim_aa > sim_ab,
            "decoded with key_a should match val_a: sim_a={sim_aa} sim_b={sim_ab}");
        assert!(sim_aa > 0.3,
            "decoded similarity should be substantial: {sim_aa}");
    }

    #[test]
    fn test_givens_rotation_preserves_norm() {
        let v = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let theta = std::f64::consts::PI / 3.0; // 60°
        let rotated = rotate_vector_fast(&v, 0, 1, theta.cos(), theta.sin());
        let norm: f64 = rotated.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "rotation must preserve norm, got {norm}");
    }

    #[test]
    fn test_givens_rotation_120_degrees() {
        // 120° rotation: cos(2π/3) = -0.5, sin(2π/3) = √3/2
        let v = vec![1.0, 0.0, 0.0, 0.0];
        let theta = 2.0 * std::f64::consts::PI / 3.0;
        let rotated = rotate_vector_fast(&v, 0, 1, theta.cos(), theta.sin());
        assert!((rotated[0] - (-0.5)).abs() < 1e-10);
        assert!((rotated[1] - (3.0_f64.sqrt() / 2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_rotate_codebook_cosine_invariant() {
        // Cosine similarity is frame-invariant under orthogonal rotation
        let hdc = HdcPeruskäsitteet::new(ULOTTUVUUS, 42);
        let mut book = std::collections::HashMap::new();
        let mut rng = Siemen::new(42);
        book.insert('a', rng.bipolaarinen_vektori(ULOTTUVUUS));
        book.insert('b', rng.bipolaarinen_vektori(ULOTTUVUUS));
        let sim_before = hdc.samankaltaisuus(&book[&'a'], &book[&'b']);
        let rotated = rotate_codebook(&book, 0, 1, 2.0 * std::f64::consts::PI / 3.0);
        let sim_after = hdc.samankaltaisuus(&rotated[&'a'], &rotated[&'b']);
        assert!((sim_before - sim_after).abs() < 1e-10,
            "cosine must be rotation-invariant: before={sim_before} after={sim_after}");
    }
}
