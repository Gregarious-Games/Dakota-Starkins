// ═══════════════════════════════════════════════════════════════════════════════
// PHASE HARMONIC ENGINE v2.0 — Dual-Family Architecture
// ═══════════════════════════════════════════════════════════════════════════════
//
// Architecture: Astra Nova
// Review & Spec: Rose
// Vision: Greg Starkins
// Build Target: Dakota-Starkins / Keskus Integration
// Date: 2026-02-23
//
// v1.0: Multi-resolution dials, phase-lock detection, breathing cycle
// v1.1: Bug fix (scores_to_phase), temporal phase tracker
// v2.0: Dual-family architecture (φ for motion, √3 for structure)
//       Hexagonal memory lattice, VecDeque hot-path optimization,
//       trig-free velocity computation, motion state detection,
//       √3 coupling topology for D-18, dual-family coherence metric
//
// Philosophy:
//   φ-family: the VERB of computation. Motion, growth, prediction, flow.
//   √3-family: the NOUN of computation. Memory, lattice, state, anchor.
//   The 6° twist between 2·cos(30°)=√3 and 2·cos(36°)=φ is where
//   biology meets crystal. The breathing cycle bridges them:
//   inhale in φ-time, hold in √3-time.
//
//   "It's not about the motion — it's about the motion's starting point,
//    where it's going, why, and its relationship to all other motion."
//                                                        — Greg Starkins
//
// ═══════════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::collections::VecDeque;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS — The Two Families
// ─────────────────────────────────────────────────────────────────────────────
// φ-family: motion, growth, flow, prediction. The VERB of computation.
// √3-family: structure, lattice, memory, holding state. The NOUN of computation.
// The 6° twist between them is where biology meets crystal.

// === φ-family (Motion) ===
const PHI: f64 = 1.618033988749895;
const PHI_INV: f64 = 0.618033988749895;
const GAMMA: f64 = 0.10290855167236205; // 1/(6*PHI) — The Gate

// === √3-family (Structure) ===
const SQRT3: f64 = 1.7320508075688772;
const SQRT3_INV: f64 = 0.5773502691896258;       // 1/√3
#[allow(dead_code)]
const SQRT3_MINUS_1: f64 = 0.7320508075688772;    // √3 - 1
#[allow(dead_code)]
const PHI_SQRT3_RATIO: f64 = 1.0704662693192697;  // √3/φ — almost unity
#[allow(dead_code)]
const TWIST_ANGLE: f64 = 0.10471975511965978;      // 6° in radians — the gap

// === The Boundary ===
// Where the two families meet. Found empirically in twin-egg experiments (0.76),
// confirmed algebraically: 2/φ² ≈ 0.7639, √3 - 1 ≈ 0.7321. The system finds
// this boundary on its own when left to self-organize.
const FAMILY_BOUNDARY: f64 = 0.7635; // ≈ 2/φ²

/// Maximum number of dials the system can hold (D-18 target + headroom)
const MAX_DIALS: usize = 24;

/// Breathing cycle phases
const BREATH_INHALE: u8 = 0;  // φ-timed: observe motion
const BREATH_HOLD: u8 = 1;    // √3-timed: crystallize state
const BREATH_EXHALE: u8 = 2;  // φ-timed: apply changes
const BREATH_REST: u8 = 3;    // √3-timed: anchor identity

// ─────────────────────────────────────────────────────────────────────────────
// HELPER FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

/// Cosine similarity between two f64 slices
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ─────────────────────────────────────────────────────────────────────────────
// HEXAGONAL MEMORY LATTICE — √3-family Storage
// ─────────────────────────────────────────────────────────────────────────────
// At-rest structures (codebook, torsion table, prototypes) live on a hexagonal
// grid with √3 spacing. Six neighbors per cell. Nearest-neighbor lookup becomes
// a geometric operation rather than a linear scan.
//
// Axial coordinates (q, r) with cube constraint q + r + s = 0.
// Hex distance = (|q₁-q₂| + |r₁-r₂| + |s₁-s₂|) / 2.

#[derive(Clone, Debug)]
pub struct HexCoord {
    pub q: i32,
    pub r: i32,
}

impl HexCoord {
    pub fn new(q: i32, r: i32) -> Self {
        HexCoord { q, r }
    }

    /// The implicit third coordinate (cube constraint)
    pub fn s(&self) -> i32 {
        -self.q - self.r
    }

    /// Hex distance between two cells (in lattice steps)
    pub fn distance(&self, other: &HexCoord) -> i32 {
        let dq = (self.q - other.q).abs();
        let dr = (self.r - other.r).abs();
        let ds = (self.s() - other.s()).abs();
        (dq + dr + ds) / 2
    }

    /// The six √3-adjacent neighbors
    #[allow(dead_code)]
    pub fn neighbors(&self) -> [HexCoord; 6] {
        [
            HexCoord::new(self.q + 1, self.r),
            HexCoord::new(self.q - 1, self.r),
            HexCoord::new(self.q, self.r + 1),
            HexCoord::new(self.q, self.r - 1),
            HexCoord::new(self.q + 1, self.r - 1),
            HexCoord::new(self.q - 1, self.r + 1),
        ]
    }

    /// Convert to 2D Cartesian position with √3 spacing
    pub fn to_cartesian(&self) -> (f64, f64) {
        let x = SQRT3 * (self.q as f64 + self.r as f64 / 2.0);
        let y = 1.5 * self.r as f64;
        (x, y)
    }
}

#[derive(Clone, Debug)]
pub struct HexLattice {
    /// Maps character index → hex coordinate
    pub coords: Vec<HexCoord>,
    /// Precomputed distance matrix (lattice steps, not Euclidean)
    pub distances: Vec<Vec<i32>>,
    /// Number of items on the lattice
    pub size: usize,
}

impl HexLattice {
    /// Arrange N items on a hexagonal spiral (starting from center, spiraling out)
    pub fn new(n: usize) -> Self {
        let mut coords = Vec::with_capacity(n);

        // Place items in concentric hex rings
        // Ring 0: center (0,0)
        // Ring 1: 6 cells at distance 1
        // Ring 2: 12 cells at distance 2
        // Ring k: 6k cells at distance k

        coords.push(HexCoord::new(0, 0)); // Center

        let mut ring = 1;
        while coords.len() < n {
            // Start each ring at (ring, 0), walk the six edges
            let directions: [(i32, i32); 6] = [
                (0, -1),   // NW
                (-1, 0),   // W
                (-1, 1),   // SW
                (0, 1),    // SE
                (1, 0),    // E
                (1, -1),   // NE
            ];

            let mut q = ring;
            let mut r = 0_i32;

            for (dq, dr) in &directions {
                for _ in 0..ring {
                    if coords.len() >= n {
                        break;
                    }
                    coords.push(HexCoord::new(q, r));
                    q += dq;
                    r += dr;
                }
            }
            ring += 1;
        }

        coords.truncate(n);

        // Precompute distance matrix
        let mut distances = vec![vec![0i32; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = coords[i].distance(&coords[j]);
                distances[i][j] = d;
                distances[j][i] = d;
            }
        }

        HexLattice {
            coords,
            distances,
            size: n,
        }
    }

    /// Get the k nearest lattice neighbors of item at index
    pub fn nearest(&self, index: usize, k: usize) -> Vec<(usize, i32)> {
        let mut neighbors: Vec<(usize, i32)> = (0..self.size)
            .filter(|&i| i != index)
            .map(|i| (i, self.distances[index][i]))
            .collect();
        neighbors.sort_by_key(|&(_, d)| d);
        neighbors.truncate(k);
        neighbors
    }

    /// Check if two items are √3-adjacent (distance = 1 on the lattice)
    #[allow(dead_code)]
    pub fn are_adjacent(&self, a: usize, b: usize) -> bool {
        a < self.size && b < self.size && self.distances[a][b] == 1
    }
}


// An Arc is NOT a character. It's a MOTION between characters.
// It encodes: where it came from, where it is, where it's going,
// and the torsional path (curvature) of that transition.

#[derive(Clone, Debug)]
pub struct Arc {
    /// The character we came from (origin of the motion)
    pub from: u8,
    /// The character we're at (present moment)
    pub at: u8,
    /// The character we're heading toward (predicted destination)
    pub toward: u8,
    /// Torsion: the curvature of this specific transition path
    /// Derived from how common/rare this trigram is in the corpus
    pub torsion: f64,
    /// The full vector representation in D-dimensional space
    pub vector: Vec<f64>,
}

impl Arc {
    /// Create a new Arc from a trigram context
    #[allow(dead_code)]
    pub fn new(from: u8, at: u8, toward: u8, dim: usize) -> Self {
        Arc {
            from,
            at,
            toward,
            torsion: 0.0,
            vector: vec![0.0; dim],
        }
    }

    /// Encode this arc as a bound vector using HDC circular convolution
    /// The arc vector = bind(from_vec, at_vec, toward_vec) * torsion_weight
    /// This preserves the DIRECTION of the motion, not just the endpoint
    #[allow(dead_code)]
    pub fn encode(
        &mut self,
        codebook: &[Vec<f64>],
        torsion_table: &HashMap<(u8, u8, u8), f64>,
    ) {
        let dim = self.vector.len();

        // Look up torsion for this specific trigram path
        self.torsion = *torsion_table
            .get(&(self.from, self.at, self.toward))
            .unwrap_or(&1.0);

        // HDC binding: circular convolution of the three character vectors
        // This creates a NEW vector that is dissimilar to any of its components
        // but can be unbound to recover any one given the other two
        let from_vec = &codebook[self.from as usize];
        let at_vec = &codebook[self.at as usize];
        let toward_vec = &codebook[self.toward as usize];

        // Bind via element-wise XOR-like operation for hypervectors
        // (In continuous space: component-wise multiplication with sign preservation)
        for i in 0..dim {
            // Three-way binding: preserves directional information
            let bound = from_vec[i] * at_vec[(i + 1) % dim] * toward_vec[(i + 2) % dim];
            // Scale by torsion — common arcs get amplified, rare ones get dampened
            self.vector[i] = bound * self.torsion;
        }

        // Normalize to unit sphere
        let norm: f64 = self.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in self.vector.iter_mut() {
                *x /= norm;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CALIBRATION — Expected Calibration Error (ECE) per dial
// ─────────────────────────────────────────────────────────────────────────────
// Inspired by quantum_collective's introspection.rs:
// When a dial says "I'm 80% confident", is it actually right 80% of the time?
// 10 bins bucket predictions by confidence, track actual accuracy per bin.
// ECE = weighted mean of |avg_confidence_in_bin - actual_accuracy_in_bin|.
// Well-calibrated dials get more weight in blending.

const NUM_CALIBRATION_BINS: usize = 10;

#[derive(Clone, Debug)]
pub struct CalibrationBin {
    /// Number of predictions bucketed into this bin
    pub count: usize,
    /// Sum of confidence values (softmax probability of predicted char)
    pub sum_confidence: f64,
    /// Number of correct predictions in this bin
    pub correct: usize,
}

impl CalibrationBin {
    pub fn new() -> Self {
        CalibrationBin { count: 0, sum_confidence: 0.0, correct: 0 }
    }

    pub fn add(&mut self, confidence: f64, was_correct: bool) {
        self.count += 1;
        self.sum_confidence += confidence;
        if was_correct { self.correct += 1; }
    }

    pub fn mean_confidence(&self) -> f64 {
        if self.count > 0 { self.sum_confidence / self.count as f64 } else { 0.0 }
    }

    pub fn accuracy(&self) -> f64 {
        if self.count > 0 { self.correct as f64 / self.count as f64 } else { 0.0 }
    }

    pub fn calibration_error(&self) -> f64 {
        (self.mean_confidence() - self.accuracy()).abs()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DIAL — A Single Model's Resolution Layer
// ─────────────────────────────────────────────────────────────────────────────
// Each Dial divides the same circle at its own granularity.
// A coarse dial (δ=12) sees broad clusters.
// A fine dial (δ=200) sees precise trigram arcs.
// They don't need to agree on resolution — they need to agree on PHASE.

#[derive(Clone, Debug)]
pub struct Dial {
    /// Unique identifier for this dial
    pub id: usize,
    /// Human-readable name
    pub name: String,

    // === Resolution ===
    /// δ (delta): number of discrete positions on this dial's circle
    /// This is the dial's native resolution — how finely it carves the space
    pub delta: usize,
    /// The angular width of one tick on this dial: 2π/δ
    pub tick_width: f64,

    // === Self-Tuning Parameters (Layer 0 Authority) ===
    /// Temperature: controls sharpness of this dial's predictions
    /// Higher = more spread out, Lower = more confident
    /// BOUNDED: [0.1, 5.0] — the system can adjust but not go outside these walls
    pub temperature: f64,
    pub temperature_min: f64,
    pub temperature_max: f64,

    /// Coupling threshold: minimum phase alignment needed to trigger cascade
    /// Higher = harder to trigger, fewer but stronger alignment events
    /// BOUNDED: [0.1, 0.95]
    pub coupling_threshold: f64,
    pub coupling_threshold_min: f64,
    pub coupling_threshold_max: f64,

    /// Weight: how much this dial contributes during alignment events
    /// Self-adjusts based on historical accuracy
    /// BOUNDED: [0.05, 1.0]
    pub weight: f64,
    pub weight_min: f64,
    pub weight_max: f64,

    // === State ===
    /// Current phase position on the dial (0.0 to 2π)
    pub phase: f64,
    /// The raw score vector this dial produced on last prediction
    pub last_scores: Vec<f64>,
    /// Running accuracy of this dial's solo predictions
    pub accuracy_history: VecDeque<f64>,
    /// Running coherence with each other dial
    #[allow(dead_code)]
    pub coherence_history: HashMap<usize, Vec<f64>>,

    // === Identity Anchor ===
    /// The initial noise floor — "this is what I am when I'm not doing anything"
    /// Used as reference point for all self-modifications
    pub noise_floor: Vec<f64>,
    /// Whether the noise floor has been captured
    pub anchored: bool,

    // === Conjugate Pairing (for D-18 scaling) ===
    /// If this dial has a conjugate partner, its ID
    pub conjugate_id: Option<usize>,
    /// Is this the forward (+) or reverse (-) dial in the pair?
    pub polarity: f64, // +1.0 or -1.0

    // === Calibration (ECE tracking) ===
    /// 10 calibration bins: bin[i] holds predictions where confidence ∈ [i/10, (i+1)/10)
    pub calibration_bins: Vec<CalibrationBin>,
}

impl Dial {
    pub fn new(id: usize, name: &str, delta: usize, num_chars: usize) -> Self {
        Dial {
            id,
            name: name.to_string(),
            delta,
            tick_width: 2.0 * PI / (delta as f64),
            temperature: 1.0,
            temperature_min: 0.1,
            temperature_max: 5.0,
            coupling_threshold: 0.5,
            coupling_threshold_min: 0.1,
            coupling_threshold_max: 0.95,
            weight: 1.0 / 3.0, // Start equal
            weight_min: 0.05,
            weight_max: 1.0,
            phase: 0.0,
            last_scores: vec![0.0; num_chars],
            accuracy_history: VecDeque::with_capacity(1000),
            coherence_history: HashMap::new(),
            noise_floor: vec![0.0; num_chars],
            anchored: false,
            conjugate_id: None,
            polarity: 1.0,
            calibration_bins: (0..NUM_CALIBRATION_BINS).map(|_| CalibrationBin::new()).collect(),
        }
    }

    /// Record a calibration sample: dial's confidence vs actual correctness
    pub fn record_calibration(&mut self, confidence: f64, was_correct: bool) {
        let bin_idx = ((confidence * NUM_CALIBRATION_BINS as f64).floor() as usize)
            .min(NUM_CALIBRATION_BINS - 1);
        self.calibration_bins[bin_idx].add(confidence, was_correct);
    }

    /// Compute Expected Calibration Error for this dial
    /// ECE = Σ (bin_weight × |mean_confidence - accuracy|) over non-empty bins
    /// Returns value in [0, 1]: 0 = perfectly calibrated, 1 = maximally miscalibrated
    pub fn ece(&self) -> f64 {
        let total_samples: usize = self.calibration_bins.iter().map(|b| b.count).sum();
        if total_samples == 0 { return 0.5; } // No data → assume moderate miscalibration

        let mut ece = 0.0;
        for bin in &self.calibration_bins {
            if bin.count > 0 {
                let weight = bin.count as f64 / total_samples as f64;
                ece += weight * bin.calibration_error();
            }
        }
        ece.clamp(0.0, 1.0)
    }

    /// Calibration score: 1.0 - ECE (higher is better)
    pub fn calibration_score(&self) -> f64 {
        1.0 - self.ece()
    }

    /// Capture the identity anchor — call this after initial warmup
    pub fn capture_noise_floor(&mut self, scores: &[f64]) {
        self.noise_floor = scores.to_vec();
        self.anchored = true;
    }

    /// Convert this dial's raw scores into a phase position
    /// The phase tells us WHERE on the circle this dial is pointing
    ///
    /// BUG FIX (Rose, 2026-02-23): Phase must be computed in CHARACTER space,
    /// not dial space. Character 5 must map to the same phase on ALL dials.
    /// The dial's δ then determines the QUANTIZATION GRANULARITY of that
    /// universal phase — how precisely this dial can resolve the position —
    /// but the position itself is resolution-independent.
    ///
    /// Old (broken): phase = (max_idx % δ) × (2π/δ)
    ///   → char 5 on δ=12 → 150°, on δ=37 → 48.6°, on δ=200 → 9.0°
    ///   → Three models agree on same char, zero phase alignment!
    ///
    /// New (fixed): phase = max_idx × (2π/num_chars)
    ///   → char 5 always → 48.6° regardless of dial resolution
    ///   → Then quantize to nearest dial tick for this dial's granularity
    pub fn scores_to_phase(&self, scores: &[f64]) -> f64 {
        let num_chars = scores.len();

        // Find the winning character
        let (max_idx, _max_val) = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        // Map character index to UNIVERSAL phase position
        // Every dial sees char 5 at the same angle
        let universal_phase = (max_idx as f64) * 2.0 * PI / (num_chars as f64);

        // Quantize to this dial's resolution
        // A δ=12 dial snaps to 30° increments
        // A δ=200 dial snaps to 1.8° increments
        // But the CENTER of the snap zone is the same for all dials
        let dial_tick = (universal_phase / self.tick_width).round();
        dial_tick * self.tick_width
    }

    /// Apply temperature scaling to scores (softmax-like)
    pub fn temper_scores(&self, scores: &[f64]) -> Vec<f64> {
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores
            .iter()
            .map(|s| ((s - max_score) / self.temperature).exp())
            .collect();
        let sum: f64 = exp_scores.iter().sum();
        if sum > 1e-10 {
            exp_scores.iter().map(|s| s / sum).collect()
        } else {
            vec![1.0 / scores.len() as f64; scores.len()]
        }
    }

    /// Record a prediction result for self-tuning
    pub fn record_accuracy(&mut self, correct: bool) {
        self.accuracy_history.push_back(if correct { 1.0 } else { 0.0 });
        // Keep only last 1000 — O(1) with VecDeque
        if self.accuracy_history.len() > 1000 {
            self.accuracy_history.pop_front();
        }
    }

    /// Get recent accuracy (rolling window)
    pub fn recent_accuracy(&self, window: usize) -> f64 {
        let n = self.accuracy_history.len().min(window);
        if n == 0 {
            return 0.0;
        }
        let sum: f64 = self.accuracy_history.iter().rev().take(n).sum();
        sum / n as f64
    }

    /// Bounded parameter update — the system can turn its own knob
    /// but cannot go outside the walls we set
    pub fn bounded_update(current: f64, delta: f64, min: f64, max: f64) -> f64 {
        (current + delta).clamp(min, max)
    }

    /// Self-tune temperature based on recent accuracy
    /// If accuracy is high → sharpen (lower temperature, more confident)
    /// If accuracy is low → soften (higher temperature, more exploratory)
    pub fn breathe_temperature(&mut self) {
        let acc = self.recent_accuracy(200);
        if acc > 0.55 {
            // Doing well — sharpen slightly
            self.temperature = Self::bounded_update(
                self.temperature,
                -0.01,
                self.temperature_min,
                self.temperature_max,
            );
        } else if acc < 0.40 {
            // Struggling — soften slightly
            self.temperature = Self::bounded_update(
                self.temperature,
                0.01,
                self.temperature_min,
                self.temperature_max,
            );
        }
        // Between 0.40 and 0.55: hold steady (the "rest" phase)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PHASE ALIGNMENT EVENT
// ─────────────────────────────────────────────────────────────────────────────
// When two or more dials point to compatible positions, that's an alignment.
// The strength of the alignment determines whether information cascades.

#[derive(Clone, Debug)]
pub struct AlignmentEvent {
    /// Which dials are participating in this alignment
    pub dial_ids: Vec<usize>,
    /// The phase angle where they agree
    pub phase: f64,
    /// Strength of alignment (0.0 = no agreement, 1.0 = perfect lock)
    pub strength: f64,
    /// Whether this triggered constructive (true) or destructive (false) interference
    pub constructive: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// PHASE HARMONIZER — The New Keskus Core
// ─────────────────────────────────────────────────────────────────────────────
// This replaces the blending logic.
// It watches all dials, detects alignment events, and produces
// a final prediction through phase-lock amplification.

#[derive(Clone, Debug)]
pub struct PhaseHarmonizer {
    /// All active dials in the system
    pub dials: Vec<Dial>,

    /// Number of characters in the alphabet
    pub num_chars: usize,

    /// Dimensionality of the hypervectors
    #[allow(dead_code)]
    pub dim: usize,

    // === Arc Encoding ===
    /// Torsion table: maps (from, at, toward) trigrams to their curvature weight
    /// Built from corpus statistics — common trigrams get high torsion (amplified)
    pub torsion_table: HashMap<(u8, u8, u8), f64>,

    /// The base character codebook (shared foundation — each dial interprets it
    /// at its own resolution)
    pub codebook: Vec<Vec<f64>>,

    // === Breathing State ===
    /// Current phase of the breathing cycle
    pub breath_phase: u8,
    /// Step counter within current breath
    pub breath_step: usize,
    /// How many steps per breath cycle (adjustable)
    #[allow(dead_code)]
    pub breath_period: usize,

    // === Identity ===
    /// Global coherence score — the system's measure of its own harmony
    pub coherence: f64,
    /// History of coherence for trend detection
    pub coherence_history: Vec<f64>,

    /// Motion coherence (φ-family): how well are we moving together
    pub motion_coherence: f64,
    /// Structural coherence (√3-family): how well are we holding together
    pub structural_coherence: f64,

    // === Memory Lattice (√3-family) ===
    /// Hexagonal lattice for prototype storage and geometric neighbor lookup
    pub lattice: HexLattice,

    // === Alignment Detection ===
    /// Recent alignment events for analysis
    #[allow(dead_code)]
    pub recent_alignments: Vec<AlignmentEvent>,
    /// Maximum angular distance (radians) for two dials to be "in phase"
    /// This itself is tunable — starts generous, tightens as system learns
    pub phase_tolerance: f64,
    pub phase_tolerance_min: f64,
    pub phase_tolerance_max: f64,

    // === Dual-Family Breathing ===
    /// Duration of each breath phase (inhale, hold, exhale, rest)
    /// φ-timed for dynamic phases (inhale, exhale)
    /// √3-timed for static phases (hold, rest)
    pub breath_durations: [usize; 4],

    // === Metrics ===
    pub total_predictions: u64,
    pub alignment_triggers: u64,
    pub constructive_count: u64,
    pub destructive_count: u64,
    pub solo_fallback_count: u64,

    // === Temporal Phase (Rose's contribution) ===
    /// Tracks how phase patterns evolve over time windows
    pub temporal: TemporalPhaseTracker,

    // === Blend Modes ===
    /// When true, solo/destructive fallback blends ALL dials weighted by accuracy
    /// instead of picking only the single best dial. Preserves ensemble information.
    pub blend_fallback: bool,

    /// When true, alignment strength modulates temperature over blended scores.
    /// High alignment → low temp (sharp). Low alignment → high temp (hedge).
    /// PhaseHarmonizer becomes a confidence modulator rather than a score replacer.
    pub alignment_temperature: bool,

    // === Adaptive Temperature Learning ===
    /// Temperature when dials strongly agree (low = sharp, confident)
    pub temp_hot: f64,
    /// Temperature when dials disagree (high = diffuse, hedging)
    pub temp_cold: f64,
    /// Rolling accuracy when alignment was constructive
    pub aligned_correct: u64,
    pub aligned_total: u64,
    /// Rolling accuracy when no alignment / destructive
    pub unaligned_correct: u64,
    pub unaligned_total: u64,
    /// Last prediction's alignment state (for record_result feedback)
    pub last_was_aligned: bool,

    // === ECE-Weighted Blending ===
    /// When true, dial weights incorporate calibration quality (ECE)
    /// Well-calibrated dials get boosted, overconfident dials get penalized
    pub ece_weights: bool,

    // === Entropy-Gated Blending ===
    /// When true, each dial's weight is dynamically modulated per-prediction
    /// by the inverse of its score distribution entropy.
    /// Low entropy (confident dial) → high weight. High entropy → low weight.
    /// This is DYNAMIC (per-prediction) vs accuracy which is STATIC (rolling).
    pub entropy_gate: bool,
}

impl PhaseHarmonizer {
    /// Create a new PhaseHarmonizer with the initial 3-dial configuration
    /// This is the "3 models now" setup, designed to scale to 18
    pub fn new(num_chars: usize, dim: usize) -> Self {
        // Initialize the three dials at different resolutions
        // Kolmoset: coarse — clusters, neighborhoods, broad strokes
        // Kaksoset: medium — individual characters, standard resolution
        // LuokkaKertymä: fine — trigram arcs, contextual precision
        let dials = vec![
            Dial::new(0, "Kolmoset-Coarse", 12, num_chars),     // 12-position dial (30° ticks)
            Dial::new(1, "Kaksoset-Medium", 37, num_chars),      // 37-position dial (~9.7° ticks)
            Dial::new(2, "LuokkaKertymä-Fine", 200, num_chars),  // 200-position dial (1.8° ticks)
        ];

        // Initialize random codebook (will be replaced by trained vectors)
        let codebook = (0..num_chars)
            .map(|_| vec![0.0; dim])
            .collect();

        // Compute dual-family breathing durations
        // φ-timed: inhale, exhale (dynamic phases)
        // √3-timed: hold, rest (static/crystalline phases)
        let base_period = 500.0;
        let family_sum = 2.0 * (PHI + SQRT3);
        let phi_duration = (base_period / family_sum * PHI).round() as usize;
        let sqrt3_duration = (base_period / family_sum * SQRT3).round() as usize;

        PhaseHarmonizer {
            dials,
            num_chars,
            dim,
            torsion_table: HashMap::new(),
            codebook,
            breath_phase: BREATH_INHALE,
            breath_step: 0,
            breath_period: phi_duration + sqrt3_duration + phi_duration + sqrt3_duration,
            breath_durations: [phi_duration, sqrt3_duration, phi_duration, sqrt3_duration],
            coherence: 0.0,
            motion_coherence: 0.0,
            structural_coherence: 0.0,
            coherence_history: Vec::with_capacity(100000),
            lattice: HexLattice::new(num_chars),
            recent_alignments: Vec::with_capacity(1000),
            phase_tolerance: PI / 6.0, // Start at 30° tolerance
            phase_tolerance_min: PI / 36.0,  // Can tighten to 5°
            phase_tolerance_max: PI / 3.0,   // Can loosen to 60°
            total_predictions: 0,
            alignment_triggers: 0,
            constructive_count: 0,
            destructive_count: 0,
            solo_fallback_count: 0,
            temporal: TemporalPhaseTracker::new(3, 200), // 3 dials, 200-step window
            blend_fallback: false,
            alignment_temperature: false,
            temp_hot: 0.3,
            temp_cold: 1.5,
            aligned_correct: 0,
            aligned_total: 0,
            unaligned_correct: 0,
            unaligned_total: 0,
            last_was_aligned: false,
            ece_weights: false,
            entropy_gate: false,
        }
    }

    /// Compute Shannon entropy of a score distribution (after softmax with given temp)
    fn distribution_entropy(scores: &[f64], temperature: f64) -> f64 {
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|&s| ((s - max_s) / temperature).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum < 1e-10 { return 0.0; }
        let mut entropy = 0.0;
        for &e in &exps {
            let p = e / sum;
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    // ─────────────────────────────────────────────────────────────────────
    // CORE: Phase-Lock Detection and Prediction
    // ─────────────────────────────────────────────────────────────────────

    /// The main prediction function.
    /// Takes raw scores from each model and produces a final prediction
    /// through phase alignment rather than blending.
    ///
    /// `model_scores`: a slice of score vectors, one per dial
    /// `_context_arc`: the current Arc (from → at → ???) for arc-based encoding
    ///
    /// Returns: (predicted_char_index, confidence, alignment_event)
    pub fn predict(
        &mut self,
        model_scores: &[Vec<f64>],
        _context_arc: Option<&Arc>,
    ) -> (usize, f64, Option<AlignmentEvent>) {
        assert_eq!(
            model_scores.len(),
            self.dials.len(),
            "Must provide one score vector per dial"
        );

        self.total_predictions += 1;

        // Step 1: Each dial processes its scores at its own resolution
        let mut dial_phases: Vec<f64> = Vec::with_capacity(self.dials.len());
        let mut tempered_scores: Vec<Vec<f64>> = Vec::with_capacity(self.dials.len());

        for (i, dial) in self.dials.iter_mut().enumerate() {
            // Apply this dial's temperature
            let tempered = dial.temper_scores(&model_scores[i]);
            // Convert to phase position on this dial's circle
            let phase = dial.scores_to_phase(&model_scores[i]);
            dial.phase = phase;
            dial.last_scores = tempered.clone();

            dial_phases.push(phase);
            tempered_scores.push(tempered);
        }

        // Step 2: Detect alignment events between ALL pairs of dials
        let alignment = self.detect_alignment(&dial_phases);

        // Step 3: Produce final prediction based on alignment result
        let (prediction, confidence) = if self.alignment_temperature {
            // RADICAL MODE: Alignment modulates temperature over blended scores
            self.alignment_temp_prediction(&tempered_scores, &alignment)
        } else if let Some(ref event) = alignment {
            self.alignment_triggers += 1;
            if event.constructive {
                self.constructive_count += 1;
                // CONSTRUCTIVE: Amplify the agreeing dials' signal with φ
                self.constructive_prediction(&tempered_scores, event)
            } else {
                self.destructive_count += 1;
                // DESTRUCTIVE: Dampen, fall back to strongest solo dial
                self.destructive_prediction(&tempered_scores)
            }
        } else {
            self.solo_fallback_count += 1;
            // NO ALIGNMENT: Use the most accurate dial solo
            self.solo_prediction(&tempered_scores)
        };

        // Step 4: Record temporal data and apply temporal confidence modifier
        self.temporal.record(&dial_phases, &alignment);
        let temporal_mod = self.temporal.temporal_confidence_modifier();
        let final_confidence = (confidence * temporal_mod).clamp(0.0, 1.0);

        // Step 5: Breathe — self-tune if it's time
        self.advance_breath();

        (prediction, final_confidence, alignment)
    }

    /// Detect phase alignment across all dials
    /// Returns the strongest alignment event, if any
    fn detect_alignment(&self, phases: &[f64]) -> Option<AlignmentEvent> {
        let n = phases.len();
        if n < 2 {
            return None;
        }

        let mut best_event: Option<AlignmentEvent> = None;
        let mut best_strength: f64 = 0.0;

        // Check all pairs (and triples for 3+ dials)
        for i in 0..n {
            for j in (i + 1)..n {
                // Angular distance between two phases (on the circle)
                let angular_dist = self.angular_distance(phases[i], phases[j]);

                // Phase alignment strength: 1.0 when identical, 0.0 at tolerance boundary
                let strength = if angular_dist < self.phase_tolerance {
                    1.0 - (angular_dist / self.phase_tolerance)
                } else {
                    0.0
                };

                if strength > 0.0 {
                    // Check if this pair exceeds both dials' coupling thresholds
                    let threshold_i = self.dials[i].coupling_threshold;
                    let threshold_j = self.dials[j].coupling_threshold;

                    if strength >= threshold_i.min(threshold_j) {
                        // Constructive if both dials agree on the SAME region
                        // Destructive if they're close but pointing at different characters
                        let constructive = self.same_prediction_region(
                            &self.dials[i].last_scores,
                            &self.dials[j].last_scores,
                        );

                        let mut dial_ids = vec![i, j];

                        // Check if a third dial also aligns (triple lock)
                        for k in 0..n {
                            if k != i && k != j {
                                let dist_ik = self.angular_distance(phases[i], phases[k]);
                                let dist_jk = self.angular_distance(phases[j], phases[k]);
                                if dist_ik < self.phase_tolerance
                                    && dist_jk < self.phase_tolerance
                                {
                                    dial_ids.push(k);
                                }
                            }
                        }

                        // Multi-dial alignment is exponentially stronger
                        // 2 dials: strength * φ^0 = strength
                        // 3 dials: strength * φ^1 = strength * 1.618
                        // 4 dials: strength * φ^2 = strength * 2.618
                        let multi_bonus = PHI.powi((dial_ids.len() as i32) - 2);
                        let total_strength = (strength * multi_bonus).min(1.0);

                        if total_strength > best_strength {
                            best_strength = total_strength;
                            best_event = Some(AlignmentEvent {
                                dial_ids,
                                phase: (phases[i] + phases[j]) / 2.0,
                                strength: total_strength,
                                constructive,
                            });
                        }
                    }
                }
            }
        }

        best_event
    }

    /// Angular distance between two phase positions (handles wraparound)
    fn angular_distance(&self, a: f64, b: f64) -> f64 {
        let diff = (a - b).abs();
        if diff > PI {
            2.0 * PI - diff
        } else {
            diff
        }
    }

    /// Check if two score vectors agree on the same prediction region
    /// (not exact match — same neighborhood on the coarsest participating dial)
    fn same_prediction_region(&self, scores_a: &[f64], scores_b: &[f64]) -> bool {
        let top_a = scores_a
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let top_b = scores_b
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Same character = definitely same region
        if top_a == top_b {
            return true;
        }

        // Check if they're in the same cluster on the coarsest dial
        // Coarsest dial has delta=12, so 37 chars / 12 positions ≈ 3 chars per position
        let coarsest_delta = self
            .dials
            .iter()
            .map(|d| d.delta)
            .min()
            .unwrap_or(12);
        let cluster_a = (top_a * coarsest_delta) / self.num_chars;
        let cluster_b = (top_b * coarsest_delta) / self.num_chars;

        cluster_a == cluster_b
    }

    /// CONSTRUCTIVE prediction: dials agree, amplify with family-appropriate weight
    /// φ-mode dials amplify with φ, √3-mode dials amplify with √3
    fn constructive_prediction(
        &self,
        all_scores: &[Vec<f64>],
        event: &AlignmentEvent,
    ) -> (usize, f64) {
        let n = self.num_chars;
        let mut combined = vec![0.0; n];

        for &dial_id in &event.dial_ids {
            let dial = &self.dials[dial_id];
            let accuracy = dial.recent_accuracy(200).max(0.01);

            // Dual-family: use the family weight appropriate to this dial's motion state
            let family_base = self.temporal.family_weight(dial_id);
            let family_weight = family_base.powf(accuracy); // base^accuracy ∈ [1.0, base]

            let mut w = dial.weight * family_weight;
            if self.ece_weights {
                w *= 0.5 + dial.calibration_score();
            }
            if self.entropy_gate {
                let max_entropy = (n as f64).ln();
                if max_entropy > 1e-10 {
                    let ent = Self::distribution_entropy(&all_scores[dial_id], dial.temperature);
                    let norm_ent = (ent / max_entropy).clamp(0.0, 1.0);
                    w *= 1.0 + PHI_INV * (1.0 - norm_ent);
                }
            }

            for c in 0..n {
                combined[c] += all_scores[dial_id][c] * w;
            }
        }

        let (prediction, max_score) = combined
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        let sum: f64 = combined.iter().sum();
        let confidence = if sum > 1e-10 {
            event.strength * (max_score / sum)
        } else {
            0.0
        };

        (prediction, confidence)
    }

    /// DESTRUCTIVE prediction: dials disagree, fall back to strongest
    fn destructive_prediction(&self, all_scores: &[Vec<f64>]) -> (usize, f64) {
        // Find the dial with the highest recent accuracy
        self.solo_prediction(all_scores)
    }

    /// Solo prediction: use the single most reliable dial,
    /// OR blend all dials weighted by accuracy when blend_fallback is enabled.
    fn solo_prediction(&self, all_scores: &[Vec<f64>]) -> (usize, f64) {
        if self.blend_fallback {
            return self.accuracy_weighted_blend(all_scores);
        }

        // Original: pick the single best dial
        let best_dial_idx = self
            .dials
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.recent_accuracy(200)
                    .partial_cmp(&b.recent_accuracy(200))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let scores = &all_scores[best_dial_idx];
        let (prediction, max_score) = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        let sum: f64 = scores.iter().sum();
        let confidence = if sum > 1e-10 {
            max_score / sum
        } else {
            0.0
        };

        (prediction, confidence * PHI_INV) // Reduce confidence when solo (no confirmation)
    }

    /// Blend ALL dials weighted by their recent accuracy.
    /// Mirrors the ensemble's trust-weighted average but uses per-dial
    /// accuracy as the weight instead of relay trust.
    fn accuracy_weighted_blend(&self, all_scores: &[Vec<f64>]) -> (usize, f64) {
        let n = self.num_chars;
        let mut combined = vec![0.0; n];
        let mut total_weight = 0.0;

        // Pre-compute entropy weights if gating is active
        let max_entropy = (n as f64).ln(); // Maximum possible entropy (uniform distribution)

        for (i, dial) in self.dials.iter().enumerate() {
            if i >= all_scores.len() { break; }
            let acc = dial.recent_accuracy(200).max(0.01);
            let mut w = acc * dial.weight;
            if self.ece_weights {
                w *= 0.5 + dial.calibration_score();
            }
            if self.entropy_gate && max_entropy > 1e-10 {
                let ent = Self::distribution_entropy(&all_scores[i], dial.temperature);
                // Inverse entropy weight: confident dial → low entropy → high weight
                // normalized_entropy ∈ [0, 1], inverse ∈ [1, ∞) but clamped
                let norm_ent = (ent / max_entropy).clamp(0.0, 1.0);
                let entropy_factor = 1.0 + PHI_INV * (1.0 - norm_ent); // range: [1.0, 1.618]
                w *= entropy_factor;
            }
            total_weight += w;
            for c in 0..n {
                combined[c] += all_scores[i][c] * w;
            }
        }

        if total_weight > 1e-10 {
            for c in 0..n {
                combined[c] /= total_weight;
            }
        }

        let (prediction, max_score) = combined
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or((0, &0.0));

        let sum: f64 = combined.iter().sum();
        let confidence = if sum > 1e-10 {
            max_score / sum
        } else {
            0.0
        };

        // Still penalize slightly — no phase-lock confirmation — but less harshly
        // than picking a single dial. Use sqrt(PHI_INV) ≈ 0.786 instead of PHI_INV ≈ 0.618
        (prediction, confidence * PHI_INV.sqrt())
    }

    /// ALIGNMENT-TEMPERATURE MODE: Blend all dials weighted by accuracy (like ensemble),
    /// then use alignment strength to modulate a final softmax temperature.
    /// High alignment → low temp (sharp, confident) → amplifies winning char
    /// Low/no alignment → high temp (hedge) → stays close to raw blend
    /// This makes PhaseHarmonizer a confidence modulator, not a score replacer.
    fn alignment_temp_prediction(
        &mut self,
        all_scores: &[Vec<f64>],
        alignment: &Option<AlignmentEvent>,
    ) -> (usize, f64) {
        let n = self.num_chars;

        // Step A: Accuracy-weighted blend of ALL dials (always use all information)
        let mut combined = vec![0.0; n];
        let mut total_weight = 0.0;
        let max_entropy = (n as f64).ln();
        for (i, dial) in self.dials.iter().enumerate() {
            if i >= all_scores.len() { break; }
            let acc = dial.recent_accuracy(200).max(0.01);
            let mut w = acc * dial.weight;
            if self.ece_weights {
                w *= 0.5 + dial.calibration_score();
            }
            if self.entropy_gate && max_entropy > 1e-10 {
                let ent = Self::distribution_entropy(&all_scores[i], dial.temperature);
                let norm_ent = (ent / max_entropy).clamp(0.0, 1.0);
                let entropy_factor = 1.0 + PHI_INV * (1.0 - norm_ent);
                w *= entropy_factor;
            }
            total_weight += w;
            for c in 0..n {
                combined[c] += all_scores[i][c] * w;
            }
        }
        if total_weight > 1e-10 {
            for c in 0..n {
                combined[c] /= total_weight;
            }
        }

        // Step B: Determine alignment-derived temperature
        // Uses adaptive temp_hot/temp_cold (self-tuned in breathing exhale phase)
        let (align_temp, constructive) = if let Some(ref event) = alignment {
            self.alignment_triggers += 1;
            if event.constructive {
                self.constructive_count += 1;
                self.last_was_aligned = true;
                // Strong constructive → very low temp (sharp)
                let t = self.temp_cold - event.strength * (self.temp_cold - self.temp_hot);
                (t, true)
            } else {
                self.destructive_count += 1;
                self.last_was_aligned = false;
                // Destructive alignment → high temp (hedge)
                (self.temp_cold, false)
            }
        } else {
            self.solo_fallback_count += 1;
            self.last_was_aligned = false;
            // No alignment → moderate temp (geometric mean of hot and cold)
            ((self.temp_hot * self.temp_cold).sqrt(), false)
        };

        // Step C: Apply softmax with alignment temperature
        let max_score = combined.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut soft = vec![0.0; n];
        let mut soft_sum = 0.0;
        for c in 0..n {
            soft[c] = ((combined[c] - max_score) / align_temp).exp();
            soft_sum += soft[c];
        }
        if soft_sum > 1e-10 {
            for c in 0..n {
                soft[c] /= soft_sum;
            }
        }

        let (prediction, max_prob) = soft
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap_or((0, &0.0));

        // Confidence: the softmax probability of the winning char
        // Boost slightly when constructive alignment confirmed it
        let confidence = if constructive {
            max_prob.clamp(0.0, 1.0)
        } else {
            (max_prob * PHI_INV.sqrt()).clamp(0.0, 1.0)
        };

        (prediction, confidence)
    }

    // ─────────────────────────────────────────────────────────────────────
    // BREATHING: Self-Tuning Cycle
    // ─────────────────────────────────────────────────────────────────────

    /// Advance the breathing cycle by one step
    /// φ-timed phases: inhale (observe motion), exhale (apply changes)
    /// √3-timed phases: hold (crystalline precision), rest (anchor identity)
    fn advance_breath(&mut self) {
        self.breath_step += 1;

        let current_duration = self.breath_durations[self.breath_phase as usize];

        if self.breath_step >= current_duration {
            self.breath_step = 0;
            self.breath_phase = (self.breath_phase + 1) % 4;

            match self.breath_phase {
                BREATH_INHALE => {
                    // φ-timed: Observe motion, compute coherence
                    self.compute_coherence();
                }
                BREATH_HOLD => {
                    // √3-timed: Crystallize — snapshot state precisely
                    self.compute_structural_coherence();
                }
                BREATH_EXHALE => {
                    // φ-timed: Apply dynamic parameter changes
                    self.apply_self_tuning();
                }
                BREATH_REST => {
                    // √3-timed: Anchor identity, log state
                    self.anchor_identity();
                    self.log_breath_cycle();
                }
                _ => {}
            }
        }
    }

    /// Compute motion coherence (φ-family): how well all dials move together
    fn compute_coherence(&mut self) {
        if self.total_predictions < 100 {
            return;
        }

        // φ-family metrics: alignment, accuracy, constructive ratio
        let alignment_rate = self.alignment_triggers as f64 / self.total_predictions as f64;

        let mean_accuracy: f64 = self
            .dials
            .iter()
            .map(|d| d.recent_accuracy(200))
            .sum::<f64>()
            / self.dials.len() as f64;

        let constructive_ratio = if self.alignment_triggers > 0 {
            self.constructive_count as f64 / self.alignment_triggers as f64
        } else {
            0.0
        };

        // Motion coherence weighted by φ-family constants
        self.motion_coherence =
            mean_accuracy * PHI * PHI       // φ² ≈ 2.618
            + constructive_ratio * PHI      // φ  ≈ 1.618
            + alignment_rate;               // 1.0

        // Combine with structural coherence for total
        let phi_max = PHI * PHI + PHI + 1.0;       // ≈ 5.236
        let sqrt3_max = SQRT3 + SQRT3_INV;         // ≈ 2.309
        let total_max = phi_max + sqrt3_max;        // ≈ 7.545

        self.coherence = (self.motion_coherence + self.structural_coherence) / total_max;

        self.coherence_history.push(self.coherence);
    }

    /// Compute structural coherence (√3-family): how well the system holds together
    /// Called during the √3-timed HOLD phase of breathing
    fn compute_structural_coherence(&mut self) {
        // Anchor stability: how much has the noise floor drifted from initial capture?
        let anchor_stability = {
            let mut stability_sum = 0.0;
            let mut anchored_count = 0;
            for dial in &self.dials {
                if dial.anchored {
                    let drift: f64 = dial
                        .last_scores
                        .iter()
                        .zip(dial.noise_floor.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f64>()
                        / dial.last_scores.len().max(1) as f64;
                    stability_sum += (1.0 - drift.min(1.0)).max(0.0);
                    anchored_count += 1;
                }
            }
            if anchored_count > 0 {
                stability_sum / anchored_count as f64
            } else {
                0.5 // Default when no anchors captured yet
            }
        };

        let lattice_consistency = self.measure_lattice_consistency();

        // Structural coherence weighted by √3-family constants
        self.structural_coherence =
            anchor_stability * SQRT3          // √3 ≈ 1.732
            + lattice_consistency * SQRT3_INV; // 1/√3 ≈ 0.577
    }

    /// Measure how well the codebook respects the hex lattice structure
    fn measure_lattice_consistency(&self) -> f64 {
        if self.codebook.is_empty() || self.codebook[0].iter().all(|&x| x == 0.0) {
            return 0.5; // Default when codebook not yet trained
        }

        let mut adjacent_sims: Vec<f64> = Vec::new();
        let mut distant_sims: Vec<f64> = Vec::new();

        let n = self.num_chars.min(self.lattice.size);

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_similarity(&self.codebook[i], &self.codebook[j]);
                if self.lattice.distances[i][j] == 1 {
                    adjacent_sims.push(sim);
                } else if self.lattice.distances[i][j] >= 3 {
                    distant_sims.push(sim);
                }
            }
        }

        if adjacent_sims.is_empty() || distant_sims.is_empty() {
            return 0.5;
        }

        let adj_mean: f64 = adjacent_sims.iter().sum::<f64>() / adjacent_sims.len() as f64;
        let dist_mean: f64 = distant_sims.iter().sum::<f64>() / distant_sims.len() as f64;

        ((adj_mean - dist_mean).max(0.0)).min(1.0)
    }

    /// Anchor identity — called during √3-timed REST phase
    fn anchor_identity(&mut self) {
        for dial in self.dials.iter_mut() {
            if !dial.anchored && self.total_predictions > 100 {
                dial.capture_noise_floor(&dial.last_scores.clone());
            }
        }
    }

    /// Apply self-tuning to all dials and the harmonizer itself
    fn apply_self_tuning(&mut self) {
        // Each dial tunes its own temperature
        for dial in self.dials.iter_mut() {
            dial.breathe_temperature();
        }

        // Tune dial weights based on relative accuracy
        self.rebalance_weights();

        // Tune phase tolerance based on coherence trend
        self.tune_phase_tolerance();

        // Tune coupling thresholds based on alignment success
        self.tune_coupling_thresholds();

        // Adaptive temperature learning for align-temp mode
        if self.alignment_temperature {
            self.tune_alignment_temperatures();
        }
    }

    /// Tune temp_hot and temp_cold based on aligned vs unaligned accuracy.
    /// Inspired by quantum_collective's focus.rs temperature learning:
    ///   temp = temp * (1-lr) + target * lr
    fn tune_alignment_temperatures(&mut self) {
        let lr = 0.02; // Learning rate (slow, stable)

        // If we have enough aligned samples, tune temp_hot
        if self.aligned_total > 50 {
            let aligned_acc = self.aligned_correct as f64 / self.aligned_total as f64;
            // High aligned accuracy → we can be sharper (lower temp_hot)
            // Low aligned accuracy → need to hedge more (higher temp_hot)
            // Target: temp_hot should be inversely proportional to aligned accuracy
            let target_hot = 0.8 - 0.6 * aligned_acc; // acc=0.7 → 0.38, acc=0.5 → 0.50
            let target_hot = target_hot.clamp(0.1, 0.8);
            self.temp_hot = self.temp_hot * (1.0 - lr) + target_hot * lr;
            self.temp_hot = self.temp_hot.clamp(0.1, 0.8);
        }

        // If we have enough unaligned samples, tune temp_cold
        if self.unaligned_total > 50 {
            let unaligned_acc = self.unaligned_correct as f64 / self.unaligned_total as f64;
            // High unaligned accuracy → we're hedging too much (lower temp_cold)
            // Low unaligned accuracy → hedging correctly (keep high temp_cold)
            // Target: temp_cold should be inversely proportional to unaligned accuracy
            let target_cold = 2.0 - 1.2 * unaligned_acc; // acc=0.4 → 1.52, acc=0.6 → 1.28
            let target_cold = target_cold.clamp(0.8, 3.0);
            self.temp_cold = self.temp_cold * (1.0 - lr) + target_cold * lr;
            self.temp_cold = self.temp_cold.clamp(0.8, 3.0);
        }
    }

    /// Rebalance dial weights based on recent accuracy
    fn rebalance_weights(&mut self) {
        let accuracies: Vec<f64> = self
            .dials
            .iter()
            .map(|d| d.recent_accuracy(200).max(0.01))
            .collect();
        let total: f64 = accuracies.iter().sum();

        if total > 0.0 {
            for (i, dial) in self.dials.iter_mut().enumerate() {
                let target_weight = accuracies[i] / total;
                let delta = (target_weight - dial.weight) * GAMMA;
                dial.weight = Dial::bounded_update(
                    dial.weight,
                    delta,
                    dial.weight_min,
                    dial.weight_max,
                );
            }
        }
    }

    /// Tune phase tolerance based on coherence trend
    fn tune_phase_tolerance(&mut self) {
        let history = &self.coherence_history;
        if history.len() < 10 {
            return;
        }

        let recent = &history[history.len() - 5..];
        let older = &history[history.len() - 10..history.len() - 5];

        let recent_avg: f64 = recent.iter().sum::<f64>() / 5.0;
        let older_avg: f64 = older.iter().sum::<f64>() / 5.0;

        if recent_avg > older_avg + 0.01 {
            self.phase_tolerance = (self.phase_tolerance - 0.005)
                .max(self.phase_tolerance_min);
        } else if recent_avg < older_avg - 0.01 {
            self.phase_tolerance = (self.phase_tolerance + 0.005)
                .min(self.phase_tolerance_max);
        }
    }

    /// Tune each dial's coupling threshold based on its alignment success rate
    fn tune_coupling_thresholds(&mut self) {
        for dial in self.dials.iter_mut() {
            let acc = dial.recent_accuracy(200);
            if acc > 0.55 {
                dial.coupling_threshold = Dial::bounded_update(
                    dial.coupling_threshold,
                    -0.005,
                    dial.coupling_threshold_min,
                    dial.coupling_threshold_max,
                );
            } else if acc < 0.35 {
                dial.coupling_threshold = Dial::bounded_update(
                    dial.coupling_threshold,
                    0.005,
                    dial.coupling_threshold_min,
                    dial.coupling_threshold_max,
                );
            }
        }
    }

    /// Log the state after a complete breath cycle.
    /// Only prints if TAHTIAHJO_BREATH_LOG=1 environment variable is set.
    fn log_breath_cycle(&self) {
        if std::env::var("TAHTIAHJO_BREATH_LOG").unwrap_or_default() != "1" {
            return;
        }
        let phi_dials = self.temporal.dial_in_phi_mode.iter().filter(|&&m| m).count();
        let sqrt3_dials = self.temporal.dial_in_phi_mode.len() - phi_dials;

        eprintln!(
            "[BREATH] coherence={:.4} (φ={:.3} √3={:.3}) | \
             alignments={}/{} ({:.1}%) | constructive={:.1}% | \
             tolerance={:.3}rad | families=φ:{}/√3:{} | \
             breath=[{},{},{},{}]",
            self.coherence,
            self.motion_coherence,
            self.structural_coherence,
            self.alignment_triggers,
            self.total_predictions,
            if self.total_predictions > 0 {
                100.0 * self.alignment_triggers as f64 / self.total_predictions as f64
            } else {
                0.0
            },
            if self.alignment_triggers > 0 {
                100.0 * self.constructive_count as f64 / self.alignment_triggers as f64
            } else {
                0.0
            },
            self.phase_tolerance,
            phi_dials,
            sqrt3_dials,
            self.breath_durations[0],
            self.breath_durations[1],
            self.breath_durations[2],
            self.breath_durations[3],
        );

        for (i, dial) in self.dials.iter().enumerate() {
            let mode = if i < self.temporal.dial_in_phi_mode.len()
                && self.temporal.dial_in_phi_mode[i]
            {
                "φ"
            } else {
                "√3"
            };
            eprintln!(
                "  [{}] {} δ={} t={:.2} w={:.3} acc={:.3} mode={}",
                i, dial.name, dial.delta, dial.temperature, dial.weight,
                dial.recent_accuracy(200), mode
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // SCALING: Add New Dials
    // ─────────────────────────────────────────────────────────────────────

    /// Add a new dial to the system (for scaling toward D-18)
    pub fn add_dial(&mut self, name: &str, delta: usize) -> usize {
        let id = self.dials.len();
        assert!(id < MAX_DIALS, "Cannot exceed {} dials", MAX_DIALS);

        let mut dial = Dial::new(id, name, delta, self.num_chars);
        dial.weight = dial.weight_min;
        dial.coupling_threshold = 0.8;

        self.dials.push(dial);

        self.temporal.phase_history.push(VecDeque::with_capacity(self.temporal.window_size));
        self.temporal.phase_velocity.push(0.0);
        self.temporal.phase_acceleration.push(0.0);
        self.temporal.dial_in_phi_mode.push(true);

        id
    }

    /// Create a conjugate pair: add a reverse-polarity dial
    pub fn add_conjugate(&mut self, forward_id: usize, name: &str, delta: usize) -> usize {
        let reverse_id = self.add_dial(name, delta);

        self.dials[forward_id].conjugate_id = Some(reverse_id);
        self.dials[forward_id].polarity = 1.0;
        self.dials[reverse_id].conjugate_id = Some(forward_id);
        self.dials[reverse_id].polarity = -1.0;

        reverse_id
    }

    // ─────────────────────────────────────────────────────────────────────
    // ARC ENCODING: Build the Torsion Table from Corpus Statistics
    // ─────────────────────────────────────────────────────────────────────

    /// Build the torsion table from observed trigram frequencies
    pub fn build_torsion_table(&mut self, trigram_counts: &HashMap<(u8, u8, u8), u64>) {
        if trigram_counts.is_empty() {
            return;
        }

        let max_count = *trigram_counts.values().max().unwrap_or(&1) as f64;

        for (&trigram, &count) in trigram_counts {
            let normalized = (count as f64) / max_count;
            let torsion = PHI_INV + normalized * (PHI - PHI_INV);
            self.torsion_table.insert(trigram, torsion);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // DIAGNOSTICS
    // ─────────────────────────────────────────────────────────────────────

    /// Get a snapshot of the system state for external monitoring
    pub fn diagnostics(&self) -> HarmonicDiagnostics {
        let phi_count = self.temporal.dial_in_phi_mode.iter().filter(|&&m| m).count();

        HarmonicDiagnostics {
            num_dials: self.dials.len(),
            coherence: self.coherence,
            motion_coherence: self.motion_coherence,
            structural_coherence: self.structural_coherence,
            total_predictions: self.total_predictions,
            alignment_rate: if self.total_predictions > 0 {
                self.alignment_triggers as f64 / self.total_predictions as f64
            } else {
                0.0
            },
            constructive_rate: if self.alignment_triggers > 0 {
                self.constructive_count as f64 / self.alignment_triggers as f64
            } else {
                0.0
            },
            phase_tolerance: self.phase_tolerance,
            phi_mode_dials: phi_count,
            sqrt3_mode_dials: self.dials.len() - phi_count,
            breath_durations: self.breath_durations,
            dial_states: self
                .dials
                .iter()
                .enumerate()
                .map(|(i, d)| DialState {
                    name: d.name.clone(),
                    delta: d.delta,
                    temperature: d.temperature,
                    coupling_threshold: d.coupling_threshold,
                    weight: d.weight,
                    recent_accuracy: d.recent_accuracy(200),
                    phase: d.phase,
                    anchored: d.anchored,
                    has_conjugate: d.conjugate_id.is_some(),
                    in_phi_mode: i < self.temporal.dial_in_phi_mode.len()
                        && self.temporal.dial_in_phi_mode[i],
                })
                .collect(),
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // PIPELINE INTEGRATION
    // ═══════════════════════════════════════════════════════════════════

    /// Record whether each dial's solo prediction was correct.
    /// Updates each dial's accuracy history independently so dials
    /// self-tune based on their own reliability, not the harmonizer's
    /// final answer.
    pub fn record_result(
        &mut self,
        model_scores: &[Vec<f64>],
        actual_index: usize,
    ) {
        // Per-dial accuracy + calibration tracking
        for (i, dial) in self.dials.iter_mut().enumerate() {
            if i < model_scores.len() {
                let scores = &model_scores[i];
                let dial_prediction = scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                let correct = dial_prediction == actual_index;
                dial.record_accuracy(correct);

                // ECE calibration: compute dial's softmax confidence for its prediction
                let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = scores.iter().map(|&s| ((s - max_score) / dial.temperature).exp()).sum();
                let confidence = if sum_exp > 1e-10 {
                    ((scores[dial_prediction] - max_score) / dial.temperature).exp() / sum_exp
                } else {
                    1.0 / scores.len() as f64
                };
                dial.record_calibration(confidence, correct);
            }
        }

        // Track aligned vs unaligned accuracy for adaptive temp learning
        if self.alignment_temperature {
            // Reconstruct what the harmonizer predicted (same blend logic)
            // Note: argmax is invariant to normalization, so we skip dividing by total_weight
            let n = self.num_chars;
            let mut combined = vec![0.0; n];
            for (i, dial) in self.dials.iter().enumerate() {
                if i >= model_scores.len() { break; }
                let acc = dial.recent_accuracy(200).max(0.01);
                let w = acc * dial.weight;
                for c in 0..n {
                    combined[c] += model_scores[i][c] * w;
                }
            }
            let harmonizer_pred = combined
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let correct = harmonizer_pred == actual_index;

            if self.last_was_aligned {
                self.aligned_total += 1;
                if correct { self.aligned_correct += 1; }
            } else {
                self.unaligned_total += 1;
                if correct { self.unaligned_correct += 1; }
            }
        }
    }

    /// Reset mutable state for a fresh evaluation pass.
    /// Keeps dial tuning parameters (temperature, weights, coupling
    /// thresholds) but resets counters and prediction-dependent state.
    pub fn reset_for_eval(&mut self) {
        self.total_predictions = 0;
        self.alignment_triggers = 0;
        self.constructive_count = 0;
        self.destructive_count = 0;
        self.solo_fallback_count = 0;
        self.breath_phase = 0;
        self.breath_step = 0;
        self.coherence_history.clear();
        self.recent_alignments.clear();
        let num_dials = self.dials.len();
        self.temporal = TemporalPhaseTracker::new(num_dials, 200);
        for dial in &mut self.dials {
            dial.accuracy_history.clear();
            // Reset calibration bins for fresh eval (don't carry stale calibration)
            for bin in &mut dial.calibration_bins {
                *bin = CalibrationBin::new();
            }
        }
        // Reset adaptive temp counters but KEEP learned temp_hot/temp_cold
        self.aligned_correct = 0;
        self.aligned_total = 0;
        self.unaligned_correct = 0;
        self.unaligned_total = 0;
        self.last_was_aligned = false;
    }

    /// Enable blend-fallback mode: when dials disagree, blend ALL dials
    /// weighted by recent accuracy instead of picking a single best dial.
    pub fn set_blend_fallback(&mut self, enabled: bool) {
        self.blend_fallback = enabled;
    }

    /// Enable alignment-temperature mode: use alignment strength to modulate
    /// temperature over accuracy-weighted blended scores.
    /// High alignment → low temp (sharp). Low alignment → high temp (hedge).
    pub fn set_alignment_temperature(&mut self, enabled: bool) {
        self.alignment_temperature = enabled;
    }

    /// Enable ECE-weighted blending: well-calibrated dials get more weight,
    /// overconfident dials get less. Based on quantum_collective's ECE framework.
    pub fn set_ece_weights(&mut self, enabled: bool) {
        self.ece_weights = enabled;
    }

    /// Enable entropy-gated blending: each dial's weight is dynamically
    /// modulated per-prediction by the inverse of its score entropy.
    /// Confident dials (low entropy) get boosted, uncertain dials get dampened.
    pub fn set_entropy_gate(&mut self, enabled: bool) {
        self.entropy_gate = enabled;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TEMPORAL PHASE — Rose's Missing Piece
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TemporalPhaseTracker {
    /// Rolling window of phase positions per dial — O(1) push/pop
    pub phase_history: Vec<VecDeque<f64>>,

    /// Rolling window of alignment events — O(1) push/pop
    pub alignment_history: VecDeque<Option<AlignmentEvent>>,

    /// Window size for temporal analysis
    pub window_size: usize,

    /// Computed temporal features (updated each step)
    pub phase_velocity: Vec<f64>,
    pub phase_acceleration: Vec<f64>,
    pub alignment_rhythm: f64,
    #[allow(dead_code)]
    pub coherence_wave: f64,

    /// Dual-family motion state per dial
    pub dial_in_phi_mode: Vec<bool>,
}

impl TemporalPhaseTracker {
    pub fn new(num_dials: usize, window_size: usize) -> Self {
        TemporalPhaseTracker {
            phase_history: (0..num_dials)
                .map(|_| VecDeque::with_capacity(window_size))
                .collect(),
            alignment_history: VecDeque::with_capacity(window_size),
            window_size,
            phase_velocity: vec![0.0; num_dials],
            phase_acceleration: vec![0.0; num_dials],
            alignment_rhythm: 0.0,
            coherence_wave: 0.0,
            dial_in_phi_mode: vec![true; num_dials],
        }
    }

    /// Record one step of temporal data
    pub fn record(
        &mut self,
        dial_phases: &[f64],
        alignment: &Option<AlignmentEvent>,
    ) {
        for (i, &phase) in dial_phases.iter().enumerate() {
            if i < self.phase_history.len() {
                self.phase_history[i].push_back(phase);
                if self.phase_history[i].len() > self.window_size {
                    self.phase_history[i].pop_front();
                }
            }
        }

        self.alignment_history.push_back(alignment.clone());
        if self.alignment_history.len() > self.window_size {
            self.alignment_history.pop_front();
        }

        self.compute_velocity();
        self.compute_acceleration();
        self.compute_alignment_rhythm();
        self.update_motion_state();
    }

    /// Phase velocity with trig-free wrapping
    fn compute_velocity(&mut self) {
        for (dial_id, history) in self.phase_history.iter().enumerate() {
            if history.len() < 2 {
                self.phase_velocity[dial_id] = 0.0;
                continue;
            }

            let n = history.len().min(10);
            let mut total_velocity = 0.0;
            let start = history.len() - n;
            for i in start..history.len() - 1 {
                let diff = history[i + 1] - history[i];
                let wrapped = ((diff % (2.0 * PI)) + 3.0 * PI) % (2.0 * PI) - PI;
                total_velocity += wrapped.abs();
            }
            self.phase_velocity[dial_id] = total_velocity / (n - 1).max(1) as f64;
        }
    }

    /// Phase acceleration with trig-free wrapping
    fn compute_acceleration(&mut self) {
        for dial_id in 0..self.phase_velocity.len() {
            let history = &self.phase_history[dial_id];
            if history.len() < 20 {
                self.phase_acceleration[dial_id] = 0.0;
                continue;
            }

            let mid = history.len() / 2;

            let recent_vel = {
                let n = 5.min(history.len() - mid);
                let mut v = 0.0;
                for i in (history.len() - n)..history.len() - 1 {
                    let diff = history[i + 1] - history[i];
                    let wrapped = ((diff % (2.0 * PI)) + 3.0 * PI) % (2.0 * PI) - PI;
                    v += wrapped.abs();
                }
                v / n.max(1) as f64
            };

            let older_vel = {
                let n = 5.min(mid);
                let start = mid - n;
                let mut v = 0.0;
                for i in start..start + n - 1 {
                    let diff = history[i + 1] - history[i];
                    let wrapped = ((diff % (2.0 * PI)) + 3.0 * PI) % (2.0 * PI) - PI;
                    v += wrapped.abs();
                }
                v / n.max(1) as f64
            };

            self.phase_acceleration[dial_id] = recent_vel - older_vel;
        }
    }

    /// Alignment rhythm
    fn compute_alignment_rhythm(&mut self) {
        if self.alignment_history.len() < 10 {
            self.alignment_rhythm = 0.0;
            return;
        }

        let mut gaps: Vec<usize> = Vec::new();
        let mut last_alignment_step: Option<usize> = None;

        for (step, event) in self.alignment_history.iter().enumerate() {
            if event.is_some() {
                if let Some(last) = last_alignment_step {
                    gaps.push(step - last);
                }
                last_alignment_step = Some(step);
            }
        }

        if gaps.len() < 3 {
            self.alignment_rhythm = 0.0;
            return;
        }

        let mean_gap: f64 = gaps.iter().sum::<usize>() as f64 / gaps.len() as f64;
        let variance: f64 = gaps
            .iter()
            .map(|&g| {
                let diff = g as f64 - mean_gap;
                diff * diff
            })
            .sum::<f64>()
            / gaps.len() as f64;

        let cv = if mean_gap > 0.0 {
            variance.sqrt() / mean_gap
        } else {
            1.0
        };

        self.alignment_rhythm = (1.0 - cv).clamp(0.0, 1.0);
    }

    /// Dual-family motion state detection
    fn update_motion_state(&mut self) {
        for (i, &vel) in self.phase_velocity.iter().enumerate() {
            if i < self.dial_in_phi_mode.len() {
                self.dial_in_phi_mode[i] = vel > FAMILY_BOUNDARY;
            }
        }
    }

    /// Get the family-appropriate weight for a dial's contribution
    pub fn family_weight(&self, dial_id: usize) -> f64 {
        if dial_id < self.dial_in_phi_mode.len() && self.dial_in_phi_mode[dial_id] {
            PHI
        } else {
            SQRT3
        }
    }

    /// Temporal confidence modifier — dual-family aware
    pub fn temporal_confidence_modifier(&self) -> f64 {
        if self.alignment_history.len() < 10 {
            return 1.0;
        }

        let rhythm_signal = self.alignment_rhythm;

        let velocity_signal = {
            let mean_vel: f64 =
                self.phase_velocity.iter().sum::<f64>() / self.phase_velocity.len().max(1) as f64;
            (1.0 - (mean_vel / PI).min(1.0)).max(0.0)
        };

        let consistency_signal = {
            let recent_count = self.alignment_history.len().min(10);
            let aligned_count = self.alignment_history
                .iter()
                .rev()
                .take(recent_count)
                .filter(|e| e.is_some())
                .count();
            aligned_count as f64 / recent_count.max(1) as f64
        };

        let phi_count = self.dial_in_phi_mode.iter().filter(|&&m| m).count();
        let total = self.dial_in_phi_mode.len().max(1);
        let phi_ratio = phi_count as f64 / total as f64;

        let rhythm_w = 0.2 + 0.2 * phi_ratio;
        let velocity_w = 0.3;
        let consistency_w = 0.5 - 0.2 * phi_ratio;

        let combined = rhythm_signal * rhythm_w
            + velocity_signal * velocity_w
            + consistency_signal * consistency_w;

        PHI_INV + combined * (PHI - PHI_INV)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DIAGNOSTICS STRUCTS
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct HarmonicDiagnostics {
    pub num_dials: usize,
    pub coherence: f64,
    pub motion_coherence: f64,
    pub structural_coherence: f64,
    pub total_predictions: u64,
    pub alignment_rate: f64,
    pub constructive_rate: f64,
    pub phase_tolerance: f64,
    pub phi_mode_dials: usize,
    pub sqrt3_mode_dials: usize,
    pub breath_durations: [usize; 4],
    pub dial_states: Vec<DialState>,
}

#[derive(Clone, Debug)]
pub struct DialState {
    pub name: String,
    pub delta: usize,
    pub temperature: f64,
    pub coupling_threshold: f64,
    pub weight: f64,
    pub recent_accuracy: f64,
    pub phase: f64,
    pub anchored: bool,
    pub has_conjugate: bool,
    pub in_phi_mode: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// D-18 BUILDER
// ─────────────────────────────────────────────────────────────────────────────

impl PhaseHarmonizer {
    /// Build the full D-18 architecture: 9 conjugate pairs
    pub fn new_d18(num_chars: usize, dim: usize) -> Self {
        let mut harmonizer = PhaseHarmonizer::new(num_chars, dim);
        harmonizer.dials.clear();

        let base_delta = 6;

        let pair_names = [
            ("Foundation",   "Foundation-Conj"),
            ("Structure",    "Structure-Conj"),
            ("Rhythm",       "Rhythm-Conj"),
            ("Texture",      "Texture-Conj"),
            ("Motion",       "Motion-Conj"),
            ("Context",      "Context-Conj"),
            ("Resonance",    "Resonance-Conj"),
            ("Harmonic",     "Harmonic-Conj"),
            ("Precision",    "Precision-Conj"),
        ];

        for (pair_idx, (fwd_name, rev_name)) in pair_names.iter().enumerate() {
            let delta = (base_delta as f64 * PHI.powi(pair_idx as i32)).round() as usize;
            let fwd_id = harmonizer.add_dial(fwd_name, delta);
            let _rev_id = harmonizer.add_conjugate(fwd_id, rev_name, delta);
        }

        harmonizer.temporal = TemporalPhaseTracker::new(18, 200);

        // √3 coupling topology — threshold scales with hex distance from center
        // Center pairs couple easily (low threshold), outer pairs are more selective
        let pair_lattice = HexLattice::new(9);
        for pair_idx in 0..9 {
            let dist_from_center = pair_lattice.distances[pair_idx][0];
            // Base: 0.2 at center, +0.1 per ring outward, scaled by 1/√3
            let threshold = SQRT3_INV * (0.2 + 0.1 * dist_from_center as f64);
            let da = pair_idx * 2;
            let db = pair_idx * 2 + 1;
            if da < harmonizer.dials.len() {
                harmonizer.dials[da].coupling_threshold = threshold.clamp(
                    harmonizer.dials[da].coupling_threshold_min,
                    harmonizer.dials[da].coupling_threshold_max,
                );
            }
            if db < harmonizer.dials.len() {
                harmonizer.dials[db].coupling_threshold = threshold.clamp(
                    harmonizer.dials[db].coupling_threshold_min,
                    harmonizer.dials[db].coupling_threshold_max,
                );
            }
        }

        harmonizer
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dial_creation() {
        let dial = Dial::new(0, "Test", 12, 37);
        assert_eq!(dial.delta, 12);
        assert!((dial.tick_width - 2.0 * PI / 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_harmonizer_creation() {
        let h = PhaseHarmonizer::new(37, 256);
        assert_eq!(h.dials.len(), 3);
        assert_eq!(h.dials[0].delta, 12);
        assert_eq!(h.dials[1].delta, 37);
        assert_eq!(h.dials[2].delta, 200);
    }

    #[test]
    fn test_d18_creation() {
        let h = PhaseHarmonizer::new_d18(37, 256);
        assert_eq!(h.dials.len(), 18);
        assert!(h.dials[0].conjugate_id.is_some());
        assert_eq!(h.dials[0].polarity, 1.0);
        assert_eq!(h.dials[1].polarity, -1.0);
    }

    #[test]
    fn test_angular_distance() {
        let h = PhaseHarmonizer::new(37, 256);
        assert!((h.angular_distance(1.0, 1.0)).abs() < 1e-10);
        assert!((h.angular_distance(0.0, PI) - PI).abs() < 1e-10);
        assert!((h.angular_distance(0.1, 2.0 * PI - 0.1) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_bounded_update() {
        assert_eq!(Dial::bounded_update(0.2, -0.5, 0.1, 1.0), 0.1);
        assert_eq!(Dial::bounded_update(0.9, 0.5, 0.1, 1.0), 1.0);
        assert!((Dial::bounded_update(0.5, 0.1, 0.1, 1.0) - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_phase_alignment_same_char() {
        let h = PhaseHarmonizer::new(37, 256);

        let mut scores = vec![0.0; 37];
        scores[5] = 1.0;

        let phase_coarse = h.dials[0].scores_to_phase(&scores);
        let phase_medium = h.dials[1].scores_to_phase(&scores);
        let phase_fine = h.dials[2].scores_to_phase(&scores);

        let universal = 5.0 * 2.0 * PI / 37.0;

        assert!(
            (phase_coarse - universal).abs() < 0.3,
            "Coarse phase {} too far from universal {}",
            phase_coarse, universal
        );
        assert!(
            (phase_medium - universal).abs() < 0.01,
            "Medium phase {} should match universal {}",
            phase_medium, universal
        );
        assert!(
            (phase_fine - universal).abs() < 0.05,
            "Fine phase {} too far from universal {}",
            phase_fine, universal
        );

        let dist_cm = h.angular_distance(phase_coarse, phase_medium);
        let dist_mf = h.angular_distance(phase_medium, phase_fine);

        assert!(
            dist_cm < h.phase_tolerance,
            "Coarse-Medium distance {} exceeds tolerance {}",
            dist_cm, h.phase_tolerance
        );
        assert!(
            dist_mf < h.phase_tolerance,
            "Medium-Fine distance {} exceeds tolerance {}",
            dist_mf, h.phase_tolerance
        );
    }

    #[test]
    fn test_constructive_alignment() {
        let mut h = PhaseHarmonizer::new(37, 256);

        let mut scores_a = vec![0.0; 37];
        let mut scores_b = vec![0.0; 37];
        let mut scores_c = vec![0.0; 37];
        scores_a[5] = 1.0;
        scores_b[5] = 0.9;
        scores_c[5] = 0.8;

        let model_scores = vec![scores_a, scores_b, scores_c];
        let (pred, _conf, alignment) = h.predict(&model_scores, None);

        assert_eq!(pred, 5);
        assert!(alignment.is_some());
        if let Some(event) = alignment {
            assert!(event.constructive);
        }
    }

    #[test]
    fn test_self_tuning_bounds() {
        let mut dial = Dial::new(0, "Test", 12, 37);
        dial.temperature = 0.15;

        for _ in 0..100 {
            dial.accuracy_history.push_back(0.9);
        }
        for _ in 0..50 {
            dial.breathe_temperature();
        }

        assert!(dial.temperature >= dial.temperature_min);
    }

    #[test]
    fn test_phi_graduated_resolutions() {
        let h = PhaseHarmonizer::new_d18(37, 256);
        let forward_deltas: Vec<usize> = h
            .dials
            .iter()
            .step_by(2)
            .map(|d| d.delta)
            .collect();

        for i in 1..forward_deltas.len() {
            assert!(
                forward_deltas[i] > forward_deltas[i - 1],
                "Resolution must increase: {} should be > {}",
                forward_deltas[i], forward_deltas[i - 1]
            );
        }

        for i in 1..forward_deltas.len() {
            let ratio = forward_deltas[i] as f64 / forward_deltas[i - 1] as f64;
            assert!(
                (ratio - PHI).abs() < 0.3,
                "Ratio {} between pair {} and {} should approximate φ",
                ratio, i - 1, i
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // DUAL-FAMILY TESTS
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_hex_lattice_creation() {
        let lattice = HexLattice::new(37);
        assert_eq!(lattice.size, 37);
        assert_eq!(lattice.coords[0].q, 0);
        assert_eq!(lattice.coords[0].r, 0);
        let neighbors = lattice.nearest(0, 6);
        assert_eq!(neighbors.len(), 6);
        for &(_, dist) in &neighbors {
            assert_eq!(dist, 1);
        }
    }

    #[test]
    fn test_hex_distance_symmetry() {
        let lattice = HexLattice::new(20);
        for i in 0..20 {
            for j in 0..20 {
                assert_eq!(lattice.distances[i][j], lattice.distances[j][i]);
            }
            assert_eq!(lattice.distances[i][i], 0);
        }
    }

    #[test]
    fn test_hex_sqrt3_spacing() {
        let lattice = HexLattice::new(7);
        let (cx, cy) = lattice.coords[0].to_cartesian();
        for i in 1..7 {
            let (nx, ny) = lattice.coords[i].to_cartesian();
            let dist = ((nx - cx).powi(2) + (ny - cy).powi(2)).sqrt();
            assert!(
                (dist - SQRT3).abs() < 0.01,
                "Neighbor {} distance {} should ≈ √3", i, dist
            );
        }
    }

    #[test]
    fn test_dual_family_breathing_timing() {
        let h = PhaseHarmonizer::new(37, 256);
        let [inhale, hold, exhale, rest] = h.breath_durations;
        assert!(inhale < hold, "φ-inhale {} < √3-hold {}", inhale, hold);
        assert!(exhale < rest, "φ-exhale {} < √3-rest {}", exhale, rest);
        assert_eq!(inhale, exhale, "Both φ-phases should match");
        assert_eq!(hold, rest, "Both √3-phases should match");
        let total = inhale + hold + exhale + rest;
        assert!((total as f64 - 500.0).abs() < 5.0, "Total ≈ 500, got {}", total);
    }

    #[test]
    fn test_dual_family_constants() {
        assert!((PHI_SQRT3_RATIO - SQRT3 / PHI).abs() < 1e-10);
        assert!((TWIST_ANGLE - 6.0_f64.to_radians()).abs() < 1e-10);
        assert!((FAMILY_BOUNDARY - 2.0 / (PHI * PHI)).abs() < 0.001);
        assert!((SQRT3_MINUS_1 - FAMILY_BOUNDARY).abs() < 0.04);
    }

    #[test]
    fn test_cosine_similarity_basic() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);
        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-10);
        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trig_free_phase_wrap() {
        // ±PI is a branch cut: atan2(0+, -1)=+PI but modular wrap gives -PI.
        // Both represent the same angle, so we test angular equivalence.
        let test_diffs = vec![0.1, -0.1, 3.5, -3.5, 6.0, -6.0, 0.0, PI, -PI];
        for diff in test_diffs {
            let trig_wrap = diff.sin().atan2(diff.cos());
            let fast_wrap = ((diff % (2.0 * PI)) + 3.0 * PI) % (2.0 * PI) - PI;
            let angular_err = ((trig_wrap - fast_wrap + PI).rem_euclid(2.0 * PI) - PI).abs();
            assert!(
                angular_err < 1e-10,
                "Mismatch for diff {}: trig={} fast={}", diff, trig_wrap, fast_wrap
            );
        }
    }

    #[test]
    fn test_d18_has_coupling_topology() {
        let h = PhaseHarmonizer::new_d18(37, 256);
        assert_eq!(h.dials.len(), 18);

        let foundation_threshold = h.dials[0].coupling_threshold;
        let precision_threshold = h.dials[16].coupling_threshold;

        assert!(
            foundation_threshold < precision_threshold,
            "Foundation (center, {:.4}) should have lower coupling threshold \
             than Precision (outer, {:.4})",
            foundation_threshold, precision_threshold
        );

        let thresholds: Vec<f64> = h.dials.iter().map(|d| d.coupling_threshold).collect();
        let all_same = thresholds.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
        assert!(!all_same, "Coupling thresholds should vary with hex topology");
    }

    #[test]
    fn test_calibration_bins() {
        let mut dial = Dial::new(0, "Test", 12, 37);

        // A perfectly calibrated dial: confidence matches accuracy
        for _ in 0..100 {
            dial.record_calibration(0.75, true);  // 75% confident, correct
        }
        for _ in 0..33 {
            dial.record_calibration(0.75, false); // 75% confident, wrong
        }
        // ~100/133 = 75% accuracy in the 0.7-0.8 bin → well calibrated
        let ece = dial.ece();
        assert!(ece < 0.05, "Well-calibrated dial should have low ECE, got {:.4}", ece);

        // A miscalibrated dial: says 90% confident but only 50% accurate
        let mut bad_dial = Dial::new(1, "Bad", 12, 37);
        for _ in 0..50 {
            bad_dial.record_calibration(0.95, true);
        }
        for _ in 0..50 {
            bad_dial.record_calibration(0.95, false);
        }
        let bad_ece = bad_dial.ece();
        assert!(bad_ece > 0.3, "Overconfident dial should have high ECE, got {:.4}", bad_ece);
        assert!(bad_ece > ece, "Overconfident ECE ({:.4}) should exceed calibrated ({:.4})", bad_ece, ece);
    }

    #[test]
    fn test_motion_state_detection() {
        let mut tracker = TemporalPhaseTracker::new(3, 200);
        for i in 0..50 {
            let phases = vec![(i as f64) * 0.5, 0.1, 0.1];
            tracker.record(&phases, &None);
        }
        assert!(
            tracker.phase_velocity[0] > tracker.phase_velocity[1],
            "Fast dial velocity {} should exceed static dial {}",
            tracker.phase_velocity[0], tracker.phase_velocity[1]
        );
    }
}
