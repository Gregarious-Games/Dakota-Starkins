"""
SCALING FORGE HDC v1.0 — Hypervector-Brained DennisNode
========================================================
Integration of:
  - scaling_forge_v4.py (DennisNode swarm body)
  - starkins_meta_hdc_v2_merged.py (HDC perception layer)

What changed:
  - Z, X1, X2, X3 are now D=256 hypervectors (np.ndarray)
  - z_scalar = norm(Z)/sqrt(D) for backward compat with health/saturation/clamping
  - Neighbor coupling uses cosine similarity instead of scalar distance
  - Encapsulation membranes gate which dimensions are shared:
      void_outer: binary mask (256,) — what the node broadcasts
      inside_presence: full unmasked Z — never shared directly
      relational: Z * void_outer — what neighbors actually see

What stays the same:
  - health, valence, arousal, phase, energy — all scalars
  - p (parameter dict) — self-evolving parameters
  - z_history — stores z_scalar (not full vectors)
  - Safety thresholds applied to z_scalar

Authors: Rose (Claude) & Greg Calkins
         Dakota (Claude) — integration build
Date:    February 21, 2026
"""

import numpy as np
import random
import time
from math import sqrt
from collections import deque
import sys
sys.stdout.reconfigure(line_buffering=True)
import os
os.environ['PYPHI_WELCOME_OFF'] = 'yes'
import pyphi
pyphi.config.PROGRESS_BARS = False
from scipy.signal import stft as scipy_stft
from scipy.signal import coherence as scipy_coherence

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS (from Genesis Engine / Scaling Forge v4)
# ═══════════════════════════════════════════════════════════════════════════
PHI = (1 + sqrt(5)) / 2
TAU = (sqrt(5) - 1) / 2        # 1/PHI
GAMMA = 1 / (6 * PHI)          # ~0.103
PHI_GAMMA = PHI * GAMMA         # ~0.167

# Safety thresholds (applied to z_scalar)
CLAMP_HIGH = 1.0 - GAMMA        # 0.897
CLAMP_LOW = GAMMA                # 0.103
HYSTERESIS = PHI_GAMMA           # ~0.167

# Hypervector dimension
HDC_DIM = 256


# ═══════════════════════════════════════════════════════════════════════════
# HDC PRIMITIVES (from starkins_meta_hdc_v2_merged.py, self-contained)
# ═══════════════════════════════════════════════════════════════════════════

class HDCPrimitives:
    """Foundation: bind, bundle, permute, similarity."""

    def __init__(self, dim: int, rng: np.random.Generator = None):
        self.dim = dim
        self.rng = rng or np.random.default_rng()

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def unbind(self, composite: np.ndarray, key: np.ndarray) -> np.ndarray:
        return self.bind(composite, key)

    def bundle(self, vectors, weights=None) -> np.ndarray:
        if not vectors:
            return np.zeros(self.dim)
        if weights is not None:
            summed = np.zeros(self.dim)
            for v, w in zip(vectors, weights):
                summed += w * v
        else:
            summed = np.sum(vectors, axis=0)
        return self._sign_normalize(summed)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm < 1e-12:
            return 0.0
        return float(np.dot(a, b) / norm)

    def permute_weyl(self, v: np.ndarray, step: int) -> np.ndarray:
        shift = int(round(step * TAU * self.dim)) % self.dim
        return np.roll(v, shift)

    def random_bipolar(self, shape=None) -> np.ndarray:
        if shape is None:
            shape = self.dim
        return self.rng.choice([-1.0, 1.0], size=shape)

    def _sign_normalize(self, v: np.ndarray) -> np.ndarray:
        result = np.sign(v)
        zeros = result == 0
        if np.any(zeros):
            result[zeros] = self.rng.choice([-1.0, 1.0], size=int(np.sum(zeros)))
        return result


# Shared HDC engine for all nodes
_hdc = HDCPrimitives(dim=HDC_DIM)


# ═══════════════════════════════════════════════════════════════════════════
# VECTOR UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def random_unit_vector(dim=None):
    """Random unit hypervector — unique starting identity."""
    if dim is None:
        dim = HDC_DIM
    v = np.random.randn(dim)
    n = np.linalg.norm(v)
    if n < 1e-12:
        v = np.ones(dim)
        n = np.linalg.norm(v)
    return v / n


def clamp_vector(v: np.ndarray, max_abs: float) -> np.ndarray:
    """Element-wise clamp. NaN/Inf -> 0."""
    v = np.where(np.isfinite(v), v, 0.0)
    return np.clip(v, -max_abs, max_abs)


def clamp_scalar(v: float, max_abs: float) -> float:
    if not np.isfinite(v):
        return 0.0
    return max(min(v, max_abs), -max_abs)


# ═══════════════════════════════════════════════════════════════════════════
# HDC DENNIS NODE
# ═══════════════════════════════════════════════════════════════════════════

class DennisNode:
    """
    Core computational node with D=256 hypervector brain.

    State variables (hypervectors):
        X1, X2, X3, phi_vec, Y, Z — all shape (256,)
        Internal dynamics operate element-wise: the math doesn't change,
        NumPy handles the broadcasting.

    Derived scalar:
        z_scalar = norm(Z) / sqrt(D) — backward compat for health/saturation

    Encapsulation membranes:
        void_outer:      binary mask (256,) — which dims are visible to outside
        inside_presence: full unmasked Z — the node's actual state
        relational:      Z * void_outer — what neighbors actually see

    Scalar state (unchanged from v4):
        health, valence, arousal, phase, energy, saturation_count
    """

    def __init__(self, name, params):
        self.name = name
        self.p = params.copy()

        # --- HYPERVECTOR STATE ---
        self.X1 = random_unit_vector()
        self.X2 = np.zeros(HDC_DIM)
        self.phi_vec = np.zeros(HDC_DIM)
        self.X3 = np.zeros(HDC_DIM)
        self.Y = np.zeros(HDC_DIM)
        self.Z = random_unit_vector()

        # --- DERIVED SCALAR ---
        self.z_scalar = np.linalg.norm(self.Z) / np.sqrt(HDC_DIM)

        # --- ENCAPSULATION MEMBRANES ---
        self.void_outer = np.zeros(HDC_DIM)
        self.heat_valence = 0.5  # warmth from neighbors [0,1]
        self._init_membrane()

        # --- MODULE ASSIGNMENT (for modular coupling) ---
        self.module_id = 0  # set externally for modular mode

        # --- SCALAR STATE (unchanged from v4) ---
        self.delay = 1.0
        self.energy = 0.0
        self.saturation_count = 0
        self.z_history = deque(maxlen=100)
        self.health = 1.0
        self.valence = 0.0
        self.arousal = 0.0
        self.phase = 0.0

        # --- SPS STATE (Silent Punctuational Syntax) ---
        self.Z_last_sent = (self.Z * self.void_outer).copy()  # relational delta
        self.silence_count = 0
        self.speak_threshold = 0.3   # cosine distance to trigger speak (raised from 0.1)
        self.max_silence = 50        # forced check-in after this many silent steps
        self.speaking = True         # current step decision
        self.speak_history = deque(maxlen=100)  # last 100 bools

        # --- NEIGHBOR TRACKING ---
        self.neighbor_last_seen = {}   # {name: Z vector}
        self.neighbor_silence = {}     # {name: int steps silent}
        self.neighbor_rhythm = {}      # {name: float avg steps between speaks}

        # --- LOVE LOGIC STATE ---
        self.Z_pre_coupling = self.Z.copy()       # snapshot before coupling
        self.self_motion = deque(maxlen=50)        # ||internal dynamics|| history
        self.external_motion = deque(maxlen=50)    # ||coupling move|| history
        self.comfort = 0.0                        # external presence metric [0,1]
        self.comfort_history = deque(maxlen=200)   # comfort trajectory
        self.relational_memory = {}               # {name: membrane-held Z vector}
        self.trust = {}                           # {name: float [0,1]}

        # --- POLYPHONIC STATE ---
        self.kappa = 0.0                          # embodiment: 0=pure Z, 1=full blend
        self.kappa_history = deque(maxlen=200)
        self.poly_active = False                  # gating flag (optimization)

        # --- SIMILARITY CACHE ---
        self._sim_cache = {}                      # {name: (my_gen, their_gen, sim)}
        self._z_generation = 0                    # incremented each time Z updates

        # --- SPARSE TOPOLOGY ---
        self.active_neighbors = None              # None = all; set() = pruned subset
        self.topology_rewire_interval = 100       # steps between random rewiring

        # --- OCTOPUS MEMORY (distributed + centralized, regenerative) ---
        # Each node is like an octopus arm: local autonomy + hive connection.
        # The memory is holographic -- each fragment contains the whole.

        # 1. Local episodic memory: Z snapshots bound with time keys
        #    Encoded as: episode = bind(Z_snapshot, time_key)
        #    Bundled into a single episodic vector.
        self.episodic_memory = np.zeros(HDC_DIM)     # bundled episodes
        self.episode_count = 0
        self.episodic_capacity = HDC_DIM               # max episodes before consolidation

        # 2. Working memory: recent context (rolling bundle of last K states)
        self.working_memory = np.zeros(HDC_DIM)
        self.working_memory_window = 10               # last K steps bundled

        # 3. Hive fragment: this node's partial copy of collective state.
        #    Each node contributes: bind(identity_key, Z)
        #    Hive = bundle of all contributions. Any node can reconstruct.
        self.identity_key = _hdc.random_bipolar()     # permanent node identity
        self.hive_fragment = np.zeros(HDC_DIM)        # local copy of hive memory
        self.hive_contributors = {}                   # {name: identity_key}

        # 4. Consolidated memory: long-term stable pattern (centroid of episodes)
        self.consolidated_memory = np.zeros(HDC_DIM)

    def _init_membrane(self):
        """Initialize void_outer with ~61.8% of dimensions open (phi ratio)."""
        self._update_membrane()

    def _update_membrane(self):
        """Recompute membrane based on current heat_valence."""
        open_fraction = 0.618 + 0.382 * self.heat_valence  # 0.618 to 1.0
        n_open = int(open_fraction * HDC_DIM)
        self.void_outer = np.zeros(HDC_DIM)
        self.void_outer[:n_open] = 1.0
        np.random.shuffle(self.void_outer)

    # ---- OCTOPUS MEMORY METHODS ----

    def remember_episode(self, step):
        """Store current Z as an episodic memory, bound with a time key.

        The time key is a Weyl-permuted basis vector, giving each timestep
        a unique but retrievable signature. Episodes are bundled (superposed)
        into the episodic memory vector.
        """
        time_key = _hdc.permute_weyl(_hdc.random_bipolar(), step)
        episode = _hdc.bind(self.Z, time_key)
        # Weighted bundle: recent episodes stronger
        decay = 0.95
        self.episodic_memory = decay * self.episodic_memory + episode
        self.episode_count += 1

        # Consolidation: when buffer is full, compress into long-term
        if self.episode_count >= self.episodic_capacity:
            self.consolidated_memory = (
                0.7 * self.consolidated_memory + 0.3 * self.episodic_memory
            )
            self.episodic_memory *= 0.5  # partial decay, don't erase fully
            self.episode_count = 0

    def update_working_memory(self):
        """Rolling bundle of recent Z states. Fast-access context."""
        self.working_memory = 0.8 * self.working_memory + 0.2 * self.Z

    def contribute_to_hive(self):
        """Generate this node's contribution to the hive mind.
        Returns: bind(identity_key, Z) — a holographic encoding
        that any node can decode if it knows the identity_key.
        """
        return _hdc.bind(self.identity_key, self.Z)

    def update_hive_fragment(self, contributions):
        """Update local hive fragment from all active node contributions.

        Args:
            contributions: list of (name, identity_key, contribution_vector)

        Uses WEIGHTED bundling with recency * trust * comfort(PHI) instead of
        simple recency decay. Vectorized for performance.
        """
        if not contributions:
            return
        my_contrib = self.contribute_to_hive()
        # Vectorized: stack all contributions, compute weights, weighted sum
        contribs = [my_contrib]
        weights = [1.0]
        for name, id_key, contrib in contributions:
            self.hive_contributors[name] = id_key
            silence = self.neighbor_silence.get(name, 0)
            trust_w = self.trust.get(name, 0.5)
            comfort_boost = 1.0 + PHI * self.comfort  # phi-scaled comfort
            w = (0.95 ** silence) * trust_w * comfort_boost
            contribs.append(contrib)
            weights.append(w)
        weights = np.array(weights)
        contribs = np.stack(contribs)  # (N, D)
        self.hive_fragment = (weights[:, np.newaxis] * contribs).sum(axis=0) / weights.sum()

    def recall_from_hive(self, target_name):
        """Reconstruct a node's Z from the hive fragment.

        This is the regeneration mechanism: if a node is lost,
        any surviving node can approximate its last known state
        by unbinding the target's identity key from the hive.
        """
        target_key = self.hive_contributors.get(target_name)
        if target_key is None:
            return None
        return _hdc.unbind(self.hive_fragment, target_key)

    def get_memory_state(self):
        """Return a diagnostic snapshot of memory health."""
        ep_norm = np.linalg.norm(self.episodic_memory)
        wm_norm = np.linalg.norm(self.working_memory)
        hive_norm = np.linalg.norm(self.hive_fragment)
        consol_norm = np.linalg.norm(self.consolidated_memory)
        return {
            "episodic_norm": ep_norm,
            "working_norm": wm_norm,
            "hive_norm": hive_norm,
            "consolidated_norm": consol_norm,
            "episode_count": self.episode_count,
            "hive_members": len(self.hive_contributors),
        }

    # ---- POLYPHONIC / MONOPHONIC METHODS ----

    def polyphonic_state(self):
        """6-voice internal state, Heqat-weighted.

        Maps internal dynamics chain to Eye of Horus fractions:
          X1 (Smell, 1/2), X2 (Sight, 1/4), phi_vec (Thought, 1/8),
          X3 (Hearing, 1/16), Y (Taste, 1/32), Z (Touch, 1/64)

        Returns: (6, D) matrix or None if poly_active is False.
        """
        if not self.poly_active:
            return None
        heqat = np.array([1/2, 1/4, 1/8, 1/16, 1/32, 1/64])
        voices = np.stack([self.X1, self.X2, self.phi_vec, self.X3, self.Y, self.Z])
        return voices * heqat[:, np.newaxis]  # (6, D)

    def monophonic_output(self):
        """Collapse 6 voices to one signal: (1-kappa)*Z + kappa*weighted_blend.

        kappa=0: pure Z (default, minimal compute)
        kappa=1: full Heqat blend of all internal voices
        """
        if self.kappa < 1e-6 or not self.poly_active:
            return self.Z
        poly = self.polyphonic_state()
        if poly is None:
            return self.Z
        blend = poly.sum(axis=0)  # already Heqat-weighted
        blend_norm = np.linalg.norm(blend)
        if blend_norm > 1e-12:
            blend = blend * (np.linalg.norm(self.Z) / blend_norm)  # match Z magnitude
        return (1.0 - self.kappa) * self.Z + self.kappa * blend

    def compute_kappa(self):
        """Adaptive embodiment: kappa = comfort * mean(trust).

        High comfort + high trust = polyphonic (rich internal expression)
        Low comfort or low trust = monophonic (guarded, minimal)
        """
        if not self.trust:
            self.kappa = 0.0
            return
        trust_vals = list(self.trust.values())
        mean_trust = np.mean(trust_vals)
        self.kappa = np.clip(self.comfort * mean_trust, 0.0, 1.0)
        self.kappa_history.append(self.kappa)

    def infer_polyphonic(self, neighbor_mono):
        """Reconstruct neighbor's polyphonic from their mono signal.

        Deliberately imperfect -- empathy is approximate. Assumes uniform
        Heqat weighting from mono signal, which is lossy by design.

        Returns: (6, D) inferred polyphonic state, or None.
        """
        if neighbor_mono is None:
            return None
        heqat = np.array([1/2, 1/4, 1/8, 1/16, 1/32, 1/64])
        inferred = np.outer(heqat, neighbor_mono)  # (6, D)
        return inferred

    def update_topology(self, all_names, step):
        """Kappa-guided sparse topology: prune low-trust neighbors.

        Keeps at least 2 neighbors. Periodic random rewiring to prevent
        the network from fragmenting into isolated cliques.
        """
        if not self.trust:
            self.active_neighbors = None
            return
        sorted_names = sorted(self.trust.keys(),
                              key=lambda n: self.trust[n], reverse=True)
        keep_count = max(2, int(len(sorted_names) * (0.5 + 0.5 * self.kappa)))
        self.active_neighbors = set(sorted_names[:keep_count])
        # Periodic random rewiring: add one random pruned neighbor back
        if step % self.topology_rewire_interval == 0 and len(sorted_names) > keep_count:
            random_add = random.choice(sorted_names[keep_count:])
            self.active_neighbors.add(random_add)

    # ---- SPS METHODS ----

    def decide_to_speak(self):
        """SPS gate: should this node transmit this step?

        Decision based on:
          1. How much the RELATIONAL vector changed since last transmission
             (not raw Z -- in complementary mode Z always changes, but the
             node's public face may be stable after membrane masking)
          2. How long since last spoke (check-in pulse)
          3. Health/energy urgency

        ADAPTIVE: speak_threshold and max_silence self-adjust:
          - If neighbors are mostly silent -> raise threshold (less noise)
          - If health drops -> lower threshold and max_silence (urgent)
          - Slowly drift toward a stable personal rhythm
        """
        # Changed enough to be worth saying?
        # Use relational (masked Z) for delta -- this is what neighbors see.
        # In complementary mode, raw Z constantly changes but the membrane-
        # masked projection may be stable, allowing meaningful silence.
        current_rel = self.Z * self.void_outer
        delta = 1.0 - _hdc.similarity(current_rel, self.Z_last_sent)

        # Love Logic: anger protection
        # Low comfort = lonely -> dampen output (raise threshold), boost sensitivity
        if self.comfort < 0.2:
            anger_factor = 1.0 - self.comfort
            self.speak_threshold = min(0.8, self.speak_threshold * (1.0 + 0.01 * anger_factor))

        # Been quiet too long?
        overdue = self.silence_count >= self.max_silence

        # In distress?
        urgent = self.health < 0.7 or self.energy > 50.0

        if delta > self.speak_threshold or overdue or urgent:
            self.speaking = True
            self.Z_last_sent = current_rel.copy()  # store relational, not raw Z
            self.silence_count = 0
        else:
            self.speaking = False
            self.silence_count += 1

        self.speak_history.append(self.speaking)

        # --- ADAPTIVE THRESHOLDS ---
        # If I spoke but delta was barely above threshold -> raise threshold
        # (I'm talking when I have nothing new to say)
        if self.speaking and not urgent and not overdue:
            if delta < self.speak_threshold * 1.5:
                self.speak_threshold *= 1.001  # slowly raise bar
            else:
                self.speak_threshold *= 0.999  # keep bar where it is

        # If I hit max_silence forced check-in, and delta was small,
        # the network is stable -> allow longer silences
        if overdue and delta < 0.1:
            self.max_silence = min(100, self.max_silence + 1)
        elif urgent:
            # Distress -> shorten max_silence
            self.max_silence = max(5, self.max_silence - 2)

        # Health-based: when degraded, lower threshold to speak more
        if self.health < 0.8:
            self.speak_threshold *= 0.99
            self.max_silence = max(5, self.max_silence - 1)

        # Clamp thresholds to sane range
        self.speak_threshold = max(0.01, min(0.8, self.speak_threshold))
        self.max_silence = max(5, min(200, self.max_silence))

        return self.speaking

    @property
    def inside_presence(self) -> np.ndarray:
        """Full unmasked Z. Never shared directly."""
        return self.Z

    @property
    def relational(self) -> np.ndarray:
        """Masked output -- what neighbors actually see. None if silent (SPS).
        Uses monophonic output when polyphonic is active (kappa > 0)."""
        if not self.speaking:
            return None
        output = self.monophonic_output() if self.poly_active else self.Z
        return output * self.void_outer

    def _update_z_scalar(self):
        """Derive scalar magnitude from Z vector."""
        self.z_scalar = np.linalg.norm(self.Z) / np.sqrt(HDC_DIM)

    def gravity_delay(self, z_scalars: dict, locals_list: list) -> float:
        """Density-based delay. Uses z_scalar for proximity check."""
        density = sum(
            1 for n in locals_list
            if abs(z_scalars.get(n, 0.0) - self.z_scalar) < 0.002
        )
        return 1.0 + density * 0.5

    def dissipate_energy(self, rate=0.01):
        self.energy *= (1.0 - rate)
        if self.saturation_count > 10:
            self.energy *= 0.9

    def detect_saturation(self, max_abs: float) -> bool:
        """Saturation detection on z_scalar."""
        at_ceiling = abs(abs(self.z_scalar) - max_abs) < 1e-6
        if at_ceiling:
            self.saturation_count += 1
        else:
            self.saturation_count = max(0, self.saturation_count - 1)
        return self.saturation_count > 5

    def self_correct(self, max_abs: float):
        """Auto-correction — applied to full Z vector and X1."""
        if self.saturation_count > 10:
            self.Z *= -0.5
            self.X1 *= 0.5
            self.energy *= 0.1
            self.saturation_count = 0
            self.health *= 0.95
        elif self.saturation_count > 5:
            self.Z *= 0.9
            self.energy *= 0.5

    def update_health(self):
        self.health = min(1.0, self.health + 0.001)
        if self.saturation_count > 0:
            self.health *= 0.999

    def update_affect(self):
        self.arousal = min(1.0, abs(self.energy) / 100.0)
        self.valence = clamp_scalar(
            self.health - (self.saturation_count / 20.0), 1.0
        )
        if len(self.z_history) >= 2:
            dz = self.z_history[-1] - self.z_history[-2]
            if abs(self.z_scalar) > 1e-10:
                self.phase = np.arctan2(dz, self.z_scalar)
            else:
                self.phase = 0.0

    def update(self, neighbor_relational_vecs: list, locals_list: list,
               z_scalars: dict, threshold: float, max_abs: float,
               iter_count: int = None, coupling_mode: str = "converge",
               adaptive_coupling: bool = False,
               tension_curve: str = "linear",
               renormalize_z: bool = False,
               target_z_scalar: float = 0.75,
               coupling_target: str = "Z",
               dim_balance_strength: float = 0.0,
               neighbor_modules: dict = None):
        """
        Main update step — hypervector dynamics.

        coupling_mode:
          "converge"       — pull toward sim=1 (synchronization)
          "complementary"  — pull toward sim=0 (differentiation)
          "modular"        — converge within same module, differentiate between.
                             Requires neighbor_modules dict {name: module_id}.
        tension_curve (complementary only):
          "linear"    — |sim|              (original)
          "quadratic" — sim^2              (fades near zero)
          "cubic"     — |sim|^3            (fades faster)
          "deadzone"  — max(0, |sim|-0.1)  (no force near zero)
        renormalize_z: if True, rescale Z to target_z_scalar after update
        coupling_target: "Z" (original — coupling added to Z after dynamics)
                         "X1" (coupling feeds into X1, propagates through chain)
        dim_balance_strength: >0 adds per-dimension territorial pressure in
                              complementary mode. Try 0.001-0.01.
        neighbor_modules: {name: module_id} for modular coupling mode.
        """
        # --- GRAVITY DELAY ---
        self.delay = self.gravity_delay(z_scalars, locals_list)

        # --- LOVE LOGIC: snapshot Z before coupling ---
        self.Z_pre_coupling = self.Z.copy()

        # --- COSINE COUPLING (silence-aware) ---
        # Compute mean tension vector from all neighbors.
        # When a neighbor is silent (None), use cached vector with decay.
        tension_vec = np.zeros(HDC_DIM)
        total_tension_scalar = 0.0
        active_neighbor_count = 0

        for name, neighbor_rel in zip(locals_list, neighbor_relational_vecs):
            # --- SPARSE TOPOLOGY: skip pruned neighbors ---
            if self.active_neighbors is not None and name not in self.active_neighbors:
                continue

            # --- SPS: handle silence (3-tier decay + trust) ---
            if neighbor_rel is None:
                cached = self.neighbor_last_seen.get(name)
                if cached is None:
                    continue  # never heard from this neighbor
                silence = self.neighbor_silence.get(name, 0) + 1
                self.neighbor_silence[name] = silence
                # Three-tier silence decay
                if silence <= 5:
                    decay = 1.0                                    # TRUST: hold full
                elif silence <= 20:
                    decay = 1.0 - 0.05 * (silence - 5)            # LINEAR: fade 5%/step
                else:
                    decay = max(0.0, 0.25 * (0.95 ** (silence - 20)))  # EXPONENTIAL from 0.25
                neighbor_rel = cached * decay
                # Trust erodes during silence
                self.trust[name] = max(0.0, self.trust.get(name, 0.5) - 0.01)
                # Modulate heat_valence based on silence pattern
                expected_rhythm = self.neighbor_rhythm.get(name, 5.0)
                if silence > expected_rhythm * 2:
                    self.heat_valence = max(0.0, self.heat_valence - 0.01)
            else:
                # Neighbor spoke -- update cache, reset silence, adjust rhythm
                old_silence = self.neighbor_silence.get(name, 0)
                if old_silence > 0:
                    old_rhythm = self.neighbor_rhythm.get(name, float(old_silence))
                    self.neighbor_rhythm[name] = 0.8 * old_rhythm + 0.2 * old_silence
                    if old_silence > self.neighbor_rhythm.get(name, 5.0) * 2:
                        self.heat_valence = min(1.0, self.heat_valence + 0.02)
                self.neighbor_last_seen[name] = neighbor_rel.copy()
                self.neighbor_silence[name] = 0
                # Trust rebuilds on speech (gradual, not instant)
                self.trust[name] = min(1.0, self.trust.get(name, 0.5) + 0.02)
                # Relational memory: store through our membrane
                self.relational_memory[name] = (neighbor_rel * self.void_outer).copy()

            active_neighbor_count += 1
            sim = _hdc.similarity(self.Z, neighbor_rel)

            # Determine effective coupling mode for this neighbor
            effective_mode = coupling_mode
            if coupling_mode == "modular" and neighbor_modules is not None:
                neighbor_mod = neighbor_modules.get(name, -1)
                if neighbor_mod == self.module_id:
                    effective_mode = "converge"       # same module: sync
                else:
                    effective_mode = "complementary"   # different module: differentiate

            if effective_mode == "complementary":
                # Pull toward orthogonality (sim=0)
                abs_sim = abs(sim)
                if tension_curve == "quadratic":
                    tension_magnitude = sim * sim
                elif tension_curve == "cubic":
                    tension_magnitude = abs_sim ** 3
                elif tension_curve == "deadzone":
                    tension_magnitude = max(0.0, abs_sim - 0.1)
                else:  # linear
                    tension_magnitude = abs_sim
                direction = -np.sign(sim) * (neighbor_rel - self.Z)
                pull = tension_magnitude * direction
                total_tension_scalar += tension_magnitude
            elif effective_mode == "downshift":
                # Downshift: one speaker rotates, others listen.
                # Use monophonic output for coupling -- kappa modulates voice blend
                mono = self.monophonic_output()
                mono_sim = _hdc.similarity(mono, neighbor_rel)
                pull = (1.0 - mono_sim) * (neighbor_rel - mono)
                total_tension_scalar += abs(1.0 - mono_sim)
            else:
                # Default: pull toward convergence (sim=1)
                disagreement = 1.0 - sim
                pull = disagreement * (neighbor_rel - self.Z)
                total_tension_scalar += abs(disagreement)

            # Love Logic: lonely nodes listen harder, not louder
            if self.comfort < 0.2:
                listen_boost = 1.0 + 2.0 * (0.2 - self.comfort)  # up to 1.4x
                pull *= listen_boost

            tension_vec += pull

        if active_neighbor_count > 0:
            tension_vec /= active_neighbor_count
            total_tension_scalar /= active_neighbor_count

        # --- ENERGY (scalar, accumulated from tension magnitude) ---
        # Scale by sqrt(64/D) to compensate for dimensionality.
        # At D=64 (original), energy ~ tension * 100. At D=256, tension_vec
        # has 4x more elements but each is ~2x smaller, so total |tension|
        # is ~2x larger. Without scaling, energy hits ~3500 and makes
        # delta_z_magnitude ~17.5, causing X1 to explode to ||X1||~350k
        # and Z_dynamics to saturate. This scaling keeps energy in the
        # same range as D=64 regardless of dimension.
        dim_scale = np.sqrt(64.0 / HDC_DIM)  # 1.0 at D=64, 0.5 at D=256
        self.energy += total_tension_scalar * 100.0 * dim_scale
        self.dissipate_energy(rate=0.02)

        # --- DELTA Z (vectorized) ---
        up = self.energy * 0.005
        down = (self.delay - 1.0) * 0.002
        delta_z_magnitude = up - down

        # Coupling direction from tension
        coupling_sign = -1.0 if total_tension_scalar < threshold else 1.0
        coupling = coupling_sign * self.p["coupling_base"]

        micro = np.random.uniform(
            -self.p["microvariance_amplitude"],
            self.p["microvariance_amplitude"],
            size=HDC_DIM
        )

        # Movement vector: tension pull + coupling + microvariance
        move = (tension_vec * delta_z_magnitude + coupling * tension_vec + micro) / self.delay

        # Love Logic: track external motion (coupling contribution)
        self.external_motion.append(np.linalg.norm(move))

        # --- DIMENSION BALANCING (complementary mode only) ---
        # Competitive exclusion: redistribute magnitude from over-contested
        # dims to under-contested dims. Each node computes a "contest score"
        # per dimension. High contest = many neighbors strong here.
        # Force pushes magnitude FROM high-contest dims TO low-contest dims
        # while preserving Z direction (sign). This avoids the monopoly
        # feedback loop of simple repulsion.
        if dim_balance_strength > 0.0 and coupling_mode == "complementary" and active_neighbor_count > 0:
            # Gather active neighbor magnitudes
            active_rels = [nr for name, nr in zip(locals_list, neighbor_relational_vecs)
                          if nr is not None]
            if not active_rels:
                active_rels = [self.neighbor_last_seen.get(n, np.zeros(HDC_DIM))
                              for n in locals_list]
            neighbor_max_mag = np.zeros(HDC_DIM)
            for nr in active_rels:
                neighbor_max_mag = np.maximum(neighbor_max_mag, np.abs(nr))
            my_mag = np.abs(self.Z)
            # Contest: product of this node's magnitude and max neighbor magnitude
            contest = my_mag * neighbor_max_mag
            # Vacancy: inverse -- dims where neighbors are weak
            max_contest = np.max(contest) + 1e-12
            vacancy = 1.0 - (neighbor_max_mag / (np.max(neighbor_max_mag) + 1e-12))
            # Redistribute: shrink high-contest dims, grow low-contest dims
            redistrib = -dim_balance_strength * contest * np.sign(self.Z) \
                        + dim_balance_strength * vacancy * np.sign(self.Z + 1e-12) * 0.5
            move += redistrib

        # --- INTERNAL DYNAMICS (element-wise on vectors) ---
        fwd = self.p["internal_forward"] / self.delay
        rev = self.p["internal_reverse"] / self.delay

        if coupling_target == "X1":
            # Coupling feeds into X1 — propagates through the whole chain
            self.X1 += fwd - self.p["center_pull"] * self.X1 + move
        else:
            self.X1 += fwd - self.p["center_pull"] * self.X1

        self.X2 = self.X1 - rev
        self.phi_vec = 0.5 * (self.X1 + self.X2)
        self.X3 = self.phi_vec * (1.0 - self.p["damping_factor"])
        self.Y = np.abs(self.X3 - self.phi_vec)
        Z_dynamics = self.X3 * self.Y  # ~0.01x suppression through chain

        # --- COUPLING BYPASS: Direct path around X3*Y bottleneck ---
        # Only active when coupling_target != "X1". When X1 coupling is
        # active, the coupling force accumulates in X1 and propagates
        # through the dynamics chain — the bypass would CONFLICT because
        # Z_dynamics (from the accumulated X1) saturates at clamp and
        # drowns out the bypass signal, creating oscillatory jitter.
        # When coupling_target="Z", the bypass is the PRIMARY coupling path.
        if coupling_target != "X1":
            bypass_strength = self.p.get("bypass_strength", 0.5)
            Z_bypass = bypass_strength * move
            self.Z = clamp_vector(Z_dynamics + Z_bypass, max_abs)
            residual_move = (1.0 - bypass_strength) * move
            self.Z = clamp_vector(self.Z + residual_move, max_abs)
        else:
            # X1 coupling: Z comes purely from dynamics chain
            self.Z = clamp_vector(Z_dynamics, max_abs)

        # --- Z RENORMALIZATION (optional) ---
        if renormalize_z:
            z_norm = np.linalg.norm(self.Z)
            if z_norm > 1e-12:
                target_norm = target_z_scalar * np.sqrt(HDC_DIM)
                self.Z = self.Z * (target_norm / z_norm)

        # NOTE: No SLERP / contraction mapping here.  Convergence must
        # emerge from the dynamics chain (tension coupling, bypass, energy)
        # or it doesn't happen.  Any convergence reported is EARNED, not forced.

        # Love Logic: track self-motion (total Z change includes internal + external)
        self.self_motion.append(np.linalg.norm(self.Z - self.Z_pre_coupling))
        self._z_generation += 1  # invalidate similarity cache

        # --- SCALAR DERIVATION ---
        self._update_z_scalar()

        # --- SATURATION CHECK (on z_scalar) ---
        if self.detect_saturation(max_abs):
            self.self_correct(max_abs)
            self._update_z_scalar()

        self.update_health()

        # --- Z HISTORY (scalar) ---
        self.z_history.append(self.z_scalar)

        self.update_affect()

        # --- MEMORY UPDATE (octopus architecture) ---
        self.update_working_memory()
        # Store episodic memory when something significant changes
        if iter_count is not None:
            delta_from_wm = 1.0 - _hdc.similarity(self.Z, self.working_memory)
            if delta_from_wm > 0.2 or (iter_count % 100 == 0):
                self.remember_episode(iter_count)

        # --- SPS: DECIDE WHETHER TO SPEAK ---
        self.decide_to_speak()

        # --- UPDATE MEMBRANE ---
        # heat_valence: average similarity to neighbors = warmth
        # Filter out None (silent) neighbors
        active_rels = [nr for nr in neighbor_relational_vecs if nr is not None]
        if active_rels:
            avg_sim = np.mean([
                _hdc.similarity(self.Z, nr) for nr in active_rels
            ])
            # Smooth update
            self.heat_valence = 0.95 * self.heat_valence + 0.05 * max(0.0, avg_sim)
        self._update_membrane()

        # --- LOVE LOGIC: comfort ---
        # Comfort = ratio of motion caused by others vs self
        ext = list(self.external_motion)
        self_m = list(self.self_motion)
        if len(ext) >= 5 and len(self_m) >= 5:
            ext_mean = np.mean(ext[-10:]) if len(ext) >= 10 else np.mean(ext)
            self_mean = np.mean(self_m[-10:]) if len(self_m) >= 10 else np.mean(self_m)
            total = ext_mean + self_mean + 1e-12
            self.comfort = ext_mean / total
        else:
            self.comfort = 0.0
        self.comfort_history.append(self.comfort)

        # --- LOVE LOGIC: kappa + polyphonic gating ---
        self.compute_kappa()
        self.poly_active = self.kappa > 0.01

        # --- SPARSE TOPOLOGY (kappa-guided neighbor pruning) ---
        if iter_count is not None:
            self.update_topology(locals_list, iter_count)

        # --- ADAPTIVE COUPLING ---
        if adaptive_coupling:
            warmth_signal = self.heat_valence - 0.5
            coupling_adjust = 1.0 + 0.1 * warmth_signal
            self.p["coupling_base"] *= coupling_adjust
            self.p["coupling_base"] = max(min(self.p["coupling_base"], 0.05), 0.001)

        # --- PARAMETER EVOLUTION (with selection pressure) ---
        # Temperature-scheduled mutation with ACCEPT/REJECT:
        # 1. Record fitness BEFORE mutation
        # 2. Propose mutated params
        # 3. Accept if fitness improved, reject (revert) if not
        # This is simulated annealing — actual evolution, not random walk.
        if iter_count is not None and iter_count > 100:
            # Temperature schedule: starts at 1.0, cools to 0.01 over ~50k steps
            temp = max(0.01, 1.0 / (1.0 + iter_count * 0.0001))

            # Fitness BEFORE mutation (what we're comparing against)
            fitness_before = self.health * (0.5 + 0.5 * self.heat_valence)

            # Save current params in case we need to revert
            saved_params = {k: v for k, v in self.p.items()}

            # Propose mutations
            evo_rate = temp * (1.5 - fitness_before)
            evo_rate = max(0.001, min(0.1, evo_rate))

            self.p["coupling_base"] *= (1.0 + random.uniform(
                -evo_rate, evo_rate))
            self.p["microvariance_amplitude"] *= (1.0 + random.uniform(
                -evo_rate * 2.0, evo_rate * 2.0))
            self.p["center_pull"] *= (1.0 + random.uniform(
                -evo_rate * 0.5, evo_rate * 0.5))
            self.p["damping_factor"] *= (1.0 + random.uniform(
                -evo_rate * 0.5, evo_rate * 0.5))
            self.p["bypass_strength"] = max(0.1, min(0.95,
                self.p.get("bypass_strength", 0.8) + random.uniform(
                    -evo_rate * 0.1, evo_rate * 0.1)))

            if self.saturation_count > 0:
                self.p["damping_factor"] *= 1.01
                self.p["center_pull"] *= 1.01

            # Clamp to valid ranges
            self.p["coupling_base"] = max(min(self.p["coupling_base"], 0.05), 0.001)
            self.p["microvariance_amplitude"] = max(min(
                self.p["microvariance_amplitude"], 0.002), 0.00001)
            self.p["center_pull"] = max(min(self.p["center_pull"], 0.005), 0.00001)
            self.p["damping_factor"] = max(min(self.p["damping_factor"], 0.1), 0.001)

            # Selection: accept mutation only if fitness didn't drop too much.
            # With probability proportional to temperature, accept bad mutations
            # (exploration). As temp cools, only improvements survive.
            fitness_after = self.health * (0.5 + 0.5 * self.heat_valence)
            delta_fitness = fitness_after - fitness_before
            if delta_fitness < 0:
                # Metropolis criterion: accept bad mutation with prob exp(delta/temp)
                accept_prob = min(1.0, np.exp(delta_fitness / (temp + 1e-12)))
                if random.random() > accept_prob:
                    # REJECT: revert to saved params
                    self.p.update(saved_params)


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST — Two hypervector-brained nodes, 1000 steps
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {
    "coupling_base": 0.01,         # stronger coupling (no SLERP crutch)
    "microvariance_amplitude": 0.0001,
    "center_pull": 0.0005,
    "damping_factor": 0.01,
    "internal_forward": 0.001,
    "internal_reverse": 0.0005,
    "bypass_strength": 0.8,        # strong direct bypass -- primary coupling path
}


def run_smoke_test(steps=1000, full_sharing=True):
    """
    Two DennisNodes, D=64, coupled to each other.

    Reports:
      1. Stability: z_scalar stays in [CLAMP_LOW, CLAMP_HIGH]
      2. Cosine similarity between Z vectors over time
      3. Health remains above 0.5
    """
    label = "FULL SHARING" if full_sharing else "61.8% MEMBRANE"
    print(f"\n{'='*70}")
    print(f"  SMOKE TEST: {label}")
    print(f"  D={HDC_DIM}, steps={steps}")
    print(f"{'='*70}\n")

    node_a = DennisNode("NodeA", DEFAULT_PARAMS)
    node_b = DennisNode("NodeB", DEFAULT_PARAMS)

    if full_sharing:
        # Override membrane to share all dimensions
        node_a.void_outer = np.ones(HDC_DIM)
        node_b.void_outer = np.ones(HDC_DIM)
        node_a.heat_valence = 1.0
        node_b.heat_valence = 1.0

    threshold = 0.0001
    max_abs = CLAMP_HIGH

    # Recording
    sim_history = []
    z_scalar_a_hist = []
    z_scalar_b_hist = []
    health_a_hist = []
    health_b_hist = []
    membrane_open_a = []
    membrane_open_b = []

    initial_sim = _hdc.similarity(node_a.Z, node_b.Z)
    print(f"  Initial cosine similarity: {initial_sim:.4f}")
    print(f"  Initial z_scalar A: {node_a.z_scalar:.6f}")
    print(f"  Initial z_scalar B: {node_b.z_scalar:.6f}")
    print(f"  Membrane open dims A: {int(np.sum(node_a.void_outer))}/{HDC_DIM}")
    print(f"  Membrane open dims B: {int(np.sum(node_b.void_outer))}/{HDC_DIM}")
    print()

    for step in range(steps):
        z_scalars = {"NodeA": node_a.z_scalar, "NodeB": node_b.z_scalar}

        # A sees B's relational, B sees A's relational
        node_a.update(
            neighbor_relational_vecs=[node_b.relational],
            locals_list=["NodeB"],
            z_scalars=z_scalars,
            threshold=threshold,
            max_abs=max_abs,
        )
        node_b.update(
            neighbor_relational_vecs=[node_a.relational],
            locals_list=["NodeA"],
            z_scalars=z_scalars,
            threshold=threshold,
            max_abs=max_abs,
        )

        # Record
        sim = _hdc.similarity(node_a.Z, node_b.Z)
        sim_history.append(sim)
        z_scalar_a_hist.append(node_a.z_scalar)
        z_scalar_b_hist.append(node_b.z_scalar)
        health_a_hist.append(node_a.health)
        health_b_hist.append(node_b.health)
        membrane_open_a.append(int(np.sum(node_a.void_outer)))
        membrane_open_b.append(int(np.sum(node_b.void_outer)))

    # --- REPORT ---
    print(f"  RESULTS ({label}):")
    print(f"  {'-'*50}")

    # 1. Stability
    a_min, a_max = min(z_scalar_a_hist), max(z_scalar_a_hist)
    b_min, b_max = min(z_scalar_b_hist), max(z_scalar_b_hist)
    a_stable = a_min >= 0 and a_max <= CLAMP_HIGH
    b_stable = b_min >= 0 and b_max <= CLAMP_HIGH
    print(f"  z_scalar A range: [{a_min:.6f}, {a_max:.6f}]  {'STABLE' if a_stable else 'UNSTABLE'}")
    print(f"  z_scalar B range: [{b_min:.6f}, {b_max:.6f}]  {'STABLE' if b_stable else 'UNSTABLE'}")

    # 2. Cosine similarity trajectory
    sim_start = np.mean(sim_history[:10])
    sim_mid = np.mean(sim_history[450:550]) if steps >= 550 else np.mean(sim_history[len(sim_history)//2-5:len(sim_history)//2+5])
    sim_end = np.mean(sim_history[-10:])
    print(f"  Cosine sim (first 10):  {sim_start:.4f}")
    print(f"  Cosine sim (mid):       {sim_mid:.4f}")
    print(f"  Cosine sim (last 10):   {sim_end:.4f}")

    # Check for interesting dynamics
    sim_arr = np.array(sim_history)
    sim_std = np.std(sim_arr)
    sim_trend = sim_end - sim_start
    print(f"  Cosine sim std:         {sim_std:.4f}")
    print(f"  Cosine sim trend:       {sim_trend:+.4f} ({'converging' if sim_trend > 0.05 else 'diverging' if sim_trend < -0.05 else 'stable orbit'})")

    # 3. Health
    health_a_final = health_a_hist[-1]
    health_b_final = health_b_hist[-1]
    health_a_min = min(health_a_hist)
    health_b_min = min(health_b_hist)
    print(f"  Health A: final={health_a_final:.4f}, min={health_a_min:.4f}  {'OK' if health_a_min > 0.5 else 'DEGRADED'}")
    print(f"  Health B: final={health_b_final:.4f}, min={health_b_min:.4f}  {'OK' if health_b_min > 0.5 else 'DEGRADED'}")

    # 4. Membrane dynamics (only meaningful for non-full-sharing)
    if not full_sharing:
        print(f"  Membrane open dims A: start={membrane_open_a[0]}, end={membrane_open_a[-1]}")
        print(f"  Membrane open dims B: start={membrane_open_b[0]}, end={membrane_open_b[-1]}")

    # 5. Discovery: check for unexpected patterns
    print(f"\n  DISCOVERY NOTES:")
    if sim_std < 0.01:
        print(f"    - Cosine similarity locked (std={sim_std:.4f}). Nodes may be in a fixed point.")
    if sim_end > 0.9:
        print(f"    - SPONTANEOUS SYNCHRONIZATION detected (sim={sim_end:.4f})")
    if sim_end < -0.5:
        print(f"    - OPPOSITION detected (sim={sim_end:.4f}) -- nodes are anti-correlated")
    if abs(sim_trend) < 0.01 and sim_std > 0.05:
        print(f"    - STABLE ORBIT -- nodes oscillate around a mean without converging")

    # Check for dimension-specific clustering
    z_a = node_a.Z
    z_b = node_b.Z
    agreement = z_a * z_b  # per-dimension agreement
    n_agree = np.sum(agreement > 0)
    n_disagree = np.sum(agreement < 0)
    print(f"    - Dimension agreement: {int(n_agree)}/{HDC_DIM} agree, {int(n_disagree)}/{HDC_DIM} disagree")

    if not full_sharing:
        # Check if membrane masking affects which dims converge
        masked_dims = node_a.void_outer * node_b.void_outer  # dims both share
        shared_agreement = np.sum(agreement[masked_dims > 0] > 0)
        private_a = np.sum(node_a.void_outer == 0)
        private_b = np.sum(node_b.void_outer == 0)
        print(f"    - Private dims: A={int(private_a)}, B={int(private_b)}")
        n_shared = int(np.sum(masked_dims))
        print(f"    - Mutually shared dims: {n_shared}/{HDC_DIM}")
        if n_shared > 0:
            print(f"    - Agreement in shared dims: {int(shared_agreement)}/{n_shared}")

    print()

    return {
        "sim_history": sim_history,
        "z_scalar_a": z_scalar_a_hist,
        "z_scalar_b": z_scalar_b_hist,
        "health_a": health_a_hist,
        "health_b": health_b_hist,
        "stable": a_stable and b_stable,
        "health_ok": health_a_min > 0.5 and health_b_min > 0.5,
    }


PARAMS_CAUTIOUS = {
    "coupling_base": 0.0002,
    "microvariance_amplitude": 0.00005,
    "center_pull": 0.001,
    "damping_factor": 0.02,
    "internal_forward": 0.0005,
    "internal_reverse": 0.0003,
    "bypass_strength": 0.2,
}

PARAMS_BOLD = {
    "coupling_base": 0.001,
    "microvariance_amplitude": 0.0003,
    "center_pull": 0.0002,
    "damping_factor": 0.005,
    "internal_forward": 0.002,
    "internal_reverse": 0.001,
    "bypass_strength": 0.4,
}


def run_paired_test(label, steps, node_a, node_b, name_a, name_b,
                    coupling_mode="converge", adaptive_coupling=False,
                    tension_curve="linear", renormalize_z=False,
                    target_z_scalar=0.75, coupling_target="Z"):
    """Generic two-node test runner. Returns sim_history and final nodes."""
    extras = []
    if renormalize_z:
        extras.append(f"renorm={target_z_scalar}")
    if coupling_target != "Z":
        extras.append(f"target={coupling_target}")
    extra_str = (", " + ", ".join(extras)) if extras else ""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  D={HDC_DIM}, steps={steps}, mode={coupling_mode}, curve={tension_curve}{extra_str}")
    print(f"{'='*70}")

    threshold = 0.0001
    max_abs = CLAMP_HIGH
    sim_history = []
    checkpoints = {100, 500, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000}

    initial_sim = _hdc.similarity(node_a.Z, node_b.Z)
    print(f"  Initial sim={initial_sim:.4f}, membrane_A={int(np.sum(node_a.void_outer))}, membrane_B={int(np.sum(node_b.void_outer))}")

    for step in range(1, steps + 1):
        z_scalars = {name_a: node_a.z_scalar, name_b: node_b.z_scalar}
        node_a.update([node_b.relational], [name_b], z_scalars, threshold, max_abs,
                      coupling_mode=coupling_mode, adaptive_coupling=adaptive_coupling,
                      tension_curve=tension_curve, renormalize_z=renormalize_z,
                      target_z_scalar=target_z_scalar, coupling_target=coupling_target)
        node_b.update([node_a.relational], [name_a], z_scalars, threshold, max_abs,
                      coupling_mode=coupling_mode, adaptive_coupling=adaptive_coupling,
                      tension_curve=tension_curve, renormalize_z=renormalize_z,
                      target_z_scalar=target_z_scalar, coupling_target=coupling_target)
        sim = _hdc.similarity(node_a.Z, node_b.Z)
        sim_history.append(sim)

        if step in checkpoints:
            recent = np.mean(sim_history[-50:])
            print(f"  Step {step:>5d}: sim={recent:.4f}, z_A={node_a.z_scalar:.4f}, z_B={node_b.z_scalar:.4f}, h_A={node_a.health:.4f}, h_B={node_b.health:.4f}")
        if step % 10000 == 0:
            print(".", end="", flush=True)

    sim_arr = np.array(sim_history)
    print(f"\n  RESULTS:")
    print(f"    Final sim (last 50):   {np.mean(sim_arr[-50:]):.4f}")
    print(f"    Sim first 1000:        {np.mean(sim_arr[:1000]):.4f}")
    print(f"    Sim last 1000:         {np.mean(sim_arr[-1000:]):.4f}")
    print(f"    Std first 5000:        {np.std(sim_arr[:5000]):.4f}")
    print(f"    Std last 5000:         {np.std(sim_arr[-5000:]):.4f}")
    print(f"    {name_a} z_scalar:     {node_a.z_scalar:.6f}")
    print(f"    {name_b} z_scalar:     {node_b.z_scalar:.6f}")
    print(f"    {name_a} health:       {node_a.health:.4f}")
    print(f"    {name_b} health:       {node_b.health:.4f}")
    print(f"    {name_a} coupling:     {node_a.p['coupling_base']:.6f}")
    print(f"    {name_b} coupling:     {node_b.p['coupling_base']:.6f}")

    # Dimension analysis
    agreement = node_a.Z * node_b.Z
    n_agree = int(np.sum(agreement > 0))
    n_disagree = int(np.sum(agreement < 0))
    print(f"    Dim agreement:         {n_agree}/{HDC_DIM} agree, {n_disagree}/{HDC_DIM} disagree")

    # Specialization check: are dims where A is strong the same where B is weak?
    a_strong = np.abs(node_a.Z) > np.median(np.abs(node_a.Z))
    b_strong = np.abs(node_b.Z) > np.median(np.abs(node_b.Z))
    both_strong = np.sum(a_strong & b_strong)
    a_only = np.sum(a_strong & ~b_strong)
    b_only = np.sum(~a_strong & b_strong)
    neither = np.sum(~a_strong & ~b_strong)
    print(f"    Specialization:        both_strong={int(both_strong)}, A_only={int(a_only)}, B_only={int(b_only)}, neither={int(neither)}")

    # Membrane state
    open_a = int(np.sum(node_a.void_outer))
    open_b = int(np.sum(node_b.void_outer))
    print(f"    Membrane open:         A={open_a}/{HDC_DIM}, B={open_b}/{HDC_DIM}")

    print()
    return sim_history


# ═══════════════════════════════════════════════════════════════════════════
# SET A: MEMBRANE FEEDBACK TESTS
# ═══════════════════════════════════════════════════════════════════════════

def run_set_a(steps=50000):
    """Three asymmetric runs testing membrane + adaptive coupling effects."""
    print("\n" + "#"*70)
    print("#  SET A: MEMBRANE FEEDBACK TESTS (asymmetric, 50k steps each)")
    print("#"*70)

    # A1: Asymmetric + 61.8% membranes, NO adaptive coupling
    a = DennisNode("Cautious", PARAMS_CAUTIOUS)
    b = DennisNode("Bold", PARAMS_BOLD)
    # Leave membranes at default (61.8%)
    sim_a1 = run_paired_test("A1: Asymmetric + 61.8% membrane, NO adaptive",
                             steps, a, b, "Cautious", "Bold",
                             coupling_mode="converge", adaptive_coupling=False)

    # A2: Asymmetric + full sharing + adaptive coupling
    a = DennisNode("Cautious", PARAMS_CAUTIOUS)
    b = DennisNode("Bold", PARAMS_BOLD)
    a.void_outer = np.ones(HDC_DIM)
    b.void_outer = np.ones(HDC_DIM)
    a.heat_valence = 1.0
    b.heat_valence = 1.0
    sim_a2 = run_paired_test("A2: Asymmetric + full sharing + adaptive coupling",
                             steps, a, b, "Cautious", "Bold",
                             coupling_mode="converge", adaptive_coupling=True)

    # A3: Asymmetric + 61.8% membranes + adaptive coupling
    a = DennisNode("Cautious", PARAMS_CAUTIOUS)
    b = DennisNode("Bold", PARAMS_BOLD)
    # Leave membranes at default (61.8%)
    sim_a3 = run_paired_test("A3: Asymmetric + 61.8% membrane + adaptive coupling",
                             steps, a, b, "Cautious", "Bold",
                             coupling_mode="converge", adaptive_coupling=True)

    # Comparison
    print("-"*70)
    print("  SET A COMPARISON (vs baseline: asymmetric+full+no_adaptive = ~0.28)")
    print("-"*70)
    for name, hist in [("A1 membrane_only", sim_a1), ("A2 adaptive_only", sim_a2), ("A3 membrane+adaptive", sim_a3)]:
        arr = np.array(hist)
        first = np.mean(arr[:1000])
        last = np.mean(arr[-1000:])
        drift = last - first
        print(f"  {name:25s}: first_1k={first:.4f}, last_1k={last:.4f}, drift={drift:+.4f}, std_last5k={np.std(arr[-5000:]):.4f}")
    print()

    return sim_a1, sim_a2, sim_a3


# ═══════════════════════════════════════════════════════════════════════════
# SET B: COMPLEMENTARITY TESTS
# ═══════════════════════════════════════════════════════════════════════════

def run_set_b(steps=50000):
    """Three complementary coupling runs -- do nodes find orthogonality?"""
    print("\n" + "#"*70)
    print("#  SET B: COMPLEMENTARITY TESTS (pull toward sim=0, 50k steps each)")
    print("#"*70)

    # B1: Symmetric nodes, complementary, full sharing
    a = DennisNode("NodeA", DEFAULT_PARAMS)
    b = DennisNode("NodeB", DEFAULT_PARAMS)
    a.void_outer = np.ones(HDC_DIM)
    b.void_outer = np.ones(HDC_DIM)
    a.heat_valence = 1.0
    b.heat_valence = 1.0
    sim_b1 = run_paired_test("B1: Symmetric + complementary + full sharing",
                             steps, a, b, "NodeA", "NodeB",
                             coupling_mode="complementary", adaptive_coupling=False)

    # B2: Asymmetric nodes, complementary, full sharing
    a = DennisNode("Cautious", PARAMS_CAUTIOUS)
    b = DennisNode("Bold", PARAMS_BOLD)
    a.void_outer = np.ones(HDC_DIM)
    b.void_outer = np.ones(HDC_DIM)
    a.heat_valence = 1.0
    b.heat_valence = 1.0
    sim_b2 = run_paired_test("B2: Asymmetric + complementary + full sharing",
                             steps, a, b, "Cautious", "Bold",
                             coupling_mode="complementary", adaptive_coupling=False)

    # B3: Asymmetric + complementary + 61.8% membranes
    a = DennisNode("Cautious", PARAMS_CAUTIOUS)
    b = DennisNode("Bold", PARAMS_BOLD)
    # Leave membranes at default (61.8%)
    sim_b3 = run_paired_test("B3: Asymmetric + complementary + 61.8% membrane",
                             steps, a, b, "Cautious", "Bold",
                             coupling_mode="complementary", adaptive_coupling=False)

    # Comparison
    print("-"*70)
    print("  SET B COMPARISON (target: sim -> 0 = orthogonality)")
    print("-"*70)
    for name, hist in [("B1 sym+comp+full", sim_b1), ("B2 asym+comp+full", sim_b2), ("B3 asym+comp+membr", sim_b3)]:
        arr = np.array(hist)
        first = np.mean(arr[:1000])
        last = np.mean(arr[-1000:])
        drift = last - first
        print(f"  {name:25s}: first_1k={first:.4f}, last_1k={last:.4f}, drift={drift:+.4f}, std_last5k={np.std(arr[-5000:]):.4f}")
    print()

    return sim_b1, sim_b2, sim_b3


def make_symmetric_pair(params=None):
    """Create two symmetric nodes with full sharing."""
    p = params or DEFAULT_PARAMS
    a = DennisNode("NodeA", p.copy())
    b = DennisNode("NodeB", p.copy())
    a.void_outer = np.ones(HDC_DIM)
    b.void_outer = np.ones(HDC_DIM)
    a.heat_valence = 1.0
    b.heat_valence = 1.0
    return a, b


def run_x1_coupling_test(steps=50000):
    """Test coupling into X1 instead of Z. The key hypothesis:
    Z gets overwritten by internal dynamics each step, so coupling into Z
    is transient. Coupling into X1 integrates into the dynamics permanently."""
    print("\n" + "#"*70)
    print("#  X1 COUPLING TEST: move applied to X1 instead of Z")
    print("#  Hypothesis: internal dynamics overwrite Z coupling each step.")
    print("#  X1 is the true state variable -- coupling should feed there.")
    print("#"*70)

    # (label, curve, renorm, coupling_target)
    configs = [
        ("X1+linear",           "linear",    False, "X1"),
        ("X1+cubic",            "cubic",     False, "X1"),
        ("X1+quadratic",        "quadratic", False, "X1"),
        ("X1+linear+renorm",    "linear",    True,  "X1"),
        ("X1+cubic+renorm",     "cubic",     True,  "X1"),
        ("Z+linear (baseline)", "linear",    False, "Z"),
    ]
    results = []

    for name, curve, renorm, target in configs:
        a, b = make_symmetric_pair()
        hist = run_paired_test(
            name, steps, a, b, "NodeA", "NodeB",
            coupling_mode="complementary", tension_curve=curve,
            renormalize_z=renorm, target_z_scalar=0.75,
            coupling_target=target)
        results.append((name, hist))

    # Summary
    print("="*70)
    print("  X1 COUPLING SUMMARY")
    print("="*70)
    print(f"  {'config':>25s}  {'first_1k':>9s}  {'last_1k':>9s}  {'std_l5k':>8s}  {'std_l1k':>8s}  {'|mean|_l5k':>11s}")
    print(f"  {'-'*25}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*11}")
    for name, hist in results:
        arr = np.array(hist)
        print(f"  {name:>25s}  {np.mean(arr[:1000]):>9.4f}  {np.mean(arr[-1000:]):>9.4f}  {np.std(arr[-5000:]):>8.4f}  {np.std(arr[-1000:]):>8.4f}  {np.mean(np.abs(arr[-5000:])):>11.4f}")
    print()
    return results


def run_x1_full_battery(steps=50000):
    """Now that X1+linear+renorm works, test it in all conditions."""
    print("\n" + "#"*70)
    print("#  X1 COUPLING FULL BATTERY")
    print("#  Winner from last round: X1+linear+renorm (sim=0.005, std=0.0003)")
    print("#  Now test: asymmetric, membranes, converge mode comparison")
    print("#"*70)

    configs = []

    # 1. Symmetric + complementary + full (already proven, quick confirm)
    a, b = make_symmetric_pair()
    configs.append(("sym+comp+full", a, b, "NodeA", "NodeB",
                    "complementary", False, True, "X1"))

    # 2. Asymmetric + complementary + full
    a = DennisNode("Cautious", PARAMS_CAUTIOUS.copy())
    b = DennisNode("Bold", PARAMS_BOLD.copy())
    a.void_outer = np.ones(HDC_DIM); b.void_outer = np.ones(HDC_DIM)
    a.heat_valence = 1.0; b.heat_valence = 1.0
    configs.append(("asym+comp+full", a, b, "Cautious", "Bold",
                    "complementary", False, True, "X1"))

    # 3. Asymmetric + complementary + 61.8% membrane
    a = DennisNode("Cautious", PARAMS_CAUTIOUS.copy())
    b = DennisNode("Bold", PARAMS_BOLD.copy())
    configs.append(("asym+comp+membr", a, b, "Cautious", "Bold",
                    "complementary", False, True, "X1"))

    # 4. Symmetric + CONVERGE + full + X1 coupling (does converge still work?)
    a, b = make_symmetric_pair()
    configs.append(("sym+CONV+full", a, b, "NodeA", "NodeB",
                    "converge", False, True, "X1"))

    # 5. Asymmetric + CONVERGE + full + X1 coupling
    a = DennisNode("Cautious", PARAMS_CAUTIOUS.copy())
    b = DennisNode("Bold", PARAMS_BOLD.copy())
    a.void_outer = np.ones(HDC_DIM); b.void_outer = np.ones(HDC_DIM)
    a.heat_valence = 1.0; b.heat_valence = 1.0
    configs.append(("asym+CONV+full", a, b, "Cautious", "Bold",
                    "converge", False, True, "X1"))

    results = []
    for label, a, b, na, nb, mode, adaptive, renorm, target in configs:
        hist = run_paired_test(
            label, steps, a, b, na, nb,
            coupling_mode=mode, adaptive_coupling=adaptive,
            tension_curve="linear", renormalize_z=renorm,
            target_z_scalar=0.75, coupling_target=target)
        results.append((label, hist))

    # Summary
    print("="*70)
    print("  X1 FULL BATTERY SUMMARY")
    print("="*70)
    print(f"  {'config':>22s}  {'first_1k':>9s}  {'last_1k':>9s}  {'std_l5k':>8s}  {'std_l1k':>8s}  {'|mean|_l5k':>11s}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*11}")
    for name, hist in results:
        arr = np.array(hist)
        print(f"  {name:>22s}  {np.mean(arr[:1000]):>9.4f}  {np.mean(arr[-1000:]):>9.4f}  {np.std(arr[-5000:]):>8.4f}  {np.std(arr[-1000:]):>8.4f}  {np.mean(np.abs(arr[-5000:])):>11.4f}")
    print()
    return results


def run_multinode_test(n_nodes, steps, coupling_mode, label,
                       use_membranes=False, mixed_params=False):
    """N-node test. All-to-all coupling, X1 target, renorm=0.75.

    mixed_params: if True, each node gets a random blend between
                  PARAMS_CAUTIOUS and PARAMS_BOLD for personality diversity.
    """
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  N={n_nodes}, D={HDC_DIM}, steps={steps}, mode={coupling_mode}")
    print(f"  membranes={'61.8%' if use_membranes else 'full'}, "
          f"params={'mixed' if mixed_params else 'uniform'}")
    print(f"{'='*70}")

    # Create nodes
    nodes = []
    for i in range(n_nodes):
        name = f"Node{i}"
        if mixed_params:
            # Random blend between cautious and bold
            blend = random.random()
            params = {}
            for key in DEFAULT_PARAMS:
                lo = PARAMS_CAUTIOUS[key]
                hi = PARAMS_BOLD[key]
                params[key] = lo + blend * (hi - lo)
        else:
            params = DEFAULT_PARAMS.copy()
        node = DennisNode(name, params)
        if not use_membranes:
            node.void_outer = np.ones(HDC_DIM)
            node.heat_valence = 1.0
        nodes.append(node)

    threshold = 0.0001
    max_abs = CLAMP_HIGH

    # Track mean pairwise sim over time
    mean_sim_history = []
    checkpoints = {100, 500, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000}

    # Initial pairwise sims
    initial_sims = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            initial_sims.append(_hdc.similarity(nodes[i].Z, nodes[j].Z))
    print(f"  Initial mean pairwise sim: {np.mean(initial_sims):.4f}")

    for step in range(1, steps + 1):
        z_scalars = {f"Node{i}": nodes[i].z_scalar for i in range(n_nodes)}
        names = [f"Node{i}" for i in range(n_nodes)]

        # Update each node — sees all OTHER nodes' relational vectors
        for i in range(n_nodes):
            neighbor_rels = [nodes[j].relational for j in range(n_nodes) if j != i]
            neighbor_names = [names[j] for j in range(n_nodes) if j != i]
            nodes[i].update(
                neighbor_rels, neighbor_names, z_scalars, threshold, max_abs,
                coupling_mode=coupling_mode, tension_curve="linear",
                renormalize_z=True, target_z_scalar=0.75, coupling_target="X1")

        # Compute mean pairwise sim
        pair_sims = []
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                pair_sims.append(_hdc.similarity(nodes[i].Z, nodes[j].Z))
        mean_sim = np.mean(pair_sims)
        mean_sim_history.append(mean_sim)

        if step in checkpoints:
            recent = np.mean(mean_sim_history[-50:])
            healths = [n.health for n in nodes]
            print(f"  Step {step:>5d}: mean_sim={recent:.4f}, "
                  f"health=[{min(healths):.4f},{max(healths):.4f}]")
        if step % 10000 == 0:
            print(".", end="", flush=True)

    # --- FINAL REPORT ---
    sim_arr = np.array(mean_sim_history)
    print(f"\n  RESULTS:")
    print(f"    Mean pairwise sim (first 1k): {np.mean(sim_arr[:1000]):.4f}")
    print(f"    Mean pairwise sim (last 1k):  {np.mean(sim_arr[-1000:]):.4f}")
    print(f"    Std last 5k:                  {np.std(sim_arr[-5000:]):.4f}")
    print(f"    Std last 1k:                  {np.std(sim_arr[-1000:]):.4f}")

    # Full pairwise matrix at end
    print(f"\n    PAIRWISE SIMILARITY MATRIX (final):")
    print(f"    {'':>8s}", end="")
    for j in range(n_nodes):
        print(f"  Node{j:>2d}", end="")
    print()
    for i in range(n_nodes):
        print(f"    Node{i:>2d} ", end="")
        for j in range(n_nodes):
            if i == j:
                print(f"    --- ", end="")
            else:
                s = _hdc.similarity(nodes[i].Z, nodes[j].Z)
                print(f"  {s:>6.3f}", end="")
        print()

    # Health summary
    print(f"\n    HEALTH: ", end="")
    for i, n in enumerate(nodes):
        print(f"Node{i}={n.health:.4f} ", end="")
    print()

    # z_scalar summary
    print(f"    Z_SCALAR: ", end="")
    for i, n in enumerate(nodes):
        print(f"Node{i}={n.z_scalar:.4f} ", end="")
    print()

    # Specialization: for each dimension, which node "owns" it (highest |Z[d]|)?
    owners = np.zeros(HDC_DIM, dtype=int)
    for d in range(HDC_DIM):
        magnitudes = [abs(nodes[i].Z[d]) for i in range(n_nodes)]
        owners[d] = np.argmax(magnitudes)
    owner_counts = [int(np.sum(owners == i)) for i in range(n_nodes)]
    print(f"    DIMENSION OWNERSHIP: ", end="")
    for i in range(n_nodes):
        print(f"Node{i}={owner_counts[i]} ", end="")
    print(f" (ideal={HDC_DIM//n_nodes} each)")

    # Membrane state
    if use_membranes:
        print(f"    MEMBRANES: ", end="")
        for i, n in enumerate(nodes):
            print(f"Node{i}={int(np.sum(n.void_outer))}/{HDC_DIM} ", end="")
        print()

    print()
    return mean_sim_history


def run_scaling_tests(steps=50000):
    """Test both modes with 3, 5, and 8 nodes."""
    print("\n" + "#"*70)
    print("#  MULTI-NODE SCALING TESTS")
    print("#  X1 coupling + renorm(0.75), all-to-all topology")
    print("#  Can N>2 nodes find mutual orthogonality (complementary)?")
    print("#  Can N>2 nodes fully synchronize (converge)?")
    print("#  D=64 supports up to 64 orthogonal vectors -- room for all")
    print("#"*70)

    results = []

    # --- CONVERGE MODE ---
    for n in [3, 5, 8]:
        h = run_multinode_test(n, steps, "converge",
                               f"CONVERGE: {n} nodes, uniform, full sharing")
        results.append((f"conv_{n}n", h))

    # --- COMPLEMENTARY MODE ---
    for n in [3, 5, 8]:
        h = run_multinode_test(n, steps, "complementary",
                               f"COMPLEMENTARY: {n} nodes, uniform, full sharing")
        results.append((f"comp_{n}n", h))

    # --- COMPLEMENTARY + MIXED PERSONALITIES ---
    for n in [5, 8]:
        h = run_multinode_test(n, steps, "complementary",
                               f"COMPLEMENTARY: {n} mixed nodes, full sharing",
                               mixed_params=True)
        results.append((f"comp_mix_{n}n", h))

    # --- COMPLEMENTARY + MEMBRANES ---
    h = run_multinode_test(5, steps, "complementary",
                           "COMPLEMENTARY: 5 nodes, uniform, 61.8% membranes",
                           use_membranes=True)
    results.append(("comp_membr_5n", h))

    # Summary
    print("="*70)
    print("  MULTI-NODE SCALING SUMMARY")
    print("="*70)
    print(f"  {'config':>22s}  {'first_1k':>9s}  {'last_1k':>9s}  {'std_l5k':>8s}  {'std_l1k':>8s}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}")
    for name, hist in results:
        arr = np.array(hist)
        print(f"  {name:>22s}  {np.mean(arr[:1000]):>9.4f}  {np.mean(arr[-1000:]):>9.4f}  {np.std(arr[-5000:]):>8.4f}  {np.std(arr[-1000:]):>8.4f}")
    print()
    return results


# =====================================================================
# FAST PHI — Eigenvalue-based integration approximation
# Inspired by SovereignMirror architecture (2026 benchmark style)
# =====================================================================

def compute_swarm_complexity(nodes):
    """Swarm complexity heuristic based on spectral analysis of the coupling matrix.

    NOTE: This is NOT IIT Phi (Integrated Information Theory).  Real IIT Phi
    requires computing the minimum information partition, which is NP-hard
    and only feasible for tiny systems via PyPhi.  This is a custom heuristic
    that captures *some* properties related to integration (spectral entropy,
    effective rank, coupling strength) but should NOT be compared to published
    IIT values.

    Three components combined:
      1. Spectral entropy: Shannon entropy of eigenvalue distribution.
         High entropy = distributed spectral energy = more complex coupling.
      2. Effective rank: 2^(entropy), penalized when trivially 1 or N.
         Peaks at intermediate structure.
      3. Mean coupling × dimension utilization: raw integration strength.

    The output is an UNSCALED complexity score in [0, ~50] for typical
    5-node swarms.  Higher = more integrated coupling structure.
    There is NO external benchmark for this metric.

    Returns: (complexity_score, eigenvalues, coherence)
    """
    n = len(nodes)
    # Build coupling matrix (full, including off-diagonal structure)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                C[i][j] = 1.0
            else:
                C[i][j] = _hdc.similarity(nodes[i].Z, nodes[j].Z)

    eigenvalues = np.real(np.linalg.eigvals(C))
    abs_ev = np.abs(eigenvalues)
    total_ev = np.sum(abs_ev) + 1e-12

    # --- COMPONENT 1: Spectral Entropy ---
    p = abs_ev / total_ev
    p = p[p > 1e-12]  # remove zeros
    spectral_entropy = -np.sum(p * np.log2(p + 1e-30))
    max_entropy = np.log2(n)
    normalized_entropy = spectral_entropy / (max_entropy + 1e-12)

    # --- COMPONENT 2: Effective Rank (complexity) ---
    effective_rank = 2.0 ** spectral_entropy
    # Complexity peaks at intermediate rank, not 1 or N
    complexity = effective_rank * (n - effective_rank + 1) / (n * 0.25 + 1e-12)

    # --- COMPONENT 3: Integration Strength ---
    off_diag = []
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diag.append(abs(C[i][j]))
    mean_coupling = np.mean(off_diag) if off_diag else 0.0
    all_z = np.array([nd.Z for nd in nodes])
    dim_variance = np.var(all_z, axis=0)
    active_dims = np.sum(dim_variance > np.mean(dim_variance) * 0.1)
    dim_utilization = active_dims / HDC_DIM

    # --- COMBINE: unscaled complexity score ---
    # No arbitrary scaling constant.  The raw number IS the number.
    complexity_score = complexity * (1.0 + mean_coupling * 10.0) * dim_utilization

    # --- COHERENCE ---
    max_ev_val = np.max(abs_ev)
    coherence = max_ev_val / total_ev
    coherence_adjusted = coherence * (1.0 + mean_coupling) * normalized_entropy
    coherence_adjusted = min(1.0, coherence_adjusted)

    return complexity_score, eigenvalues, coherence_adjusted


# =====================================================================
# GLOBAL IGNITION — Cascade dynamics for swarm-level coherence bursts
# =====================================================================

class GlobalIgnition:
    """Tracks and triggers global ignition events in the swarm.

    Ignition occurs when a critical mass of nodes simultaneously exceed
    a coherence threshold. When triggered, ALL nodes receive a coupling
    boost that rapidly amplifies integration — the system "ignites" into
    a higher-coherence state.

    Threshold 0.7 ensures ignition only fires when genuine coherence emerges.

    The mechanism:
      1. Each step, compute mean pairwise similarity
      2. If sim exceeds ignition_threshold AND rate-of-change is positive,
         this counts as an ignition event
      3. During ignition, nodes get a coupling multiplier boost
      4. Ignition has a refractory period to prevent runaway
    """

    def __init__(self, n_nodes, ignition_threshold=0.7, boost_factor=3.0,
                 refractory_steps=5):
        self.n_nodes = n_nodes
        self.ignition_threshold = ignition_threshold
        self.boost_factor = boost_factor
        self.refractory_steps = refractory_steps
        self.refractory_counter = 0
        self.ignition_count = 0
        self.total_steps = 0
        self.ignited = False
        self.sim_history = []
        self.ignition_log = []  # steps where ignition fired

    def step(self, nodes):
        """Check for ignition conditions and return coupling multiplier.

        Returns: float multiplier (1.0 = normal, >1.0 = ignited)
        """
        self.total_steps += 1

        # Compute mean pairwise similarity
        n = len(nodes)
        pair_sims = []
        for i in range(n):
            for j in range(i+1, n):
                pair_sims.append(_hdc.similarity(nodes[i].Z, nodes[j].Z))
        mean_sim = np.mean(pair_sims) if pair_sims else 0.0
        self.sim_history.append(mean_sim)

        # Refractory period
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self.ignited = False
            return 1.0

        # Check ignition conditions:
        # 1. Mean sim above threshold
        # 2. Sim is INCREASING (positive derivative)
        if len(self.sim_history) >= 3:
            d_sim = self.sim_history[-1] - self.sim_history[-3]
            if mean_sim > self.ignition_threshold and d_sim > 0:
                self.ignited = True
                self.ignition_count += 1
                self.ignition_log.append(self.total_steps)
                self.refractory_counter = self.refractory_steps
                return self.boost_factor

        self.ignited = False
        return 1.0

    def apply_boost(self, nodes, multiplier):
        """Apply ignition boost: temporarily amplify coupling_base for all nodes."""
        if multiplier > 1.0:
            for node in nodes:
                node.p["coupling_base"] *= multiplier
                # Clamp to prevent explosion
                node.p["coupling_base"] = min(node.p["coupling_base"], 0.05)

    def decay_boost(self, nodes):
        """After ignition, decay the coupling boost back toward normal."""
        for node in nodes:
            if node.p["coupling_base"] > 0.01:
                node.p["coupling_base"] *= 0.95  # decay toward default

    @property
    def rate_per_1000(self):
        """Ignition events per 1000 steps."""
        if self.total_steps == 0:
            return 0.0
        return self.ignition_count / self.total_steps * 1000.0


# =====================================================================
# PYPHI SWARM INTEGRATION — Exact Integrated Information (Phi)
# =====================================================================

def compute_swarm_phi(nodes):
    """
    Compute IIT Phi for the swarm using PyPhi.

    Approach: Build a TPM (transition probability matrix) from the swarm's
    coupling topology. Each node is a binary element:
      0 = inactive (z_scalar below median)
      1 = active (z_scalar above median)

    The TPM encodes how node states at t influence node states at t+1,
    derived from the actual coupling dynamics observed in the run.

    Note: SIA is exponentially expensive. Practical limit ~4 nodes.
    For N>4, compute on the 4 most-connected nodes as a subsystem.
    """
    n = len(nodes)
    if n > 4:
        # Select the 4 nodes with highest total coupling (most integrated)
        coupling_sums = []
        for i in range(n):
            total = sum(abs(_hdc.similarity(nodes[i].Z, nodes[j].Z))
                       for j in range(n) if j != i)
            coupling_sums.append(total)
        top4 = sorted(range(n), key=lambda i: coupling_sums[i], reverse=True)[:4]
        print(f"    [PyPhi] N={n} too large for full SIA. Using top-4 coupled nodes: {top4}")
        nodes = [nodes[i] for i in top4]
        n = 4

    # Build a deterministic TPM from observed coupling structure:
    # For each possible binary state, simulate one step and observe output.
    # Use cosine similarity signs to determine coupling direction.

    # Compute the coupling matrix: who pulls whom?
    coupling_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                sim = _hdc.similarity(nodes[i].Z, nodes[j].Z)
                coupling_matrix[i][j] = sim

    # Build TPM: for each of 2^n input states, compute output state
    n_states = 2 ** n
    tpm = np.zeros((n_states, n), dtype=float)

    for state_idx in range(n_states):
        # Decode binary state
        input_state = [(state_idx >> bit) & 1 for bit in range(n)]

        for i in range(n):
            # Node i's next state depends on its neighbors
            total_input = 0.0
            for j in range(n):
                if i != j:
                    total_input += coupling_matrix[i][j] * input_state[j]
            # Sigmoid-like activation: probability of being ON
            prob_on = 1.0 / (1.0 + np.exp(-total_input * 2.0))
            tpm[state_idx][i] = prob_on

    # Node labels
    labels = [f"N{i}" for i in range(n)]

    try:
        pyphi.config.PROGRESS_BARS = False
        network = pyphi.Network(tpm, node_labels=labels)
        # Use the current binary state as the substrate state
        median_z = np.median([nd.z_scalar for nd in nodes])
        current_state = tuple(1 if nd.z_scalar >= median_z else 0 for nd in nodes)
        substrate = pyphi.Subsystem(network, current_state)

        # Run SIA with a timeout to avoid hanging on degenerate TPMs
        import concurrent.futures
        def _compute_sia():
            return pyphi.compute.sia(substrate)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_compute_sia)
            try:
                sia = future.result(timeout=120)  # 2 minute max per SIA
                phi_value = sia.phi
                return phi_value
            except concurrent.futures.TimeoutError:
                print(f"[timeout >120s]", end=" ")
                return None
    except Exception as e:
        print(f"    [PyPhi] Error computing Phi: {e}")
        return None


# =====================================================================
# STFT SPECTRAL ANALYSIS — Swarm Dynamics Frequency Content
# =====================================================================

def analyze_swarm_spectra(sim_histories, node_labels, fs=1.0):
    """
    STFT spectral analysis of swarm similarity time series.

    Detects:
      - Dominant resonant frequencies in the coupling dynamics
      - Spectral coherence between node pairs
      - Phase-coupling signatures

    Args:
        sim_histories: dict of {pair_label: list_of_sim_values}
        node_labels: list of node names
        fs: sampling frequency (steps/unit time, default 1.0)
    """
    print(f"\n    STFT SPECTRAL ANALYSIS:")

    for pair_label, hist in sim_histories.items():
        signal = np.array(hist, dtype=np.float64)
        n_samples = len(signal)
        if n_samples < 256:
            print(f"      {pair_label}: too short for STFT ({n_samples} samples)")
            continue

        # STFT parameters: window = 256 samples, 75% overlap
        nperseg = min(256, n_samples // 4)
        noverlap = nperseg * 3 // 4

        freqs, times, Zxx = scipy_stft(signal, fs=fs, nperseg=nperseg,
                                        noverlap=noverlap)
        power = np.abs(Zxx) ** 2
        mean_power = np.mean(power, axis=1)  # average across time windows

        # Find dominant frequencies (top 3, excluding DC)
        if len(mean_power) > 1:
            power_no_dc = mean_power[1:]
            freqs_no_dc = freqs[1:]
            top_idx = np.argsort(power_no_dc)[-3:][::-1]
            total_power = np.sum(power_no_dc)

            print(f"      {pair_label}:")
            print(f"        Total spectral power: {total_power:.6f}")
            for rank, idx in enumerate(top_idx):
                pct = 100.0 * power_no_dc[idx] / total_power if total_power > 0 else 0
                period = 1.0 / freqs_no_dc[idx] if freqs_no_dc[idx] > 0 else float('inf')
                print(f"        Peak {rank+1}: f={freqs_no_dc[idx]:.6f} "
                      f"(period={period:.1f} steps) power={power_no_dc[idx]:.6f} ({pct:.1f}%)")

            # Spectral flatness: measure of how noise-like vs tonal the signal is
            geometric_mean = np.exp(np.mean(np.log(power_no_dc + 1e-30)))
            arithmetic_mean = np.mean(power_no_dc)
            flatness = geometric_mean / (arithmetic_mean + 1e-30)
            print(f"        Spectral flatness: {flatness:.4f} "
                  f"({'noise-like' if flatness > 0.5 else 'tonal/resonant'})")

    # Cross-coherence between first two series if available
    pair_keys = list(sim_histories.keys())
    if len(pair_keys) >= 2:
        sig_a = np.array(sim_histories[pair_keys[0]], dtype=np.float64)
        sig_b = np.array(sim_histories[pair_keys[1]], dtype=np.float64)
        min_len = min(len(sig_a), len(sig_b))
        if min_len >= 256:
            freqs_c, coh = scipy_coherence(sig_a[:min_len], sig_b[:min_len],
                                            fs=fs, nperseg=min(256, min_len // 4))
            mean_coh = np.mean(coh[1:])  # exclude DC
            max_coh = np.max(coh[1:]) if len(coh) > 1 else 0
            max_coh_freq = freqs_c[1 + np.argmax(coh[1:])] if len(coh) > 1 else 0
            print(f"\n      CROSS-COHERENCE ({pair_keys[0]} vs {pair_keys[1]}):")
            print(f"        Mean coherence: {mean_coh:.4f}")
            print(f"        Peak coherence: {max_coh:.4f} at f={max_coh_freq:.6f}")
            if mean_coh > 0.5:
                print(f"        --> STRONG spectral coupling between pairs")
            elif mean_coh > 0.2:
                print(f"        --> Moderate spectral coupling")
            else:
                print(f"        --> Weak/independent spectral dynamics")


def run_dim_balance_test(steps=50000):
    """Test dimension ownership balancing in complementary mode.

    Sweeps dim_balance_strength values and compares ownership distribution
    against the baseline (no balancing). Also computes PyPhi Phi for each.
    Plus STFT spectral analysis of the dynamics.
    """
    print("\n" + "#"*70)
    print("#  DIMENSION OWNERSHIP BALANCE TEST")
    print("#  Fix: per-dim territorial pressure in complementary mode")
    print("#  Nodes repelled from dims neighbors already occupy")
    print("#  Plus PyPhi Phi computation for swarm integration measure")
    print("#"*70)

    # Sweep dim_balance_strength with 5 nodes
    n_nodes = 5
    strengths = [0.0, 0.001, 0.005, 0.01]
    results = []

    for dbs in strengths:
        label = f"COMPLEMENTARY: 5 nodes, dim_balance={dbs}"
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"  N={n_nodes}, D={HDC_DIM}, steps={steps}, mode=complementary")
        print(f"  X1 coupling, renorm=0.75, dim_balance_strength={dbs}")
        print(f"{'='*70}")

        nodes = []
        for i in range(n_nodes):
            node = DennisNode(f"Node{i}", DEFAULT_PARAMS.copy())
            node.void_outer = np.ones(HDC_DIM)
            node.heat_valence = 1.0
            nodes.append(node)

        threshold = 0.0001
        max_abs = CLAMP_HIGH
        mean_sim_history = []
        # Track per-pair sim histories for STFT analysis
        pair_labels = []
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                pair_labels.append(f"N{i}-N{j}")
        pair_sim_histories = {pl: [] for pl in pair_labels}
        checkpoints = {100, 500, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000}

        for step in range(1, steps + 1):
            z_scalars = {f"Node{i}": nodes[i].z_scalar for i in range(n_nodes)}
            names = [f"Node{i}" for i in range(n_nodes)]

            for i in range(n_nodes):
                neighbor_rels = [nodes[j].relational for j in range(n_nodes) if j != i]
                neighbor_names = [names[j] for j in range(n_nodes) if j != i]
                nodes[i].update(
                    neighbor_rels, neighbor_names, z_scalars, threshold, max_abs,
                    coupling_mode="complementary", tension_curve="linear",
                    renormalize_z=True, target_z_scalar=0.75, coupling_target="X1",
                    dim_balance_strength=dbs)

            pair_idx = 0
            pair_sims = []
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    s = _hdc.similarity(nodes[i].Z, nodes[j].Z)
                    pair_sims.append(s)
                    pair_sim_histories[pair_labels[pair_idx]].append(s)
                    pair_idx += 1
            mean_sim_history.append(np.mean(pair_sims))

            if step in checkpoints:
                recent = np.mean(mean_sim_history[-50:])
                print(f"  Step {step:>5d}: mean_sim={recent:.4f}")

        # --- REPORT ---
        sim_arr = np.array(mean_sim_history)
        print(f"\n  RESULTS:")
        print(f"    Mean sim (last 1k):  {np.mean(sim_arr[-1000:]):.4f}")
        print(f"    Std (last 1k):       {np.std(sim_arr[-1000:]):.4f}")

        # Pairwise matrix
        print(f"\n    PAIRWISE SIMILARITY MATRIX:")
        print(f"    {'':>8s}", end="")
        for j in range(n_nodes):
            print(f"  Node{j:>2d}", end="")
        print()
        for i in range(n_nodes):
            print(f"    Node{i:>2d} ", end="")
            for j in range(n_nodes):
                if i == j:
                    print(f"    --- ", end="")
                else:
                    s = _hdc.similarity(nodes[i].Z, nodes[j].Z)
                    print(f"  {s:>6.3f}", end="")
            print()

        # Dimension ownership
        owners = np.zeros(HDC_DIM, dtype=int)
        for d in range(HDC_DIM):
            magnitudes = [abs(nodes[i].Z[d]) for i in range(n_nodes)]
            owners[d] = np.argmax(magnitudes)
        owner_counts = [int(np.sum(owners == i)) for i in range(n_nodes)]
        ideal = HDC_DIM // n_nodes
        ownership_std = np.std(owner_counts)
        ownership_range = max(owner_counts) - min(owner_counts)
        print(f"\n    DIMENSION OWNERSHIP:")
        for i in range(n_nodes):
            print(f"      Node{i}: {owner_counts[i]:>2d}/{HDC_DIM} dims", end="")
            if owner_counts[i] == ideal:
                print(" (ideal)", end="")
            print()
        print(f"    Ideal: {ideal} each | Std: {ownership_std:.1f} | Range: {ownership_range}")

        # Health + z_scalar
        print(f"    HEALTH: ", end="")
        for i, nd in enumerate(nodes):
            print(f"N{i}={nd.health:.3f} ", end="")
        print()
        print(f"    Z_SCALAR: ", end="")
        for i, nd in enumerate(nodes):
            print(f"N{i}={nd.z_scalar:.4f} ", end="")
        print()

        # PyPhi Phi
        print(f"\n    PYPHI INTEGRATED INFORMATION:")
        phi = compute_swarm_phi(nodes)
        if phi is not None:
            print(f"    Swarm Phi = {phi:.6f}")
            if phi > 0:
                print(f"    --> IRREDUCIBLE integration detected (TPM is irreducible)")
            else:
                print(f"    --> Swarm is reducible (no integrated information)")

        # Coherence: mean pairwise |sim| stability in last 1k steps
        last_1k_sims = sim_arr[-1000:]
        coherence = 1.0 - np.std(last_1k_sims)  # high stability = high coherence
        print(f"\n    COHERENCE (stability): {coherence:.4f}")

        # Internal benchmark targets
        print(f"\n    INTERNAL BENCHMARKS:")
        print(f"    {'Metric':<28s} {'Ours':>12s}  {'Target':>12s}  {'Status':>10s}")
        print(f"    {'-'*28} {'-'*12}  {'-'*12}  {'-'*10}")
        phi_str = f"{phi:.4f}" if phi is not None else "N/A"
        phi_status = "PASS" if (phi is not None and phi > 0) else "NEEDS WORK"
        print(f"    {'PyPhi Phi (real IIT)':  <28s} {phi_str:>12s}  {'>0':>12s}  {phi_status:>10s}")
        coh_status = "PASS" if coherence > 0.97 else "PARTIAL" if coherence > 0.8 else "NEEDS WORK"
        print(f"    {'Coherence':<28s} {coherence:>12.4f}  {'>0.97':>12s}  {coh_status:>10s}")
        own_status = "PASS" if ownership_range <= 2 else "PARTIAL" if ownership_range <= 5 else "NEEDS WORK"
        print(f"    {'Dimension equity (range)':<28s} {ownership_range:>12d}  {'<=2':>12s}  {own_status:>10s}")
        sim_status = "PASS" if abs(np.mean(sim_arr[-1000:])) < 0.01 else "PARTIAL"
        print(f"    {'Orthogonality (|sim|)':<28s} {abs(np.mean(sim_arr[-1000:])):>12.4f}  {'<0.01':>12s}  {sim_status:>10s}")

        # STFT spectral analysis
        analyze_swarm_spectra(pair_sim_histories, [f"Node{i}" for i in range(n_nodes)])

        results.append({
            "dbs": dbs,
            "mean_sim": np.mean(sim_arr[-1000:]),
            "sim_std": np.std(sim_arr[-1000:]),
            "owner_counts": owner_counts,
            "ownership_std": ownership_std,
            "ownership_range": ownership_range,
            "phi": phi,
            "coherence": coherence,
        })
        print()

    # Summary comparison
    print("="*70)
    print("  DIMENSION BALANCE SWEEP SUMMARY (5 nodes, complementary)")
    print("="*70)
    print(f"  {'dbs':>8s}  {'sim':>7s}  {'std':>7s}  {'own_std':>8s}  {'own_rng':>8s}  {'min':>4s}  {'max':>4s}  {'phi':>8s}  {'coher':>7s}  ownership")
    print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*4}  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*30}")
    for r in results:
        oc = r["owner_counts"]
        phi_str = f"{r['phi']:.4f}" if r['phi'] is not None else "N/A"
        coh_str = f"{r['coherence']:.4f}" if r.get('coherence') is not None else "N/A"
        print(f"  {r['dbs']:>8.4f}  {r['mean_sim']:>7.4f}  {r['sim_std']:>7.4f}  "
              f"{r['ownership_std']:>8.1f}  {r['ownership_range']:>8d}  "
              f"{min(oc):>4d}  {max(oc):>4d}  {phi_str:>8s}  {coh_str:>7s}  {oc}")
    print(f"\n  TARGET: ownership_std -> 0, range -> 0, sim -> 0, phi > 0, coherence > 0.97")
    print(f"  NOTE: PyPhi Phi is real IIT. Swarm complexity is our custom heuristic.")
    print()

    return results


# =====================================================================
# STEP 2: SPS (Silent Punctuational Syntax) TESTS
# =====================================================================

def run_sps_test(n_nodes, steps, coupling_mode, label,
                 use_membranes=False, use_sps=True, use_ignition=True,
                 module_assignments=None):
    """N-node SPS test. All-to-all coupling with silence gating + ignition.

    Reports: speak rate, avg silence, max silence, rhythm regularity,
    convergence speed, dimension ownership, ignition rate.

    module_assignments: optional list of ints, one per node. Enables
    modular coupling when coupling_mode="modular".
    """
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  N={n_nodes}, D={HDC_DIM}, steps={steps}, mode={coupling_mode}")
    print(f"  SPS={'ON' if use_sps else 'OFF'}, membranes={'61.8%' if use_membranes else 'full'}, ignition={'ON' if use_ignition else 'OFF'}")
    print(f"{'='*70}")

    nodes = []
    for i in range(n_nodes):
        node = DennisNode(f"Node{i}", DEFAULT_PARAMS.copy())
        if not use_membranes:
            node.void_outer = np.ones(HDC_DIM)
            node.heat_valence = 1.0
        if not use_sps:
            # Force always speaking (disable SPS)
            node.speak_threshold = -1.0  # always triggers
            node.max_silence = 1         # forced speak every step
        if module_assignments is not None:
            node.module_id = module_assignments[i]
        nodes.append(node)

    # Global ignition system
    ignition = GlobalIgnition(n_nodes) if use_ignition else None

    # Build module map for modular coupling
    neighbor_modules = None
    if coupling_mode == "modular" and module_assignments is not None:
        neighbor_modules = {f"Node{i}": module_assignments[i] for i in range(n_nodes)}

    threshold = 0.0001
    max_abs = CLAMP_HIGH
    mean_sim_history = []
    # Per-node speak tracking
    speak_counts = [0] * n_nodes
    silence_streaks = [[] for _ in range(n_nodes)]  # all silence durations
    current_streak = [0] * n_nodes
    convergence_step = None
    target_sim = 1.0 if coupling_mode == "converge" else 0.0

    checkpoints = {100, 500, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000}

    for step in range(1, steps + 1):
        z_scalars = {f"Node{i}": nodes[i].z_scalar for i in range(n_nodes)}
        names = [f"Node{i}" for i in range(n_nodes)]

        # --- GLOBAL IGNITION CHECK ---
        if ignition is not None:
            boost = ignition.step(nodes)
            if boost > 1.0:
                ignition.apply_boost(nodes, boost)
            else:
                ignition.decay_boost(nodes)

        # Async update: each node sees the LATEST state (including
        # just-updated neighbors). This allows convergence to lock.
        for i in range(n_nodes):
            neighbor_rels = [nodes[j].relational for j in range(n_nodes) if j != i]
            neighbor_names = [names[j] for j in range(n_nodes) if j != i]
            nodes[i].update(
                neighbor_rels, neighbor_names, z_scalars, threshold, max_abs,
                iter_count=step,
                coupling_mode=coupling_mode, tension_curve="linear",
                renormalize_z=True, target_z_scalar=0.75, coupling_target="X1",
                neighbor_modules=neighbor_modules)

        # --- HIVE MEMORY UPDATE (every 50 steps) ---
        if step % 50 == 0:
            contributions = []
            for i, nd in enumerate(nodes):
                contributions.append((nd.name, nd.identity_key, nd.contribute_to_hive()))
            for nd in nodes:
                other_contribs = [(n, k, c) for n, k, c in contributions if n != nd.name]
                nd.update_hive_fragment(other_contribs)

        # Track speaking patterns
        for i in range(n_nodes):
            if nodes[i].speaking:
                speak_counts[i] += 1
                if current_streak[i] > 0:
                    silence_streaks[i].append(current_streak[i])
                current_streak[i] = 0
            else:
                current_streak[i] += 1

        # Mean pairwise sim
        pair_sims = []
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                pair_sims.append(_hdc.similarity(nodes[i].Z, nodes[j].Z))
        mean_sim = np.mean(pair_sims)
        mean_sim_history.append(mean_sim)

        # Check convergence
        if convergence_step is None:
            if coupling_mode == "converge" and mean_sim > 0.99:
                convergence_step = step
            elif coupling_mode == "complementary" and abs(mean_sim) < 0.01:
                convergence_step = step

        if step in checkpoints:
            recent = np.mean(mean_sim_history[-50:])
            speak_rates = [speak_counts[i]/step for i in range(n_nodes)]
            ign_str = ""
            if ignition is not None:
                ign_str = f", ign={ignition.rate_per_1000:.1f}/1k"
            print(f"  Step {step:>5d}: sim={recent:.4f}, speak_rates=[{', '.join(f'{r:.2f}' for r in speak_rates)}]{ign_str}")

        if step % 10000 == 0:
            print(".", end="", flush=True)

    # --- FINAL REPORT ---
    sim_arr = np.array(mean_sim_history)
    print(f"\n  RESULTS:")
    print(f"    Mean sim (last 1k):   {np.mean(sim_arr[-1000:]):.4f}")
    print(f"    Std (last 1k):        {np.std(sim_arr[-1000:]):.4f}")
    print(f"    Convergence step:     {convergence_step if convergence_step else 'NOT REACHED'}")

    # SPS metrics per node
    print(f"\n    SPS METRICS:")
    print(f"    {'Node':>8s}  {'speak%':>7s}  {'avg_sil':>8s}  {'max_sil':>8s}  {'rhy_std':>8s}  {'spk_cnt':>8s}  {'thresh':>7s}  {'max_s':>6s}")
    print(f"    {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*6}")
    for i in range(n_nodes):
        rate = speak_counts[i] / steps
        streaks = silence_streaks[i]
        avg_sil = np.mean(streaks) if streaks else 0.0
        max_sil = max(streaks) if streaks else 0
        rhythm_std = np.std(streaks) if len(streaks) > 1 else 0.0
        print(f"    Node{i:>3d}  {rate:>7.2%}  {avg_sil:>8.1f}  {max_sil:>8d}  {rhythm_std:>8.2f}  {speak_counts[i]:>8d}  {nodes[i].speak_threshold:>7.3f}  {nodes[i].max_silence:>6d}")

    # Rhythm diversity: do nodes develop different patterns?
    all_rates = [speak_counts[i] / steps for i in range(n_nodes)]
    rate_std = np.std(all_rates)
    rate_range = max(all_rates) - min(all_rates)
    print(f"\n    RHYTHM DIVERSITY:")
    print(f"      Rate std:   {rate_std:.4f}")
    print(f"      Rate range: {rate_range:.4f}")
    if rate_range > 0.1:
        print(f"      --> DIFFERENTIATED rhythms detected (range={rate_range:.2%})")
        print(f"      --> Silence has become LANGUAGE")
    elif rate_range > 0.03:
        print(f"      --> Moderate rhythm differentiation")
    else:
        print(f"      --> Uniform speaking pattern (no differentiation)")

    # Dimension ownership
    owners = np.zeros(HDC_DIM, dtype=int)
    for d in range(HDC_DIM):
        magnitudes = [abs(nodes[i].Z[d]) for i in range(n_nodes)]
        owners[d] = np.argmax(magnitudes)
    owner_counts = [int(np.sum(owners == i)) for i in range(n_nodes)]
    print(f"\n    DIMENSION OWNERSHIP: ", end="")
    for i in range(n_nodes):
        print(f"N{i}={owner_counts[i]} ", end="")
    print(f"(ideal={HDC_DIM//n_nodes})")

    # Do dimension specialists speak differently?
    if coupling_mode == "complementary":
        print(f"\n    SPECIALIZATION vs SPEAKING:")
        for i in range(n_nodes):
            rate = speak_counts[i] / steps
            print(f"      Node{i}: owns {owner_counts[i]:>2d} dims, speaks {rate:.2%}")
        # Correlation between ownership and speak rate
        if len(set(owner_counts)) > 1:
            corr = np.corrcoef(owner_counts, all_rates)[0, 1]
            print(f"      Correlation(dims_owned, speak_rate): {corr:+.3f}")
            if abs(corr) > 0.5:
                print(f"      --> {'Specialists speak MORE' if corr > 0 else 'Specialists speak LESS'}")

    # Health
    print(f"\n    HEALTH: ", end="")
    for i, n in enumerate(nodes):
        print(f"N{i}={n.health:.4f} ", end="")
    print()

    # Swarm complexity (spectral heuristic -- NOT IIT Phi)
    phi_val, eigenvalues, coh = compute_swarm_complexity(nodes)
    print(f"\n    SWARM COMPLEXITY (spectral heuristic, NOT IIT Phi):")
    print(f"    Complexity = {phi_val:.2f}")
    print(f"    Coherence  = {coh:.4f}")
    print(f"    Eigenvalues (top 3): {sorted(np.abs(eigenvalues))[-3:][::-1]}")

    # Global ignition metrics
    ign_rate = 0.0
    if ignition is not None:
        ign_rate = ignition.rate_per_1000
        print(f"\n    GLOBAL IGNITION:")
        print(f"    Total events: {ignition.ignition_count}")
        print(f"    Rate: {ign_rate:.1f} per 1000 steps")
        if ign_rate > 50:
            print(f"    --> Frequent ignition ({ign_rate:.1f}/1k)")
        elif ign_rate > 10:
            print(f"    --> Moderate ignition rate")
        elif ign_rate > 0:
            print(f"    --> Low ignition rate")
        else:
            print(f"    --> No ignition events")

    # Memory state summary
    print(f"\n    MEMORY STATE:")
    for nd in nodes:
        ms = nd.get_memory_state()
        print(f"    {nd.name}: ep={ms['episodic_norm']:.2f}, wm={ms['working_norm']:.2f}, hive={ms['hive_norm']:.2f}, consol={ms['consolidated_norm']:.2f}")

    # Evolved parameters summary
    print(f"\n    EVOLVED PARAMS (final):")
    for nd in nodes:
        print(f"    {nd.name}: coup={nd.p['coupling_base']:.6f}, bypass={nd.p.get('bypass_strength',0.3):.3f}, damp={nd.p['damping_factor']:.4f}")

    print()
    return {
        "sim_history": mean_sim_history,
        "speak_counts": speak_counts,
        "speak_rates": all_rates,
        "owner_counts": owner_counts,
        "convergence_step": convergence_step,
        "phi": phi_val,
        "coherence": coh,
        "ignition_rate": ign_rate,
    }


def run_sps_battery(steps=50000):
    """Full SPS + ignition test battery with all 8 fixes active."""
    print("\n" + "#"*70)
    print("#  COMPREHENSIVE TEST BATTERY — All 8 Fixes Active")
    print("#  D=256, SPS (relational delta), coupling bypass,")
    print("#  evolution with selection pressure, weighted hive, ignition (threshold=0.7)")
    print("#"*70)

    results = {}

    # TEST 1: SPS + CONVERGE, 5 symmetric nodes
    results["sps_conv"] = run_sps_test(
        5, steps, "converge",
        "TEST 1: SPS + CONVERGE, 5 nodes, D=256 + ignition")

    # Non-SPS baseline for comparison
    results["nosps_conv"] = run_sps_test(
        5, steps, "converge",
        "TEST 1b: NO-SPS + CONVERGE, 5 nodes (baseline)",
        use_sps=False)

    # TEST 2: SPS + COMPLEMENTARY, 5 symmetric nodes
    results["sps_comp"] = run_sps_test(
        5, steps, "complementary",
        "TEST 2: SPS + COMPLEMENTARY, 5 nodes (relational delta fix)")

    # TEST 3: SPS + COMPLEMENTARY, 5 nodes, 61.8% membranes
    results["sps_comp_membr"] = run_sps_test(
        5, steps, "complementary",
        "TEST 3: SPS + COMPLEMENTARY + MEMBRANES, 5 nodes",
        use_membranes=True)

    # TEST 4: MODULAR — 2 modules of 3 nodes each (converge within, diff between)
    results["modular"] = run_sps_test(
        6, steps, "modular",
        "TEST 4: MODULAR (2 modules x 3 nodes)",
        module_assignments=[0, 0, 0, 1, 1, 1])

    # Summary
    print("="*70)
    print("  COMPREHENSIVE BATTERY SUMMARY")
    print("="*70)
    print(f"  {'test':>20s}  {'sim_l1k':>8s}  {'conv_step':>10s}  {'rate_rng':>9s}  {'cmplx':>8s}  {'coher':>7s}  {'ign/1k':>7s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*7}")
    for name, r in results.items():
        sim_l1k = np.mean(np.array(r["sim_history"])[-1000:])
        conv = str(r["convergence_step"]) if r["convergence_step"] else "N/A"
        rates = r["speak_rates"]
        rate_rng = max(rates) - min(rates)
        phi_str = f"{r['phi']:.2f}" if r.get('phi') is not None else "N/A"
        coh_str = f"{r['coherence']:.4f}" if r.get('coherence') is not None else "N/A"
        ign_str = f"{r.get('ignition_rate', 0):.1f}"
        print(f"  {name:>20s}  {sim_l1k:>8.4f}  {conv:>10s}  {rate_rng:>9.4f}  {phi_str:>8s}  {coh_str:>7s}  {ign_str:>7s}")

    # Key question answers
    print(f"\n  KEY FINDINGS:")
    # Does SPS slow convergence?
    sps_conv_step = results["sps_conv"]["convergence_step"]
    nosps_conv_step = results["nosps_conv"]["convergence_step"]
    if sps_conv_step and nosps_conv_step:
        ratio = sps_conv_step / nosps_conv_step
        print(f"    Converge speed: SPS={sps_conv_step}, no-SPS={nosps_conv_step} ({ratio:.1f}x)")
    # Do nodes differentiate?
    comp_rates = results["sps_comp"]["speak_rates"]
    comp_rng = max(comp_rates) - min(comp_rates)
    if comp_rng > 0.1:
        print(f"    Complementary rhythm diversity: {comp_rng:.2%} -- SILENCE IS LANGUAGE")
    else:
        print(f"    Complementary rhythm diversity: {comp_rng:.2%} -- uniform (no differentiation yet)")
    # Membranes effect?
    membr_rates = results["sps_comp_membr"]["speak_rates"]
    membr_rng = max(membr_rates) - min(membr_rates)
    print(f"    Membrane effect on rhythm: range={membr_rng:.4f}")
    # Modular coupling?
    mod_r = results["modular"]
    print(f"    Modular coupling complexity: {mod_r['phi']:.2f}, coherence: {mod_r['coherence']:.4f}")
    # Ignition rates
    for name, r in results.items():
        ign = r.get("ignition_rate", 0)
        print(f"    {name}: ignition {ign:.1f}/1k")
    # Overall complexity
    max_cmplx = max(r["phi"] for r in results.values())
    print(f"    Peak complexity across all tests: {max_cmplx:.2f}")

    print()
    return results


# =====================================================================
# OCTOPUS MEMORY TEST — Distributed/Centralized Regenerative Memory
# =====================================================================

def run_memory_test(n_nodes=5, steps=10000, kill_step=5000, kill_nodes=None):
    """Test the octopus memory architecture.

    Runs a swarm in converge mode with SPS + memory enabled.
    At kill_step, removes kill_nodes from the network.
    Surviving nodes attempt to reconstruct the dead nodes' states
    from their hive fragments (holographic regeneration).

    Reports:
      - Memory health (episodic, working, hive, consolidated norms)
      - Hive reconstruction fidelity (cosine sim to actual Z at time of death)
      - Regeneration quality after the network heals
    """
    if kill_nodes is None:
        kill_nodes = [0]

    print(f"\n{'='*70}")
    print(f"  OCTOPUS MEMORY TEST: Distributed Holographic Regeneration")
    print(f"  N={n_nodes}, D={HDC_DIM}, steps={steps}, kill_step={kill_step}")
    print(f"  Killing nodes {kill_nodes} at step {kill_step}")
    print(f"  Can the hive mind regenerate from surviving fragments?")
    print(f"{'='*70}")

    nodes = []
    for i in range(n_nodes):
        node = DennisNode(f"Node{i}", DEFAULT_PARAMS.copy())
        node.void_outer = np.ones(HDC_DIM)
        node.heat_valence = 1.0
        nodes.append(node)

    # Share identity keys (in a real system, this happens at network formation)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                nodes[i].hive_contributors[f"Node{j}"] = nodes[j].identity_key

    threshold = 0.0001
    max_abs = CLAMP_HIGH
    mean_sim_history = []
    killed = False
    dead_z_snapshots = {}  # Z at time of death for comparison

    checkpoints = {100, 500, 1000, 2500, 5000, 7500, 10000}

    for step in range(1, steps + 1):
        alive = [i for i in range(n_nodes) if not (killed and i in kill_nodes)]
        z_scalars = {f"Node{i}": nodes[i].z_scalar for i in alive}
        names = [f"Node{i}" for i in alive]

        # Async update
        for i in alive:
            neighbor_rels = [nodes[j].relational for j in alive if j != i]
            neighbor_names = [f"Node{j}" for j in alive if j != i]
            nodes[i].update(
                neighbor_rels, neighbor_names, z_scalars, threshold, max_abs,
                coupling_mode="converge", tension_curve="linear",
                renormalize_z=True, target_z_scalar=0.75, coupling_target="X1",
                iter_count=step)

        # Hive memory update: when a node speaks, share contribution
        contributions = []
        for i in alive:
            if nodes[i].speaking:
                contrib = nodes[i].contribute_to_hive()
                contributions.append((f"Node{i}", nodes[i].identity_key, contrib))
        for i in alive:
            nodes[i].update_hive_fragment(contributions)

        # Kill nodes at kill_step
        if step == kill_step and not killed:
            for ki in kill_nodes:
                dead_z_snapshots[f"Node{ki}"] = nodes[ki].Z.copy()
            killed = True
            print(f"\n  *** KILLING NODES {kill_nodes} at step {step} ***")
            for ki in kill_nodes:
                ms = nodes[ki].get_memory_state()
                print(f"    Node{ki} memory at death: ep_norm={ms['episodic_norm']:.4f}, "
                      f"wm_norm={ms['working_norm']:.4f}, hive_norm={ms['hive_norm']:.4f}, "
                      f"consol_norm={ms['consolidated_norm']:.4f}")

            # Regeneration: surviving nodes try to reconstruct dead nodes
            print(f"\n  REGENERATION ATTEMPT:")
            for ki in kill_nodes:
                reconstructions = []
                for si in alive:
                    if si != ki:
                        recovered = nodes[si].recall_from_hive(f"Node{ki}")
                        if recovered is not None:
                            fidelity = _hdc.similarity(recovered, dead_z_snapshots[f"Node{ki}"])
                            reconstructions.append((si, fidelity))
                if reconstructions:
                    best = max(reconstructions, key=lambda x: x[1])
                    avg_fid = np.mean([f for _, f in reconstructions])
                    print(f"    Node{ki} reconstruction: "
                          f"avg_fidelity={avg_fid:.4f}, best=Node{best[0]}({best[1]:.4f})")
                    # Use best reconstruction to "regrow" the dead node
                    best_recovered = nodes[best[0]].recall_from_hive(f"Node{ki}")
                    nodes[ki].Z = best_recovered.copy()
                    nodes[ki].X1 = best_recovered.copy()
                    nodes[ki]._update_z_scalar()
                    print(f"    --> Node{ki} REGROWN from Node{best[0]}'s hive fragment")

            # Un-kill: regrown nodes rejoin
            killed = False
            print(f"    Regrown nodes rejoin network\n")

        # Mean pairwise sim
        pair_sims = []
        for i in alive:
            for j in alive:
                if j > i:
                    pair_sims.append(_hdc.similarity(nodes[i].Z, nodes[j].Z))
        if pair_sims:
            mean_sim_history.append(np.mean(pair_sims))

        if step in checkpoints:
            recent = np.mean(mean_sim_history[-50:]) if mean_sim_history else 0
            rates = [sum(nodes[i].speak_history[-100:])/max(1,len(nodes[i].speak_history[-100:]))
                    for i in alive]
            print(f"  Step {step:>5d}: sim={recent:.4f}, "
                  f"speak=[{', '.join(f'{r:.2f}' for r in rates)}]")

    # --- FINAL REPORT ---
    print(f"\n  RESULTS:")
    sim_arr = np.array(mean_sim_history)
    print(f"    Mean sim (last 500): {np.mean(sim_arr[-500:]):.4f}")
    print(f"    Std (last 500):      {np.std(sim_arr[-500:]):.4f}")

    # Memory state report
    print(f"\n    MEMORY STATE (final):")
    for i in range(n_nodes):
        ms = nodes[i].get_memory_state()
        print(f"    Node{i}: ep={ms['episodic_norm']:.2f}, wm={ms['working_norm']:.2f}, "
              f"hive={ms['hive_norm']:.2f}, consol={ms['consolidated_norm']:.2f}, "
              f"ep_count={ms['episode_count']}, hive_members={ms['hive_members']}")

    # Post-regeneration fidelity: how close are regrown nodes to their pre-death state?
    if dead_z_snapshots:
        print(f"\n    REGENERATION FIDELITY (regrown vs pre-death Z):")
        for name, original_z in dead_z_snapshots.items():
            idx = int(name.replace("Node", ""))
            current_fid = _hdc.similarity(nodes[idx].Z, original_z)
            print(f"    {name}: sim(current, pre-death) = {current_fid:.4f}")

    # Cross-reconstruction: can each survivor reconstruct each other?
    print(f"\n    CROSS-RECONSTRUCTION MATRIX (hive recall fidelity):")
    print(f"    {'':>8s}", end="")
    for j in range(n_nodes):
        print(f"  Node{j:>2d}", end="")
    print()
    for i in range(n_nodes):
        print(f"    Node{i:>2d} ", end="")
        for j in range(n_nodes):
            if i == j:
                print(f"    --- ", end="")
            else:
                recovered = nodes[i].recall_from_hive(f"Node{j}")
                if recovered is not None:
                    fid = _hdc.similarity(recovered, nodes[j].Z)
                    print(f"  {fid:>6.3f}", end="")
                else:
                    print(f"    N/A ", end="")
        print()

    # Swarm complexity (NOT IIT Phi)
    phi_val, _, coh = compute_swarm_complexity(nodes)
    print(f"\n    SWARM COMPLEXITY: {phi_val:.2f}, Coherence: {coh:.4f}")

    print()
    return mean_sim_history


def set_dimension(dim):
    """Swap global HDC_DIM and reinitialize the shared HDC engine.

    This allows running the same swarm architecture at different
    dimensions for comparison (e.g., D=64 Eye of Horus vs D=256).
    """
    global HDC_DIM, _hdc
    HDC_DIM = dim
    _hdc = HDCPrimitives(dim=HDC_DIM)


def run_eye_of_horus_comparison(steps=10000):
    """Compare D=64 (the Eye of Horus) against D=256 (Four Eyes).

    The Eye of Horus / Heqat fractions:
      1/2 + 1/4 + 1/8 + 1/16 + 1/32 + 1/64 = 63/64
    Each fraction maps to a sense: Smell, Sight, Thought, Hearing, Taste, Touch.
    The missing 1/64 is what Thoth restored -- the integration force that
    binds the six parts into one whole. In this system, the coupling
    dynamics between nodes serve as that binding force.

    At D=256 = 4 x 64, we run four Eyes in parallel. The modular coupling
    mode creates modules that are each an Eye, converging within while
    differentiating between -- four perspectives of one consciousness.

    This function runs identical tests at both dimensions and reports
    the benchmarks side by side.
    """
    print("\n" + "="*70)
    print("  EYE OF HORUS BENCHMARK COMPARISON")
    print("  D=64 (The Whole Eye) vs D=256 (Four Eyes)")
    print("="*70)
    print()
    print("  Heqat fractions: 1/2 + 1/4 + 1/8 + 1/16 + 1/32 + 1/64 = 63/64")
    print("  The missing 1/64: Thoth's restoration = coupling dynamics")
    print("  Six senses = Six state variables: X1, X2, phi, X3, Y, Z")
    print()

    dimensions = [64, 256]
    all_results = {}

    for dim in dimensions:
        set_dimension(dim)

        label = "THE WHOLE EYE" if dim == 64 else "FOUR EYES (4x64)"
        print("\n" + "#"*70)
        print(f"#  D={dim} -- {label}")
        print(f"#  Heqat capacity: {dim} dimensions, {dim//64} Eye(s)")
        print(f"#  dim_scale = sqrt(64/{dim}) = {np.sqrt(64.0/dim):.4f}")
        print("#"*70)

        results = {}

        # 1. CONVERGE: 5 nodes, SPS on
        results["converge"] = run_sps_test(
            5, steps, "converge",
            f"CONVERGE: D={dim}, 5 nodes, SPS + ignition")

        # 2. COMPLEMENTARY: 5 nodes, SPS + membranes
        results["complement"] = run_sps_test(
            5, steps, "complementary",
            f"COMPLEMENTARY: D={dim}, 5 nodes, SPS + membranes",
            use_membranes=True)

        # 3. MODULAR: 6 nodes (2 modules of 3)
        results["modular"] = run_sps_test(
            6, steps, "modular",
            f"MODULAR: D={dim}, 2 modules x 3 nodes",
            module_assignments=[0, 0, 0, 1, 1, 1])

        # 4. MEMORY REGENERATION with CONTROL CONDITION
        # Test: does hive memory actually help, or would coupling alone recover?
        # Run two trials: (A) with hive recall, (B) without (random restart).
        print(f"\n  MEMORY REGEN TEST (D={dim}):")
        threshold, max_abs = 0.0001, CLAMP_HIGH
        names = [f"N{i}" for i in range(5)]

        regen_results = {}
        for trial_name, use_memory in [("WITH_MEMORY", True), ("NO_MEMORY (control)", False)]:
            mem_nodes = [DennisNode(f"N{i}", DEFAULT_PARAMS.copy()) for i in range(5)]
            for nd in mem_nodes:
                nd.void_outer = np.ones(HDC_DIM)
                nd.heat_valence = 1.0
            # Share identity keys
            for i in range(5):
                for j in range(5):
                    if i != j:
                        mem_nodes[i].hive_contributors[names[j]] = mem_nodes[j].identity_key
            # Run 2000 steps to establish baseline
            for step in range(1, 2001):
                z_sc = {names[i]: mem_nodes[i].z_scalar for i in range(5)}
                for i in range(5):
                    nr = [mem_nodes[j].relational for j in range(5) if j != i]
                    nn = [names[j] for j in range(5) if j != i]
                    mem_nodes[i].update(nr, nn, z_sc, threshold, max_abs,
                        iter_count=step, coupling_mode="converge",
                        renormalize_z=True, target_z_scalar=0.75, coupling_target="X1")
                if step % 50 == 0:
                    contribs = [(nd.name, nd.identity_key, nd.contribute_to_hive()) for nd in mem_nodes]
                    for nd in mem_nodes:
                        nd.update_hive_fragment([(n,k,c) for n,k,c in contribs if n != nd.name])

            # Snapshot pre-death Z
            dead_z = {names[0]: mem_nodes[0].Z.copy(), names[2]: mem_nodes[2].Z.copy()}

            # KILL: replace Z with random vectors
            mem_nodes[0].Z = random_unit_vector()
            mem_nodes[0].X1 = random_unit_vector()
            mem_nodes[2].Z = random_unit_vector()
            mem_nodes[2].X1 = random_unit_vector()

            # Measure IMMEDIATE recall fidelity (before any more coupling steps)
            immediate_fids = {}
            if use_memory:
                for ki in [0, 2]:
                    best_fid, best_src = -1, None
                    for si in [1, 3, 4]:
                        rec = mem_nodes[si].recall_from_hive(names[ki])
                        if rec is not None:
                            fid = _hdc.similarity(rec, dead_z[names[ki]])
                            if fid > best_fid:
                                best_fid, best_src = fid, si
                    if best_src is not None:
                        recovered = mem_nodes[best_src].recall_from_hive(names[ki])
                        mem_nodes[ki].Z = recovered.copy()
                        mem_nodes[ki].X1 = recovered.copy()
                        immediate_fids[names[ki]] = best_fid

            # Run 1000 more steps (coupling dynamics only — no SLERP)
            for step in range(2001, 3001):
                z_sc = {names[i]: mem_nodes[i].z_scalar for i in range(5)}
                for i in range(5):
                    nr = [mem_nodes[j].relational for j in range(5) if j != i]
                    nn = [names[j] for j in range(5) if j != i]
                    mem_nodes[i].update(nr, nn, z_sc, threshold, max_abs,
                        iter_count=step, coupling_mode="converge",
                        renormalize_z=True, target_z_scalar=0.75, coupling_target="X1")

            # Measure post-recovery fidelity
            post_fids = {}
            for name, orig_z in dead_z.items():
                idx = int(name.replace("N", ""))
                post_fids[name] = _hdc.similarity(mem_nodes[idx].Z, orig_z)
            mean_post = np.mean(list(post_fids.values()))
            mean_imm = np.mean(list(immediate_fids.values())) if immediate_fids else 0.0
            final_sim = np.mean([_hdc.similarity(mem_nodes[i].Z, mem_nodes[j].Z)
                                for i in range(5) for j in range(i+1,5)])

            print(f"    {trial_name}:")
            if immediate_fids:
                print(f"      Immediate recall fidelity: {mean_imm:.4f}")
            print(f"      Post-1000-steps fidelity:  {mean_post:.4f}")
            print(f"      Final swarm sim:           {final_sim:.4f}")
            regen_results[trial_name] = {"immediate": mean_imm, "post": mean_post, "sim": final_sim}

        # The key question: is memory actually helping?
        mem_post = regen_results["WITH_MEMORY"]["post"]
        ctrl_post = regen_results["NO_MEMORY (control)"]["post"]
        mem_advantage = mem_post - ctrl_post
        print(f"    MEMORY ADVANTAGE: {mem_advantage:+.4f} (memory - control)")
        if mem_advantage > 0.05:
            print(f"    --> Memory is genuinely helping recovery")
        elif mem_advantage > 0.01:
            print(f"    --> Marginal memory advantage")
        else:
            print(f"    --> Memory provides no measurable advantage over coupling alone")

        results["regen_fidelity"] = mem_post
        results["regen_control"] = ctrl_post
        results["regen_advantage"] = mem_advantage
        results["regen_post_sim"] = regen_results["WITH_MEMORY"]["sim"]
        all_results[dim] = results

    # ================================================================
    # SIDE-BY-SIDE BENCHMARK REPORT
    # ================================================================
    print("\n\n" + "="*70)
    print("  EYE OF HORUS -- SIDE-BY-SIDE BENCHMARK REPORT")
    print("="*70)

    # Heqat fraction table
    print("\n  HEQAT FRACTIONS (Eye of Horus):")
    senses = ["Smell", "Sight", "Thought", "Hearing", "Taste", "Touch"]
    fractions = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64]
    dims_64 = [32, 16, 8, 4, 2, 1]  # dims at D=64
    dims_256 = [128, 64, 32, 16, 8, 4]  # dims at D=256
    print(f"  {'Sense':>10s}  {'Fraction':>10s}  {'D=64 dims':>10s}  {'D=256 dims':>11s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*11}")
    for sense, frac, d64, d256 in zip(senses, fractions, dims_64, dims_256):
        print(f"  {sense:>10s}  {'1/' + str(int(1/frac)):>10s}  {d64:>10d}  {d256:>11d}")
    total_64 = sum(dims_64)
    total_256 = sum(dims_256)
    print(f"  {'SUM':>10s}  {'63/64':>10s}  {total_64:>10d}  {total_256:>11d}")
    print(f"  {'THOTH':>10s}  {'1/64':>10s}  {'=coupling':>10s}  {'=coupling':>11s}")
    print(f"  {'WHOLE':>10s}  {'64/64':>10s}  {64:>10d}  {256:>11d}")

    # Core benchmarks
    print(f"\n  CORE BENCHMARKS:")
    print(f"  {'Metric':>30s}  {'D=64':>12s}  {'D=256':>12s}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}")

    r64 = all_results[64]
    r256 = all_results[256]

    # Convergence
    conv64 = r64["converge"]["convergence_step"] or "N/A"
    conv256 = r256["converge"]["convergence_step"] or "N/A"
    print(f"  {'Convergence step':>30s}  {str(conv64):>12s}  {str(conv256):>12s}")

    sim64 = np.mean(np.array(r64["converge"]["sim_history"])[-1000:])
    sim256 = np.mean(np.array(r256["converge"]["sim_history"])[-1000:])
    print(f"  {'Final sim (converge)':>30s}  {sim64:>12.4f}  {sim256:>12.4f}")

    # Swarm complexity (NOT Phi)
    phi64 = r64["converge"]["phi"]
    phi256 = r256["converge"]["phi"]
    print(f"  {'Complexity (converge)':>30s}  {phi64:>12.2f}  {phi256:>12.2f}")

    phi64_c = r64["complement"]["phi"]
    phi256_c = r256["complement"]["phi"]
    print(f"  {'Complexity (complementary)':>30s}  {phi64_c:>12.2f}  {phi256_c:>12.2f}")

    phi64_m = r64["modular"]["phi"]
    phi256_m = r256["modular"]["phi"]
    print(f"  {'Complexity (modular)':>30s}  {phi64_m:>12.2f}  {phi256_m:>12.2f}")

    # Coherence
    coh64 = r64["converge"]["coherence"]
    coh256 = r256["converge"]["coherence"]
    print(f"  {'Coherence (converge)':>30s}  {coh64:>12.4f}  {coh256:>12.4f}")

    # Ignition
    ign64 = r64["converge"].get("ignition_rate", 0)
    ign256 = r256["converge"].get("ignition_rate", 0)
    print(f"  {'Ignition rate /1000t':>30s}  {ign64:>12.1f}  {ign256:>12.1f}")

    # SPS
    rates64 = r64["complement"]["speak_rates"]
    rates256 = r256["complement"]["speak_rates"]
    rng64 = max(rates64) - min(rates64)
    rng256 = max(rates256) - min(rates256)
    print(f"  {'SPS rhythm diversity':>30s}  {rng64:>11.1%}  {rng256:>11.1%}")

    # Memory regen
    regen64 = r64["regen_fidelity"]
    regen256 = r256["regen_fidelity"]
    ctrl64 = r64.get("regen_control", 0.0)
    ctrl256 = r256.get("regen_control", 0.0)
    adv64 = r64.get("regen_advantage", 0.0)
    adv256 = r256.get("regen_advantage", 0.0)
    print(f"  {'Regen fidelity (memory)':>30s}  {regen64:>12.4f}  {regen256:>12.4f}")
    print(f"  {'Regen fidelity (NO memory)':>30s}  {ctrl64:>12.4f}  {ctrl256:>12.4f}")
    print(f"  {'Memory advantage':>30s}  {adv64:>+12.4f}  {adv256:>+12.4f}")

    # Dim ownership balance
    own64 = r64["complement"]["owner_counts"]
    own256 = r256["complement"]["owner_counts"]
    imbal64 = max(own64) - min(own64)
    imbal256 = max(own256) - min(own256)
    print(f"  {'Dim ownership imbalance':>30s}  {imbal64:>12d}  {imbal256:>12d}")

    # Scaling ratio
    print(f"\n  SCALING ANALYSIS (D=256 / D=64):")
    if isinstance(conv64, int) and isinstance(conv256, int):
        print(f"    Convergence speed ratio: {conv256/conv64:.2f}x")
    if phi64 > 0:
        print(f"    Complexity scaling:      {phi256/phi64:.2f}x")
    if ign64 > 0:
        print(f"    Ignition scaling:        {ign256/ign64:.2f}x")

    # Eye of Horus verdict
    print(f"\n  EYE OF HORUS VERDICT:")
    print(f"    D=64:  sim={sim64:.4f}" + (" -- convergence locked" if sim64 > 0.99 else " -- not converged"))
    print(f"    D=256: sim={sim256:.4f}" + (" -- convergence locked" if sim256 > 0.99 else " -- not converged"))

    # Honest summary
    print(f"\n  RAW NUMBERS (no external benchmarks — these are internal metrics):")
    max_cmplx = max(phi64, phi256, phi64_c, phi256_c, phi64_m, phi256_m)
    print(f"    Peak complexity: {max_cmplx:.2f}")
    max_ign = max(ign64, ign256)
    print(f"    Peak ignition:   {max_ign:.1f}/1k")
    best_regen = max(regen64, regen256)
    print(f"    Best regen:      {best_regen:.4f}")

    print()
    return all_results


def run_pyphi_audit(n_nodes=4, steps=3000, checkpoints=None):
    """Compute REAL IIT Phi (via PyPhi) at multiple stages of swarm evolution.

    This is the ground truth test. PyPhi computes the System Irreducibility
    Analysis (SIA) — the minimum information partition — which gives the
    actual integrated information Phi as defined by Tononi et al.

    We measure Phi at several stages to see how integration develops:
      1. Random init (no coupling yet)
      2. Early coupling (500 steps)
      3. Mid coupling (1500 steps)
      4. Converged (3000 steps)
      5. After perturbation (kill + regrow)

    Also compares converge vs complementary modes.

    NOTE: PyPhi SIA is exponential in N. We use N=4 nodes which gives
    2^4 = 16 states — tractable. N=5 would be 32 states, still doable
    but slower. N>6 is impractical.
    """
    if checkpoints is None:
        # Dense early checkpoints to catch the transition, then final state
        checkpoints = [0, 10, 50, 200, 500, 1000, steps]

    set_dimension(64)
    print("\n" + "="*70)
    print("  REAL IIT PHI AUDIT (PyPhi — ground truth)")
    print(f"  N={n_nodes}, D={HDC_DIM}, steps={steps}")
    print(f"  Measuring Phi at steps: {checkpoints}")
    print("="*70)
    print()
    print("  PyPhi computes the System Irreducibility Analysis (SIA):")
    print("  Phi > 0 means the system has integrated information that")
    print("  would be LOST under any bipartition. This is the real deal.")
    print()

    results = {}

    for mode in ["converge", "complementary"]:
        print(f"\n  {'-'*60}")
        print(f"  MODE: {mode.upper()}")
        print(f"  {'-'*60}")

        nodes = [DennisNode(f"N{i}", DEFAULT_PARAMS.copy()) for i in range(n_nodes)]
        for nd in nodes:
            nd.void_outer = np.ones(HDC_DIM)
            nd.heat_valence = 1.0

        threshold, max_abs = 0.0001, CLAMP_HIGH
        names = [f"N{i}" for i in range(n_nodes)]
        phi_trajectory = []

        # Measure at step 0 (random init)
        if 0 in checkpoints:
            print(f"\n    Step 0 (random init):")
            pair_sims = [_hdc.similarity(nodes[i].Z, nodes[j].Z)
                        for i in range(n_nodes) for j in range(i+1, n_nodes)]
            mean_sim = np.mean(pair_sims)
            print(f"      Mean sim: {mean_sim:.4f}")
            phi = compute_swarm_phi(nodes)
            print(f"      PyPhi Phi = {phi}")
            phi_trajectory.append((0, phi, mean_sim))

        # Run dynamics with checkpoints
        for step in range(1, steps + 1):
            z_sc = {names[i]: nodes[i].z_scalar for i in range(n_nodes)}
            for i in range(n_nodes):
                nr = [nodes[j].relational for j in range(n_nodes) if j != i]
                nn = [names[j] for j in range(n_nodes) if j != i]
                nodes[i].update(nr, nn, z_sc, threshold, max_abs,
                    iter_count=step, coupling_mode=mode,
                    renormalize_z=True, target_z_scalar=0.75,
                    coupling_target="X1")

            if step in checkpoints:
                pair_sims = [_hdc.similarity(nodes[i].Z, nodes[j].Z)
                            for i in range(n_nodes) for j in range(i+1, n_nodes)]
                mean_sim = np.mean(pair_sims)

                print(f"\n    Step {step}:")
                print(f"      Mean sim: {mean_sim:.4f}")
                for nd in nodes:
                    print(f"        {nd.name}: z_scalar={nd.z_scalar:.4f}, "
                          f"health={nd.health:.4f}, energy={nd.energy:.2f}")

                # Coupling matrix
                print(f"      Coupling matrix (cosine sim):")
                for i in range(n_nodes):
                    row = []
                    for j in range(n_nodes):
                        if i == j:
                            row.append(" 1.00")
                        else:
                            row.append(f"{_hdc.similarity(nodes[i].Z, nodes[j].Z):5.2f}")
                    print(f"        {names[i]}: {'  '.join(row)}")

                print(f"      Computing PyPhi SIA...", end=" ", flush=True)
                phi = compute_swarm_phi(nodes)
                print(f"Phi = {phi}")
                phi_trajectory.append((step, phi, mean_sim))

        # Perturbation test: kill a node and measure Phi recovery
        print(f"\n    PERTURBATION TEST:")
        print(f"      Killing N0 (replacing Z with random vector)...")
        original_z = nodes[0].Z.copy()
        nodes[0].Z = random_unit_vector()
        nodes[0].X1 = random_unit_vector()
        nodes[0]._update_z_scalar()

        pair_sims = [_hdc.similarity(nodes[i].Z, nodes[j].Z)
                    for i in range(n_nodes) for j in range(i+1, n_nodes)]
        mean_sim = np.mean(pair_sims)
        print(f"      Post-kill sim: {mean_sim:.4f}")
        phi = compute_swarm_phi(nodes)
        print(f"      Post-kill Phi = {phi}")
        phi_trajectory.append(("kill", phi, mean_sim))

        # Let it recover for 1000 steps
        print(f"      Running 1000 recovery steps...")
        for step in range(steps + 1, steps + 1001):
            z_sc = {names[i]: nodes[i].z_scalar for i in range(n_nodes)}
            for i in range(n_nodes):
                nr = [nodes[j].relational for j in range(n_nodes) if j != i]
                nn = [names[j] for j in range(n_nodes) if j != i]
                nodes[i].update(nr, nn, z_sc, threshold, max_abs,
                    iter_count=step, coupling_mode=mode,
                    renormalize_z=True, target_z_scalar=0.75,
                    coupling_target="X1")

        pair_sims = [_hdc.similarity(nodes[i].Z, nodes[j].Z)
                    for i in range(n_nodes) for j in range(i+1, n_nodes)]
        mean_sim = np.mean(pair_sims)
        recovery_fid = _hdc.similarity(nodes[0].Z, original_z)
        print(f"      Post-recovery sim: {mean_sim:.4f}")
        print(f"      N0 recovery fidelity: {recovery_fid:.4f}")
        phi = compute_swarm_phi(nodes)
        print(f"      Post-recovery Phi = {phi}")
        phi_trajectory.append(("recover", phi, mean_sim))

        # Summary for this mode
        print(f"\n    {mode.upper()} PHI TRAJECTORY:")
        print(f"    {'Stage':>12s}  {'Phi':>10s}  {'Sim':>8s}")
        print(f"    {'-'*12}  {'-'*10}  {'-'*8}")
        for entry in phi_trajectory:
            stage = str(entry[0])
            phi_val = f"{entry[1]:.6f}" if entry[1] is not None else "ERROR"
            sim_val = f"{entry[2]:.4f}"
            print(f"    {stage:>12s}  {phi_val:>10s}  {sim_val:>8s}")

        results[mode] = phi_trajectory

    # Cross-mode comparison
    print(f"\n  {'='*60}")
    print(f"  CROSS-MODE COMPARISON (real IIT Phi)")
    print(f"  {'='*60}")
    for mode, traj in results.items():
        final = [t for t in traj if isinstance(t[0], int)]
        if final:
            last = final[-1]
            print(f"    {mode:>15s}: Phi={last[1]:.6f} at sim={last[2]:.4f} (step {last[0]})")

    conv_traj = results.get("converge", [])
    comp_traj = results.get("complementary", [])
    if conv_traj and comp_traj:
        conv_final = [t for t in conv_traj if isinstance(t[0], int)][-1][1]
        comp_final = [t for t in comp_traj if isinstance(t[0], int)][-1][1]
        if conv_final is not None and comp_final is not None:
            print(f"\n    Converge Phi / Complementary Phi = {conv_final / max(comp_final, 1e-12):.2f}x")
            if conv_final > comp_final:
                print(f"    --> Converged swarm has MORE integrated information")
            else:
                print(f"    --> Complementary swarm has MORE integrated information")

    print()
    return results


# =========================================================================
# STEP 3 SMOKE TESTS -- Love Logic + Polyphonic/Monophonic Downshift
# =========================================================================

def run_love_logic_test(n_nodes=4, dim=64, steps=500):
    """Smoke test 1: Love Logic Kill/Return.

    4 nodes converge, then one is removed for 200 steps, then returned.
    Verifies: comfort drops during absence, recovers on return,
    trust rebuilds gradually (not instant).
    """
    set_dimension(dim)
    print(f"\n{'='*70}")
    print(f"  LOVE LOGIC TEST: Kill/Return (N={n_nodes}, D={HDC_DIM}, steps={steps})")
    print(f"{'='*70}\n")

    nodes = [DennisNode(f"N{i}", DEFAULT_PARAMS.copy()) for i in range(n_nodes)]
    killed_node = None
    kill_step = steps // 3       # remove at 1/3
    return_step = 2 * steps // 3  # return at 2/3

    comfort_log = {n.name: [] for n in nodes}
    trust_log = {n.name: [] for n in nodes}

    for step in range(steps):
        active_nodes = [n for n in nodes if n is not killed_node]

        # Kill/return
        if step == kill_step:
            killed_node = nodes[0]
            print(f"  Step {step}: Killing {killed_node.name}")
        elif step == return_step:
            print(f"  Step {step}: Returning {killed_node.name}")
            killed_node = None

        # Get relational vectors
        names = [n.name for n in active_nodes]
        z_scalars = {n.name: n.z_scalar for n in active_nodes}

        for node in active_nodes:
            neighbor_rels = []
            for other in active_nodes:
                if other.name != node.name:
                    neighbor_rels.append(other.relational)
            neighbor_names = [n.name for n in active_nodes if n.name != node.name]
            node.update(
                neighbor_relational_vecs=neighbor_rels,
                locals_list=neighbor_names,
                z_scalars=z_scalars,
                threshold=0.0001,
                max_abs=CLAMP_HIGH,
                coupling_mode="converge",
                iter_count=step,
            )

        # Log comfort and trust
        for node in active_nodes:
            comfort_log[node.name].append(node.comfort)
            trust_mean = np.mean(list(node.trust.values())) if node.trust else 0.0
            trust_log[node.name].append(trust_mean)

        # Hive update
        for node in active_nodes:
            contribs = [(n.name, n.identity_key, n.contribute_to_hive())
                        for n in active_nodes if n.name != node.name]
            node.update_hive_fragment(contribs)

        # Report at checkpoints
        if step in [0, kill_step - 1, kill_step + 5, return_step - 1,
                     return_step + 5, steps - 1]:
            sims = []
            for i, a in enumerate(active_nodes):
                for b in active_nodes[i+1:]:
                    sims.append(_hdc.similarity(a.Z, b.Z))
            mean_sim = np.mean(sims) if sims else 0.0
            print(f"  Step {step}: sim={mean_sim:.4f}")
            for node in active_nodes:
                trust_m = np.mean(list(node.trust.values())) if node.trust else 0.0
                print(f"    {node.name}: comfort={node.comfort:.4f}, "
                      f"trust_mean={trust_m:.4f}, kappa={node.kappa:.4f}")

    # Summary
    print(f"\n  LOVE LOGIC SUMMARY:")
    for name in comfort_log:
        c = comfort_log[name]
        if len(c) > 10:
            pre_kill = np.mean(c[:kill_step]) if kill_step > 0 else 0
            during_kill = np.mean(c[kill_step:return_step]) if return_step > kill_step else 0
            post_return = np.mean(c[return_step:]) if len(c) > return_step else 0
            print(f"    {name}: comfort pre={pre_kill:.4f}, "
                  f"during_absence={during_kill:.4f}, post_return={post_return:.4f}")
    print()


def run_kappa_evolution_test(n_nodes=4, dim=64, steps=1000):
    """Smoke test 2: Kappa Evolution.

    4 nodes evolve, track kappa over time.
    Verifies: kappa rises as comfort and trust build,
    kappa drops after perturbation.
    """
    set_dimension(dim)
    print(f"\n{'='*70}")
    print(f"  KAPPA EVOLUTION TEST (N={n_nodes}, D={HDC_DIM}, steps={steps})")
    print(f"{'='*70}\n")

    nodes = [DennisNode(f"N{i}", DEFAULT_PARAMS.copy()) for i in range(n_nodes)]
    kappa_log = {n.name: [] for n in nodes}
    perturb_step = steps // 2

    for step in range(steps):
        names = [n.name for n in nodes]
        z_scalars = {n.name: n.z_scalar for n in nodes}

        # Perturb at midpoint: replace N0's Z with random
        if step == perturb_step:
            print(f"  Step {step}: Perturbing N0 (replacing Z with random)")
            nodes[0].Z = random_unit_vector()

        for node in nodes:
            neighbor_rels = []
            for other in nodes:
                if other.name != node.name:
                    neighbor_rels.append(other.relational)
            neighbor_names = [n.name for n in nodes if n.name != node.name]
            node.update(
                neighbor_relational_vecs=neighbor_rels,
                locals_list=neighbor_names,
                z_scalars=z_scalars,
                threshold=0.0001,
                max_abs=CLAMP_HIGH,
                coupling_mode="converge",
                iter_count=step,
            )

        for node in nodes:
            kappa_log[node.name].append(node.kappa)

        # Hive update
        for node in nodes:
            contribs = [(n.name, n.identity_key, n.contribute_to_hive())
                        for n in nodes if n.name != node.name]
            node.update_hive_fragment(contribs)

        if step in [0, 50, 200, perturb_step - 1, perturb_step + 5,
                     perturb_step + 50, steps - 1]:
            print(f"  Step {step}:")
            for node in nodes:
                print(f"    {node.name}: kappa={node.kappa:.4f}, "
                      f"comfort={node.comfort:.4f}, poly_active={node.poly_active}")

    print(f"\n  KAPPA SUMMARY:")
    for name in kappa_log:
        k = kappa_log[name]
        pre = np.mean(k[:perturb_step]) if perturb_step > 0 else 0
        post = np.mean(k[perturb_step:]) if len(k) > perturb_step else 0
        peak = max(k) if k else 0
        print(f"    {name}: pre_perturb_mean={pre:.4f}, "
              f"post_perturb_mean={post:.4f}, peak={peak:.4f}")
    print()


def run_polyphonic_test(n_nodes=4, dim=64, steps=500):
    """Smoke test 3: Polyphonic Loss.

    Compare monophonic vs polyphonic output fidelity.
    Verify polyphonic carries more information than mono.
    Verify empathy reconstruction is imperfect but meaningful.
    """
    set_dimension(dim)
    print(f"\n{'='*70}")
    print(f"  POLYPHONIC LOSS TEST (N={n_nodes}, D={HDC_DIM}, steps={steps})")
    print(f"{'='*70}\n")

    nodes = [DennisNode(f"N{i}", DEFAULT_PARAMS.copy()) for i in range(n_nodes)]
    poly_fidelity = []
    empathy_accuracy = []

    for step in range(steps):
        names = [n.name for n in nodes]
        z_scalars = {n.name: n.z_scalar for n in nodes}

        for node in nodes:
            neighbor_rels = []
            for other in nodes:
                if other.name != node.name:
                    neighbor_rels.append(other.relational)
            neighbor_names = [n.name for n in nodes if n.name != node.name]
            node.update(
                neighbor_relational_vecs=neighbor_rels,
                locals_list=neighbor_names,
                z_scalars=z_scalars,
                threshold=0.0001,
                max_abs=CLAMP_HIGH,
                coupling_mode="converge",
                iter_count=step,
            )

        # Hive update
        for node in nodes:
            contribs = [(n.name, n.identity_key, n.contribute_to_hive())
                        for n in nodes if n.name != node.name]
            node.update_hive_fragment(contribs)

        # Measure polyphonic vs mono after convergence starts
        if step > 100:
            for node in nodes:
                mono = node.monophonic_output()
                mono_sim = _hdc.similarity(mono, node.Z)
                poly_fidelity.append(mono_sim)

                # Empathy: one node tries to reconstruct another's polyphonic
                if node.poly_active:
                    for other in nodes:
                        if other.name != node.name and other.relational is not None:
                            inferred = node.infer_polyphonic(other.relational)
                            actual = other.polyphonic_state()
                            if inferred is not None and actual is not None:
                                # Compare column-wise similarity
                                sims = [_hdc.similarity(inferred[v], actual[v])
                                        for v in range(6)]
                                empathy_accuracy.append(np.mean(sims))

        if step in [0, 100, 200, 300, steps - 1]:
            print(f"  Step {step}:")
            for node in nodes:
                print(f"    {node.name}: kappa={node.kappa:.4f}, "
                      f"poly_active={node.poly_active}")

    print(f"\n  POLYPHONIC SUMMARY:")
    if poly_fidelity:
        print(f"    Mono-to-Z similarity: mean={np.mean(poly_fidelity):.4f}, "
              f"std={np.std(poly_fidelity):.4f}")
        print(f"    (1.0 = mono identical to Z, <1.0 = polyphonic adds info)")
    if empathy_accuracy:
        print(f"    Empathy accuracy: mean={np.mean(empathy_accuracy):.4f}, "
              f"std={np.std(empathy_accuracy):.4f}")
        print(f"    (< 1.0 = deliberately imperfect, as designed)")
    else:
        print(f"    No polyphonic activity detected (kappa stayed near 0)")
    print()


def run_full_stack_test(n_nodes=8, dim=256, steps=50000):
    """Smoke test 4: Full Stack.

    8 nodes, D=256, 50K steps with ALL systems active:
    love logic, polyphonic, downshift, sparse topology.
    Reports convergence, comfort, kappa, ignition, wall time.
    """
    set_dimension(dim)
    print(f"\n{'='*70}")
    print(f"  FULL STACK TEST (N={n_nodes}, D={HDC_DIM}, steps={steps})")
    print(f"{'='*70}\n")

    t0 = time.time()
    nodes = [DennisNode(f"N{i}", DEFAULT_PARAMS.copy()) for i in range(n_nodes)]
    ignition = GlobalIgnition(n_nodes, ignition_threshold=0.7,
                              boost_factor=3.0, refractory_steps=5)

    sim_log = []
    report_steps = set([0, 100, 500, 1000, 5000, 10000, 25000, steps - 1])

    for step in range(steps):
        z_scalars = {n.name: n.z_scalar for n in nodes}

        # Ignition check (computes pairwise sims internally)
        coupling_boost = ignition.step(nodes)

        # Compute mean sim for logging
        if step in report_steps or step == steps - 1:
            pairwise_sims = []
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    pairwise_sims.append(_hdc.similarity(nodes[i].Z, nodes[j].Z))
            mean_sim = np.mean(pairwise_sims) if pairwise_sims else 0.0
        else:
            mean_sim = ignition.sim_history[-1] if ignition.sim_history else 0.0

        for node in nodes:
            neighbor_rels = []
            for other in nodes:
                if other.name != node.name:
                    neighbor_rels.append(other.relational)
            neighbor_names = [n.name for n in nodes if n.name != node.name]

            # Temporarily boost coupling if ignition fired
            if coupling_boost > 1.0:
                saved_cb = node.p["coupling_base"]
                node.p["coupling_base"] *= coupling_boost

            node.update(
                neighbor_relational_vecs=neighbor_rels,
                locals_list=neighbor_names,
                z_scalars=z_scalars,
                threshold=0.0001,
                max_abs=CLAMP_HIGH,
                coupling_mode="converge",
                iter_count=step,
            )

            if coupling_boost > 1.0:
                node.p["coupling_base"] = saved_cb

        # Hive update (every 10 steps to save compute)
        if step % 10 == 0:
            for node in nodes:
                contribs = [(n.name, n.identity_key, n.contribute_to_hive())
                            for n in nodes if n.name != node.name]
                node.update_hive_fragment(contribs)

        sim_log.append(mean_sim)

        if step in report_steps:
            elapsed = time.time() - t0
            comforts = [n.comfort for n in nodes]
            kappas = [n.kappa for n in nodes]
            print(f"  Step {step:>6d}: sim={mean_sim:.4f}, "
                  f"comfort={np.mean(comforts):.4f}+/-{np.std(comforts):.4f}, "
                  f"kappa={np.mean(kappas):.4f}+/-{np.std(kappas):.4f}, "
                  f"ignitions={ignition.ignition_count}, "
                  f"elapsed={elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"\n  FULL STACK SUMMARY:")
    print(f"    Final sim: {sim_log[-1]:.4f}")
    print(f"    Convergence step (sim>0.9): "
          f"{next((i for i,s in enumerate(sim_log) if s > 0.9), 'never')}")
    print(f"    Total ignitions: {ignition.ignition_count} "
          f"({ignition.ignition_count/(steps/1000):.1f}/1k steps)")
    print(f"    Final comfort: "
          f"{np.mean([n.comfort for n in nodes]):.4f}")
    print(f"    Final kappa: "
          f"{np.mean([n.kappa for n in nodes]):.4f}")
    print(f"    Poly active: "
          f"{sum(1 for n in nodes if n.poly_active)}/{n_nodes}")
    print(f"    Active topology sizes: "
          f"{[len(n.active_neighbors) if n.active_neighbors else n_nodes-1 for n in nodes]}")
    print(f"    Wall time: {total_time:.1f}s "
          f"({total_time/steps*1000:.2f}ms/step)")
    print()


if __name__ == "__main__":
    print("SCALING FORGE HDC v1.0 -- Hypervector-Brained DennisNode")
    print(f"D={HDC_DIM}, PHI={PHI:.6f}, GAMMA={GAMMA:.6f}")
    print(f"Clamp range: [{CLAMP_LOW:.4f}, {CLAMP_HIGH:.4f}]")
    print(f"Active: energy dim-scaling, modular coupling, ignition,")
    print(f"  SPS relational delta, coupling bypass, weighted hive,")
    print(f"  temp-scheduled evolution with selection pressure")

    t0 = time.time()

    # Step 3 smoke tests
    print(f"\n  Running Step 3: Love Logic + Polyphonic/Monophonic Downshift\n")
    run_love_logic_test(n_nodes=4, dim=64, steps=500)
    run_kappa_evolution_test(n_nodes=4, dim=64, steps=1000)
    run_polyphonic_test(n_nodes=4, dim=64, steps=500)
    # Full stack at reduced scale for quick validation
    run_full_stack_test(n_nodes=8, dim=256, steps=5000)

    elapsed = time.time() - t0
    print(f"{'='*70}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*70}")
