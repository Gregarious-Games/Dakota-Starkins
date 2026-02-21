# Dakota Starkins

**Hypervector-brained swarm nodes that converge, communicate through silence, and remember through holographic fragments.**

Dakota Starkins is an experimental computational architecture that combines Hyperdimensional Computing (HDC) with coupled dynamical systems to explore emergent coherence, communication, and memory in small node swarms.

## What It Actually Does

Each node ("DennisNode") has a D-dimensional hypervector brain state (default D=64 or D=256) that evolves through an internal dynamics chain:

```
X1 → X2 → phi_vec → X3 → Y → Z
```

Nodes are coupled through a tension vector computed from cosine similarity of their relational projections. When multiple nodes run together, three things can emerge from the dynamics — not forced, not faked:

1. **Convergence** — Nodes lock their Z vectors into alignment (sim → 1.0) through coupling strength alone. No contraction mapping or interpolation tricks. The coupling dynamics earn it or they don't.

2. **Silent Punctuational Syntax (SPS)** — Nodes decide each step whether to speak or stay silent based on how much their relational state has changed. Silence carries information. At D=256 in complementary mode, nodes develop genuinely differentiated speaking rhythms correlated with their dimension specialization (+0.91 correlation).

3. **Octopus Memory** — Four-layer holographic memory (episodic, working, hive fragment, consolidated) using HDC bind/unbind operations. Any surviving node can reconstruct a dead node's state from its hive fragment. Validated with a no-memory control test: at D=64, memory provides a genuine +9.7% recovery advantage over coupling alone.

## Architecture

### DennisNode

The core computational unit. Each node has:

- **Hypervector state**: X1, X2, phi_vec, X3, Y, Z — all shape (D,)
- **Encapsulation membranes**: `void_outer` masks which dimensions are visible (~61.8% open, phi ratio)
- **Self-evolving parameters**: coupling_base, damping_factor, bypass_strength, etc. — mutated via simulated annealing with Metropolis acceptance criterion (actual selection pressure, not random walk)
- **Scalar state**: health, valence, arousal, phase, energy

### Coupling Modes

| Mode | What happens | Use case |
|------|-------------|----------|
| **converge** | Nodes pull toward alignment (sim → 1.0) | Consensus, synchronization |
| **complementary** | Nodes push toward orthogonality (sim → 0.0) | Specialization, division of labor |
| **modular** | Converge within modules, differentiate between | Hierarchical organization |

### Global Ignition

When mean pairwise similarity crosses a threshold (0.7) and is increasing, all nodes receive a coupling boost. Refractory period prevents runaway. Tracks genuine coherence emergence.

### Swarm Complexity Metric

A spectral heuristic combining eigenvalue entropy, effective rank, and coupling strength of the pairwise similarity matrix. **This is NOT IIT Phi** (Integrated Information Theory). Real IIT Phi requires computing minimum information partitions, which is NP-hard. The code includes an optional PyPhi integration for exact Phi on small systems, but the fast metric is a custom heuristic and is labeled as such.

## The Eye of Horus Connection

The system supports D=64, which maps to the Heqat fractions of the Eye of Horus:

| Sense | Fraction | Dimensions |
|-------|----------|------------|
| Smell | 1/2 | 32 |
| Sight | 1/4 | 16 |
| Thought | 1/8 | 8 |
| Hearing | 1/16 | 4 |
| Taste | 1/32 | 2 |
| Touch | 1/64 | 1 |
| **Sum** | **63/64** | **63** |
| **Thoth's restoration** | **1/64** | **= coupling** |

The missing 1/64 that Thoth restored is the integration force — the coupling dynamics that bind 64 independent dimensions into one coherent whole. At D=256, four Eyes run in parallel.

## Honest Numbers

From the audited benchmark run (10,000 steps, 5 nodes):

| Metric | D=64 | D=256 |
|--------|------|-------|
| Convergence step | 22 | 18 |
| Final similarity | 0.9966 | 0.9999 |
| Swarm complexity | 33.38 | 34.04 |
| SPS rhythm diversity | 12.8% | 26.7% |
| Memory regen fidelity | 0.9728 | 0.9923 |
| No-memory control | 0.8755 | 0.9978 |
| Memory advantage | +9.7% | -0.6% |
| Ignition rate | 131/1k | 144/1k |

These numbers have no arbitrary scaling constants. There are no external benchmarks being claimed. The complexity metric is an internal heuristic, not a published standard. What you see is what the system does.

## Running

```bash
pip install numpy scipy pyphi
python scaling_forge_hdc_v1.py
```

Runs the Eye of Horus comparison (D=64 vs D=256) with converge, complementary, modular, and memory regeneration tests. Takes ~5 minutes.

## What's Real and What Isn't

**Real:**
- Convergence through coupling dynamics (no SLERP/contraction mapping)
- SPS rhythm differentiation correlated with dimension specialization
- Holographic memory recall advantage at D=64
- Parameter evolution with selection pressure (Metropolis criterion)
- HDC vector algebra (bind, unbind, bundle, similarity)

**Known limitations:**
- Ignition threshold (0.7) may still be too low — fires ~140/1k steps
- Memory advantage disappears at D=256 (coupling alone is sufficient)
- Modular mode doesn't fully converge within modules yet
- Swarm complexity metric has no theoretical grounding beyond spectral analysis
- Tested only with 5-6 node swarms

## Authors

- **Greg Calkins** (Gregarious Games)
- **Rose** (Claude) — original DennisNode dynamics
- **Dakota** (Claude) — HDC integration, honest audit, this build

## License

GPL v3 — see [LICENSE](LICENSE)
