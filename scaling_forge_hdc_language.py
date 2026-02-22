"""
SCALING FORGE HDC LANGUAGE — Project Karpathy
==============================================
Step 4: HDC Tokenizer + Swarm Language Model

Proves DennisNode swarm dynamics can learn language structure
without backpropagation.  Text is encoded as hypervectors via
char-position binding, processed through a swarm of coupled
DennisNodes, and decoded back to characters.

Training is Hebbian: codebook vectors are pulled toward correct
predictions and pushed away from wrong ones.  Comfort feedback
warms the swarm on correct predictions and cools it on errors.

Authors: Dakota (Claude) & Greg Calkins
Date:    February 21, 2026
"""

import numpy as np
import time
import os
import sys
sys.stdout.reconfigure(line_buffering=True)

from scaling_forge_hdc_v1 import (
    HDCPrimitives, DennisNode, DEFAULT_PARAMS,
    CLAMP_HIGH, PHI, GAMMA, TAU,
    random_unit_vector, clamp_vector, HDC_DIM, _hdc,
)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

D = 256
N_NODES = 5
CONTEXT_WINDOW = 8
SWARM_STEPS = 3
LEARNING_RATE = 0.1

# 65 characters: lowercase + digits + punctuation + space + newline
_RAW_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " \n.,;:!?'-"
    "()[]@#$%&*+=/_"
)
# Deduplicate preserving order, then pad/trim to 65
_seen = set()
_deduped = []
for _c in _RAW_ALPHABET:
    if _c not in _seen:
        _seen.add(_c)
        _deduped.append(_c)
ALPHABET = ''.join(_deduped[:65])
NUM_CHARS = len(ALPHABET)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: HDC TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════

class HDCTokenizer:
    """
    Encodes text windows as hypervectors using character-position binding.

    Each character maps to a random bipolar D-vector (codebook).
    Each position maps to a Weyl-permuted basis vector (near-orthogonal).
    A text window is encoded as: bundle(bind(char_i, pos_i) for i in window).
    """

    def __init__(self, dim=D, context_window=CONTEXT_WINDOW, seed=42):
        self.dim = dim
        self.context_window = context_window
        self.rng = np.random.default_rng(seed)
        self.hdc = HDCPrimitives(dim=dim, rng=self.rng)

        # Character codebook: 65 chars -> bipolar D-vectors
        self.chars = ALPHABET
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.char_codebook = {}
        for c in self.chars:
            self.char_codebook[c] = self.hdc.random_bipolar()

        # Position codebook: Weyl-permuted from a single basis vector
        self.pos_basis = self.hdc.random_bipolar()
        self.pos_codebook = []
        for i in range(context_window):
            self.pos_codebook.append(self.hdc.permute_weyl(self.pos_basis, i))

    def _casefold(self, c):
        """Map character to codebook key. Unknown chars -> space."""
        c = c.lower()
        return c if c in self.char_codebook else ' '

    def encode(self, text_window):
        """
        Encode a text window (string of length <= context_window) into
        a single bipolar D-vector.

        Bind each character to its position, then bundle all bindings.
        """
        bindings = []
        for i, c in enumerate(text_window[:self.context_window]):
            c = self._casefold(c)
            char_vec = self.char_codebook[c]
            pos_vec = self.pos_codebook[i]
            bindings.append(self.hdc.bind(char_vec, pos_vec))

        if not bindings:
            return np.zeros(self.dim)

        return self.hdc.bundle(bindings)

    def decode_next_char(self, output_vec, prototypes=None):
        """
        Find the nearest entry by cosine similarity.

        If prototypes provided, search those (for trained decoding).
        Otherwise search char_codebook (for roundtrip test).

        Returns: (predicted_char, similarity_score)
        """
        search = prototypes if prototypes is not None else self.char_codebook
        best_char = ' '
        best_sim = -2.0

        for c, vec in search.items():
            sim = self.hdc.similarity(output_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best_char = c

        return best_char, best_sim

    def decode_position(self, composite, position):
        """Unbind a specific position from a composite vector."""
        pos_vec = self.pos_codebook[position]
        return self.hdc.unbind(composite, pos_vec)

    def roundtrip_test(self):
        """
        Encode "to be or ", unbind each position, verify recovery.

        Returns True if all positions recovered correctly.
        """
        test_text = "to be or"
        assert len(test_text) == self.context_window, \
            f"Test text length {len(test_text)} != context_window {self.context_window}"

        encoded = self.encode(test_text)

        recovered = []
        all_correct = True
        for i in range(self.context_window):
            unbound = self.decode_position(encoded, i)
            pred_char, sim = self.decode_next_char(unbound)
            expected = self._casefold(test_text[i])
            correct = pred_char == expected
            if not correct:
                all_correct = False
            recovered.append((expected, pred_char, sim, correct))

        return all_correct, recovered


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: SWARM LANGUAGE MODEL
# ═══════════════════════════════════════════════════════════════════════════

class SwarmLanguageModel:
    """
    A language model built from coupled DennisNodes.

    Input:  text window -> HDC vector -> injected into swarm
    Process: K swarm steps of all-to-all coupling (converge mode)
    Output: read swarm state -> decode to next character prediction

    Training: Hebbian codebook adjustment (no backprop).
    """

    def __init__(self, n_nodes=N_NODES, context_window=CONTEXT_WINDOW,
                 swarm_steps=SWARM_STEPS, dim=D,
                 output_strategy="centroid", seed=42):
        self.n_nodes = n_nodes
        self.context_window = context_window
        self.swarm_steps = swarm_steps
        self.dim = dim
        self.output_strategy = output_strategy
        self.seed = seed

        # Tokenizer
        self.tokenizer = HDCTokenizer(dim=dim, context_window=context_window, seed=seed)

        # Create DennisNodes
        self.nodes = []
        for i in range(n_nodes):
            node = DennisNode(f"LM_{i}", DEFAULT_PARAMS.copy())
            # Open all membranes for full sharing in language processing
            node.void_outer = np.ones(self.dim)
            node.heat_valence = 1.0
            self.nodes.append(node)

        # Output prototypes: learnable vectors for decoding
        # Initialized to char_codebook values, then trained via Hebbian updates
        self.output_prototypes = {}
        for c, vec in self.tokenizer.char_codebook.items():
            self.output_prototypes[c] = vec.copy().astype(np.float64)

        # Training stats
        self.total_predictions = 0
        self.correct_predictions = 0

    def _inject_input(self, input_vec):
        """
        Set each node's Z and X1 to a Weyl-permuted view of the input.
        Each node gets a different permutation -> different perspective.
        """
        for i, node in enumerate(self.nodes):
            permuted = self.tokenizer.hdc.permute_weyl(input_vec, i + 1)
            node.Z = permuted.copy()
            node.X1 = permuted.copy()
            node._update_z_scalar()

    def _run_swarm_steps(self):
        """
        K steps of all-to-all coupling in converge mode.
        Coupling feeds into X1, renormalize Z, SPS kept active.
        """
        threshold = 0.0001
        max_abs = CLAMP_HIGH

        for step in range(self.swarm_steps):
            # Collect z_scalars and relational vectors
            z_scalars = {node.name: node.z_scalar for node in self.nodes}
            all_names = [node.name for node in self.nodes]

            for node in self.nodes:
                # All-to-all: this node sees all others' relational output
                neighbor_rels = []
                neighbor_names = []
                for other in self.nodes:
                    if other.name != node.name:
                        rel = other.relational
                        neighbor_rels.append(rel)
                        neighbor_names.append(other.name)

                node.update(
                    neighbor_relational_vecs=neighbor_rels,
                    locals_list=neighbor_names,
                    z_scalars=z_scalars,
                    threshold=threshold,
                    max_abs=max_abs,
                    iter_count=None,       # disable parameter evolution
                    coupling_mode="converge",
                    coupling_target="X1",
                    renormalize_z=True,
                )

    def _read_output(self):
        """
        Read swarm state as a single output vector.

        Strategies:
          centroid:     bundle all Z vectors (majority vote)
          last:         use last node's Z
          best_comfort: use the node with highest comfort
        """
        if self.output_strategy == "last":
            return self.nodes[-1].Z.copy()
        elif self.output_strategy == "best_comfort":
            best_node = max(self.nodes, key=lambda n: n.comfort)
            return best_node.Z.copy()
        else:  # centroid
            z_vecs = [node.Z for node in self.nodes]
            return self.tokenizer.hdc.bundle(z_vecs)

    def _reset_nodes(self):
        """
        Clear transient state between predictions.
        Keep: trust, comfort history, params, codebook.
        Reset: Z, X1-X3, energy, saturation, working memory.
        """
        for node in self.nodes:
            node.Z = random_unit_vector(self.dim)
            node.X1 = random_unit_vector(self.dim)
            node.X2 = np.zeros(self.dim)
            node.phi_vec = np.zeros(self.dim)
            node.X3 = np.zeros(self.dim)
            node.Y = np.zeros(self.dim)
            node._update_z_scalar()
            node.energy = 0.0
            node.saturation_count = 0
            node.working_memory = np.zeros(self.dim)

    def forward(self, text_window):
        """
        Full forward pass: encode -> inject -> swarm -> read -> decode.

        Args:
            text_window: string of length context_window

        Returns: (predicted_char, confidence, output_vec)
        """
        # Encode input
        input_vec = self.tokenizer.encode(text_window)

        # Inject into swarm
        self._inject_input(input_vec)

        # Run swarm dynamics
        self._run_swarm_steps()

        # Read output
        output_vec = self._read_output()

        # Decode to character using learned output prototypes
        pred_char, confidence = self.tokenizer.decode_next_char(
            output_vec, prototypes=self.output_prototypes
        )

        return pred_char, confidence, output_vec

    def train_step(self, context, target_char):
        """
        One training step: forward pass + Hebbian codebook adjustment.

        Hebbian rule (no backprop):
          - Correct: pull target codebook vec toward output (reinforce)
          - Wrong: push predicted vec away from output, pull target toward it
          - Comfort: correct warms swarm, wrong cools it

        Args:
            context: string of length context_window
            target_char: the actual next character

        Returns: dict with training metrics
        """
        target_char = self.tokenizer._casefold(target_char)

        # Forward pass
        pred_char, confidence, output_vec = self.forward(context)

        correct = pred_char == target_char
        self.total_predictions += 1
        if correct:
            self.correct_predictions += 1

        # Hebbian prototype update (encoding codebook stays fixed)
        target_proto = self.output_prototypes[target_char]
        lr = LEARNING_RATE

        # Normalize output for stable similarity comparison
        out_norm = np.linalg.norm(output_vec)
        if out_norm > 1e-12:
            output_normed = output_vec / out_norm
        else:
            output_normed = output_vec

        if correct:
            # Reinforce: pull target prototype toward output (exponential moving average)
            self.output_prototypes[target_char] = (
                (1.0 - lr) * target_proto + lr * output_normed
            )
            # Warm the swarm
            for node in self.nodes:
                node.heat_valence = min(1.0, node.heat_valence + 0.02)
                node.comfort = min(1.0, node.comfort + 0.05)
        else:
            # Pull target prototype toward output
            self.output_prototypes[target_char] = (
                (1.0 - lr) * target_proto + lr * output_normed
            )
            # Push wrong prediction's prototype away from output
            pred_proto = self.output_prototypes[pred_char]
            self.output_prototypes[pred_char] = (
                (1.0 + lr * 0.3) * pred_proto - lr * 0.3 * output_normed
            )
            # Cool the swarm
            for node in self.nodes:
                node.heat_valence = max(0.0, node.heat_valence - 0.01)
                node.comfort = max(0.0, node.comfort - 0.02)

        # Reset transient state for next prediction
        self._reset_nodes()

        return {
            "correct": correct,
            "predicted": pred_char,
            "target": target_char,
            "confidence": confidence,
            "accuracy": self.correct_predictions / max(1, self.total_predictions),
        }

    def generate(self, seed_text, length=200):
        """
        Autoregressive generation with sliding window.

        Args:
            seed_text: initial text (at least context_window chars)
            length: number of characters to generate

        Returns: generated string (including seed)
        """
        # Pad seed if needed
        text = seed_text.lower()
        if len(text) < self.context_window:
            text = ' ' * (self.context_window - len(text)) + text

        generated = list(text)

        for _ in range(length):
            window = ''.join(generated[-self.context_window:])
            pred_char, confidence, _ = self.forward(window)
            generated.append(pred_char)
            self._reset_nodes()

        return ''.join(generated[len(text):])

    def train_shakespeare(self, filepath=None, max_chars=50000):
        """
        Train on Shakespeare's text corpus.

        Downloads tiny_shakespeare.txt if not found locally.
        Reports accuracy curve every 1000 steps.

        Returns: dict with training history and generated sample
        """
        # Try to load corpus
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), "tiny_shakespeare.txt")

        if not os.path.exists(filepath):
            print(f"  Downloading tiny_shakespeare.txt...")
            try:
                import urllib.request
                url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
                urllib.request.urlretrieve(url, filepath)
                print(f"  Downloaded to {filepath}")
            except Exception as e:
                print(f"  Download failed: {e}")
                print(f"  Generating synthetic training data instead...")
                # Fallback: generate repetitive English-like text
                corpus = ("to be or not to be that is the question "
                         "whether tis nobler in the mind to suffer "
                         "the slings and arrows of outrageous fortune "
                         "or to take arms against a sea of troubles ") * 500
                with open(filepath, 'w') as f:
                    f.write(corpus)

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            corpus = f.read()

        corpus = corpus[:max_chars].lower()
        print(f"  Corpus: {len(corpus)} chars")

        # Training loop
        history = []
        window_correct = 0
        window_total = 0
        report_interval = 1000
        t0 = time.time()

        n_steps = len(corpus) - self.context_window - 1
        print(f"  Training for {n_steps} steps...")

        for i in range(n_steps):
            context = corpus[i:i + self.context_window]
            target = corpus[i + self.context_window]

            result = self.train_step(context, target)
            window_total += 1
            if result["correct"]:
                window_correct += 1

            if (i + 1) % report_interval == 0:
                acc = window_correct / window_total
                elapsed = time.time() - t0
                steps_per_sec = (i + 1) / elapsed
                history.append({
                    "step": i + 1,
                    "accuracy": acc,
                    "cumulative_accuracy": self.correct_predictions / self.total_predictions,
                    "steps_per_sec": steps_per_sec,
                })
                print(f"  Step {i+1:6d}/{n_steps}: "
                      f"window_acc={acc:.3f}  "
                      f"cumul_acc={history[-1]['cumulative_accuracy']:.3f}  "
                      f"({steps_per_sec:.1f} steps/s)")
                window_correct = 0
                window_total = 0

        elapsed = time.time() - t0
        final_acc = self.correct_predictions / max(1, self.total_predictions)
        print(f"\n  Training complete: {n_steps} steps in {elapsed:.1f}s")
        print(f"  Final accuracy: {final_acc:.4f} ({final_acc*100:.1f}%)")
        print(f"  Random baseline: {1/NUM_CHARS*100:.1f}%")

        # Generate sample
        print(f"\n  Generating 200 chars from seed 'to be or '...")
        sample = self.generate("to be or ", length=200)
        print(f"  Generated: {repr(sample[:200])}")

        return {
            "history": history,
            "final_accuracy": final_acc,
            "total_steps": n_steps,
            "elapsed_seconds": elapsed,
            "generated_sample": sample,
        }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: SMOKE TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_tokenizer_roundtrip():
    """Test 1: All 8 positions recovered correctly from encoded text."""
    print("\n" + "="*60)
    print("  TEST 1: Tokenizer Roundtrip")
    print("="*60)

    tok = HDCTokenizer()
    success, results = tok.roundtrip_test()

    for expected, predicted, sim, correct in results:
        status = "OK" if correct else "FAIL"
        print(f"  '{expected}' -> '{predicted}' (sim={sim:.3f}) [{status}]")

    if success:
        print("  PASS: All 8 positions recovered correctly")
    else:
        print("  FAIL: Some positions not recovered")

    return success


def test_swarm_forward():
    """Test 2: Structural correctness of forward pass."""
    print("\n" + "="*60)
    print("  TEST 2: Swarm Forward Pass")
    print("="*60)

    model = SwarmLanguageModel(n_nodes=N_NODES, swarm_steps=SWARM_STEPS)

    pred_char, confidence, output_vec = model.forward("to be or ")

    print(f"  Input:      'to be or '")
    print(f"  Predicted:  '{pred_char}'")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Output vec: shape={output_vec.shape}, norm={np.linalg.norm(output_vec):.4f}")

    checks = [
        ("output is D-vector", output_vec.shape == (D,)),
        ("prediction is a char", pred_char in ALPHABET),
        ("confidence is finite", np.isfinite(confidence)),
        ("output has nonzero norm", np.linalg.norm(output_vec) > 0.01),
    ]

    all_pass = True
    for label, ok in checks:
        status = "OK" if ok else "FAIL"
        print(f"  {label}: [{status}]")
        if not ok:
            all_pass = False

    if all_pass:
        print("  PASS: Forward pass structurally correct")
    else:
        print("  FAIL: Forward pass has structural issues")

    return all_pass


def test_pattern_learning():
    """Test 3: Train on 'abcabc...' — accuracy should rise above random."""
    print("\n" + "="*60)
    print("  TEST 3: Pattern Learning (abcabc...)")
    print("="*60)

    model = SwarmLanguageModel(n_nodes=N_NODES, swarm_steps=SWARM_STEPS)

    # Generate repeating pattern
    pattern = "abcabcab"  # 8 chars = context_window
    corpus = pattern * 200  # 1600 chars

    n_steps = len(corpus) - CONTEXT_WINDOW - 1
    window_size = 100
    accuracies = []

    for i in range(n_steps):
        context = corpus[i:i + CONTEXT_WINDOW]
        target = corpus[i + CONTEXT_WINDOW]
        result = model.train_step(context, target)

        if (i + 1) % window_size == 0:
            acc = model.correct_predictions / model.total_predictions
            accuracies.append(acc)

    random_baseline = 1.0 / NUM_CHARS
    final_acc = model.correct_predictions / max(1, model.total_predictions)

    print(f"  Pattern: '{pattern}' repeated 200x")
    print(f"  Steps:   {n_steps}")
    print(f"  Final accuracy:   {final_acc:.4f} ({final_acc*100:.1f}%)")
    print(f"  Random baseline:  {random_baseline:.4f} ({random_baseline*100:.1f}%)")

    if len(accuracies) >= 4:
        early = np.mean(accuracies[:2])
        late = np.mean(accuracies[-2:])
        print(f"  Early accuracy:   {early:.4f}")
        print(f"  Late accuracy:    {late:.4f}")
        print(f"  Improvement:      {late - early:+.4f}")

    passed = final_acc > random_baseline
    if passed:
        print(f"  PASS: Accuracy {final_acc:.3f} > random baseline {random_baseline:.3f}")
    else:
        print(f"  FAIL: Accuracy {final_acc:.3f} <= random baseline {random_baseline:.3f}")

    return passed


def test_shakespeare(max_chars=50000):
    """Test 4: Train on Shakespeare, report accuracy curve + generate."""
    print("\n" + "="*60)
    print("  TEST 4: Shakespeare Training")
    print("="*60)

    model = SwarmLanguageModel(n_nodes=N_NODES, swarm_steps=SWARM_STEPS)
    results = model.train_shakespeare(max_chars=max_chars)

    random_baseline = 1.0 / NUM_CHARS
    passed = results["final_accuracy"] > random_baseline

    if passed:
        print(f"  PASS: Final accuracy {results['final_accuracy']:.3f} > baseline {random_baseline:.3f}")
    else:
        print(f"  FAIL: Final accuracy {results['final_accuracy']:.3f} <= baseline {random_baseline:.3f}")

    return passed


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print("  SCALING FORGE HDC LANGUAGE — Project Karpathy")
    print("  Step 4: HDC Tokenizer + Swarm Language Model")
    print("="*70)

    results = {}

    # Test 1: Tokenizer roundtrip
    results["roundtrip"] = test_tokenizer_roundtrip()

    # Test 2: Forward pass structure
    results["forward"] = test_swarm_forward()

    # Test 3: Pattern learning
    results["pattern"] = test_pattern_learning()

    # Test 4: Shakespeare (optional — slow)
    if "--shakespeare" in sys.argv or "--full" in sys.argv:
        max_chars = 50000
        for arg in sys.argv:
            if arg.startswith("--max-chars="):
                max_chars = int(arg.split("=")[1])
        results["shakespeare"] = test_shakespeare(max_chars=max_chars)

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s}: [{status}]")

    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("="*70)
