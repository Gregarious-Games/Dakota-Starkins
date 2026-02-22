"""
QAMS Phonetic Codebook for HDC Language Model
==============================================
Drop-in replacement for random codebook in scaling_forge_hdc_language.py

Every symbol is a motion primitive. Encode the motion, not an arbitrary label.

6 phonetic parameters per character:
  [aperture, duration, voicing, place, manner, frequency]

Characters with similar phonetics get similar hypervectors.
Characters with different phonetics get dissimilar hypervectors.

Usage:
    from qams_codebook import generate_qams_codebook, PHONETIC_SIGNATURES
    codebook = generate_qams_codebook(D=256)

Authors: Rose (Claude) & Greg Calkins
Date:    February 21, 2026
"""

import numpy as np

# ============================================================
# PHONETIC MOTION SIGNATURES — TERNARY ENCODING
# 6 parameters per character:
#   [aperture, duration, voicing, place, manner, frequency]
#
# TERNARY: each axis is [-1, 0, +1]
#   Zero is NOT absence — zero is the SPS silence channel.
#   The dimensions where a character has parameter=0 are the
#   dimensions where that character is SILENT on that phonetic axis.
#
# aperture:  -1=closed (active blockage), 0=neutral, +1=open
# duration:  -1=impulse/burst, 0=medium, +1=sustained
# voicing:   -1=unvoiced (cords held open), 0=neutral, +1=voiced
# place:     -1=lips (bilabial), 0=palate (alveolar), +1=throat (velar/glottal)
# manner:    -1=stop (complete blockage), 0=fricative, +1=approximant/open
# frequency: -1=low, 0=mid, +1=high
# ============================================================

PHONETIC_SIGNATURES = {
    # Vowels — open, sustained, voiced, approximant
    # Place axis encodes vowel harmony: back(+) / neutral(0) / front(-)
    #          [aper, dur,  voic, place, mann, freq]
    'a': [+1.0, +1.0, +1.0,  0.0, +1.0, -0.4],  # open, central, low-mid
    'e': [+0.5, +0.8, +1.0, -0.2, +1.0,  0.0],  # mid-front (neutral in Finnish)
    'i': [+0.0, +0.7, +1.0, -0.4, +1.0, +0.6],  # close-front, high (neutral in Finnish)
    'o': [+0.6, +0.8, +1.0, +0.2, +1.0, -0.5],  # mid-back, low
    'u': [+0.0, +0.7, +1.0, +0.4, +1.0, -0.6],  # close-back, lowest

    # Finnish front vowels — same mouth shape as back counterpart, front place
    '\u00e4': [+1.0, +1.0, +1.0, -0.6, +1.0, +0.1],  # ä: open front (counterpart of a)
    '\u00f6': [+0.6, +0.8, +1.0, -0.4, +1.0, -0.1],  # ö: mid front rounded (counterpart of o)
    '\u00e5': [+0.9, +0.9, +1.0, +0.3, +1.0, -0.5],  # å: back open rounded (Swedish loans)

    # Plosives — closed, impulse, stop
    #          [aper, dur,  voic, place, mann, freq]
    'b': [-1.0, -1.0, +1.0, -1.0, -1.0, -0.7],  # voiced bilabial stop
    'p': [-1.0, -1.0, -1.0, -1.0, -1.0, -0.6],  # UNvoiced bilabial stop
    'd': [-1.0, -1.0, +1.0,  0.0, -1.0, -0.4],  # voiced alveolar stop
    't': [-1.0, -1.0, -1.0,  0.0, -1.0, +0.2],  # UNvoiced alveolar stop
    'g': [-1.0, -1.0, +1.0, +0.8, -1.0, -0.6],  # voiced velar stop
    'k': [-1.0, -1.0, -1.0, +0.8, -1.0, -0.5],  # UNvoiced velar stop

    # Fricatives — narrow aperture, sustained, fricative manner
    #          [aper, dur,  voic, place, mann, freq]
    'f': [-0.7, +0.5, -1.0, -0.8, 0.0, +0.3],   # unvoiced labiodental
    'v': [-0.7, +0.5, +1.0, -0.8, 0.0, +0.1],   # voiced labiodental
    's': [-0.8, +0.6, -1.0,  0.0, 0.0, +0.7],   # unvoiced alveolar (high freq)
    'z': [-0.8, +0.6, +1.0,  0.0, 0.0, +0.5],   # voiced alveolar
    'h': [ 0.0, +0.3, -1.0, +1.0, 0.0, -0.4],   # unvoiced glottal

    # Nasals — closed mouth, voiced, nasal manner (~+0.5 between fric and approx)
    #          [aper, dur,  voic, place, mann, freq]
    'm': [-1.0, +0.6, +1.0, -1.0, +0.5, -0.7],  # bilabial nasal
    'n': [-1.0, +0.5, +1.0,  0.0, +0.5, -0.5],  # alveolar nasal

    # Approximants / Liquids — partially open, voiced, approximant
    #          [aper, dur,  voic, place, mann, freq]
    'l': [ 0.0, +0.5, +1.0,  0.0, +1.0, -0.3],  # lateral approximant
    'r': [-0.3, +0.4, +1.0, +0.1, +1.0, -0.4],  # alveolar approximant
    'w': [-0.4, +0.2, +1.0, -1.0, +1.0, -0.7],  # labial-velar approximant
    'y': [-0.3, +0.2, +1.0, -0.4, +1.0, +0.4],  # palatal approximant

    # Affricates / Special consonants
    #          [aper, dur,  voic, place, mann, freq]
    'c': [-0.9, -0.5, -1.0,  0.0, -0.5, +0.1],  # k/s blend
    'j': [-0.8, -0.4, +1.0, -0.1, -0.5,  0.0],  # voiced affricate
    'q': [-1.0, -1.0, -1.0, +0.8, -1.0, -0.5],  # like k
    'x': [-0.9, -0.3, -1.0,  0.0, -0.5, +0.4],  # k+s blend

    # Punctuation — SILENCE PRIMITIVES (SPS for text)
    # All axes at 0 = pure silence. Frequency differentiates energy level.
    #          [aper, dur,  voic, place, mann, freq]
    ' ':  [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # space = pure silence
    '.':  [ 0.0,  0.0,  0.0,  0.0,  0.0, -0.1],  # period = terminal silence
    ',':  [ 0.0,  0.0,  0.0,  0.0,  0.0, +0.1],  # comma = brief pause
    '!':  [ 0.0,  0.0,  0.0,  0.0,  0.0, +0.9],  # exclamation = energy burst
    '?':  [ 0.0,  0.0,  0.0,  0.0,  0.0, +0.6],  # question = rising energy
    ':':  [ 0.0,  0.0,  0.0,  0.0,  0.0, +0.2],  # colon = sustained pause
    ';':  [ 0.0,  0.0,  0.0,  0.0,  0.0, +0.15], # semicolon = medium pause
    '-':  [ 0.0,  0.0,  0.0,  0.0,  0.0, -0.2],  # dash = connecting silence
    "'":  [-0.5, -0.8, -1.0, +1.0, -0.5, +0.0],  # apostrophe = glottal flick
    '\n': [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # newline = paragraph silence

    # Digits — phonetic centroids of Finnish spoken forms
    # Each digit is the MEAN of its constituent phonemes' signatures.
    # "yksi" and "kolme" are completely different mouth motions,
    # so digits get genuinely different signatures.
    #          [aper,  dur,   voic,  place, mann,  freq ]
    '0': [+0.12, +0.66, +1.00, +0.04, +0.90, -0.40],  # nolla
    '1': [-0.53, +0.12, +0.00, +0.00, +0.25, +0.30],  # yksi
    '2': [-0.36, +0.06, -0.20, +0.24, +0.00, -0.02],  # kaksi
    '3': [-0.18, +0.34, +0.60, -0.04, +0.50, -0.40],  # kolme
    '4': [-0.06, +0.48, +1.00, -0.06, +0.60, -0.24],  # neljä
    '5': [-0.30, +0.64, +0.60, -0.40, +0.60, +0.52],  # viisi
    '6': [-0.36, +0.34, +0.20, +0.24, +0.40, -0.08],  # kuusi
    '7': [-0.29, +0.51, +0.33, -0.20, +0.44, +0.07],  # seitsemän
    '8': [-0.26, +0.13, +0.11, +0.27, +0.06, -0.27],  # kahdeksan
    '9': [-0.33, +0.17, +0.25, +0.15, +0.19, -0.14],  # yhdeksän
}


def phonetic_similarity(char_a, char_b):
    """Cosine similarity between two characters' phonetic signatures."""
    if char_a not in PHONETIC_SIGNATURES or char_b not in PHONETIC_SIGNATURES:
        return 0.0
    a = np.array(PHONETIC_SIGNATURES[char_a])
    b = np.array(PHONETIC_SIGNATURES[char_b])
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0 if norm_a < 1e-12 and norm_b < 1e-12 else 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def transition_probability(char_a, char_b):
    """How physically easy is it to go from sound A to sound B?
    Smooth transitions = high probability, awkward articulations = low."""
    if char_a not in PHONETIC_SIGNATURES or char_b not in PHONETIC_SIGNATURES:
        return 0.5  # neutral for unknown chars
    params_a = np.array(PHONETIC_SIGNATURES[char_a])
    params_b = np.array(PHONETIC_SIGNATURES[char_b])
    distance = np.sum((params_a - params_b) ** 2)
    GAMMA_SCALE = 1.0 / (6 * 1.618033988749895)  # Γ = 1/(6φ)
    return float(np.exp(-distance * GAMMA_SCALE))


def generate_qams_codebook(D=256, seed=42):
    """
    Generate character codebook from phonetic motion signatures.

    Characters with similar phonetics get similar hypervectors.
    Characters with different phonetics get dissimilar hypervectors.

    Method: BLOCK-DIAGONAL basis — each of the 6 phonetic parameters
    owns its own D/6 dimensions. This guarantees that flipping one
    axis (e.g., voicing) only affects ~D/6 dimensions, leaving the
    other 5D/6 identical.  A single-axis flip yields cosine ~0.67.

    Cross-terms use a shared interaction block (remaining dims after
    the 6 blocks) for parameter co-occurrence encoding.

    Returns: dict mapping char -> bipolar np.ndarray of shape (D,)
    """
    rng = np.random.RandomState(seed)

    n_params = 6
    block_size = D // n_params       # 42 for D=256
    remainder = D - block_size * n_params  # 4 extra dims

    # Block-diagonal basis: each parameter gets its own random bipolar block
    # basis[i] is nonzero only in dims [i*block_size : (i+1)*block_size]
    basis_blocks = []
    for i in range(n_params):
        block = rng.choice([-1, 1], size=block_size).astype(np.float64)
        basis_blocks.append(block)

    # Cross-term interaction vectors: use the remainder dims + overlay
    # These are full-D random vectors but weighted very lightly
    cross_basis = []
    for i in range(n_params - 1):
        cv = rng.choice([-1, 1], size=D).astype(np.float64)
        cross_basis.append(cv)

    codebook = {}
    for char, params in PHONETIC_SIGNATURES.items():
        vec = np.zeros(D, dtype=np.float64)

        # Block-diagonal linear terms: each param writes only to its block
        for i in range(n_params):
            start = i * block_size
            end = start + block_size
            vec[start:end] += params[i] * basis_blocks[i]

        # Cross-terms: weak full-D overlay for parameter co-occurrence
        for i in range(n_params - 1):
            cross_weight = abs(params[i] * params[i + 1])
            vec += cross_weight * cross_basis[i] * 0.1

        # Character-specific noise for disambiguation
        char_seed = abs(hash(char)) % (2**31)
        char_rng = np.random.RandomState(char_seed)
        noise = char_rng.choice([-1, 1], size=D).astype(np.float64)
        vec += noise * 0.2

        # Bipolarize
        result = np.sign(vec)
        result[result == 0] = 1.0
        codebook[char] = result

    return codebook


def verify_phonetic_structure(codebook):
    """
    Print similarity matrix for key phonetic groups.
    Verify that phonetically similar characters have similar vectors.
    """
    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("=== QAMS PHONETIC STRUCTURE VERIFICATION ===\n")

    # Vowels should cluster
    vowels = 'aeiou'
    print("Vowel similarities (should be >0.3):")
    for i in range(len(vowels)):
        for j in range(i + 1, len(vowels)):
            a, b = vowels[i], vowels[j]
            sim = cos_sim(codebook[a], codebook[b])
            print(f"  {a}-{b}: {sim:.3f}")

    # Voiced/unvoiced pairs should be related
    print("\nVoiced/unvoiced pairs (should be >0.4):")
    pairs = [('b', 'p'), ('d', 't'), ('g', 'k'), ('v', 'f'), ('z', 's')]
    for v, u in pairs:
        sim = cos_sim(codebook[v], codebook[u])
        print(f"  {v}-{u}: {sim:.3f}")

    # Vowel vs consonant should be distant
    print("\nVowel vs consonant (should be <0.2):")
    cross = [('a', 't'), ('e', 'k'), ('i', 's'), ('o', 'p')]
    for v, c in cross:
        sim = cos_sim(codebook[v], codebook[c])
        print(f"  {v}-{c}: {sim:.3f}")

    # Silence group
    print("\nSilence/punctuation (should be >0.5):")
    silence = [' ', '.', ',', '\n']
    for i in range(len(silence)):
        for j in range(i + 1, len(silence)):
            a, b = silence[i], silence[j]
            sim = cos_sim(codebook[a], codebook[b])
            label_a = repr(a)
            label_b = repr(b)
            print(f"  {label_a}-{label_b}: {sim:.3f}")

    # Nasals should cluster
    print("\nNasals m-n (should be >0.4):")
    sim = cos_sim(codebook['m'], codebook['n'])
    print(f"  m-n: {sim:.3f}")

    # Finnish vowel harmony pairs (if present)
    if '\u00e4' in codebook:
        print("\nFinnish vowel harmony (back/front pairs, should be >0.3):")
        fi_pairs = [('a', '\u00e4'), ('o', '\u00f6')]
        for back, front in fi_pairs:
            if front in codebook:
                sim = cos_sim(codebook[back], codebook[front])
                print(f"  {back}-{front}: {sim:.3f} (same shape, different place)")

        print("\nFinnish harmony groups (within-group, should be >0.2):")
        back_vowels = ['a', 'o', 'u']
        front_vowels = ['\u00e4', '\u00f6']
        for i in range(len(back_vowels)):
            for j in range(i + 1, len(back_vowels)):
                a, b = back_vowels[i], back_vowels[j]
                sim = cos_sim(codebook[a], codebook[b])
                print(f"  {a}-{b}: {sim:.3f} (back group)")
        for i in range(len(front_vowels)):
            for j in range(i + 1, len(front_vowels)):
                a, b = front_vowels[i], front_vowels[j]
                sim = cos_sim(codebook[a], codebook[b])
                print(f"  {a}-{b}: {sim:.3f} (front group)")

        print("\nFinnish cross-harmony (back vs front, should be LOW):")
        for bv in ['a', 'o']:
            for fv in ['\u00e4', '\u00f6']:
                if bv != fv and fv in codebook:
                    sim = cos_sim(codebook[bv], codebook[fv])
                    print(f"  {bv}-{fv}: {sim:.3f}")

    print()


if __name__ == "__main__":
    codebook = generate_qams_codebook(D=256)
    verify_phonetic_structure(codebook)

    # Count unique vectors
    print(f"Codebook size: {len(codebook)} characters")
    print(f"Vector dimension: {len(list(codebook.values())[0])}")

    # Show that ALL vectors are bipolar
    all_bipolar = all(
        set(np.unique(v)).issubset({-1.0, 1.0})
        for v in codebook.values()
    )
    print(f"All vectors bipolar: {all_bipolar}")
