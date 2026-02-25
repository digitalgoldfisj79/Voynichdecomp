# S2: Information Budget Derivation

> **Source code:** `src/p70c_connect.py` (information budget derivation)  
> **Data:** `data/transition_lookup.json` (8×8 matrix), `data/enriched_records.pkl` (ground truth)


## S2.1 Entropy Framework

The information budget decomposes word-selection entropy H(quad) into a grammatical component (explained by external conditioning axes) and a lexical content component (unexplained).

**Notation.** Let quad = (prefix, gallows, m_core_class, sfx_fam). H(quad) denotes the empirical entropy of the quad distribution over the 37,465-token corpus:

    H(quad) = 9.124 bits

This is lower than the sum of marginal quad-slot entropies (10.306 bits, see table below) by 1.182 bits (11.5% redundancy), confirming that PGCS quad-level slots are approximately 89% independent. (At full character-level resolution, the paper §3.1 reports a higher redundancy of 21.7% because full cores and full suffixes carry more mutual information than the classified abstractions used here.)

## S2.2 Slot-Level Decomposition

By the chain rule of entropy:

    H(word) = H(P) + H(G|P) + H(C|P,G) + H(S|P,G,C)

Empirical values:

| Quantity | Bits | % of H(word) |
|----------|------|-------------|
| H(prefix) = H(P) | 2.628 | 25.5% |
| H(gallows) = H(G) | 1.660 | 16.1% |
| H(m_core) = H(C) | 3.855 | 37.4% |
| H(sfx_fam) = H(SF) | 2.163 | 21.0% |
| Sum of marginals | 10.306 | — |
| H(quad) observed | 9.124 | — |
| Redundancy | 1.182 | 11.5% |

### Pairwise Couplings (MI)

| Pair | MI (bits) | Notes |
|------|-----------|-------|
| Core–Suffix | 0.976 | Strongest within-word coupling |
| Prefix–Core | 0.428 | |
| Prefix–Gallows | 0.393 | Strongest pairwise (Cramér's V = 0.266) |
| Gallows–Suffix | 0.187 | |
| Gallows–Core | 0.156 | |
| Prefix–Suffix | 0.094 | Weakest |

## S2.3 Five-Axis Conditioning

Axes are added in decreasing order of marginal contribution. At each step, we compute MI(quad; axis₁, ..., axisₖ) using the joint distribution of quad with all conditioning axes up to k.

### Axis 1: Section (9 categories)

VMS sections: Herbal-A, Herbal-B, Pharmaceutical, Balneological, Astronomical, Zodiac, Cosmological, Stars, Rosettes.

    MI(quad; section) = 0.810 bits (8.88% of H(quad))

This confirms that section identity predicts word choice, consistent with section classification at 76–81% accuracy using token frequencies.

### Axis 2: Line Position (3 categories)

FIRST (word 1 of line), MID (words 2 through n−1), LAST (final word of line).

    MI(quad; section, position) = 1.329 bits (14.56%)
    Δ position beyond section = 0.518 bits (5.68%)

Position adds 63.9% additional information beyond section alone. The three-zone model identifies:

- **Opener zone** (FIRST): Enriched for d-prefix (15.3% vs 10.1%, Z=12.5), depleted for empty cores (39% vs 57%), elevated y-prefix (13.4%).
- **Closer zone** (LAST): M-suffix line-ending marker at 14.9% vs 1.8% penultimate (Z=57.8), the strongest positional signal in the manuscript.
- **Middle zone** (MID): Default register.

### Axis 3: Previous Suffix Family (7 categories + LINE_START)

    MI(quad; section, position, prev_sfx) = 2.086 bits (22.86%)
    Δ prev_sfx beyond section+position = 0.757 bits (8.30%)

This is the single largest marginal increment. Key transitions:

| Previous suffix | → Next prefix | Probability | Enrichment |
|----------------|---------------|-------------|------------|
| Y | → qo | 26.0% | 1.9× |
| BARE | → ∅ | 39.5% | 1.8× |
| N | → o | 29.7% | 1.8× |
| R | → o | 26.1% | 1.6× |
| LINE_START | → y | 13.4% | 2.1× |
| LINE_START | → d | 14.3% | 1.2× |

The transition grammar also explains the length autocorrelation (AC = +0.160): suffix-family constrains following prefix, and prefix constrains word length, propagating the coupling through PGCS slots. The effect resets at line boundaries (within-line AC = 0.151; cross-line AC = 0.062).

### Axis 4: Paragraph Flag (binary)

    MI(quad; section, position, prev_sfx, para_flag) = 2.116 bits (23.19%)
    Δ para_flag = 0.030 bits (0.33%)

Negligible marginal contribution despite a striking distributional signature: 71.7% of paragraph-initial tokens carry ∅-prefix and 84.5% bear gallows (rising to 79.6% and 88.7% in the five main text sections). The low MI reflects the small sample (226 tokens, 0.6% of corpus).

### Axis 5: Quire / Production Unit (16 categories)

    MI(quad; all five axes) = 2.634 bits (28.87%)
    Δ quire beyond all prior = 0.518 bits (5.68%)

Quire identity captures within-section vocabulary variation across production units. The f42–f49–f56 cross-quire cluster (Jaccard = 0.108 with other Herbal folios) has 98 exclusive quads, validating the production-unit finding at the quad level.

## S2.4 The Residual: 71.1% Unexplained

    H(quad | all five axes) = 6.490 bits (71.13%)

This is the content layer. It is not noise: it represents the specific lexical choices that distinguish one passage from another within the same section, position, and grammatical context. The grammar (29%) is fully reproducible without semantic encoding. The content (71%) is what resists recovery.

## S2.5 Robustness

The information budget is stable under:

- **Core coarsening:** 2-character cores preserve 80% of the MI pattern
- **Suffix abstraction level:** Full suffixes add 1.096 bits (12.0%) but at the cost of 1,794 additional entries and worse generalisation
- **Transcription sensitivity:** Sensitivity analysis under 5% character corruption (Supplement S4) confirms structural findings are robust, though all analyses are conditioned on a single transcription system (ZLZI)
- **Bootstrap:** 95% CIs for cumulative MI are within ±0.05 bits at each axis
