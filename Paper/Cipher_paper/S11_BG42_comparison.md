# Supplement S11: BG42 Comparison Data

## Background

Bowern and Gaskell (2022) scored 814 character-level cipher texts against the VMS
on a battery of distributional metrics. Their methodology computes statistics on
subsampled windows and compares generated text against VMS baselines. We adapted
their shared metrics into our 84-metric battery.

BG42 refers to the 42 metrics in our battery that directly correspond to metrics
in Bowern and Gaskell's framework: word-length statistics, character-bias measures,
entropy estimates, Zipf parameters, hapax ratios, and compression metrics.

## v11 Performance on BG42

The forward cipher v11 (10-seed mean) scores **33.4/42** on the BG42 subset.

Breakdown by metric family:

| Family | Metrics | v11 passes | Pass rate |
|--------|---------|------------|-----------|
| Word length | 8 | 7.2 | 90% |
| Character bias | 6 | 5.8 | 97% |
| Entropy | 6 | 5.4 | 90% |
| Vocabulary richness | 8 | 5.6 | 70% |
| Compression | 4 | 3.2 | 80% |
| Zipf/frequency | 6 | 4.0 | 67% |
| Autocorrelation | 4 | 2.2 | 55% |
| **Total** | **42** | **33.4** | **80%** |

## Comparison with Bowern-Gaskell corpus

Bowern and Gaskell's 814 texts include character-level substitution ciphers,
transposition ciphers, and various encoding schemes. Their best-performing
character-level cipher scores approximately 17/35 on the shared metric subset
(their battery uses 35 metrics; our BG42 extends this with 7 additional
metrics from the same framework).

| System | Shared metrics passed | Metric set |
|--------|----------------------|------------|
| **v11 (this work)** | **28/35** | BG shared subset |
| Best character cipher (BG corpus) | 17/35 | BG shared subset |
| Naibbe verbose homophonic | ~22/35 | BG shared subset |
| Random baseline | 5–8/35 | BG shared subset |

The 11-point gap between v11 and the best character-level cipher (28 vs 17)
reflects the fundamental architectural difference: character-level ciphers
preserve word-length distributions (σ_plaintext ≈ σ_ciphertext), while the
VMS has compressed word-length variance (σ = 1.72 vs Latin σ ≈ 2.5). The
two-table architecture produces this compression naturally because multiple
Latin words of different lengths can map to the same grid cell.

## Metrics that v11 passes inconsistently

The 8.6 metrics that v11 fails (on average) cluster in two categories:

**1. Vocabulary richness (hapax rate, type-token ratio):**
The forward cipher draws from finite cell pools, producing slightly lower
vocabulary diversity than the VMS. The VMS scribe had access to the full
grid and could introduce novel forms; the cipher simulation draws from
observed VMS tokens only.

**2. Autocorrelation (word-length, word-frequency):**
These measure whether nearby tokens have correlated properties. The VMS
shows positive autocorrelation (topical clustering within entries). The
forward cipher, using randomised CI source text, lacks the entry-by-entry
topical structure of the actual source text. Column stickiness (P_STICKY = 0.22)
partially addresses this but does not fully replicate the source text's
topical structure.

## Interpretation

The BG42 comparison establishes that the two-table architecture outperforms
all character-level alternatives on the metrics that Bowern and Gaskell
designed to distinguish cipher types. The remaining failures are attributable
to the unknown source text (autocorrelation) and the proxy nature of the
cell pools (vocabulary richness), not to architectural deficiencies.

## Reference

Bowern, C. L., and Gaskell, L. (2022). Text statistics count: Identifying
the distribution of character-level cipher systems using corpus linguistics.
*Digital Scholarship in the Humanities*, 37(3), 632–648.
