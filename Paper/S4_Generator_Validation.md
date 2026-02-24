# S4: Generator Validation — Bowern-Gaskell Benchmarks

> **Source code:** `src/p70c_full.py` (`generate_line()`, `score_line()`)  
> **Data:** `data/p70c_full_layer.pkl` (pre-built layer for generation)


## S4.1 Generator Architecture

The stochastic generator implements the PGCS grammar as a card-based production system with five components:

1. **Orthographic grammar**: P70 bigram legality constraints (210 rules)
2. **Stratified lexicon**: Corpus-derived token frequency distribution with Zipfian structure
3. **Positional templates**: Line-position-specific sampling pools (FIRST/MID/LAST)
4. **Length coupling**: 30% probability of selecting a word within ±1 character of the previous word's length (disabled at line boundaries)
5. **Sequential transitions**: Suffix-family → prefix reweighting via the 8×8 transition matrix

No semantic content is encoded at any stage.

## S4.2 Bowern-Gaskell Benchmark Results

The Bowern-Gaskell (BG) benchmark suite (Bowern and Lindemann 2021) provides 40+ statistical metrics against which generators are evaluated. Our generator passes 83.7% of tested metrics.

### Core Metrics (7/8 pass)

| Metric | VMS | Generator Mean | Generator SD | Status |
|--------|-----|----------------|------------|--------|
| Type-token ratio | 0.214 | 0.207 | 0.008 | ✓ Pass |
| Bigram entropy (h₂) | 2.13 | 2.10 | 0.05 | ✓ Pass |
| Zipf R² | 0.915 | 0.88 | 0.03 | ✓ Pass |
| Length AC(1) | +0.138 | +0.105 | 0.025 | ✓ Pass |
| Hapax ratio | 0.69 | 0.74 | 0.04 | ✓ Pass |
| Gallows-initial rate | 21.1% | 21.1% | 1.2% | ✓ Pass |
| Length gradient | 0.73 | 0.74 | 0.08 | ✓ Pass |
| **Position-frequency gradient** | **−41.2** | **−1.5** | **0.8** | **✗ Fail** |

The position-frequency gradient is the only core metric that fails. VMS shows a steep decline in word frequency from line-initial to line-final position (−41.2), while all generators produce flat gradients (−1 to −2). This points toward non-stationary production mechanisms that remain unmodelled.

## S4.3 Factorial Decomposition

A 2×2+1 factorial design confirms that the generator's components are independently necessary:

| Cell | Grammar | Lexicon | h₂ | Zipf R² | AC | Gallows |
|------|---------|---------|-----|---------|-----|---------|
| Full VMS | VMS | VMS | 2.10 | 0.88 | +0.105 | 21.1% |
| Grammar only | VMS | Random | 2.10 | 0.45 | +0.03 | 21.1% |
| Lexicon only | Random | VMS | 3.35 | 0.88 | +0.01 | 7.5% |
| Neither | Random | Random | 3.35 | 0.45 | −0.01 | 7.5% |
| Morph-random | VMS | VMS (shuffled) | 2.12 | 0.82 | +0.08 | 20.5% |

**Key finding**: Only the full model (VMS grammar + VMS lexicon) passes all metrics. Procedurally generated vocabularies fail on Zipf and hapax distributions. This means the VMS lexicon is learned/accumulated, not generated de novo from rules.

## S4.4 The 83.7% Ceiling

The generator achieves 83.7% of BG metrics without encoding any semantic content. The VMS's self-consistency across manuscript partitions (split-half analysis) yields an 86% ceiling — the maximum any model could achieve given transcription noise and within-manuscript variation.

The gap between 83.7% (structural ceiling) and 86% (self-consistency ceiling) represents the space available for genuine content encoding. This is a narrow margin (~2.3%), consistent with the information budget's finding that 71.1% of entropy is unexplained but structurally unconstrained.

## S4.5 Failed Metric: Position-Frequency Gradient

The position-frequency gradient (PFG) measures how average word frequency changes across line positions:

    PFG = Σ (position_rank × mean_log_frequency) / n_positions

| System | PFG | Interpretation |
|--------|-----|----------------|
| VMS | −41.2 | Strong: rare words cluster at line edges |
| All generators | −1 to −2 | Flat: uniform frequency across positions |
| Natural language | −3 to −8 | Mild: some positional frequency variation |

The VMS PFG is 20× steeper than any tested system. This implies a non-stationary production mechanism where the scribe's word-selection distribution shifts during line production — consistent with a lookup or card-drawing process where the deck composition changes through the line.

## S4.6 Character-Level vs Word-Level Independence

Maximum joint satisfaction of character-level and word-level constraints across 30+ configurations is 67%. This is incompatible with hierarchical lexical encoding, where optimising one layer should preserve the other. The independence suggests the two constraint layers (character grammar and word-level statistics) are enforced by separate mechanisms.
