# S5: Cross-Validation, Red Team, and Robustness Checks

> **Source code:** `src/p70c_full.py` (`validate()`), `src/p70c_connect.py` (cross-val)  
> **Data:** `data/p70c_full_layer.pkl`, `data/enriched_records.pkl`


## S5.1 PGCS Grammar Cross-Validation

### Held-Out Generalisation

40% of the corpus was held out by folio during grammar development. The held-out set contains 3,930 unique types not seen during rule development. Zero types produce PGCS violations. The grammar generalises perfectly to unseen data.

### Tier-by-Tier Cross-Validation (50 trials, 80/20 split)

| Tier | n range | Quads | Novel type % | Novel token % |
|------|---------|-------|-------------|--------------|
| T1 | ≥50 | 118 | 0.0% ± 0.0% | 0.0% ± 0.0% |
| T2 | 10–49 | 317 | 0.0% ± 0.0% | 0.0% ± 0.0% |
| T3 | 4–9 | 511 | 0.1% ± 0.2% | 0.2% ± 0.4% |
| T4 | 1–3 | 4,226 | 66.1% ± 1.3% | 65.2% ± 1.4% |

T1+T2 generalise perfectly. T4 novel rates match the mathematical prediction from the hapax distribution (Zipf's law). This is normal lexical behavior, not structural failure.

## S5.2 Red Team Analysis (9 Attacks)

### Summary

| Attack | Target | Verdict | Severity |
|--------|--------|---------|----------|
| 1. Circularity | p70-c generalisation | ✓ PASS | None (Zipf's law) |
| 2. m_core granularity | Core representation | ✓ PASS | Low (acknowledged) |
| 3. Line-length confound | Positional effects | ✓ PASS | None |
| 4. Transcription dependence | All findings | ⚠ CAVEAT | Medium (inherent) |
| 5. Tautology risk | Skeleton/content split | ✓ PASS | None |
| 6. Over-generation framing | 58.6% claim | ✗ FIXED | Was high → corrected |
| 7. Decomposition validity | PGCS structure | ✓ PASS | None |
| 8. Significance inflation | Effect sizes | ⚠ CAVEAT | Medium (reporting) |
| 9. Token-quad ambiguity | Parse determinism | ✓ PASS | None |

### Attack 3 Detail: Line-Length Confound

M-suffix enrichment at LAST position:
- Short lines (≤5 words): 4.2× enrichment
- Long lines (≥8 words): 21.6× enrichment

y-prefix enrichment at FIRST position:
- Short lines: 3.7× enrichment
- Long lines: 3.5× enrichment

Effects hold and are actually STRONGER on long lines. Not a confound.

### Attack 5 Detail: Tautology Risk

Random 4-slot decomposition gives 12,563 quads with 76.7% hapax. PGCS gives 5,172 quads with 63.5% hapax. The skeleton/content distinction is non-trivially a product of the PGCS grammar, not an artefact of any arbitrary decomposition.

### Attack 6 Detail: Over-Generation Correction

Two distinct metrics must be separated:

**Vocabulary precision** (which types the model accepts):

| Model | Types accepted | Over-generation |
|-------|---------------|----------------|
| PGCS unconstrained | ~42,522 | 5.6× |
| PGCS-C weighted | ~14,768 | 1.9× |
| Observed | 7,598 | 1.0× |

**Placement precision** (where types can appear):

| Model | (type, position) pairs | Over-generation |
|-------|----------------------|----------------|
| Quad (no position) | 22,794 | 2.41× |
| Quint (with position) | 9,446 | 1.00× |

Position conditioning constrains WHERE tokens appear, not WHICH tokens exist.

## S5.3 Transcription Robustness

### Sensitivity Analysis (5% Character Corruption)

| Slot | Hit rate at 5% corruption | Interpretation |
|------|--------------------------|----------------|
| Prefix | 5.8% | Robust (short, high-frequency) |
| Gallows | 2.5% | Very robust (distinctive characters) |
| Suffix | 10.4% | Moderate (longer, more variation) |
| m_core | 7.2% | Moderate (depends on exact characters) |

### Tiered Robustness Statement

- **ROBUST** (T1+T2, 435 quads, 77.6% tokens): Cross-transcription validated. Short slots, high counts, absorbs noise.
- **MODERATELY ROBUST** (T3, 511 quads, 7.8% tokens): n = 4–9, survives isolated errors.
- **FRAGILE** (T4, 4,226 quads, 14.6% tokens): Hapax entries. Class existence is robust (Zipf); individual membership is not.

Structural findings (4-slot architecture, entropy distribution, M-suffix marker, position effects, skeleton/content split) are transcription-independent. Specific T4 m_cores and exact MI values (±0.01–0.05 bits) may shift.

## S5.4 Effect Size Reporting

All χ² tests in the paper operate on N = 37,465 tokens. At this sample size, significance is expected even for small effects. Effect sizes (Cramér's V and MI as % of marginal H) are the appropriate measures of substantive importance.

| Relationship | χ² | V | MI (bits) | MI/H% | Size |
|-------------|-----|-------|----------|-------|------|
| Prefix × Position | 2,963 | 0.199 | 0.054 | 1.9% | small |
| Gallows × Position | 1,013 | 0.116 | 0.018 | 1.0% | small |
| m_core × Position | 8,742 | 0.342 | 0.128 | 3.1% | medium |
| SfxFam × Position | 3,273 | 0.209 | 0.040 | 1.7% | small |
| Prefix × Section | 3,396 | 0.114 | 0.067 | 2.4% | small |
| Gallows × Section | 1,376 | 0.068 | 0.024 | 1.4% | negligible |
| SfxFam × Section | 1,849 | 0.091 | 0.037 | 1.6% | negligible |
| Prefix × Gallows | 18,526 | 0.266 | 0.393 | 14.1% | small |
| Core × SfxFam | 28,705 | 0.357 | 0.371 | 9.1% | medium |

Individual slot-level effects are small to medium. The information budget captures their joint contribution: 28.9% of quad entropy is explained by the five conditioning axes together.
