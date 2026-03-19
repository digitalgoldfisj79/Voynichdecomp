# Supplement S1: Forward Cipher v11 — Architecture, Calibration, and Ablation

## Overview

This supplement documents the forward cipher v11 (`S1_v11_nomenclator.py`), the generative model that routes pharmaceutical Latin through the two-table cipher architecture described in the main text. The cipher takes two inputs — VMS Herbal-A tokens (for cell pool construction) and Circa Instans Latin (as source text) — and produces synthetic VMS-like output scored against the 84-metric battery.

The cipher has two parts. Part A (Babuini routing) is the cipher-class contribution of this paper: each Latin word is classified as function or content, then routed to a grid cell. Part B (copy-mutate scribe) is the production model from Bozzard (2026a): given a cell address, the scribe selects a token from the pool using preferential reuse, suffix avoidance, and column stickiness.

The code is self-contained. All parameters are inlined; no external pickles beyond the two inputs are required. Reproduction: `python S1_v11_nomenclator.py 42` produces a single run; `python S1_v11_nomenclator.py --ablation` runs the full ablation study.

---

## 1. Part A: Babuini Routing

Each Latin word is routed through one of two tables.

**Nomenclator (function words).** Twelve words have fixed suffix-family assignments, cross-validated on Ald.211 and Circa Instans (§5 of main text):

| Family | Words |
|--------|-------|
| Y | et, postea |
| N | in, cum, hoc |
| L | de, habet, uel/vel, que, supra, ad |

These assignments bypass all scribe rules: family is fixed by the cipher designer, never rebalanced or overridden by column stickiness. The token is drawn from the EC pool for that family.

**Grid (content words).** The initial consonant determines the row; the first vowel determines the column (suffix family):

| Row | Consonants |
|-----|-----------|
| o | c, s, p |
| c | vowel-initial, v |
| e | f, d |
| a | m, l |
| d | r, q, h, n, g |
| l | t |
| r | b, z, x, j, k, w, y |

| Vowel | Family |
|-------|--------|
| a | Y |
| e | R |
| i | N |
| o | L |
| u | BARE |

This is the identity permutation (no keyword). Exhaustive search over 7! = 5,040 row permutations confirms that the identity produces the best fit for Herbal-A (§10 of main text).

**EC/FC classification.** Words appearing in the top 53% of the CI frequency distribution are classified as EC (function/high-frequency) and routed to empty-core pools. All others are FC (content) and routed through the grid. The 53% threshold matches the manuscript's observed 52.7% EC rate.

---

## 2. Part B: Copy-Mutate Scribe

Four production rules operate on the cell-addressed token stream.

### 2.1 Preferential reuse (COPY_ALPHA = 1.3)

When vocabulary saturation is reached (≥1,430 types produced), new tokens are drawn from past output weighted by frequency raised to the power COPY_ALPHA. A token seen 100 times gets 100^1.3 = 501× the weight of a token seen once. This produces the Zipfian concentration observed in the manuscript.

**Calibration.** COPY_ALPHA = 1.3 was calibrated to match the manuscript's top-50 token concentration (generated 0.411 vs manuscript 0.406). The original value of 2.0 was lowered because it squared past counts, creating excessive EC concentration (§Finding 5 in session log).

### 2.2 Suffix avoidance (AVOIDANCE = 15)

When selecting from a cell pool, tokens appearing in the recently produced set have their sampling weight divided by AVOIDANCE. This discourages exact repetition while preserving the cell's frequency distribution. The effect is per-triple: the scribe avoids repeating the same surface form from the same cell, producing the suffix diversity documented in Bozzard (2026a, §4.5).

**Calibration.** AVOIDANCE = 15 produces type counts matching the manuscript (generated ~1,430 vs manuscript 1,430). Without avoidance, types collapse to ~1,120.

### 2.3 Column stickiness (P_STICKY = 0.22)

With probability P_STICKY, a non-nomenclator token's suffix family is overridden to match the previous token's family, regardless of what the grid routing specified. This models the tendency of a scribe glancing at the previous token and picking from a nearby grid column.

**Calibration.** P_STICKY = 0.22 was calibrated to match the suffix-family bigram rate (generated 0.253 vs manuscript 0.252 under the same measurement method used in the scoring pipeline). Nomenclator-routed words are exempt: their family is fixed by the cipher designer.

### 2.4 Suffix-family rebalancing (REBAL_STR = 8.0)

When the running family distribution drifts more than 3% below target for any family, the scribe probabilistically redirects to an underrepresented family. This prevents drift accumulation over 4,033 tokens. Target distribution is from Herbal-A: Y 31.3%, R 17.4%, N 17.1%, L 16.1%, BARE 14.0%, M 3.0%.

### 2.5 Boundary innovation

At line beginnings and endings, the copy-mutate rate is reduced by factor 0.35, producing elevated novelty at structural boundaries. This matches the manuscript's 2.0× hapax rate at line-initial positions (24.4% vs 12.2% elsewhere).

### 2.6 Line structure

Output is segmented into lines drawn from the Herbal-A line-length distribution (613 lines, range 1–13 tokens, mode 5). Line breaks reset the copy-mutate context.

---

## 3. Cell Pools

Each grid cell (row, family) is populated with VMS Herbal-A tokens observed in that cell, grouped by (m_core first character, suffix family). These are proxies for the unknown grid contents. If we had the actual keyword and grid, the cipher would construct tokens from first principles without needing the manuscript at all.

Pools are augmented with constructed tokens assembled from GRID_PRIMARY core strings (the most common core per cell), crossed with the prefix and gallows inventories, filtered by the character-level finite state machine derived from Herbal-A bigrams. Constructed tokens receive 10% of the weight of observed tokens (SEED_WEIGHT = 0.10).

Ten rare tokens are injected at random positions to maintain digraph coverage for metrics sensitive to rare character combinations.

---

## 4. Scoring

v11 is scored against the 84-metric battery (S5_score_85_metrics.py) using binary pass/fail with empirically derived tolerances from bootstrap resampling of VMS split-half partitions.

**10-seed results** (seeds 42, 404, 501, 606, 808, 909, 101, 202, 303, 505):

| Metric | Value |
|--------|-------|
| n/84 | 62.9 (σ = 2.6) |
| CORE-15 | 12.6/15 |
| BG42 | 33.4/42 |
| Types | 1,432 (VMS: 1,430) |
| top50 | 0.411 (VMS: 0.406) |
| sfx_bi | 0.253 (VMS: 0.252) |

Twelve of 84 metrics fail on all ten seeds, clustering in vocabulary spectrum measures (hapax ratio, frequency spectrum, Sichel's S, Honoré's R) and vocabulary-diversity autocorrelation (segmented TTR and windowed hapax rate). These measure how the mix of unique and repeated words varies across the token stream — topical structure that depends on which plant entry is being enciphered in which order. Word-length autocorrelation passes on all ten seeds, confirming that column stickiness correctly reproduces length-level sequential structure.

---

## 5. Ablation Study

To measure each component's contribution, we systematically disabled one component per run while holding everything else at default values. Three seeds (42, 404, 501) per configuration, scored on the full 84-metric battery.

### 5.1 Configurations

| # | Configuration | Change from full v11 |
|---|--------------|---------------------|
| 1 | Full v11 | Baseline — all components active |
| 2 | Minus nomenclator | Replace inferred assignments with random (et→Y, in→N fixed as anchors) |
| 3 | Minus stickiness | P_STICKY = 0.0 |
| 4 | Minus reuse | COPY_ALPHA = 0.0 (uniform weighting in reuse) |
| 5 | Minus avoidance | AVOIDANCE = 1.0 (no penalty for recent tokens) |
| 6 | Architecture only | S4 clean baseline: two-table routing with uniform random pool selection, no scribe rules |

### 5.2 Results

| Configuration | n/84 | C15 | BG42 | Δ from full |
|--------------|------|-----|------|-------------|
| Full v11 | 62.0 | 12.3 | 33.3 | — |
| Minus nomenclator | 60.7 | 11.0 | 33.3 | −1.3 |
| Minus stickiness | 65.3 | 13.3 | 34.7 | +3.3 |
| Minus reuse | 38.3 | 10.0 | 23.0 | −23.7 |
| Minus avoidance | 65.0 | 14.0 | 36.7 | +3.0 |
| Architecture only (S4) | 48.3 | 8.0 | 30.3 | −13.7 |

### 5.3 Per-seed detail

**Full v11:** seed 42: 61/84; seed 404: 65/84; seed 501: 60/84

**Minus nomenclator:** seed 42: 59/84; seed 404: 61/84; seed 501: 62/84

**Minus stickiness:** seed 42: 64/84; seed 404: 66/84; seed 501: 66/84

**Minus reuse:** seed 42: 40/84; seed 404: 37/84; seed 501: 38/84

**Minus avoidance:** seed 42: 66/84 (1,129 types); seed 404: 64/84 (1,115 types); seed 501: 65/84 (1,124 types)

**Architecture only (S4):** seed 42: 49/84 (893 types); seed 404: 50/84 (876 types); seed 501: 46/84 (887 types)

### 5.4 Interpretation

**Preferential reuse is the dominant scribe component.** Removing it drops the score by 24 points, from 62 to 38. Without frequency-weighted reuse, the cipher cannot reproduce the manuscript's vocabulary concentration: Zipf structure, top-token dominance, and frequency spectrum all collapse. This is the single mechanism that transforms grid output into VMS-like text.

**The two-table architecture provides the base.** The S4 clean baseline scores 48.3/84 with nothing but routing and uniform random selection from pools. This establishes what the cipher architecture contributes before any production modelling: correct word-length distribution (σ within range), correct EC/FC ratio, correct consonant-row distribution, and approximately correct suffix-family proportions.

**Column stickiness trades general performance for a specific structural match.** Removing stickiness improves the general score by +3.3 points. Stickiness was calibrated to match the suffix-family bigram rate (sfx_bi = 0.252), a specific VMS property that captures the scribe's tendency to stay in the same grid column. The metric battery does not include sfx_bi as a scored metric, so the 3-point cost is not recovered by the sfx_bi match. The justification for including stickiness is structural: the manuscript exhibits column clustering (sfx_bi = 0.252 vs 0.204 expected under independence), and the copy-mutate production model (Bozzard 2026a) predicts it.

**Suffix avoidance trades general performance for vocabulary size.** Removing avoidance improves the general score by +3.0 points but collapses type count from ~1,430 to ~1,120. The manuscript has 1,430 Herbal-A types; without avoidance, the cipher underproduces by 22%. Avoidance is the mechanism that generates the suffix diversity documented in Bozzard (2026a, §4.5): 629 excess hapax beyond proportional sampling, at a constant rate across all nine sections.

**The nomenclator contributes minimally to the metric score** (−1.3 points when randomised). This confirms the main text's finding (§6.2) that the 84-metric battery validates the cipher architecture, not the specific function-word assignments. The nomenclator's validation comes from a separate test: bigram correlation r = 0.96 on training data, r = 0.89 on cross-validation (CI), exceeding all 10,000 random assignments (p < 0.0001).

### 5.5 Component interaction

The components are not fully additive. Architecture only (48.3) + the reuse delta (23.7) = 72.0, which exceeds the full v11 score (62.0) by 10 points. This implies that stickiness and avoidance interact negatively with reuse: they constrain the reuse mechanism in ways that reduce its effectiveness on the general battery while steering it toward specific VMS properties. The full v11 represents a calibrated compromise between general distributional accuracy and targeted structural matching.

---

## 6. Reproduction

All results can be reproduced from two input files:

- `enriched_records.pkl` — VMS PGCS decomposition (37,465 tokens, archived at DOI 10.5281/zenodo.18827922)
- `ci_corpus_parsed.pkl` — Circa Instans tokenisation (52,004 tokens, archived at same DOI)

Commands:
```
python S1_v11_nomenclator.py              # single run, default seed
python S1_v11_nomenclator.py 42           # single run, seed 42
python S1_v11_nomenclator.py --ablation   # full ablation study (requires S4, S5 scripts)
```

The ablation study additionally requires `S4_forward_cipher_clean.py`, `score_85_metrics.py`, and `metric_defs.py` in the same directory.
