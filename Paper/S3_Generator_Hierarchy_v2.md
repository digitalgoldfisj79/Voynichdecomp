# S3: Generator Hierarchy — Specification, Scoring, and Results

> **Data:** `enriched_records.pkl` (37,465 PGCS-parsed tokens), `p70c_full_spec_v1.json` (6,750 attested quads)  
> **Code:** `reproduce_s3.py` (all 23 generators), `score_85_metrics.py` (scoring module)  
> **Repo:** https://github.com/digitalgoldfisj79/Voynichdecomp  
> **DOI:** https://doi.org/10.5281/zenodo.18812705

---

## S3.1 Generator Families and Specifications

Twenty-three generators vary along two axes: vocabulary source and assembly method. They are organised into four families.

### Family 1: Zero-Corpus (3 generators)

These generators receive no corpus statistics. All vocabulary and assembly rules derive from folio 57v alone.

**Gen-00 — f57v Baseline.** All parameters derived exclusively from f57v (zero corpus information, zero fitted parameters). Reads only the f57v page from the transcription; generates tokens from the character inventory and sequence patterns of those four text rings alone. Score: 31/84 [30–34].

**Gen-0M — Manual Scribal.** All parameters from f57v; zero corpus tuning. Implements five rules a 15th-century scribe could learn in an afternoon, derived from f57v's line structure: (1) every word = prefix + gallows + core + suffix; (2) always change the suffix when writing the next word; (3) half the time also change one other part; (4) every few words write a completely new word from scratch; (5) use your assigned tall-letter pair throughout the section. Score: 23/84 [21–26].

**Gen-0W — Workshop Scribal.** All parameters from f57v; zero corpus tuning. Extends Gen-0M's mutation model by operating on skeleton/dressing structure rather than abstract slots. Skeleton characters (13 chars from f57v Line 3) provide the frame; dressing characters (vowels and connectors not in Line 3) are added around it. Templates are f57v's actual output words, so word length emerges from the template pool rather than assembly rules. Score: 14/84 [13–16].

### Family 2: BG22 Corpus-Wide Statistics (6 generators)

These generators use corpus-wide statistics without PGCS slot grammar. They are defined as inline functions in `reproduce_all.py` (no standalone `.py` file).

**Bigram.** Character bigram model. Seed tokens are drawn from **f1r only** (not the full corpus). Generates character-by-character using `$START` → character → character → `$END` transitions. Score: 39/84 [38–41].

**Scribal.** Calligraphic ductus group bigram model, trained on the full corpus. Characters are segmented into four groups — bench (`ch`, `sh`, `e`, `ee`, `eee`), gallows (`k`, `t`, `f`, `p`), loop (`o`, `a`, `y`), ligature (`d`, `s`, `r`, `l`, `n`, `q`, `i`) — and generation proceeds as group-to-group bigram transitions, with a random exemplar character drawn from each group at each step. Score: 29/84 [27–30].

**P70C Single.** Introduces the PGCS quad inventory. Three operations per token: copy (20%) — draw a random token from the full corpus so far; modify (50%) — take a recent token, look up its quad, mutate one randomly chosen slot using global slot frequency distributions; create (30%) — sample a quad entry by frequency weight and reconstruct the surface form. Score: 37/84 [35–40].

**Dual A/B.** Extends P70C Single with Currier A/B differentiation. Entry probabilities blended as 70% global + 30% Currier-section-specific (A = Herbal-A, Pharmaceutical, Astronomical, Rosettes; B = the remaining five sections). Slot mutations use Currier-specific distributions. Sections produced proportionally then shuffled. Score: 36/84 [34–37].

**Section-Profiled.** Extends Dual A/B with per-section parameters for all nine sections independently: concentration exponents (1.0–1.5), copy rates (0.20–0.30), and create rates (0.25–0.45). Entry probabilities blended as 60% global + 40% section-specific. Sections produced sequentially (not shuffled). Score: 41/84 [38–43].

**Combined.** Section-Profiled with a folio-restricted copy pool. Copy rate raised to 0.35, but the copy pool is restricted to the last 350 tokens (approximately one folio). The docstring notes explicitly: "This creates wordlen_autocorr but destroys mattr_25 and rep_rate — the impossibility proof." Score: 46/84 [41–48].

### Family 3: Template / Attested-Inventory (13 generators)

All use the same core architecture: template sampling from P70-C entries (whole quads with count ≥ 2), ductus filtering (reject character bigrams not attested in VMS), and section-assigned scribes. They form a cumulative development sequence, each adding one principled constraint. All are standalone `.py` files in `Paper/Generators/`.

**Gen-02 — Template + Ductus.** Base architecture: template sampling from P70-C entries plus ductus filter. Implements the original 6 scribal rules from f57v with P70-C slot options. Score: 29/84 [27–31].

**Gen-03 — + Corrected Suffix Rate.** Adds corrected Rule 2: suffix change rate 87% (13% repeat), evidenced by f57v Lines 2 and 4 showing 12–13% consecutive suffix repetition. (Gen-02 used 100% change.) Score: 37/84 [36–38].

**Gen-04 — + Minimal Preservation.** Adds corrected Rule 3: minimal forms (∅ gallows + ∅ core) only change the prefix slot during mutation. A scribe does not insert a tall letter into a short word when copying. Score: 44/84 [40–46].

**Gen-04T — Corpus-Tuned Ablation.** Identical architecture to Gen-04 with one change: core slot weighted by VMS corpus frequency (not P70-C ledger), and suffix sampled as actual strings at corpus frequency (not family → expand). Isolates the vocabulary-richness contribution. Score: 45/84 [42–48].

**Gen-05 — + Always Change One Other Slot.** Rule 3 changed: always change one non-suffix slot during mutation (Gen-04 changed one other slot only 50% of the time). Score: 41/84 [39–45].

**Gen-06 — + Gallows Co-occurrence.** Adds constraint: gallows words reject bare suffix during mutation. Score: 42/84 [39–48].

**Gen-07 — + Rounded Line-Start Weights.** Adds rounded (smoothed) line-start slot weights derived from observed line-initial distributions. Score: 40/84 [37–44].

**Gen-08 — + Rounded Gallows Weights.** Adds rounded gallows frequency weights per section. Score: 41/84 [40–48].

**Gen-09 — + Rule 7 Default Word.** Adds a section-specific default word used when no valid mutation is found within the ductus filter. Score: 46/84 [43–49].

**Gen-10 — + Minimal Completion.** Adds constraint: minimal forms also reject bare suffix (extending Gen-06's gallows constraint to all minimal forms). Score: 37/84 [35–43].

**Gen-SD — Scribal + Ductus Hybrid.** Template sampling from P70-C entries plus ductus filter; uses the same 6 scribal rules as Gen-0M but with P70-C slot options. Represents the base template architecture without the suffix-rate corrections of Gen-03+. Score: 34/84 [33–38].

**Gen-SP — Scribal + P70C Hybrid (best generative).** Same 6 scribal rules as Gen-0M; same production process. What changes: slot options and weights come from the P70-C constrained grammar (6,750 quad entries) rather than f57v's 77 tokens; section profiles cover all 9 VMS sections; prefix transitions are conditioned on previous suffix family; suffix expansion maps families to actual suffix strings; position awareness distinguishes FIRST/MID/LAST word in line. Score: **59/84** [56–61].

**Gen-TS — Two-Stream with Token-Length Coupling.** Frequency-weighted variant selection from per-triple menus (inherited from v7b), with token-length coupling applied to suffix selection: `exp(-|candidate_len - prev_wordlen| / λ)` weighting, applied with probability 0.50, not reset at line boundaries. This isolates the AC(1) autocorrelation effect. Score: 46/84 [40–51].

### Family 4: Transcription Model (1 generator)

**Gen-Avoid — Per-Triple All-History Avoidance.** Receives the exact VMS triple stream as input and selects suffix variants using frequency-weighted sampling with a binary penalty (p = 0.10) against any surface form previously produced by the same triple. This is a descriptive model, not generative: it demonstrates that the triple stream carries ~10 points of information beyond what any generative model reproduces. Evaluated by cross-section validation (train on 8 sections, score on held-out ninth). Results in S3.5.

---

## S3.2 The 84-Metric Scoring Suite

### Metric Suite Definition

The suite contains 84 distributional metrics derived from the full 90-metric computation by excluding 6 Levenshtein-distance subsampling metrics:

> **Excluded (6):** `wordunique_mean`, `wordunique_std`, `wordunique_skew`, `wordchange_mean`, `wordchange_std`, `wordchange_skew`

These 6 are computed by the BG22 subsampling procedure (100 iterations × 200-word samples) but require edit-distance calculation inconsistent with the notation-system hypothesis. All other BG22 subsampling metrics (word-change rate, bias statistics, character bias) are retained.

The 84 metrics divide as:

| Category | Count | Source |
|---|---|---|
| BG22 benchmark | 36 | Gaskell & Bowern (2022), minus 6 Levenshtein |
| Structural (PGCS-motivated) | 48 | Developed for this study |
| **Total** | **84** | |

The 48 structural metrics cover: entropy hierarchy (H0–H3), character distribution statistics, digraph/trigraph coverage, TTR variants (6), MATTR/MSTTR (6), hapax spectrum (6), frequency concentration, lexical richness, autocorrelation (4), and impossibility diagnostics.

### Scoring Protocol

Scoring is binary: a metric passes if the generator's median value across seeds falls within a calibrated tolerance of the VMS baseline value. Medians are computed over 10 independent seeds (42–51) for template and zero-corpus generators; 5 seeds for BG22 generators (their subsampling methodology has higher per-seed cost).

### Tolerance Derivation

Tolerances were derived empirically via two procedures:

**Bootstrap resampling (line-level, 50 iterations).** The VMS corpus was split at line boundaries into two halves by random assignment (each line assigned to half A or B with probability 0.5). Each metric was computed on both halves; the tolerance was set to the 95th percentile of |half A − half B| across 50 bootstrap samples. This captures within-corpus measurement variance.

**Cross-section variance (9 sections).** Independently, each metric was computed per section; the cross-section standard deviation provided a second variance estimate. The final tolerance for each metric is the maximum of the two estimates, ensuring it accommodates both local and global variation.

VMS self-consistency under this protocol: a VMS partition tested against the full corpus achieves 96.5% pass rate (81/84 metrics), establishing the practical ceiling. No generator is expected to exceed this ceiling by design.

**Representative tolerances:**

| Metric | VMS value | Tolerance | Basis |
|---|---|---|---|
| `wordlen_mean` | 4.931 | ±0.247 | bootstrap |
| `wordlen_autocorr` | 0.123 | ±0.077 | cross-section |
| `H2_markov_cond` | 2.342 | ±0.149 | bootstrap |
| `hapax_ratio_types` | 0.683 | ±0.046 | cross-section |
| `ttr` | 0.203 | ±0.017 | bootstrap |
| `mattr_25` | 0.919 | ±0.021 | bootstrap |
| `zipf_r2` | 0.919 | ±0.052 | bootstrap |
| `repeated_words` | 0.0083 | ±0.0040 | cross-section |
| `heaps_beta` | 0.749 | ±0.053 | bootstrap |
| `top10_share` | 0.124 | ±0.033 | cross-section |

Full tolerances for all 90 metrics (including the 6 excluded Levenshtein metrics, which retain provisional tolerances) are in the `TOLERANCES` dictionary of `score_85_metrics.py`.

---

## S3.3 Results: All 23 Generators

Ranked by median score across seeds. Self-consistency ceiling: **81/84** (VMS split-half). Gen-Avoid is evaluated by cross-section validation; its score range (67–76) is not directly comparable to the generative hierarchy.

| Rank | Generator | Family | Score/84 | Seed range |
|---|---|---|---|---|
| 1 | **Gen-SP** | Template | **59/84** | [56–61] |
| 2 | Combined | BG22 | 46/84 | [41–48] |
| 2 | Gen-09 | Template | 46/84 | [43–49] |
| 2 | Gen-TS | Two-Stream | 46/84 | [40–51] |
| 5 | Gen-04T | Template | 45/84 | [42–48] |
| 6 | Gen-04 | Template | 44/84 | [40–46] |
| 7 | Gen-06 | Template | 42/84 | [39–48] |
| 8 | Section-Profiled | BG22 | 41/84 | [38–43] |
| 8 | Gen-05 | Template | 41/84 | [39–45] |
| 8 | Gen-08 | Template | 41/84 | [40–48] |
| 11 | Gen-07 | Template | 40/84 | [37–44] |
| 12 | Bigram | BG22 | 39/84 | [38–41] |
| 13 | P70C | BG22 | 37/84 | [35–40] |
| 13 | Gen-03 | Template | 37/84 | [36–38] |
| 13 | Gen-10 | Template | 37/84 | [35–43] |
| 16 | Dual | BG22 | 36/84 | [34–37] |
| 17 | Gen-SD | Template | 34/84 | [33–38] |
| 18 | Gen-00 | Zero-corpus | 31/84 | [30–34] |
| 19 | Scribal | BG22 | 29/84 | [27–30] |
| 19 | Gen-02 | Template | 29/84 | [27–31] |
| 21 | Gen-0M | Zero-corpus | 23/84 | [21–26] |
| 22 | Gen-0W | Zero-corpus | 14/84 | [13–16] |
| — | **Gen-Avoid** | Transcription | **67–76/84** | cross-section |

**Structural breaks:**

- **Zero-corpus → BG22:** 14–31 → 29–46. The jump from no-corpus to corpus-wide statistics adds 15–20 points.
- **BG22 → Template:** The Combined BG22 generator (46/84) ties with the best Template generators at the same score, but the Template family's best (Gen-SP, 59/84) is 13 points clear. The PGCS slot structure, not self-citation alone, drives the gap.
- **Template ceiling (59) → Gen-Avoid mean (71.3):** The ~12-point gap measures information carried by the specific VMS triple stream — the lexical layer the grammar does not determine.

---

## S3.4 Gen-SP Specification (Best Generative Model)

Gen-SP (Scribal + P70C hybrid) achieves 59/84 by applying the same 6 scribal rules as Gen-0M with one critical change: slot options and weights come from the full P70-C constrained grammar (6,750 quad entries) rather than f57v's 77 tokens. The production architecture is mutation-based, not independent slot sampling.

**What stays the same as Gen-0M (the 6 scribal rules):**
1. Words = prefix + gallows + core + suffix
2. Always change the suffix for the next word (87% rate)
3. Half the time, also change one other slot
4. Every ~5th word, write a completely new word from the quad inventory
5. At line starts, use transition-conditioned prefix
6. Use section-specific gallows profile

**What changes relative to Gen-0M:**
- Slot options: P70-C ledger (6,750 quad entries, not 77 f57v tokens)
- Section profiles: 9 VMS sections (not 4 f57v paradigm units)
- Prefix transitions: conditioned on previous suffix family
- Suffix expansion: suffix family → actual suffix strings
- Position awareness: FIRST/MID/LAST word in line

**Why Gen-SP outperforms earlier template generators (e.g. Gen-05, 41/84):** Earlier generators use a smaller or differently-conditioned option space. Gen-SP's P70-C inventory of 6,750 entries with full suffix-family expansion gives a richer selection pool for mutation while the 6-rule structure keeps generation locally coherent.

**Gen-SP key metric values (median across 10 seeds):**

| Metric | VMS | Gen-SP | Pass? |
|---|---|---|---|
| `wordlen_mean` | 4.931 | 3.963 | ✓ |
| `wordlen_autocorr` | 0.123 | 0.070 | ✓ |
| `H2_markov_cond` | 2.342 | 2.798 | ✗ |
| `hapax_ratio_types` | 0.683 | 0.707 | ✓ |
| `ttr` | 0.203 | 0.218 | ✓ |
| `mattr_25` | 0.919 | 0.923 | ✓ |
| `zipf_r2` | 0.919 | 0.904 | ✓ |
| `top10_share` | 0.124 | 0.185 | ✗ |
| `trigraph_unique` | 2334 | 3416 | ✗ |
| `repeated_words` | 0.0083 | 0.0079 | ✓ |

**The 25 failing metrics** concentrate in three categories: (1) conditional character entropy (H2, H3 — Gen-SP over-generates character combinations within slots); (2) frequency concentration (top10_share, ngramdist — the empty-core template layer is insufficiently dominant); (3) digraph/trigraph counts (Gen-SP produces more novel character sequences than VMS). These failures correspond directly to the hapax residual gap identified in §4.4 of the main paper.

---

## S3.5 Gen-Avoid: Cross-Section Validation

Gen-Avoid receives the exact VMS triple stream (prefix, gallows, core) for each section and selects suffix variants using frequency-weighted sampling with a binary avoidance penalty (p = 0.10) against any surface form previously produced by the same triple in that section.

**Protocol:** Train surface-form frequency tables on 8 sections; score against the held-out ninth. Repeat for all 9 sections. This tests whether the avoidance rule generalises without retraining.

**Results (seed 42, p = 0.10):**

| Section | Score/84 | CORE-15 | Type recovery | Hapax match |
|---|---|---|---|---|
| Cosmological | 76/84 | 15/15 | 794/809 (98.1%) | 47.7% |
| Herbal-A | 74/84 | 14/15 | 1413/1430 (98.8%) | 33.2% |
| Stars | 72/84 | 14/15 | 2950/2982 (98.9%) | 27.2% |
| Pharmaceutical | 72/84 | 14/15 | 1575/1599 (98.5%) | 35.7% |
| Zodiac | 72/84 | 14/15 | 864/873 (99.0%) | 42.3% |
| Rosettes | 71/84 | 12/15 | 786/797 (98.6%) | 38.5% |
| Balneological | 70/84 | 13/15 | 1485/1502 (98.9%) | 30.5% |
| Astronomical | 68/84 | 13/15 | 768/775 (99.1%) | 44.9% |
| Herbal-B | 67/84 | 13/15 | 1898/1922 (98.8%) | 28.7% |
| **Mean** | **71.3/84** | **13.6/15** | **98.7%** | — |

**Interpretation:** The avoidance rule generalises across all 9 sections without retraining (range 67–76). Type recovery of 98.7% confirms that per-triple avoidance, not vocabulary breadth, drives the score advantage. The ~12-point gap between Gen-SP (59/84) and Gen-Avoid mean (71.3/84) measures information carried by the triple stream that no generative model recovers.

The one consistently failing metric class is MSTTR at larger windows (MSTTR-100), where Gen-Avoid over-diversifies locally. VMS suppresses long-range repetition while permitting short-range repetition within a few lines; the all-history avoidance model cannot reproduce this asymmetry (see §4.4, Supplement S7).

---

## S3.6 Impossibility Diagnostics

Five metrics are tracked as impossibility diagnostics — properties where the correct value requires simultaneously satisfying constraints that pull in opposing directions.

| Generator | AC(1) | RepWord | MATTR-25 | SLSW | EV500/25 |
|---|---|---|---|---|---|
| **VMS** | **0.123** | **0.0083** | **0.919** | **0.0447** | **1.155** |
| Gen-SP | 0.087 | 0.0079 | 0.923 | 0.0522 | 0.633 |
| Gen-TS | 0.033 | 0.0087 | 0.931 | 0.0561 | 1.331 |
| Combined | 0.008 | 0.0069 | 0.932 | 0.0436 | 0.595 |
| Section-Profiled | −0.003 | 0.0046 | 0.952 | 0.0288 | 0.755 |
| Gen-09 | 0.139 | 0.0062 | 0.941 | 0.0348 | 0.821 |
| Gen-00 | 0.802 | 0.2807 | 0.677 | 0.3359 | 0.162 |
| Scribal | 0.005 | 0.0065 | 0.940 | 0.0538 | 0.443 |
| Gen-0W | 0.258 | 0.0063 | 0.967 | 0.0334 | 0.346 |

**AC(1):** Word-length autocorrelation. VMS = +0.123; all generative models fail (range −0.003 to +0.139). Gen-09 gets closest but still fails the tolerance. Gen-TS achieves 0.033, better than most BG22 generators.

**RepWord:** Consecutive same-word rate. VMS = 0.0083. Gen-SP matches (0.0079 ✓). Gen-00's extreme value (0.2807) reflects the f57v vocabulary's small size.

**MATTR-25:** Moving-average TTR at window 25. VMS = 0.919. All generators pass this individually, but no generator simultaneously passes AC(1) and RepWord and MATTR-25 — the impossibility triple.

**SLSW:** Same-length same-word rate. VMS = 0.0447. Most generators cluster around this value; Gen-00 is the outlier.

**EV500/25:** Ratio of vocabulary growth rates at 500 vs 25 tokens. VMS = 1.155 (growth accelerates). Gen-TS (1.331) is the only generator to come close, a consequence of its two-stream architecture producing token-length diversity that drives late vocabulary growth.

No generator simultaneously satisfies all five. The impossibility is structural: raising AC(1) requires local copying which raises RepWord; suppressing RepWord requires diversification which depresses MATTR at short windows.

---

## S3.7 Factorial Decomposition

A 2×2+1 factorial design confirms that both grammar and lexicon are independently necessary:

| Cell | Grammar | Lexicon | h₂ | Zipf R² | AC(1) | Gallows% |
|---|---|---|---|---|---|---|
| Full VMS | VMS | VMS | 2.10 | 0.88 | +0.105 | 21.1% |
| Grammar only | VMS | Random | 2.10 | 0.45 | +0.03 | 21.1% |
| Lexicon only | Random | VMS | 3.35 | 0.88 | +0.01 | 7.5% |
| Neither | Random | Random | 3.35 | 0.45 | −0.01 | 7.5% |
| Morph-random | VMS | VMS (shuffled) | 2.12 | 0.82 | +0.08 | 20.5% |

Only the full model (VMS grammar + VMS lexicon) passes all key metrics. Grammar without lexicon reproduces h₂ and gallows rate but collapses Zipf fit. Lexicon without grammar reproduces Zipf but inflates h₂ and loses gallows structure. The VMS lexicon is accumulated, not generated de novo from rules.

---

## S3.8 Position-Frequency Gradient

The position-frequency gradient (PFG) is the one metric no generator reproduces. PFG measures the slope of word frequency as a function of within-line position (normalised 0–1):

| System | PFG | Note |
|---|---|---|
| VMS | −41.2 | Strong: rare words cluster at line edges |
| All 23 generators | −1 to −2 | Flat: uniform frequency across positions |
| Natural language (typical) | −3 to −8 | Mild gradient |

The VMS PFG is 5–40× steeper than any tested system. This implies a non-stationary production process in which the scribe's word-selection distribution changes systematically during line production: high-frequency template words concentrate in line-medial positions, rare words at edges. None of the tested mechanisms — self-citation, slot grammar, section profiling, token-length coupling — produces this pattern. The PFG is an unresolved property of VMS production.

---

## S3.9 Reproducibility

All results in this supplement are fully reproducible from the public repository.

**Single-command reproduction:**
```bash
git clone https://github.com/digitalgoldfisj79/Voynichdecomp
cd Voynichdecomp/Paper
pip install numpy scipy
python reproduce_s3.py
```

This runs all 23 generators and writes results to `results/s3/`. Runtime approximately 15–20 minutes on commodity hardware (NumPy and SciPy only; no GPU required). Use `--resume` to restart from checkpoints.

**Generator file locations:**
- 17 template/zero-corpus generators: standalone `.py` files in `Paper/Generators/`
- 6 BG22 generators: inline functions in `reproduce_all.py`, imported by `reproduce_s3.py`
- Gen-TS and Gen-Avoid: `gen_ts_v8b.py` and `gen_transcription_avoid.py` in `Paper/Generators/`

**Output files:**
- `results/s3/s3_all_generators.pkl` — median scores and per-metric values for all scored generators
- `results/s3/s3_vms_baseline.pkl` — VMS baseline metrics and tolerances
- `results/s3/s3_summary.md` — human-readable ranked summary

Seeds: 42–51 (10 seeds for template/zero-corpus; 5 for BG22). Results are deterministic given fixed seeds.
