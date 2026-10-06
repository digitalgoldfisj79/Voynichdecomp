# EDLC1 — Edit-Distance and Length Comparanda Benchmark

**Version:** 1.0 preregistration  
**Date:** 2026-10-06  
**Status:** frozen before primary results

## Purpose

Test whether the Voynich Manuscript's token-length distribution and ED1/ED2 lexical neighbourhood structure are unusual relative to historically appropriate German and Latin comparanda.

This programme is intentionally narrower than the existing Voynich grammar, CF, DAIIN, and NeuroDecipher programmes. It does **not** infer language identity or decipherment. It establishes a calibrated historical baseline for surface-form geometry.

## Primary questions

Q1. Is Voynich token length unusually concentrated relative to 15th-century German and Latin manuscript text?

Q2. Is Voynich vocabulary unusually dense under raw Levenshtein distance 1 and <=2?

Q3. Does any ED excess survive controls for vocabulary size, token frequency, token length, and orthographic alphabet structure?

Q4. In morphologically annotated historical German, what share of ED1/ED2 edges connect forms of the **same lemma** versus **different lemmas**?

Q5. In document-blocked held-out evaluation, how often can a genuinely unseen token type be “repaired” to a training type within ED1/ED2/ED3?

## Corpus hierarchy

### Tier A — primary confirmatory controls

1. **Voynich ZLZI**, strict +P0 running text.  
   - Same pinned slim corpus used in DAIIN/CF work.
   - ZLZI is the primary EVA-like representation.
   - TTLI is the primary independent representation check.
   - ZLZB is a transcription check only and MUST NOT be treated as an independent corpus.

2. **German: ReF v1.0.2, 15th-century manuscripts only.**
   - Filter: metadata medium contains Handschrift; metadata time begins 15,.
   - Primary surface channel: tok_dipl diplomatic UTF form.
   - Lemma channel: tok_anno/lemma where alignment is unambiguous.
   - Main pooled panel plus predeclared regional panels where sample size permits: Bavarian, Alemannic, East Upper German, West Upper German, North Upper German.
   - Documents remain separate blocks for uncertainty and OOV evaluation.

3. **Latin: DEEDS HTR Dataset, BL Cotton MS Nero E VI.**
   - Dataset commit pinned in source manifest.
   - DEEDS sample is from ff. 200–221 of the Prima Camera. Gervers (1974) records that the cartulary was begun in 1442 and that the Prima Camera (ff. 3–288v) was completed by 1447; therefore this sampled Latin surface is securely 1442–1447. Primary control uses hand-corrected/ground-truth plain ALTO XML only, never .chocomufin.xml model output.
   - Only MainZone text blocks.
   - Physical page is the resampling/OOV block.
   - This is a diplomatic surface control and preserves abbreviation signs.

### Tier B — predeclared replications, not required for EDLC1 primary adjudication

- RIDGES Herbology, *Gart der Gesunthait* (1487), diplomatic layer.
- Bonner Frühneuhochdeutschkorpus (FnhdC), 1450–1500 slice.
- Honkapohja–Suomela John of Burgundy Latin medical-manuscript transcriptions, 1450s–1490s, if source XML can be obtained reproducibly.
- eFontes 1400–1500 Medieval Latin slice if a stable export/API snapshot is available.

Tier B failures of access MUST be reported as missing replications, never silently substituted with modern or Classical corpora.

## Surface normalization channels

### Primary: diplomatic grapheme channel

- Unicode NFC.
- Casefold/lowercase.
- Leading/trailing punctuation removed.
- Internal alphabetic characters, combining marks, and historically meaningful abbreviation signs retained.
- Token length measured in Unicode grapheme clusters, not Unicode code points.
- Tokens must contain at least one letter.
- Voynich tokens use their frozen transliteration strings; no FORM decomposition, ED normalization, or glyph-family folding.

### Sensitivity: conservative folded channel

Language-specific, predeclared:
- German: long-s -> s; standard Unicode compatibility folding where unambiguous.
- Latin: long-s -> s; j -> i; v -> u; standard ligature expansion.
- Combining abbreviation marks are **not expanded into guessed letters**.
- The folded channel is sensitivity only and cannot replace diplomatic results.

### Annotated normalized channel

For ReF only, where an unambiguous tok_anno alignment exists:
- compute the same length and ED metrics on annotated normalized word forms;
- use this to estimate how much surface ED density is attributable to historical orthography versus ordinary morphology/lexicon.

## Core metrics

All ED metrics are repeated at minimum token-frequency thresholds **1, 2, 3, and 5**. The **predeclared primary ED threshold is frequency >=3**; thresholds 1, 2, and 5 are sensitivity panels. This choice is frozen before the primary run to reduce singleton/transcription-error leverage without discarding genuinely recurrent vocabulary.

### Length metrics

Report token-weighted and type-weighted:
- mean
- median
- standard deviation
- variance
- coefficient of variation
- Fano ratio = variance / mean
- q10, q25, q75, q90
- complete integer length histogram
- alphabet size
- character entropy
- effective alphabet size = 2^H

### ED graph metrics

For unique types at each frequency threshold:
- vocabulary size V
- fraction of types with any ED1 neighbour
- fraction of types with any ED<=2 neighbour
- ED<=3 sensitivity
- mean and median ED1 degree
- mean ED<=2 degree
- pair density at ED1 = ED1 unordered pairs / C(V,2)
- pair density at ED<=2
- nearest-neighbour-distance distribution, capped at >=4
- largest connected-component share at ED1 and ED<=2
- degree Gini coefficient

Pair density is a primary statistic because raw degree scales with vocabulary size.

### Morphology-aware decomposition (ReF)

For every ED1 and ED2 edge among surface types, using the set of observed lemmas for each type:
- SAME_LEMMA: non-empty lemma-set intersection
- DIFFERENT_LEMMA: disjoint non-empty lemma sets
- AMBIGUOUS: either form lacks a usable lemma

Report shares by edge count and token-frequency-weighted edge mass.

Also report:
- within-lemma pairwise ED distribution;
- proportion of same-lemma type pairs at ED0/1/2/3/>3;
- examples selected only by deterministic frequency ranking, never by interpretive interest.

This directly separates ein/einem/einer-like morphology/orthography from accidental Haus/Maus-like string neighbours.

## Matched controls

Raw cross-corpus ED density is not sufficient because ED neighbourhood density depends strongly on token length, frequency spectrum, vocabulary size, and alphabet structure.

### M1 — joint length x frequency matched resampling

For each historical corpus and each Voynich transcription:
- define strata by exact grapheme length and frequency bin:
  - 1
  - 2–3
  - 4–7
  - 8–15
  - 16+
- draw without replacement from the historical type inventory to reproduce the Voynich stratum counts where support exists;
- use the **largest common support subset** if an exact match is impossible;
- report retained Voynich type count and coverage fraction;
- 1,000 publication draws (200 allowed for development only).
- compute all ED graph metrics in each draw.

No extrapolation into unsupported strata is allowed.

### M2 — orthographic positional-shuffle null

Within each corpus:
- preserve each token's exact grapheme length;
- for every character position conditional on length, shuffle characters across types;
- preserve as much of the empirical positional character distribution as possible;
- duplicates are retained as ED0 collisions and explicitly counted;
- 1,000 publication draws.

Purpose: estimate ED clustering in excess of that produced by length and alphabet/positional structure alone.

### M3 — frequency-spectrum sensitivity

Repeat primary ED graph statistics at min-frequency thresholds 1, 2, 3, 5.
A conclusion that exists only at singleton-inclusive threshold is flagged as transcription/OCR-sensitive.

## OOV repairability

Use deterministic block cross-validation.

- Voynich: physical bifolium blocks.
- ReF: entire manuscript/source documents.
- Cotton Nero: physical pages.

Five folds assigned by stable hash of block id.

For each fold:
- training vocabulary = types in the other four folds;
- held-out events = tokens whose type is absent from training;
- find nearest training type by Levenshtein distance;
- report event-weighted and type-weighted fraction with nearest distance 1, <=2, <=3, and >=4;
- report the full nearest-distance distribution;
- no candidate morphology or context is used.

This is the direct historical comparator for prior Voynich “repairability” results.

## Statistical uncertainty

### Historical corpora
Primary uncertainty uses cluster bootstrap at the highest independent physical/textual unit:
- ReF: document;
- Cotton Nero: page;
- Voynich: bifolium.

At least 2,000 bootstrap replicates for publication tables.

### Matched-null comparison
For every headline matched statistic report:
- observed Voynich value;
- matched-control mean;
- matched-control SD;
- standardized difference z = (obs-null_mean)/null_SD;
- two-sided empirical p from the matched draws;
- 95% percentile interval of control distribution.

If |z| < 2, reporting MUST begin: **“the metric does not resolve this”.**

No multiplicity-corrected “significance” claim will be made from an exploratory family of correlated ED metrics. The predeclared primary endpoints are:
1. token-weighted token-length CV (all accepted tokens);
2. ED1 pair density at minimum corpus frequency 3;
3. ED<=2 pair density at minimum corpus frequency 3;
4. event-weighted OOV ED<=2 repairability.

Type-weighted length CV and ED thresholds 1/2/5 are predeclared sensitivities.

## Comparability and exclusions

- No Classical Caesar/Cicero corpus is permitted as a primary Latin control.
- No modern standardized German is permitted as a primary German control.
- Printed 1487 German material is a replication, not a substitute for manuscript German.
- HTR/OCR model output is excluded where manually corrected ground truth exists.
- Marginalia, numbering, running titles, stamps, and non-main text zones are excluded from Cotton Nero.
- ReF punctuation and foreign-material tokens are excluded where annotation marks them.
- Named entities are **not** removed from the primary corpus; a no-proper-name sensitivity may be added later.
- Abbreviations are not expanded in the primary diplomatic channel because the Voynich surface likewise provides no expansion key.

## Independence

- ZLZI and ZLZB are not independent.
- Regional ReF panels are subsets of the same corpus and are robustness panels, not independent replications.
- Pages from Cotton Nero are not independent manuscripts; they are blocks within one manuscript.
- Claims of cross-language robustness require agreement of German and Latin Tier-A controls, not merely many pages/documents.

## Reproducibility requirements

Every run must emit:
- source URL/repository;
- immutable version or commit when available;
- SHA-256 of downloaded archive/file where practical;
- parser version/commit;
- exact filters;
- token counts, type counts, excluded-token counts;
- random seed;
- source manifest;
- JSON result with machine-readable metric definitions.

The runner must fail closed if the pinned Voynich SHA mismatches or if an expected source version cannot be verified.

## Publication stop rule

EDLC1 may support a claim only of the form:

> “Under matched 15th-century manuscript controls, Voynich [does/does not] exhibit unusually concentrated token lengths / unusually dense low-edit-distance lexical neighbourhoods.”

It may **not** by itself support:
- German or Latin identification;
- cipher/notation identification;
- synonymy/homophony;
- lexical-family interpretation;
- decipherment.

Those require independent evidence.
