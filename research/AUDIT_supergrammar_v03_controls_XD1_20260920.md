# Super-Grammar v03 Control Admissibility Audit — XD1

**Date:** 2026-09-20  
**Parent release:** `supergrammar_v03_release_20260920`  
**Protocol:** `XD1_20260920`  
**Protocol freeze commit:** `5c5bf98447d64dd8bf6dddbd8fb7e85e8f62fe27`

## Purpose

This document separates genuine external controls from obsolete, target-tuned, incomplete, or mechanism-dependent material accumulated during earlier Voynich programmes.

The rule is simple: a source is not admitted because it is historical, cipher-related, medical, or superficially Voynich-like. It is admitted only for a named measurement for which its source contract, representation, sample structure, and independence are adequate.

The authoritative machine-readable register is:

- `public.vms_supergrammar_control_registry_v03`
- `public.vms_supergrammar_control_metric_map_v03`

Both tables are RLS-protected.

## Fully qualified executed control

### SG1.1 Middle Dutch multi-witness source control — ADMISSIBLE_EXECUTED

Four independently aligned manuscript traditions were evaluated under held-out-witness validation, including an opaque256 representation that removes readable lexical identity. All four source-bearing controls resolved. This is a valid positive control for the **source-bearing detection instrument**.

It is not evidence for a Dutch source, a cipher, or a historical Voynich mechanism.

## Primary prospective external controls

### Nuremberg Letterbooks 2–5 — ADMISSIBLE_PENDING_INGEST

Dates 1408–1423. Four archival books, 1711 labelled pages, approximately 50k annotated physical lines and ten writer IDs. PAGE XML provides line polygons and writer labels. Diplomatic/regularized variants and abbreviation-expanded/unexpanded representations are available.

Admitted for:
- within-token memory depth;
- SPACE versus physical LINE_BREAK attenuation;
- previous-token morphology carryover;
- incremental exact previous-token identity;
- lagged exact-form recurrence;
- writer effects where support is crossed;
- paired abbreviation intervention.

This is the strongest newly identified general-purpose control because several relevant variables are observed rather than reconstructed.

### CREMMA Medii Aevi — ADMISSIBLE_PENDING_INGEST

Open Latin manuscript HTR ground truth, 12th–16th centuries, 7274 physical lines. Graphematic transcription preserves abbreviation signs, manuscript spacing and physical lines. Contains medical as well as non-medical witnesses.

Admitted prospectively for the portable surface metrics, with manuscripts stratified by witness/century/genre rather than pooled indiscriminately.

## Admissible only for specific questions

### Nuremberg Letterbooks 6–14 — ADMISSIBLE_WITH_SCOPE

Dates 1423–1441 and therefore exceptionally attractive chronologically. However, the project explicitly states that these automatic transcriptions are **unverified** and intended as a navigation aid. They are not admitted for fine-grained quantitative inference until a frozen QC sample establishes transcription error bounds.

### Burchards Dekret Digital abbreviation pairs — ADMISSIBLE_WITH_SCOPE

Potentially strong paired abbreviation intervention, but chronologically early. Only natural manuscript sequence is admissible. Augmented duplicate rows created for machine-learning balance are prohibited.

### PeDoCo — ADMISSIBLE_WITH_SCOPE

Strong control for formulaic professional Latin and document-level repertoire/recurrence. It derives from editions rather than physical manuscript line transcription, so it cannot support line-break or as-written abbreviation claims.

### Gaskell–Bowern human pseudo-writing — ADMISSIBLE_WITH_SCOPE

A real human production experiment, not a historical corpus. Useful for testing whether instructed asemantic writing creates portable low-level features. Short 1–3 page samples are likely underpowered for manuscript-scale recurrence. Voynich-aware participant strata must be kept separate where possible.

### Bowern et al. systematic text transformations — ADMISSIBLE_WITH_SCOPE

Useful algorithmic intervention corpus because the transformations are independently specified. It is not a historical-cipher corpus and cannot establish historical use. Old nearest-distance conclusions are not inherited; XD1 must be run prospectively.

### Florence MS 106 — ADMISSIBLE_WITH_SCOPE

A valid positive control for the **specific semantic-clustering/Jaccard method**: specialised therapeutic categories produced detectable clustering under the archived method. Because its transcription was normalized, lemmatized and stopword-filtered, it is not admitted for the portable v03 surface grammar measurements.

### Tacuinum / Regimen seasonal-row test — ADMISSIBLE_WITH_SCOPE

Keep the narrow negative result. The archived work explicitly corrected an initially circular cherry-picked test, then ran actual Tacuinum entries and a source-assigned Regimen de Mensibus test. The proposed seasonal consonant-row flip failed, including wrong-direction and flat effects. This does not generalize to all medical/regimen traditions.

## High-value corpora requiring provenance recovery

### 35 NE Italian / Venetian herbal Transkribus corpus — PENDING_PROVENANCE_QC

Archived March handoff reports 35 manuscript transcriptions, 12,390 words and 1,408 physical lines. Potentially highly relevant as a region/genre control.

The old lexical-anchor/frequency conclusions are contaminated by hypothesis development and are not admitted. The raw witness inventory, transcription files and QC history must be recovered before the corpus enters XD1.

### Circa Instans / Wellcome MS 624 — PENDING_PROVENANCE_QC

The March package reports a parsed 52,004-token Transkribus corpus. It was heavily used in the old nomenclator/Babuini programme and therefore cannot independently validate those models.

It may still be valid for **newly frozen XD1 measurements that were not selected using CI outcomes**, once the raw transcription/source contract is recovered and checked.

## Discovery pool only

### DECODE / DECRYPT — POOL_ONLY_NOT_ADMITTED

DECODE is heterogeneous: cipher keys and ciphertexts, multiple centuries, languages, mechanisms, lengths, transcription states and cryptanalytic outcomes.

The database as a whole is **not** a control corpus.

A historical-cipher subset can be admitted only after prospective selection requiring, at minimum:
- known mechanism;
- ciphertext rather than key-only record;
- reliable machine-readable transcription;
- adequate sequence length for the chosen XD1 metric;
- date/language/provenance metadata;
- preferably a known key or plaintext;
- no selection by resemblance to Voynich.

## Model comparators, not external controls

### P70C / 84-metric generator hierarchy — MODEL_COMPARATOR_ONLY

Target-derived engineering/ablation framework. Valid for asking whether a specified mechanism can reproduce measurements. Invalid as independent evidence that the mechanism existed historically.

### Timm / Schinner self-citation — MODEL_COMPARATOR_ONLY

The external hypothesis remains a legitimate mechanism class to test. Old local approximate implementations are not sufficient authority. Any future result must use a faithful independently specified implementation or the authors' code/protocol.

## Excluded from independent-control inference

### Legacy v11 Babuini / nomenclator model — EXCLUDED_CONTAMINATED_FOR_MODEL_VALIDATION

Severe target and corpus contamination: Voynich structure informed the model; Ald.211 was used in training/optimization; Circa Instans was repeatedly used for selection/validation. Later adversarial audit found search inflation and unresolved score discrepancies. Retain for provenance and falsifiable subclaims only.

### Synthetic Chinese JCZS generator — EXCLUDED_SYNTHETIC_TUNED

The archived v2 plan explicitly proposed scaling vocabulary, page-topic structure and corpus size to repair specific failed Voynich metrics. This is target-directed simulation, not an external control.

### Legacy S6/S6a language battery — EXCLUDED_MODEL_DEPENDENT

Underlying language corpora may be useful, but the rankings were generated by mapping source-language consonants through the proposed Babuini/grid mechanism. The result therefore presupposes the very mechanism it purported to discriminate. Old rankings/kills are not inherited.

### Legacy BG42/v11 derivative — EXCLUDED_DERIVATIVE

The archived 28/35 versus 17/35 headline combines a tuned v11 model with a derived/non-identical metric subset and insufficiently distinguishes the separate Bowern/Gaskell human-gibberish and algorithmic-encipherment programmes.

The **underlying public author datasets remain usable prospectively**. The old derivative headline does not.

### Ald.211 as cipher validation — EXCLUDED_CONTAMINATED_FOR_MODEL_VALIDATION

Ald.211 was directly used in nomenclator training/optimization. Later adversarial work found the greedy search-inclusive null weakened the apparent assignment result substantially. It cannot serve as independent validation of the trained architecture.

## Exploratory only / restart from primary data

### Pseudo-Apuleius fragment null — EXPLORATORY_ONLY

The archive itself says the text consists of incomplete fragments assembled from web and scholarly snippets. Replace with a complete edition/transcription before formal use.

### Legacy Arabic Qanun/Aqrabadhin comparison — EXPLORATORY_ONLY

Potentially useful real historical corpus, but the archived numerical comparisons mixed synthetic and real controls and used VMS target definitions that changed across sessions. Recover source/edition/tokenization, freeze them, and rerun from raw text.

### Taiping Huimin Heji Jufang — NOT_EXECUTED

The Drive artifact is a task specification, not a completed control. It remains a candidate only if a verified historical text and frozen tokenization protocol are obtained.

## Cross-domain protocol

The portable v03 comparison is frozen separately as `XD1_20260920`.

XD1 deliberately excludes Voynich-specific q/i rules, the named hard-zero inventory, Currier, Davis hand and PAGE_CODEBOOK. It reruns both Voynich and controls through generic models for:

1. within-token memory depth;
2. SPACE versus physical LINE_BREAK attenuation;
3. previous-token morphology carryover;
4. incremental exact previous-token identity;
5. lagged exact-form recurrence geometry;
6. paired abbreviation intervention where available;
7. writer effects only under crossed support.

There is **no aggregate similarity score and no nearest-corpus winner**.

The first external run order was frozen before outcome inspection:
Voynich reference → Nuremberg 2–5 → Nuremberg abbreviation pair → CREMMA → Gaskell–Bowern human pseudo-writing → Bowern algorithmic transformations.

## Audit conclusion

The archive contains substantially more genuine comparison material than a fresh start would suggest, but older project labels such as “control”, “validation”, “language battery” and “cipher comparison” cannot be trusted at face value.

The strongest path forward is therefore **salvage without inheritance**:

- retain raw historical data and independently defined experiments;
- preserve narrow negative controls that were fairly run;
- recover high-value manuscript transcriptions;
- discard old target-tuned rankings as authority;
- rerun only the portable, prospectively frozen XD1 measurements.


## Corrected Nuremberg Letterbooks 2–5 XD1 result

**Implementation:** XD1 v2 orderfix, commit `f20a03f3d559bf723e11a25be800cd613a22993e`  
**Source:** Zenodo record 13881575, labels.zip SHA256 `59e5264acb4546477567e78c8b3d444c472f1a0a5256ee0ee7d0407a70904652`  
**Ingest check:** 48,322 diplomatic physical lines reproduced exactly; 1,637 physical page images; 436,950 unexpanded and 436,958 expanded tokens.  
**P2 verifier:** HF `6aafb6be51992417dfccca90`  
**P5 verifier:** HF `6aafb6c252d0dbd7f1d73a5d`

The earlier XD1 v1 sequence outputs remain invalidated by XD1_CORR_001/002 and are not used.

### P2 — SPACE versus physical LINE_BREAK

Primary unexpanded representation, alpha=64:

- Voynich: effect +0.1321147960 bits, null SD 0.0133708000, ratio 9.88.
- Nuremberg: effect +0.0979685056 bits, null SD 0.0042264973, ratio 23.18.
- VMS minus Nuremberg: +0.0341462905; combined null SD 0.0140228946; ratio 2.44.

Nuremberg therefore exhibits the same strong qualitative attenuation of local edge dependence across physical line breaks. Presence of SPACE>LINE_BREAK is **not** Voynich-diagnostic by itself. The magnitude is larger in Voynich under this frozen model.

The paired expanded Nuremberg representation remains strongly positive at alpha=64: +0.0926513669, null SD 0.0043577654, ratio 21.26. Abbreviation expansion therefore does not remove the effect.

Cross-correspondence physical adjacencies were excluded from Nuremberg P2 via the v2 segment reset.

### P5 — page-level exact recurrence profile

Primary unexpanded representation:

| Band | Voynich effect / null SD | Nuremberg effect / null SD | Direction |
|---|---|---|---|
| lag 1 | +0.0007182 / 0.0006011 = 1.19, unresolved | -0.0074993 / 0.0001444 = 51.92 | divergent |
| lags 2–5 | +0.0076466 / 0.0010183 = 7.51 | -0.0023876 / 0.0002846 = 8.39 | opposite |
| lags 6–16 | +0.0042893 / 0.0014352 = 2.99 | +0.0113518 / 0.0003563 = 31.86 | same sign, different magnitude |
| lags 17–64 | -0.0045004 / 0.0018541 = 2.43 | +0.0116285 / 0.0005116 = 22.73 | opposite |

Independent-difference ratios using sqrt(sd_VMS² + sd_Nuremberg²) are approximately 13.29, 9.49, 4.78 and 8.39 for the four bands respectively.

The expanded representation preserves the same Nuremberg sign profile:
lag1 negative; lags2–5 negative; lags6–16 positive; lags17–64 positive.

This is currently the clearest XD1 distinction between Voynich and the near-period professional scribal control.

### Required caution

P5 is deliberately page-level under the frozen XD1 definition. Nuremberg physical pages may contain multiple correspondence segments. Therefore the long-range Nuremberg recurrence profile can partly reflect page composition across letters. This does **not** invalidate the registered page-level result, but a segment-aware recurrence sensitivity is required before the P5 difference is interpreted as a production mechanism.

Accordingly:

- P2 presence is downgraded as a diagnostic feature.
- Generic short-range recurrence is already known from CREMMA and remains non-diagnostic in isolation.
- The **joint recurrence geometry** remains a candidate discriminator, not a mechanism identification.
- No historical/cipher/generative mechanism is promoted from the Nuremberg result.


## Recurrence segmentation falsifier

The page-level P5 contrast was challenged for two obvious unit-composition confounds:

1. Nuremberg physical pages can contain multiple independent correspondence records.
2. Voynich pages contain multiple natural paragraphs.

The registered P5 statistic and 200-permutation within-block token-multiset null were therefore rerun with only the **block unit** changed.

### Frozen sensitivity sources

- VMS paragraphs: `research/xd1_vms_paragraphs_20260920.json`, commit `91198829b83e2941f78eff62b20e91cc9618e7ee`; 1,694 paragraph units, 29,886 words, source SHA `bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc`. This is a sensitivity subset, not a canonical replacement for the 34,087-event v03 corpus.
- Nuremberg correspondence units: 3,176 diplomatic XML records, source SHA `59e5264acb4546477567e78c8b3d444c472f1a0a5256ee0ee7d0407a70904652`.
- Exact sensitivity workflow: GitHub Actions run `35507147040`, result SHA `60f249eba0593671de2d5d1f8fe7f545627aac513c6313b1a96858a2ca00a3f7`.
- Independent vectorized implementation: run `35507469254`, result SHA `849bddc1d6ac968b1ff7f43714aeda848672aebd596346a1c5dd189d07a23f06`; it reproduces all signs and closely matching Monte Carlo null estimates.

### Results

| Band | VMS paragraph effect / null SD | Nuremberg correspondence effect / null SD | Interpretation |
|---|---|---|---|
| lag 1 | +0.000540 / 0.000674 = 0.80 | -0.007647 / 0.000185 = 41.27 | VMS unresolved; Nuremberg strong deficit |
| lags 2–5 | +0.002308 / 0.000907 = 2.54 | -0.001837 / 0.000286 = 6.43 | opposite signs |
| lags 6–16 | -0.003916 / 0.001780 = 2.20 | +0.010426 / 0.000422 = 24.68 | opposite signs; VMS page-level positive does not survive paragraph restriction |
| lags 17–64 | -0.010754 / 0.002499 = 4.30 | +0.010147 / 0.000555 = 18.27 | opposite signs |

The VMS-minus-Nuremberg differences relative to combined null SD are approximately 11.71, 4.36, 7.84 and 8.17 respectively.

### Consequence

The multi-correspondence-page confound does **not** explain the Nuremberg/Voynich recurrence contrast.

However, the VMS lag-6–16 page-level excess is not segmentation-robust and is removed from the robust candidate signature.

The surviving candidate signature is narrower:

- lag 1: no resolved VMS excess;
- lag 2–5: VMS excess versus Nuremberg deficit;
- lag 17–64: VMS deficit versus Nuremberg excess.

This remains a structural discriminator against the tested near-period scribal control, **not a mechanism identification**.

A remaining fairness issue is block length: VMS paragraphs have median length 9 tokens while Nuremberg correspondence records have median length 121. A length-stratified test was frozen before interpreting the recurrence contrast mechanistically.
