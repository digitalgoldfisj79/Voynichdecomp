# Compression-Transfer Distance Programme v0.1

**Internal identifier:** `compression-transfer-v0.1`  
**Date frozen:** 2026-07-30  
**Repository:** `digitalgoldfisj79/Voynichdecomp`  
**Research branch:** `experiment/compression-transfer-v0.1-20260730`  
**Base commit:** `f86759d6651fdde135c427682c79a07ef8df38f9`  
**Voynich status at programme creation:** sealed.

## 1. Scientific question

Can directional compression transfer and compression-based distance provide a calibrated, representation-robust measure of predictive similarity between unknown texts, known languages, historical cipher surfaces, notation-like systems, structured non-message generation and human meaningless writing?

The programme does **not** assume that compression proximity identifies plaintext, language or mechanism. It tests whether the method can earn bounded inferential use through known-text, fresh-key and generator-disjoint calibration.

## 2. Relation to prior work

The repository already contains:

- character entropy and conditional entropy measures;
- held-out n-gram codelength and MDL source-transfer tests;
- a one-document zlib compression ratio among the generator-comparison metrics;
- compression-derived features in blind cipher-family recognition.

Those analyses measure internal compressibility or model-specific predictive fit. They did not run the pairwise construction inspired by *Language Trees and Zipping*: conditioning a compressor on reference corpus A and measuring the incremental cost of a held-out probe from B, with B's own-source cost as the baseline.

This programme is therefore a new comparator, not a relabelling of the previous entropy work.

## 3. Registered metrics

Let `C_K(x)` be the compressed byte length of byte string `x` under compressor K. A fixed binary boundary is inserted between corpora.

### 3.1 Internal compression ratio

`R_K(x) = C_K(x) / |x|`.

This is descriptive only and is not a pairwise source-distance measure.

### 3.2 Directional conditional cost

For reference A and held-out probe b:

`H_K(b | A) = 8 [C_K(A || boundary || b) - C_K(A || boundary)] / |b|`

in compressed bits per probe byte.

### 3.3 Self-normalized directional excess

If `B_train` is an independent training reference from the same registered source as b:

`E_K(A -> b) = H_K(b | A) - H_K(b | B_train)`.

Positive values mean A predicts b less efficiently than its own-source reference. Negative values are retained and reported; they are not clipped.

### 3.4 Normalized compression distance

For corpus references x and y:

`NCD_K(x,y) = [C_K(x || boundary || y) - min(C_K(x),C_K(y))] / max(C_K(x),C_K(y))`.

Real compressors are order-sensitive. The programme therefore preserves both concatenation orders and uses their arithmetic mean as the registered symmetric value. The absolute order gap is a required diagnostic.

## 4. Registered compressors

### Mandatory

1. `zlib9`
2. `bz2_9`
3. `lzma9e`

### Optional but preregistered

4. `zstd19`
5. `ppmd6`

An optional compressor may be omitted only because the pinned package is unavailable before formal execution. It may not be introduced after results are seen. Scientific conclusions require agreement from at least two mandatory compressors.

## 5. Registered representations

### Primary

`codepoint_u32_ws`: NFC-normalized text with collapsed whitespace, serialized as fixed-width unsigned 32-bit Unicode codepoints. This prevents variable UTF-8 byte width from acting as an uncharged cross-script predictor.

### Sensitivity panel

- `surface_utf8`: normalized UTF-8 surface with spaces retained;
- `codepoint_u32_nospace`: fixed-width codepoints with whitespace removed;
- `token_recurrence_u32`: document-local first-occurrence token IDs;
- `char_recurrence_u32`: document-local first-occurrence non-space character IDs;
- `token_length_u32`: token-length sequence only.

Representation results are reported separately. They may not be selected post hoc or averaged into one opaque score. Recurrence representations are structural controls and cannot support lexical or source-language claims.

## 6. Data and split rules

1. Every source file is identified in a CSV manifest by document, corpus, class, language, family, licence and SHA-256.
2. Splits are document-grouped. No document, duplicate edition or overlapping excerpt may occur in more than one split.
3. Known-source corpora use train/development/locked-test partitions. Formal test documents remain untouched until configuration and gates are frozen.
4. Historical cipher test items use fresh independently sampled keys. Reused-key or shared-code-pool items are positive controls only.
5. Generator tests are generator-disjoint: locked-test generator parameterisations and latent documents are absent from development.
6. Voynich folios and representations remain in split `sealed` until the opening rule is satisfied.
7. All corpus transformations are deterministic and hash-recorded.

## 7. Probe and reference geometry

Primary formal probe length: 4,096 representation units.  
Registered length sensitivity: 1,024, 2,048 and 4,096 units.  
Maximum registered reference tail: 131,072 units per corpus.  
Primary probes are non-overlapping. Overlapping windows may be used only as an explicitly labelled sensitivity analysis and may not increase the effective sample size in confidence intervals.

The reference and probe must come from different documents. A probe is never compressed after an earlier probe from the same locked document.

## 8. Null and leakage controls

Each eligible source panel must include:

- non-space character shuffle, preserving character counts and whitespace positions;
- token shuffle, preserving the token multiset;
- fixed block shuffle;
- at least one trained Markov or grammar control fitted on training material only;
- length-matched unrelated corpora;
- duplicate and near-duplicate screening;
- author/work holdout where metadata permit it.

A method that classifies shuffled controls almost as well as intact texts is treated as relying mainly on alphabet or unigram composition.

## 9. Stage 0 — implementation qualification

Required before corpus acquisition is treated as scientific execution:

1. unit tests for directional cost, excess cost, fixed-width encoding and NCD order retention;
2. deterministic synthetic fixture with four distinct Markov sources;
3. 100% top-1 recovery on the primary surface representation for all mandatory compressors used in the smoke run;
4. independent arithmetic validator passes every emitted row;
5. machine-readable compressor versions and parameters recorded.

Stage 0 does not validate the scientific method.

## 10. Stage 1 — known-source and language calibration

### 10.1 Panel

At least eight source-language classes, including medieval or early-modern Latin and at least one non-Latin script. Each class requires:

- at least 12 independently attributable documents;
- at least two works and, where possible, two authors;
- sufficient material for 4,096-unit probes;
- documented licence and provenance.

### 10.2 Primary question

Does a held-out document receive lower conditional cost from its own source-language reference than from competing references under document and author shift?

### 10.3 Locked gates

At 4,096 units on `codepoint_u32_ws`:

1. macro top-1 language accuracy at least 0.80 for at least two mandatory compressors;
2. consensus accuracy conditional on acceptance at least 0.85;
3. consensus coverage at least 0.80;
4. worst-language recall at least 0.50;
5. median own-source rank equal to 1;
6. median NCD concatenation-order gap no greater than 0.05;
7. leave-author-out accuracy drop no greater than 0.15 absolute;
8. unigram-shuffled control top-1 accuracy no greater than 0.50;
9. no primary conclusion reverses between 2,048 and 4,096 units.

Failure closes source-language use of the method. It does not prohibit Stage 2 surface-class calibration, but Voynich source-language comparison remains permanently blocked under v0.1.

## 11. Stage 2 — cipher, notation and generator calibration

### 11.1 Required classes

- ordinary plaintext;
- fresh-key monoalphabetic substitution;
- fresh-key homophonic substitution;
- nomenclator-style substitution;
- substitution plus transposition;
- periodic or changing-alphabet family represented by the existing Family P construction;
- notation-like or procedural corpora, including the existing CoReMA control where licensing permits;
- human-produced meaningless writing;
- structured generators from the repository;
- matched shuffled and Markov controls.

### 11.2 Two separate targets

**Surface-class recognition:** identify bounded surface families without claiming source recovery.

**Source transfer through transformation:** identify the source language of fresh-key cipher text. This is the stronger claim and has separate gates.

### 11.3 Surface-class locked gates

1. macro top-1 family accuracy at least 0.80;
2. worst-class recall at least 0.60;
3. consensus coverage at least 0.75;
4. generator-disjoint accuracy at least 0.75;
5. matched-null false-positive rate no greater than 0.05;
6. agreement in at least four registered representations, including the primary representation and one recurrence representation;
7. agreement from at least two mandatory compressors.

### 11.4 Source-transfer locked gates

1. fresh-key source-language accuracy at least 0.70 overall;
2. at least 0.50 in every cipher family designated required before execution;
3. no reused-key positive control may be counted toward the gate;
4. the source-language margin must remain positive under author and document holdout;
5. the result must survive correction across the registered family-by-representation contrasts.

If surface-class gates pass but source-transfer gates fail, Stage 3 may report only surface compatibility. It may not report source-language proximity as evidence about plaintext.

## 12. Stage 3 — sealed Voynich analysis

Voynich data may be opened only after:

1. the Stage 1 source calibration decision is frozen;
2. the Stage 2 surface-class decision is frozen;
3. all code, manifests, thresholds, tree rules and output schemas are committed;
4. a clean-clone smoke run reproduces the frozen payload hash.

### 12.1 Target units

- folio is the minimum independent unit;
- Currier A/B and section metadata are used only for preregistered stratification;
- no folio contributes to both reference and probe roles within a comparison;
- multiple EVA/transcription variants, if used, are separate sensitivity conditions;
- unknown or uncertain glyphs are retained under a documented deterministic policy.

### 12.2 Registered outputs

- directional excess matrices with bootstrap intervals;
- NCD matrices and UPGMA trees by compressor and representation;
- cluster-support values;
- nearest family and nearest source reference only when consensus rules pass;
- Currier and section within-target distances;
- explicit abstention state.

### 12.3 Allowed decisions

- `NON_IDENTIFIABLE`
- `SURFACE_COMPATIBILITY_ONLY`
- `SOURCE_FAMILY_COMPATIBILITY_ONLY`

`SOURCE_FAMILY_COMPATIBILITY_ONLY` is available only if Stage 1 and Stage 2 source-transfer gates both passed. It is not a plaintext-language identification.

### 12.4 Forbidden claims

- decipherment;
- semantic reading;
- plaintext-language identification;
- proof that plaintext exists;
- rejection of untested historical cipher classes;
- selection of a preferred representation after seeing the target result.

## 13. Consensus and uncertainty

1. Primary classification is minimum conditional bits per probe byte.
2. Consensus is a preregistered vote across compressor × representation cells.
3. Ties abstain.
4. A Voynich family claim requires agreement from at least four representations and two mandatory compressors.
5. Bootstrap resampling is at document/folio level, not overlapping-window level.
6. Cluster support must be at least 0.75 for any named grouping.
7. Family-margin intervals must exclude zero.
8. Registered multiple comparisons use Holm correction with familywise alpha 0.05.

## 14. Stop rules

- Failed Stage 0: repair implementation only; no scientific result exists.
- Failed Stage 1: no source-language interpretation under v0.1.
- Failed Stage 2 surface gate: Voynich remains sealed and the programme closes.
- Passed surface gate but failed source-transfer gate: Voynich may be opened only for surface-class proximity.
- Failed representation or compressor consensus on Voynich: `NON_IDENTIFIABLE`.
- No threshold, compressor, representation, reference corpus, chunk length or exclusion may be changed after a locked result without a new version and untouched data.

## 15. Required artefacts

Each formal stage emits:

- frozen config and manifest;
- corpus and code SHA-256 records;
- compressor version record;
- row-level directional observations;
- pairwise NCD rows;
- consensus decision rows;
- Newick trees;
- independent arithmetic validation;
- scientific-payload hash separated from runtime metadata;
- result report with exact claim boundary.

## 16. Interpretation

Compression transfer can reveal shared reusable structure. It does not identify what generated that structure. Similarity may arise from language, orthography, symbol inventory, renderer, key schedule, scribal convention, genre, notation or a generator trained on related material. The programme is therefore designed to earn narrow claims through calibrated transfer, not to convert a visually attractive language tree into a decipherment claim.
