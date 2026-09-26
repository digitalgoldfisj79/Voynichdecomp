# Frozen protocol — historical notation calibration v0.4

**Frozen:** 2026-07-24, before downloading or inspecting the Ammerbach annotations in this phase.  
**Repository:** `digitalgoldfisj79/Voynichdecomp`  
**Branch:** `experiment/voynich-notation-historical-calibration-v0.4-20260724`  
**Status:** external calibration protocol; Ammerbach is not period evidence for the fifteenth-century Voynich Manuscript.

## 1. Question

Can a surface-only, transcription-invariant model recognise historical musical notation against matched language, cipher and pseudo-text controls, and—only if that external gate passes—how does the sealed Voynich feature table score?

This phase tests recognisability, not semantic recovery. No glyph is assigned a pitch, duration, action or meaning.

## 2. External corpora

### 2.1 Organ tablature calibration

Use the public **AmmerbachReal** data set from the University of Marburg / DeepTab project:

- 1,200 annotated staves from *Ein new künstlich Tabulaturbuch* (1575);
- 1,200 annotated staves from *Orgel oder Instrument Tabulaturbuch* (1583);
- paired duration/special and pitch/rest label sequences.

The data set is later than Voynich and printed rather than handwritten. It is used only to establish that the pipeline can recognise a genuine alphabetic tablature family.

### 2.2 Neumatic calibration

Use every parseable Aquitanian and square-notation `.gabc` file in the public ECHOES `GABCtoMEI` repository. Group all windows by source manuscript so derived controls from the same manuscript cannot cross train/test folds.

### 2.3 Matched controls

Construct without target tuning:

1. Latin lyric words from the same GABC files;
2. monoalphabetic substitutions of those lyrics;
3. character-shuffled notation preserving event lengths;
4. first-order character-Markov surfaces preserving event lengths;
5. synthetic weak-state procedural packets, treated as a separate family rather than historical notation.

## 3. Surface representation

All sources are reduced to linear event strings. No image, staff-position coordinate, modern pitch interpretation or editorial semantic reconstruction enters the classifier.

For Ammerbach, evaluate three representations:

1. `paired`: duration/special and pitch/rest labels paired by sequence position;
2. `pitch`: pitch/rest sequence only;
3. `flattened`: duration/special followed by pitch/rest.

Representation selection uses external cross-validation only. Select by ensemble historical-notation ROC AUC, then calibrated balanced accuracy, then the fixed tie order `paired > pitch > flattened`.

Within every 48-event window, map characters to a deterministic frequency-rank alphabet before feature extraction. This removes source-specific glyph identities and transcription alphabets while retaining equality, order, event boundaries and frequency rank.

## 4. Features

The sealed 41-dimensional surface feature vector contains only:

- event-length distribution and lag-1 dependence;
- symbol and event inventory measures;
- type/token, hapax and repetition measures;
- character and event conditional entropies through order two;
- initial/final-symbol entropy and mutual information;
- positional character dependence;
- adjacent prefix/suffix overlap;
- within-event character diversity;
- non-semantic character-class diagnostics.

Window size is 48 events, stride 24, minimum 48.

## 5. Splitting and leakage controls

Use `StratifiedGroupKFold` with at most five folds. The group is the source manuscript or Ammerbach book. GABC notation, lyrics, cipher and null transformations from one manuscript always remain in the same fold.

No random event or stave split is admissible for the primary result.

## 6. Models

Fit two frozen classifiers:

1. standardised class-balanced logistic regression, `C=1`;
2. class-balanced random forest, 500 trees, maximum depth 10, minimum leaf size 3.

Use the mean probability as the ensemble.

### Broad task

Positive:

- organ tablature;
- Aquitanian neumes;
- square neumes.

Negative:

- Latin / monoalphabetic substitution;
- shuffled and Markov surface nulls.

Synthetic procedural packets are excluded from the broad calibration task.

### Family task

Classes:

- organ tablature;
- Aquitanian neumes;
- square neumes;
- language or monoalphabetic substitution;
- surface null;
- synthetic procedural packets.

## 7. Frozen gates

The sealed Voynich table may be passed to the classifiers only when all broad conditions hold for the externally selected representation:

1. logistic ROC AUC >= 0.80;
2. random-forest ROC AUC >= 0.80;
3. ensemble balanced accuracy >= 0.70 at the externally calibrated threshold;
4. held-out Ammerbach recall >= 0.60.

The threshold is chosen from external cross-validated predictions to maximise recall subject to false-positive rate <= 0.10.

Historical family labels may be interpreted only if the grouped multiclass macro-F1 is >= 0.55.

Failure means `ABSTAIN`; thresholds are not revised.

## 8. Sealed Voynich target

The target contains 346 deterministic windows from 226 folios (the first and, where available, last non-overlapping 48-token window per folio). It was generated before external Ammerbach inspection from 37,465 Voynich tokens.

- source SHA-256: `dbf87cf5525e065da881b06a26c9d411543ff8ef3f5f8e15a9e4b557808f1174`
- sealed feature CSV SHA-256: `c94ad2c96e76ba89efe67f776cfc0b4d820b2b9c21824d44cfc43323a05bb1f1`

The target table carries folio and section only for post-prediction aggregation. These fields are not classifier inputs.

## 9. Interpretation boundary

A positive historical-notation probability is not a decipherment and does not identify music. The classifier can establish resemblance to the calibrated surface family only. A procedural-synthetic assignment supports structured packet production but remains compatible with recipes, mnemonics, cipher machinery, tables and pseudo-text.

Ammerbach results are calibration evidence, not fifteenth-century provenance evidence.
